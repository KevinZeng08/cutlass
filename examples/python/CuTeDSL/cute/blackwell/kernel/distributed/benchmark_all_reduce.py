# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Orchestrator that benchmarks the four intra-node (NVLink-domain) all-reduce
# examples in this directory and prints a side-by-side comparison.
#
#   1. simple   - all_reduce_simple.py           : SIMT peer ld.global + register accumulate
#   2. tma      - all_reduce_tma.py              : TMA G2S load + register reduce + multicast TMA store
#   3. multimem - all_reduce_two_shot_multimem.py: multimem.ld_reduce + multimem.st (NVLS in-network)
#   4. lamport  - all_reduce_one_shot_lamport.py : Lamport flag-based lock-free one-shot
#
# Each (impl, size) is launched as an *independent* `torchrun` subprocess, so the
# runs cannot interfere with one another (no shared NVSHMEM state, no cross-rank
# desync). The script parses each run's "Kernel execution time" / "Achieved
# memory throughput" lines and aggregates them into latency / throughput tables.
#
# This is a plain host-side script -- run it directly (NOT under torchrun):
#
#   python examples/python/CuTeDSL/cute/blackwell/kernel/distributed/benchmark_all_reduce.py
#
#   # custom sweep / subset
#   python .../benchmark_all_reduce.py --sizes 1024,2048,4096,8192 --impls simple,tma,multimem,lamport
#   python .../benchmark_all_reduce.py --nproc-per-node 8 --iterations 100

import os
import re
import sys
import argparse
import subprocess

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# impl short name -> example script filename
IMPLS = {
    "simple": "all_reduce_simple.py",
    "tma": "all_reduce_tma.py",
    "multimem": "all_reduce_two_shot_multimem.py",
    "lamport": "all_reduce_one_shot_lamport.py",
}

_TIME_RE = re.compile(r"Kernel execution time:\s*([0-9.]+)\s*ms")
_BW_RE = re.compile(r"Achieved memory throughput:\s*([0-9.]+)\s*GB/s")


def run_one(impl_file, M, N, nproc, master_addr, master_port, warmup, iters, timeout):
    """Launch a single torchrun benchmark and parse (time_ms, gbps) from stdout."""
    cmd = [
        "torchrun",
        f"--nproc-per-node={nproc}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        os.path.join(_THIS_DIR, impl_file),
        "--M", str(M),
        "--N", str(N),
        "--benchmark",
        "--warmup_iterations", str(warmup),
        "--iterations", str(iters),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=_THIS_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, None, "timeout"

    out = proc.stdout + "\n" + proc.stderr
    t_match = _TIME_RE.search(out)
    bw_match = _BW_RE.search(out)
    if proc.returncode != 0 and t_match is None:
        # Surface a short tail of the log to help debugging.
        tail = "\n".join(out.strip().splitlines()[-8:])
        return None, None, f"rc={proc.returncode}\n{tail}"

    time_ms = float(t_match.group(1)) if t_match else None
    gbps = float(bw_match.group(1)) if bw_match else None
    if time_ms is None:
        return None, None, "no timing in output"
    return time_ms, gbps, None


def _fmt_table(title, sizes, impls, cell_getter, unit):
    lines = []
    lines.append("")
    lines.append(f"### {title} ({unit})")
    col_w = 13
    header = f"{'M=N':>8}" + "".join(f"{name:>{col_w}}" for name in impls)
    lines.append(header)
    lines.append("-" * len(header))
    for s in sizes:
        row = f"{s:>8}"
        # determine best (min time / max bw) for highlighting
        vals = {name: cell_getter(name, s) for name in impls}
        numeric = {k: v for k, v in vals.items() if isinstance(v, float)}
        best_key = None
        if numeric:
            if unit == "ms":
                best_key = min(numeric, key=numeric.get)
            else:
                best_key = max(numeric, key=numeric.get)
        for name in impls:
            v = vals[name]
            if isinstance(v, float):
                txt = f"{v:.4f}" if unit == "ms" else f"{v:.1f}"
                if name == best_key:
                    txt = "*" + txt
            else:
                txt = str(v)
            row += f"{txt:>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark & compare intra-node all-reduce examples via torchrun"
    )
    parser.add_argument(
        "--sizes",
        default="1024,2048,4096,8192",
        type=str,
        help="comma-separated square sizes (M==N), e.g. 1024,2048,4096,8192",
    )
    parser.add_argument(
        "--impls",
        default="simple,tma,multimem,lamport",
        type=str,
        help="comma-separated subset of: simple,tma,multimem,lamport",
    )
    parser.add_argument("--nproc-per-node", default=4, type=int)
    parser.add_argument("--master-addr", default="127.0.0.1", type=str)
    parser.add_argument("--master-port", default=16989, type=int)
    parser.add_argument("--warmup_iterations", default=5, type=int)
    parser.add_argument("--iterations", default=50, type=int)
    parser.add_argument(
        "--timeout", default=600, type=int, help="per-run timeout in seconds"
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    impls = [s.strip() for s in args.impls.split(",") if s.strip()]
    for name in impls:
        if name not in IMPLS:
            print(f"error: unknown impl '{name}' (valid: {', '.join(IMPLS)})")
            sys.exit(1)

    # results[name][size] = (time_ms, gbps) or error string
    times = {name: {} for name in impls}
    bws = {name: {} for name in impls}

    # Use a distinct port per run to avoid TIME_WAIT collisions between launches.
    port = args.master_port

    print("=" * 72)
    print(
        f"Sweep: impls={impls} sizes={sizes} nproc={args.nproc_per_node} "
        f"warmup={args.warmup_iterations} iters={args.iterations}"
    )
    print("=" * 72)

    for name in impls:
        impl_file = IMPLS[name]
        for s in sizes:
            print(f"[run] {name:<9} M=N={s:<6} (port {port}) ...", end="", flush=True)
            time_ms, gbps, err = run_one(
                impl_file,
                s,
                s,
                args.nproc_per_node,
                args.master_addr,
                port,
                args.warmup_iterations,
                args.iterations,
                args.timeout,
            )
            port += 1
            if err is not None:
                times[name][s] = "FAIL"
                bws[name][s] = "FAIL"
                print(f" FAILED ({err.splitlines()[0]})")
            else:
                times[name][s] = time_ms
                bws[name][s] = gbps
                print(f" {time_ms:.4f} ms  {gbps:.1f} GB/s")

    print("\n" + "=" * 72)
    print(f"Intra-node all-reduce comparison | fp32 | world_size={args.nproc_per_node}")
    print("(* marks the best impl for each size; lower ms / higher GB/s is better)")
    print("=" * 72)
    print(_fmt_table("Latency", sizes, impls, lambda n, s: times[n][s], "ms"))
    print(_fmt_table("Throughput", sizes, impls, lambda n, s: bws[n][s], "GB/s"))
    print("=" * 72)


if __name__ == "__main__":
    main()
