# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Minimal NVSHMEM device-API RMA demo (CuTe DSL), runnable in this environment.

This is the ``put`` / ``get`` example from the NVSHMEM CuTe DSL device docs
  https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/device/cute/index.html
adapted so it runs in /opt/tiger/cutlass/.venv, which:
  * has no mpi4py  -> bootstrap with torchrun + a UID broadcast instead of MPI;
  * ships CUTLASS-DSL 4.5.2 whose compiled kernels take no ``stream=`` kwarg;
  * ships nvshmem4py 0.3.0 that imports the older ``dtype`` / ``Constexpr`` names
    from ``cutlass.cute.typing`` (added back below as a small shim).

Each PE writes its value into the *right* neighbour's symmetric ``remote_buf``
(ring put), everyone barriers, then reads it back into ``dst`` (get). After the
ring, every PE's ``remote_buf`` holds its *left* neighbour's value.

Run (one process per GPU):

    cd /opt/tiger/cutlass
    .venv/bin/torchrun --standalone --nproc-per-node 4 \
        examples/python/CuTeDSL/cute/blackwell/kernel/distributed/nvshmem_rma_demo.py
"""

import os

import numpy as np
import torch
import torch.distributed as dist

import cutlass
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

try:
    from cuda.core import Device
except ImportError:
    from cuda.core.experimental import Device
from cuda.pathfinder import load_nvidia_dynamic_lib

# --- compat shim: nvshmem4py 0.3.0 expects names that moved in newer CUTLASS DSL
import cutlass.cute.typing as _t
if not hasattr(_t, "dtype"):
    _t.dtype = _t.Numeric
if not hasattr(_t, "Constexpr"):
    _t.Constexpr = cutlass.Constexpr

load_nvidia_dynamic_lib("nvshmem_host")
import nvshmem.core
import nvshmem.core.device.cute as nvshmem_cute
import nvshmem.core.interop.cute as nvshmem_cute_interop


@cute.kernel
def rma_kernel(src: cute.Tensor, dst: cute.Tensor, remote_buf: cute.Tensor, pe: cutlass.Int32):
    # Push src -> remote_buf on PE `pe` (CTA-scoped, blocking).
    nvshmem_cute.put_block(remote_buf, src, pe)
    # Make sure every PE's put has landed before anyone reads.
    nvshmem_cute.barrier_all_block()
    # Read remote_buf on PE `pe` back into our local dst.
    nvshmem_cute.get_block(dst, remote_buf, pe)
    nvshmem_cute.barrier_all_block()


@cute.jit
def rma_launcher(src, dst, remote_buf, pe):
    rma_kernel(src, dst, remote_buf, pe).launch(grid=[1, 1, 1], block=[1, 1, 1])


def main():
    # ---- bootstrap NVSHMEM via torchrun + UID broadcast (no MPI needed) ------- #
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dev = Device(local_rank)
    dev.set_current()

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    world_size = dist.get_world_size()

    uid = nvshmem.core.get_unique_id(empty=(local_rank != 0))
    uid_tensor = torch.from_numpy(uid._data.view(np.uint8).copy()).cuda()
    dist.broadcast(uid_tensor, src=0)
    dist.barrier()
    uid._data[:] = uid_tensor.cpu().numpy().view(uid._data.dtype)
    nvshmem.core.init(
        device=dev, uid=uid, rank=local_rank, nranks=world_size, initializer_method="uid"
    )

    me = nvshmem.core.my_pe()
    n_pes = nvshmem.core.n_pes()
    right = (me + 1) % n_pes
    left = (me - 1 + n_pes) % n_pes

    # ---- symmetric buffers (torch-backed NVSHMEM memory, easy to init/verify) - #
    src = nvshmem.core.tensor((1,), dtype=torch.int32)
    dst = nvshmem.core.tensor((1,), dtype=torch.int32)
    remote_buf = nvshmem.core.tensor((1,), dtype=torch.int32)
    src.fill_(me + 100)
    dst.fill_(-1)
    remote_buf.fill_(-1)

    # ---- compile (links libnvshmem_device.bc) and launch ---------------------- #
    compiled_fn, nvshmem_kernel = nvshmem_cute_interop.cute_compile_helper(
        rma_launcher, from_dlpack(src), from_dlpack(dst), from_dlpack(remote_buf), right
    )

    dist.barrier()
    compiled_fn(from_dlpack(src), from_dlpack(dst), from_dlpack(remote_buf), right)
    torch.cuda.synchronize()
    dist.barrier()

    # After the ring put, our remote_buf holds the LEFT neighbour's src value.
    got_remote = int(remote_buf.cpu().item())
    got_dst = int(dst.cpu().item())
    exp_remote = left + 100
    ok = got_remote == exp_remote

    for r in range(n_pes):
        dist.barrier()
        if r == me:
            print(
                f"[PE {me}] put {me + 100} -> PE {right}; "
                f"remote_buf={got_remote} (from PE {left}, expect {exp_remote}) "
                f"{'OK' if ok else 'MISMATCH'}; get-back dst={got_dst}",
                flush=True,
            )
    dist.barrier()

    flag = torch.tensor([0 if ok else 1], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if me == 0:
        print("\nRESULT: " + ("PASSED" if flag.item() == 0 else "FAILED"), flush=True)

    # ---- cleanup ------------------------------------------------------------- #
    nvshmem.core.free_tensor(src)
    nvshmem.core.free_tensor(dst)
    nvshmem.core.free_tensor(remote_buf)
    nvshmem.core.library_finalize(nvshmem_kernel)
    nvshmem_cute_interop.cleanup_cute()
    nvshmem.core.finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
