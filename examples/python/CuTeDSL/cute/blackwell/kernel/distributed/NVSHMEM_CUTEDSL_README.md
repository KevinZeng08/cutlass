# NVSHMEM Device API + CuTe DSL: capabilities, demos, and porting notes

This document covers the **NVSHMEM device-side Python bindings for the CuTe DSL**
(`nvshmem.core.device.cute`), the two runnable demos in this directory, and the
full set of pitfalls hit while making them work in `/opt/tiger/cutlass/.venv`.

Reference docs:
https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/device/cute/index.html

---

## 1. What NVSHMEM device APIs give you

NVSHMEM exposes a **PGAS (Partitioned Global Address Space)** model on top of GPUs:

- A **PE (Processing Element)** is a participant in the job. In these demos
  **1 PE = 1 process = 1 GPU**. PEs are numbered `0 .. n_pes()-1`.
- **Symmetric memory**: a buffer allocated with the same layout on *every* PE.
  Any PE can read/write the corresponding buffer on a *remote* PE by passing
  `(buffer, pe)` — no explicit send/recv handshake on the remote side.
- **GPU-initiated communication**: the data movement is issued *from inside a
  CUDA kernel*, not from the host. This is the key difference vs NCCL/MPI and the
  whole point of the device API.

### Available device primitives (CuTe DSL)

All callable from inside a `@cute.kernel`, in `thread` / `block` / `warp` scopes
(suffix `_block` / `_warp`; bare name = thread scope), blocking or non-blocking
(`_nbi`):

| Family | Functions |
|---|---|
| **RMA** | `put` / `get`, `put_nbi` / `get_nbi`, scalar `p` / `g`, `put_signal` |
| **Signalling** | `signal_op`, `signal_wait` (+ `put_signal*` writes data then signals) |
| **Collectives** | `barrier` / `barrier_all`, `sync` / `sync_all`, `reduce` (all-reduce), `reducescatter`, `fcollect` (all-gather), `broadcast`, `alltoall` |
| **Reduction ops** | `sum`, `prod`, `min`, `max`, `and`, `or`, `xor` |
| **Utilities** | `my_pe`, `n_pes`, `team_my_pe`, `team_n_pes` |
| **Memory (host)** | `tensor()` (symmetric alloc), `get_peer_tensor()`, `get_multicast_tensor()` |

Supported dtypes: int/uint 8/16/32/64, float16/bf16/float32/float64 (per-op).

---

## 2. How the device-API JIT path works

```
@cute.kernel  (calls nvshmem_cute.put_block / reduce_block / ...)
   └─ nvshmem.core.interop.cute.cute_compile_helper()
        ├─ links libnvshmem_device.bc into your kernel
        └─ CUTLASS-DSL libNVVM: read bitcode → verify → emit PTX/SASS
             └─ runtime: nvshmem_host lib + symmetric memory move data over NVLink
```

Two interop modules, keep them straight:

- `import nvshmem.core.interop.cute as nvshmem_cute_interop` — **host side**:
  `tensor(...)` (symmetric alloc), `cute_compile_helper(...)` (compile + link bc).
- `import nvshmem.core.device.cute as nvshmem_cute` — **device side**: the
  primitives you call inside the kernel.

---

## 3. Demos in this directory

| File | What it shows |
|---|---|
| `nvshmem_rma_demo.py` | Minimal ring **`put`/`get`** RMA exchange across PEs + verification. |
| `nvshmem_device_api_demo.py` | Fuller demo: `put_signal`/`signal_wait`, all-reduce (`reduce_block`), all-gather (`fcollect_block`), `barrier_all`. |

### Run

```bash
cd /opt/tiger/cutlass
.venv/bin/torchrun --standalone --nproc-per-node 4 \
    examples/python/CuTeDSL/cute/blackwell/kernel/distributed/nvshmem_rma_demo.py
```

- `--nproc-per-node` = number of PEs / GPUs (2, 4, …).
- `--standalone` is **required** on this host (see pitfall #2).

### Expected output (`nvshmem_rma_demo.py`)

```
[PE 0] put 100 -> PE 1; remote_buf=103 (from PE 3, expect 103) OK; get-back dst=100
[PE 1] put 101 -> PE 2; remote_buf=100 (from PE 0, expect 100) OK; get-back dst=101
[PE 2] put 102 -> PE 3; remote_buf=101 (from PE 1, expect 101) OK; get-back dst=102
[PE 3] put 103 -> PE 0; remote_buf=102 (from PE 2, expect 102) OK; get-back dst=103
RESULT: PASSED
```

Logic: each PE pushes its value to the **right** neighbour's symmetric
`remote_buf`; after a `barrier_all`, every PE's `remote_buf` holds its **left**
neighbour's value (a ring `0→1→2→3→0`).

---

## 4. Environment

| Component | Version |
|---|---|
| GPUs | 4 × GB200 (sm_100, Blackwell) |
| Python | 3.12 (`/opt/tiger/cutlass/.venv`) |
| `nvshmem4py-cu13` | 0.3.0 |
| `nvidia-cutlass-dsl` | 4.5.2 (libNVVM = LLVM 20.0.0) |
| `nvidia-nvshmem-cu13` | **3.5.21** (was 3.3.24; see pitfall #4) |
| CUDA | 13 |

> No `mpi4py` and no `pip` in the venv; package management via `uv`.

---

## 5. Pitfalls hit while porting (and fixes)

### Pitfall 1 — bootstrap: no `mpi4py`
The official snippet uses `from mpi4py import MPI` + MPI init, which isn't
available here.
**Fix:** bootstrap with **torchrun + UID broadcast** (NCCL broadcasts the
`get_unique_id()` UID, then `nvshmem.core.init(..., initializer_method="uid")`).

### Pitfall 2 — torchrun: `EADDRINUSE` / "failed to listen on any local address"
Default rendezvous endpoint fails to bind on this host.
**Fix:** add `--standalone` to `torchrun`.

### Pitfall 3 — nvshmem4py vs cutlass-dsl name drift (import-time crash)
```
ImportError: cannot import name 'dtype' from 'cutlass.cute.typing'
ImportError: cannot import name 'Constexpr' from 'cutlass.cute.typing'
NameError: name 'dtype' is not defined
```
nvshmem4py 0.3.0 was written against an older CUTLASS DSL: `dtype` → `Numeric`,
`Constexpr` moved to `cutlass.base_dsl.typing`.
**Fix:** shim before importing the nvshmem device modules:
```python
import cutlass, cutlass.cute.typing as _t
if not hasattr(_t, "dtype"):     _t.dtype = _t.Numeric
if not hasattr(_t, "Constexpr"): _t.Constexpr = cutlass.Constexpr
```

### Pitfall 4 — device bitcode ⟷ libNVVM LLVM version mismatch (the hard one)
`libnvshmem_device.bc` is JIT-linked into your kernel and must (a) be **parseable**
by the local libNVVM bitcode reader and (b) **pass NVVM verification**. The reader
is backward-compatible (reads older bitcode) but **not forward-compatible**.

| nvshmem | bitcode LLVM | result |
|---|---|---|
| 3.3.24 (original) | 18.1.1 | verify fails: `Explicit section marker .text.compute is not allowed` |
| 3.6.5 (doc version) | 20.0.0**git** (newer) | parse fails: `Unknown attribute kind (102) (Producer LLVM20.0.0git, Reader LLVM 20.0.0)` |
| **3.5.21** | no `.text.compute`, ≤ 20.0.0 reader | **works** |

**Root cause:** cutlass-dsl 4.5.2's libNVVM is LLVM 20.0.0. 3.3.24 is too old
(illegal section), 3.6.5 is too new (dev-build LLVM with an unknown attribute).
**Fix:** align the leaf native lib to the compatible window:
```bash
uv pip install --python /opt/tiger/cutlass/.venv/bin/python "nvidia-nvshmem-cu13==3.5.21"
```
(Chose this over bumping cutlass-dsl, which is shared and higher-risk.)
**Diagnostics:** `strings libnvshmem_device.bc | grep -c text.compute`; the error's
`Producer/Reader` fields reveal the LLVM version gap.

> Alternative valid fix: keep nvshmem 3.6.5 and instead **upgrade cutlass-dsl** to a
> build whose libNVVM uses ≥ LLVM 20.0.0git. Pick *one* consistent combination.

### Pitfall 5 — `compiled_fn(..., stream=stream)` rejected
```
DSLRuntimeError: unexpected keyword argument: stream
```
This CUTLASS-DSL build's compiled functions take no `stream=` kwarg.
**Fix:** call without `stream=`, then `torch.cuda.synchronize()` (matches the
`all_reduce_tma.py` sample).

### Minor API differences vs the docs
- `nvshmem.core.init(device=..., ...)` (not `dev=`); `nvshmem.core.finalize()`
  takes **no** args (docs show `dev=`/`stream=`).
- Prefer `kernel(...).launch(grid=..., block=...)` over the docs'
  `kernel[grid, block](...)`.
- For easy host-side init/verify, allocate with
  `nvshmem.core.tensor(..., dtype=torch.int32)` (torch-backed symmetric memory)
  and pass `from_dlpack(t)` into the kernel, rather than `interop.cute.tensor()`
  (which returns a `cute.Tensor` that's awkward to fill/verify on the host).

---

## 6. Minimal working recipe (checklist)

1. Add the typing shim (pitfall 3) at the top of the file.
2. Bootstrap with torchrun + UID; launch with `--standalone` (pitfalls 1, 2).
3. Ensure `nvidia-nvshmem-cu13==3.5.21` in the venv (pitfall 4).
4. Launch without `stream=`; sync via `torch.cuda.synchronize()` (pitfall 5).
5. `.venv/bin/torchrun --standalone --nproc-per-node 4 <demo>.py`.

## 7. Key takeaway

> ~90% of the friction is **not** in the kernel code but in **three-way version
> alignment**: `nvshmem4py` (Python API names) ⟷ `cutlass-dsl` (libNVVM's LLVM
> version) ⟷ `nvidia-nvshmem` device bitcode (the LLVM it was built with).
> Lock down a compatible combination first, then write the communication logic.
