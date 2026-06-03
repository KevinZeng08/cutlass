# CuTeDSL 通信支持现状

本文基于以下三个来源，系统梳理 **CuTeDSL（`cutlass.cute`）目前在 device 侧能用的跨 GPU 通信能力**：

- 当前仓库内的 CuTeDSL 代码（`python/CuTeDSL/cutlass/` 与 `examples/python/CuTeDSL/cute/blackwell/kernel/distributed/`）
- NVSHMEM Device API for CuTe DSL：<https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/device/cute/index.html>
- NCCL GIN Device API（nccl4py 接入 CuTeDSL 的提交）：<https://github.com/NVIDIA/nccl/commit/21287f89d75ab8d05fa7f8b5259cdbfe53072198>

## 1. 总览：三条技术路线

CuTeDSL 自身只内建 **NVLink 域内**的底层通信原语；要做完整的（尤其跨节点）集合/点对点通信，目前依赖 CuTeDSL 的 **FFI + bitcode 链接能力**接入 NVSHMEM 或 NCCL 的 device 库。

| 路线 | 实现机制 | 覆盖范围 | 在 kernel 内能做什么 | 成熟度 |
|------|----------|----------|----------------------|--------|
| **A. 原生 PTX/NVVM 内建** | inline asm + `cute.arch.*` / `cutlass.utils.distributed` | 仅 NVLink/NVSwitch 域内 | multimem 归约/广播、P2P ld/st、TMA 搬运、原子、fence、spin-lock | 已合入仓库，examples 可跑 |
| **B. NVSHMEM device API** | nvshmem4py 的 CuTe 绑定 | 机内 NVLink + 机间 RDMA | put/get、全套 collectives、atomics、signal | 官方文档已发布 |
| **C. NCCL GIN device API** | `cute.ffi` + `cute.BitCode` 链接 `libnccl_device.bc` | 机内 LSA + 机间 GIN(网络) + hybrid | GIN put / wait_signal、teams、barrier session、coop | 上述提交刚加入 |

> 关键判断：**A 是 CuTeDSL 真正"自带"的能力；B 和 C 都是外部通信库借助 CuTeDSL 的 `cute.ffi`（FFI 原型）、`cute.BitCode`（bitcode 链接）、`cute.native_struct`（C 结构映射）三件套，把各自的 device 函数注入到 `@cute.kernel` 中。**

---

## 2. 路线 A：CuTeDSL 原生 device-side 通信原语

这部分围绕 **NVLink/NVSwitch 域内的对称内存（symmetric memory）**，host 侧通常用 NVSHMEM/PyTorch symmetric memory 完成分配与映射，device 侧通信全部由 PTX/NVVM 原语完成。

### 2.1 Multimem（NVLS / NVLink SHARP 在网计算）

位置：`python/CuTeDSL/cutlass/utils/distributed.py`。通过 `llvm.inline_asm` 直接发射 `multimem.*` PTX，分三类：

- **`multimem.ld_reduce`**：从 multicast 地址读取并返回跨 GPU 的归约结果（在 NVSwitch 内完成 sum）。
  - 覆盖 32 / 64 / 128-bit 宽度，dtype：`f16` / `bf16` / `f32` / `e4m3` / `e5m2`；支持高精度累加器（`.acc::f32` / `.acc::f16`）。
  - 入口函数：`multimem_ld_reduce(mc_ptr, dtype, num_elements)`（dispatch），及具体变体 `multimem_ld_reduce_8xf16` 等。
- **`multimem.st`**：写 multicast 地址 → 广播到所有 GPU。变体 `multimem_st_1xb32 / _2xb32 / _4xb32` 与 dispatch `multimem_st(mc_ptr, *regs)`。
- **`multimem.red.{relaxed,release}.{gpu,sys}.add`**：对 multicast 地址做原子归约，主要用于跨 GPU 信号/同步（`multimem_red_add1`）。

### 2.2 跨 GPU 同步原语（spin-lock / barrier）

同样在 `cutlass/utils/distributed.py`：

- `spin_lock_atom_cas_relaxed_wait` / `spin_lock_atom_cas_acquire_wait`：基于 `atomic_cas` 的自旋等待 + 复位。
- `spin_lock_ld_lt_relaxed_wait`：基于 `load` 的轮询。
- `red_add1`（unicast 原子加）+ `multimem_red_add1`（multicast 原子加）实现计数式 barrier。

### 2.3 底层 `cute.arch` 内存 / 原子 / fence / 异步拷贝

位置：`python/CuTeDSL/cutlass/cute/arch/`（见 `__init__.py` 的 `__all__`）。这些是构建任何 device 通信的通用积木：

- **load / store**：`cute.arch.load` / `cute.arch.store`，支持 `sem`（relaxed/acquire/release）、`scope`（cta/gpu/cluster/**sys**）、cache 提示。配合 peer-mapped 地址即可做 NVLink P2P 直读直写。
- **原子 / 归约**：`atomic_add` / `atomic_cas` / `atomic_exch` / `atomic_and|or|xor|min|max`、`red`（add/min/max/and/or/xor）。
- **fence**：`fence_acq_rel_{cta,cluster,gpu,sys}`、`fence_proxy`、`fence_view_async_tmem_*`。
- **异步拷贝 / TMA**：`cp_async_*`、`cp_async_bulk_commit_group` / `cp_async_bulk_wait_group`；TMA tile copy 经 `cute.nvgpu.cpasync` + `cute.copy` 发起。
- **mbarrier**：`mbarrier_init` / `mbarrier_arrive_and_expect_tx` / `mbarrier_wait` / `mbarrier_try_wait` 等，用于 TMA 异步完成跟踪。
- **warp / 同步**：`barrier` / `barrier_arrive` / `sync_threads` / `sync_warp`、`shuffle_sync*`、`vote_*_sync`、`warp_redux_sync`、`elect_one`。

### 2.4 三种 device 搬运/通信原语对比（同在 NVLink 域内）

| 维度 | `cute.arch.load/store`（SIMT） | `multimem.ld_reduce/st/red` | TMA（`cp.async.bulk.tensor`） |
|------|-------------------------------|-----------------------------|-------------------------------|
| 谁发起 | 每线程各发 `ld/st.global` | 每线程发 multimem 指令 | 单线程发起整 tile 传输 |
| 搬运执行者 | SM 线程 | SM 线程，但归约/广播在 NVSwitch | 独立 TMA 异步 DMA 引擎 |
| 归约/广播 | 本地 ALU（读 N 次自加） | **在网做**：一条指令读全部 rank 求和 / 一条 store 广播 | 只搬运、不归约；广播借 multicast 地址 |
| 数据落点 | 寄存器 ↔ 远端 global | 寄存器 ↔ multicast 地址 | 远端 global ↔ SMEM（不直达寄存器） |
| 地址要求 | peer-mapped 普通指针 | 必须是 multicast(MC/NVLS) 地址 | descriptor（可编码 peer/multicast 地址） |
| 同步 | fence + 原子/flag | weak + fence/flag | mbarrier + bulk wait group |
| 适用 | 细粒度、不规则访问 | 大规模 all-reduce/broadcast | 大块吞吐、通信计算融合 |

### 2.5 仓库内 examples（`.../distributed/`）

| 文件 | 演示的 device 通信方式 |
|------|------------------------|
| `all_reduce_simple.py` | peer 张量 + 普通 `cute.copy`（`ld.global`）直读远端 + 寄存器累加 |
| `all_reduce_tma.py` | TMA G2S 从各 rank 远端搬入 SMEM + 寄存器归约 + multicast TMA store 广播 + `multimem.red` barrier |
| `all_reduce_two_shot_multimem.py` | `multimem.ld_reduce` + `multimem.st` 两段式 all-reduce + spin-lock barrier |
| `all_reduce_one_shot_lamport.py` | Lamport flag 无锁 one-shot all-reduce |
| `distributed_gemm_all_reduce_blackwell.py` 等 | GEMM 与 all-reduce / all-gather / reduce-scatter 融合 |

> 仓库 README 明确：这些自带 example 中 **NVSHMEM 只用于 host 侧分配/映射对称内存**，device 侧通信完全靠上述 PTX 原语，不调用 NVSHMEM device 函数。

---

## 3. 路线 B：NVSHMEM Device API for CuTe DSL

NVSHMEM4Py 提供可在 `@cute.kernel` 内直接调用的 device 绑定，能力远超仓库自带 multimem，且**机内（NVLink）与机间（IB/RoCE + GPUDirect RDMA）通用**。

- **Collectives**：`barrier` / `barrier_all`、`sync` / `sync_all`、`reduce`、`reducescatter`、`fcollect`、`broadcast`、`alltoall`（含 `*_{block,warp}` 变体）。
- **RMA（单边）**：向量 `put` / `get`（阻塞与非阻塞 `_nbi`）、`put_signal`、标量 `p` / `g`。
- **Atomics**：`fetch` / `set` / `swap` / `compare_swap`、`inc` / `fetch_inc`、`add` / `fetch_add`、位运算 `and` / `or` / `xor` 及其 `fetch_*`。
- **Signalling**：`signal_op`、`signal_wait_until`。
- **Utilities**：`n_pes`、`my_pe`、`team_n_pes`、`team_my_pe`。
- **执行 scope**：`device` / `block` / `warp`；**语义**：blocking 与 `_nbi` 非阻塞。
- **数据类型**：API 接受 CuTe `Tensor`（也可用 DLPack 把 torch tensor 转 CuTe tensor）。

特点：PGAS / 对称内存模型，覆盖最完整的 SHMEM 集合通信 + 点对点 + 原子 + 信号语义；适合细粒度、不规则、动态（MoE dispatch、producer-consumer、远端读写）场景。

---

## 4. 路线 C：NCCL GIN Device API（nccl4py 接入 CuTeDSL）

实现机制是 **CuTeDSL 的 FFI + bitcode 链接**：`cute.BitCode("libnccl_device.bc")` 加载 NCCL device 库，`cute.ffi(source=_BC, ...)` 为每个 `__device__` 符号生成 1:1 原型，`@cute.native_struct` 1:1 映射 C 结构（`ncclTeam` / `ncclGin_C` / `ncclCoopAny` / 各 barrier handle）。

提供的 device 能力：

- **GIN（跨节点网络）RMA + 信号**：
  - `Gin.put(team, peer, dst_win, dst, src_win, src, coop, is_signal=..., signal_id=..., signal_op=..., is_counter=..., is_descriptor=...)` —— 直接吃 `cute.Tensor`（由 `Window.tensor(dtype, layout)` 构造），从 tensor 的 iterator 地址与 layout 自动推导 byte offset 与 size。
  - `Gin.wait_signal(coop, signal, least, bits, ord)`。
- **对称窗口 / 指针翻译**：`Window.local_pointer` / `lsa_pointer`（LSA = 节点内 NVLink/peer-access）/ `peer_pointer`（按 team 寻址远端）。
- **Teams**：`team_world` / `team_lsa` / `team_rail`，及 `DevComm.rank / n_ranks / lsa_rank / lsa_size`。
- **Barrier sessions（三种）**：
  - `LsaBarrierSession`（节点内，可选 multimem）：`arrive` / `wait` / `sync`。
  - `GinBarrierSession`（节点间网络）：`sync`，带 `GinFenceLevel(PUT/GET)`。
  - `BarrierSession`（hybrid：LSA inner + GIN outer）：`sync`。
  - 便捷工厂：`lsa_default` / `world_gin` / `rail_gin` / `world_hybrid`。
- **Cooperative groups**：`cta()` / `warp()` / `thread()` / `lanes(mask)` / `warp_span(...)` → `ncclCoopAny`，所有通信按 coop 粒度协作。
- **后端选择**：`GinBackendMask.{PROXY, GDAKI, GPI, ALL}`；内存序：`MemoryOrder`（libcu++ 对齐）。

注意：kernel 需 `launch(..., cooperative=True)`；指针参数必须用 `cutlass.Int64` 注解（否则默认 `int`→`Int32` 截断 >4GB 地址）。

特点：聚焦单边 `put` + `signal/counter` 与多层 barrier；无现成 collective，靠 barrier + put 组合实现。

---

## 5. 三条路线横向对比

| 对比项 | A. 原生 PTX/NVVM | B. NVSHMEM | C. NCCL GIN |
|--------|------------------|------------|-------------|
| 通信域 | 仅 NVLink/NVSwitch 域内 | 机内 + 机间全覆盖 | LSA(机内) + GIN(机间网络) + hybrid |
| 编程模型 | 底层指令（自己拼） | one-sided + collectives | one-sided put/signal + barrier |
| 集合通信 | 需手写 | 完整内建 | 无现成，靠 barrier+put 组合 |
| 接入机制 | inline asm（`llvm.inline_asm`） | `cute.ffi` + bitcode | `cute.ffi` + `cute.BitCode` |
| 数据接口 | `cute.Tensor` / 指针 | `cute.Tensor`（DLPack 互转） | `cute.Tensor`（自动算 offset/size） |
| 依赖 | 无额外依赖 | nvshmem4py + nvidia-nvshmem | nccl4py[cute] + libnccl_device.bc |

---

## 6. 结论

- **CuTeDSL 自带的 device 通信只有 NVLink 域内的一层**：`multimem`（在网归约/广播）、peer-mapped 的 `ld/st` 与 `atomic/red`、TMA 大块搬运，以及 `fence` / spin-lock / mbarrier 等同步原语。
- **要做完整的、尤其是跨节点的集合 / 点对点通信**，目前是通过 CuTeDSL 的 **`cute.ffi` + `cute.BitCode` + `cute.native_struct`** 这套 FFI/bitcode 链接能力，接入 **NVSHMEM** 或 **NCCL GIN** 的 device API 来实现。
- 三者可叠加使用：典型如 `all_reduce_tma.py` —— TMA 负责远端→SMEM 大块搬运与流水线 overlap，寄存器负责归约，multicast TMA store 负责广播，`multimem.red` 负责跨卡 barrier。

## 7. 参考

- 原生原语：`python/CuTeDSL/cutlass/utils/distributed.py`、`python/CuTeDSL/cutlass/cute/arch/`
- Examples：`examples/python/CuTeDSL/cute/blackwell/kernel/distributed/`
- FFI/bitcode：`python/CuTeDSL/cutlass/cute/ffi.py`、`python/CuTeDSL/cutlass/base_dsl/ffi.py`、`python/CuTeDSL/cutlass/base_dsl/native_struct.py`
- NVSHMEM Device API：<https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/device/cute/index.html>
- NCCL GIN 接入提交：<https://github.com/NVIDIA/nccl/commit/21287f89d75ab8d05fa7f8b5259cdbfe53072198>
