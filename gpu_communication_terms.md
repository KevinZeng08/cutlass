# GPU 通信术语说明

本文整理分布式注意力、集合通信和 device-side communication 中常见的 GPU 通信术语。整体按**三层**理解：**硬件网络层、指令和内存抽象层、通信库与编程模型层**。最后总结为什么各类通信库都在向 **device-side 通信**演进。

## 硬件网络层

### NVLink

NVLink 是 NVIDIA 的 **GPU 高速互联链路**，用于 GPU 和 GPU 之间的**高带宽、低延迟通信**。相比 PCIe，NVLink 更适合高频 peer access、集合通信、张量并行、上下文并行等多 GPU 场景。

在单机内，如果拓扑和 CUDA runtime 配置支持，**一个 GPU 可以通过 peer access 访问另一个 GPU 的显存**。NCCL、NVSHMEM 以及框架层的 fused communication kernel 都可以把 NVLink 作为单机通信的基础路径。

### NVSwitch

NVSwitch 是 **NVLink 的交换芯片**。点对点 NVLink 连接的是两个端点，而 **NVSwitch 把多条 NVLink 汇聚成一个高带宽 fabric**，让同一台机器内多块 GPU 能高效互联。

在 HGX、DGX 等系统中，NVSwitch 提供**接近 all-to-all 的 GPU 连接能力**，通常比纯 PCIe 拓扑有更高的聚合带宽。现代 NVIDIA 多 GPU 服务器上的高性能单机 collective 很大程度依赖 NVSwitch。

### NVLink SHARP

NVLink SHARP 指 **NVLink/NVSwitch fabric 中的 in-network reduction 能力**。SHARP 是 **Scalable Hierarchical Aggregation and Reduction Protocol** 的缩写，核心思想是**把一部分规约计算放到网络 fabric 内部完成**，而不是让 GPU 之间反复搬运中间结果，并由 GPU 完成所有 reduction 步骤。

对 AllReduce 这类操作来说，NVLink SHARP 可以**减少通信流量和 GPU 侧规约开销**。用户通常**不会直接调用** NVLink SHARP，它一般由 **NCCL、NVSHMEM 或 PTX multimem** 等更高层或更底层的软件路径间接利用，具体取决于硬件和软件支持。

### RDMA 和 GPUDirect RDMA

RDMA（Remote Direct Memory Access）允许**网卡直接读写远端机器的内存，数据路径不需要远端 CPU 参与**。常见 RDMA 网络包括 **InfiniBand 和 RoCE**。

GPUDirect RDMA 把 RDMA 能力扩展到 GPU 显存：**NIC 可以直接 DMA 读写 GPU memory，避免经过 host memory 中转**。这是**高效跨节点 GPU 通信的关键基础**。NCCL 和 NVSHMEM 在机间通信时通常会通过 InfiniBand 或 RoCE 搭配 GPUDirect RDMA。

需要注意的是，**GPUDirect RDMA 只表示 NIC 可以直接访问 GPU memory，并不自动意味着 GPU kernel 可以完整地发起和推进任意网络操作**。真正的 GPU-initiated networking 还需要 **device-side queue、doorbell、同步机制**以及 runtime 或通信库支持。

## 指令和内存抽象层

这一层有**三种常被对比的 device-side 数据搬运/通信原语：普通 `ld/st`、`multimem`、TMA**。它们的核心区别在于「**谁来搬运、谁来归约、作用范围多大**」。

### 普通 ld/st（global memory 访问）

`ld.global` / `st.global` / `atom.global` / `red.global` 是**最基础的内存访问指令，作用在线程级**。**配合 peer-mapped 的 global 地址，它们也可以直接访问 NVLink domain 内其他 GPU 的显存**。

特点：

- **线程级、同步、寄存器直达**：load 结果直接落寄存器，后续依赖指令需要等待。
- **最灵活**：任意访问模式、任意粒度、可与计算自由穿插。
- 代价是**搬运和归约都压在 SM 上，指令数多**；跨 GPU 还需自己用 `**.sys` scope 与 `fence.sys`** 处理可见性和内存序。

适合**细粒度、不规则、需要与计算紧耦合**的访问，但**不擅长大块吞吐和跨多 GPU 的聚合**。

### multimem

multimem 是 **PTX 层面的特殊内存操作机制，用来访问 multimem address**。**multimem address 是一种虚拟地址，可以映射到多个不同的内存位置**，这些位置可能分布在多个设备上。**普通 `ld`、`st` 不能访问 multimem address，必须使用 `multimem.*` 指令**。

重要的 PTX 操作包括：

- `**multimem.ld_reduce`**：从多个位置 load，并把多个值**规约成一个结果**。
- `**multimem.st`**：把同一个值 store 到 multimem address 指向的**所有位置**（多播）。
- `**multimem.red`**：对 multimem address 指向的位置执行 **reduction** 操作。

从抽象上看，multimem 是**硬件辅助的多位置内存操作在 PTX 层的接口**。在支持的 NVLink/NVSwitch 系统上，它可以作为软件暴露 **in-fabric reduction 或 multicast 行为**的一种方式。

特点：

- **把归约和广播卸载到 NVSwitch 在网计算（NVLink SHARP）**：`multimem.ld_reduce`/`multimem.red` 在 fabric 内完成规约，`multimem.st` 完成多播。
- **一条指令作用于 multicast 地址，覆盖该 multicast group 内所有 GPU**，流量和指令数相比逐 GPU 搬运**大幅降低**。
- 约束是**只支持定型的 reduce op**（整型 `.add/.min/.max/.and/.or/.xor`，浮点规约以 `.add` 为主），且**地址必须是 multicast/multimem 地址**；指令仍由单线程发起，覆盖范围取决于该地址映射了哪些设备。

### TMA（`cp.async.bulk.tensor`）

TMA（Tensor Memory Accelerator，sm_90+）是**异步大块张量搬运引擎**。它依赖 **host 侧创建的 tensor map descriptor**，**按 tile 在 global memory 和 shared memory 之间搬运数据**，并通过 **mbarrier（`complete_tx`）做异步完成通知**。

特点：

- **把搬运卸载到异步 DMA 引擎**：按 tile 大块搬、用 mbarrier 异步通知，从而**释放 SM 去做计算，利于通信与计算 overlap**。
- 当 descriptor 的 global 地址是 **NVLink 可达的 peer/multicast 对称内存**（通常由 NVSHMEM 建立）时，**TMA 也能用于 NVLink domain 内的跨 GPU 搬运**。
- 局限是**它只负责移动数据、不做归约**：因此 all-reduce 里的**归约仍需在寄存器中完成**，**广播则要借助 multicast 地址**（或多次 store）。

需要区分的是 `**.shared::cluster` / DSMEM** 这类跨 CTA 访问，其**作用域是单 GPU 内的 thread block cluster，并不跨 NVLink**。

### 三者对比


| 维度          | `ld/st`（含 `atom/red`）         | `multimem`                 | TMA                       |
| ----------- | ----------------------------- | -------------------------- | ------------------------- |
| 搬运执行者       | SM 线程                         | NVSwitch fabric            | 异步 DMA 引擎                 |
| 归约能力        | SM 在寄存器里做                     | fabric 在网归约（定型 op）         | 不做归约                      |
| 广播能力        | 逐目标 store                     | multicast 地址一次广播           | 借 multicast 地址或多次 store   |
| 粒度          | 线程级、任意                        | multicast 地址、定型 op         | tile 级大块                  |
| 与计算 overlap | 同步、占用 SM                      | 单指令、占用少                    | 异步、释放 SM                  |
| 跨 GPU 前提    | peer-mapped 地址 + `.sys` scope | multicast/multimem 地址      | descriptor 指向 peer/对称内存地址 |
| 适用场景        | 细粒度、不规则访问                     | all-reduce/broadcast 的在网加速 | 大块吞吐、计算通信融合               |


实践中**三者常组合使用**：例如用 **TMA 做大块跨 GPU 搬运 + 寄存器归约**，用 **multicast 地址或 `multimem.`* 做广播和跨 GPU flag 同步**，用 `**ld/st` 处理零散的细粒度访问**。

### symmetric memory

symmetric memory 是一种**内存分配和寻址模型**，不是某一个单独产品。基本思想是：**每个 rank 或 processing element 都分配布局一致的对应 buffer**，让软件可以用**统一方式理解本地和远端 buffer**。

常见例子包括：

- NVSHMEM 的 symmetric heap。
- PyTorch 的 `torch.distributed._symmetric_memory`。
- 某些框架或 kernel 内部用于通信与计算融合的 symmetric buffer。

symmetric memory 的价值在于**解决 device-side 通信里的地址可达性问题**。**GPU kernel 通常不能随便拿另一个 rank 的普通 tensor 指针就安全访问**；这需要提前完成**映射、注册、权限控制和同步约定**。symmetric memory 让**远端地址变得稳定、可发现、可被通信库管理**。

## 通信库与编程模型层

### NVSHMEM

NVSHMEM 是 NVIDIA 的 **OpenSHMEM 风格 GPU 通信库**。它提供 **symmetric heap**，以及 **put、get、atomic 等 one-sided 通信 API**。一个关键特点是**这些 API 可以在 GPU kernel 内调用，从而允许 device 直接发起通信**。

NVSHMEM 可以覆盖机内和机间通信：

- 机内路径通常使用 NVLink、NVSwitch 或 PCIe peer access，具体取决于拓扑。
- 机间路径通常使用 InfiniBand 或 RoCE 等 RDMA transport，并搭配 GPUDirect RDMA。

NVSHMEM 适合**细粒度、不规则、动态的通信模式**，尤其是 **one-sided 语义比 collective 更自然**的场景。例如 MoE token dispatch、producer-consumer queue、远端读、远端写，以及自定义通信与计算融合。

### NCCL

NCCL 是 NVIDIA 的**集合通信库**，提供 **AllReduce、ReduceScatter、AllGather、Broadcast、AllToAll 等 collective**。PyTorch 等框架通常用 **NCCL 作为分布式训练后端**。

NCCL 同样可以覆盖机内和机间通信：

- 机内路径使用 NVLink、NVSwitch 或 PCIe。
- 机间路径使用 InfiniBand 或 RoCE，通常搭配 GPUDirect RDMA。

**NCCL 和 NVSHMEM 的核心区别是编程模型**。**NCCL 暴露 collective**，所有参与 rank 按 collective contract 进入同一个通信操作；**NVSHMEM 暴露 one-sided memory operation**，一个 GPU 可以直接 put 到或 get 自远端 peer 的 symmetric memory。

### NCCL GIN

NCCL GIN 通常指 **GPU-Initiated Networking** 或 GPU-initiated NCCL 风格通信。具体公开 API 和可用性取决于 NCCL 版本、平台和框架集成，但方向很明确：**减少 CPU 参与，让 GPU 侧执行更直接地触发或推进通信**。

**GIN 并不从根本上改变 NCCL 的通信语义**。操作仍然主要是 AllReduce、AllGather、ReduceScatter 等 collective。**变化的是执行模型**：通信可以更贴近产生或消费数据的 GPU work，而不是每个通信阶段都由 CPU 作为独立步骤 enqueue。

### PyTorch symmetric memory

PyTorch symmetric memory 是**框架层的 symmetric memory 抽象**，目标是**把 symmetric buffer 以 PyTorch tensor 的形式暴露出来**，并支持 PyTorch 生态里的**通信与计算融合**。

它尤其适合单机 fused operation，例如：

- AllGather + matmul。
- ReduceScatter + matmul。
- 自定义分布式 kernel 中的 peer-buffer access。

广义的 symmetric memory 可以在 NVSHMEM 等后端支持下覆盖机间通信。但 **PyTorch symmetric memory 当前更常被放在单机 NVLink/NVSwitch 和框架集成 fused kernel 的语境下讨论**。

## 分层关系

可以用下面的栈来理解这些概念：

```text
应用和框架
  PyTorch DDP / FSDP / tensor parallelism / context parallelism / MoE

通信库与编程模型
  NCCL collectives
  NVSHMEM one-sided communication
  PyTorch symmetric memory 与 fused communication-compute operators

指令和内存抽象
  symmetric memory
  ld.global / st.global
  TMA load / store
  multimem addresses
  multimem.ld_reduce / multimem.st / multimem.red

硬件网络层
  NVLink
  NVSwitch
  NVLink SHARP
  InfiniBand 或 RoCE
  GPUDirect RDMA
```

简要总结：

- **NVLink 和 NVSwitch** 提供**机内 GPU 通信 fabric**。
- **NVLink SHARP** 在支持的 fabric 上提供 **in-network reduction** 能力。
- **RDMA 和 GPUDirect RDMA** 提供**高效机间数据搬运**能力。
- **multimem** 在 PTX 层暴露特殊的**多位置内存操作**。
- **symmetric memory** 提供**稳定、结构化的跨 rank 地址可达性**。
- **NVSHMEM** 提供 **device-callable 的 one-sided 通信**。
- **NCCL** 提供**优化过的 collective 通信**。
- **NCCL GIN** 把 collective 的触发和推进**进一步靠近 device-side execution**。

## 为什么需要 device-side 通信

传统 GPU 通信通常由 CPU 侧发起：

```text
GPU kernel 结束
  -> CPU 观察或等待
  -> CPU enqueue 通信操作
  -> GPU 或 NIC 执行通信
  -> 下一个 GPU kernel 运行
```

这个模型对大块、粗粒度 collective 很有效。但**现代 AI workload 的通信越来越频繁、细粒度、动态，并且和计算强耦合，CPU-side 发起通信逐渐成为瓶颈**。

device-side 通信的主要动机包括：

- **降低 launch 和 enqueue 开销**。频繁 CPU 参与会引入 kernel launch、stream 调度、runtime 调用、host-device synchronization 等延迟。
- **更细粒度地 overlap 通信和计算**。GPU kernel 可以在 tile、block、expert bucket 或 partial result 就绪后立即通信，而不是必须等整个 kernel 边界。
- **更好支持动态通信**。MoE routing、sparse all-to-all、context-parallel attention 和 serving workload 的通信决策经常在 GPU 上运行时产生。
- **减少大规模场景下的 CPU 瓶颈**。当很多 GPU 都依赖 host CPU 发起和排序通信时，CPU 可能进入关键路径。
- **更自然地实现 fused kernel**。AllGather + matmul、ReduceScatter + matmul、MoE dispatch + expert compute、remote KV fetch + attention 等模式，在 device execution 内能直接通信时更容易实现和优化。

## device-side 通信的前提

**device-side 通信不只是需要一条高速链路**，还需要以下条件：

- **硬件访问能力**：机内需要 NVLink/NVSwitch 或 PCIe peer access；机间需要 RDMA-capable NIC 和 GPUDirect RDMA。
- **可注册、可寻址的内存**：需要 symmetric allocation、peer mapping、memory registration 和稳定的 remote address。
- **device-callable 通信机制**：例如 NVSHMEM device API、multimem PTX 操作、框架提供的 peer-buffer access，或 device-side NCCL/GIN 机制。
- **正确的同步和内存序**：远端写、远端读、reduction、fence、barrier、signal、wait、acquire/release 语义都必须被明确处理。
- **progress 机制**：通信必须由 GPU execution、persistent kernel、NIC offload、NVSwitch/NVLink fabric 特性，或 runtime 管理的 queue 机制推进。

总体趋势是，**现代分布式 GPU workload 需要通信成为计算调度的一部分，而不只是 kernel 之间由 CPU enqueue 的独立操作**。