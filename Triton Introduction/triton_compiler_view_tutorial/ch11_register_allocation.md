# 第 11 章：寄存器分配与内存管理

> 参考：*Engineering a Compiler* Chapter 12 (Register Allocation)
>
> **Triton 对应模块：** `lib/Analysis/Allocation.cpp`, `lib/Analysis/BufferRegion.cpp`, `lib/Analysis/Alias.cpp`, `lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp`, `lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp`

---

## 11.1 章节导引

### 本章在全书中的位置

第 10 章讨论了软件流水线和 Warp Specialization——编译器如何重排指令和划分 warp 角色以隐藏访存延迟。那些优化在"有地方存数据"的前提下才有意义。本章回答一个更根本的问题：**程序中产生的数据应该放在内存的哪个位置？**

在 GPU 上，内存层次分为全局内存（global memory / HBM）、共享内存（shared memory / LDS）、和寄存器文件（register file）。Triton 编译器在这三个层次上执行不同的内存管理策略：**寄存器分配委托给 LLVM/NVPTX 后端**，**共享内存分配由 Triton 自己的 Allocation 分析完成**，**当共享内存不足时，使用 Global Scratch Memory 作为溢出空间**。

本章位于全书第四部分（后端——代码生成与运行时）的中段，承接指令选择（第 9 章）和流水线优化（第 10 章），为第 12 章的后端代码发射（LLVM -> PTX -> CUBIN）做好准备——共享内存的 layout 和偏移量必须在 PTX 生成前完全确定。

### 学习目标

完成本章后，你将能够：

1. 掌握寄存器分配的核心理论：活跃范围分析（数据流方程）、干涉图、图着色算法（Chaitin-Briggs）和线性扫描分配
2. 理解 Triton 在共享内存层面的内存管理：buffer 分类、活跃范围计算、first-fit 图着色分配、分区内存管理
3. 理解 Alias 分析的原理及其在共享内存复用中的作用
4. 掌握 BufferRegion 分析如何追踪共享内存的细化访问
5. 理解 Global Scratch Memory 作为共享内存溢出空间的机制
6. 理解 tile 大小、num_warps、layout encoding 如何影响寄存器压力

### 先修知识

- **第 4 章**：TTGIR 设计（`ttg.local_alloc`、`ttg.local_load`、`ttg.local_store` 及内存空间概念）
- **第 6 章**：Lowering（TTIR -> TTGIR 转换中 layout 的传播）
- **第 9 章**：指令选择（TTGIR -> LLVM IR 中如何使用共享内存基地址）
- **数据流分析基础**：不动点迭代、lattice 理论的基本概念

---

## 11.2 编译器基础知识

### 11.2.1 寄存器分配理论（*EaC* Ch.12）

#### 活跃范围（Live Range）与活跃变量分析

在讨论寄存器分配之前，必须确定每个值在程序中的"生命周期"——即它从定义到不再被使用的区间。这个区间称为**活跃范围**（live range）。

**定义（活跃变量）：** 在程序的某个执行点 p，变量 v 是活跃的，当且仅当存在一条从 p 开始的路径，在该路径上 v 的值被使用并且在该使用之前没有被重新定义。

活跃变量分析（Live Variable Analysis）是一种**后向数据流分析**（backward data-flow analysis），其数据流方程为：

$$LIVE_OUT[n] = \bigcup_{s \in succ(n)} LIVE_IN[s]$$

$$LIVE_IN[n] = USE[n] \cup (LIVE_OUT[n] - DEF[n])$$

其中：
- $USE[n]$ 是基本块 $n$ 中在定义之前就使用的变量集合
- $DEF[n]$ 是基本块 $n$ 中定义的变量集合
- $succ(n)$ 是 $n$ 的后继基本块集合

这些方程通过**不动点迭代**求解：将所有集合初始化为空，反复应用方程直到没有集合发生变化。收敛性由两个性质保证：（1）集合操作是单调的（只增不减）；（2）变量集合是有限的，因此包含在一个有限的 lattice 中。

**直观理解：** 下面的代码展示了活跃范围的直观含义：

```
v1 = load(x)           # def v1
v2 = load(y)           # def v2
v3 = v1 + v2           # def v3, use v1, v2  --> v1 last use, v2 last use
v4 = v3 * 2            # def v4, use v3        --> v3 last use
store(z, v4)           # use v4                --> v4 last use
```

值的活跃范围：
- v1: [1, 3) -- 从定义到第一次（也是最后一次）在 v3 处使用
- v2: [2, 3) -- 从定义到在 v3 处使用
- v3: [3, 4) -- 从定义到在 v4 处使用
- v4: [4, 5) -- 从定义到在 store 处使用

#### 干涉图（Interference Graph）

**定义：** 两个值的活跃范围有重叠，称它们**干涉**（interfere）。干涉的两个值不能共享同一个物理寄存器（或同一块内存）。

**干涉图**是一个无向图 $G = (V, E)$，其中：
- 节点 $V$ 对应每个虚拟寄存器（值）
- 边 $(u, v) \in E$ 当且仅当 $u$ 和 $v$ 的活跃范围存在重叠

如果我们可以用 $K$ 种颜色对干涉图进行着色（相邻节点不同色），那么就可以用 $K$ 个物理寄存器完成分配。这正是图着色寄存器分配的核心洞察。

```mermaid
graph TD
    v1["v1: [1,3)"] --- v3["v3: [3,4)"]
    v2["v2: [2,3)"] --- v3
    v3 --- v4["v4: [4,5)"]
    v1 -.- v2
    style v1 fill:#e1f5fe
    style v2 fill:#e1f5fe
    style v3 fill:#ce93d8
    style v4 fill:#c8e6c9
```

> 上图中实现表示干涉（活跃范围重叠），虚线表示两个值活跃范围虽在同一区间但刚好不重叠（v1 的 last use 在 v2 的 def 之前，边界情况取决于分析精度）。v1 和 v2 不干涉，可以共享同一寄存器；v1 和 v3 干涉，不能共享。

#### 图着色寄存器分配（Chaitin-Briggs Algorithm）

图着色寄存器分配是最经典的寄存器分配算法，其流程为：

1. **构建（Build）：** 计算活跃范围和干涉图
2. **合并（Coalesce）：** 消除因 copy 指令产生的干涉边，将 copy 相关的节点合并（减少 move 指令）
3. **稀疏化（Spill Cost）：** 为每个节点计算溢出代价（spill cost）——如果溢出这个值，会引入多少额外的 load/store
4. **简化（Simplify）：** 重复移除度数小于 $K$ 的节点（将它们压栈），这些节点显然可以被着色
5. **溢出（Spill）：** 当图中没有度数小于 $K$ 的节点时，选择一个节点溢出——将其插入 load/store 指令后从图中移除
6. **选择（Select）：** 按栈的逆序弹出节点，为每个节点分配一个不同于其邻居的颜色

**Briggs 的改进**在于：Chaitin 原始算法在 simplify 阶段后直接 spill，而 Briggs 观察到一个度数 >= K 的节点仍可能被着色（如果其邻居中存在共享颜色的情况），因此加入了 optimistic coloring 策略——先假设所有节点都可着色，到 select 阶段再决定是否真正需要溢出。

#### 线性扫描分配（Linear Scan Allocation）

线性扫描是另一种广泛使用的寄存器分配算法，复杂度为 $O(n \log n)$（n 为活跃范围数量），远低于图着色的 $O(n^2)$。

**算法核心：**
1. 将所有活跃范围按起始位置排序
2. 维护一个"活跃列表"（active list）——当前正在某个寄存器中的值
3. 当遇到一个新的活跃范围时：
   - 从活跃列表中移除已结束的值（释放寄存器）
   - 如果有空闲寄存器，分配给新值
   - 如果没有空闲寄存器，选择**结束位置最远**的值溢出（spill heuristic）

线性扫描的劣势是着色质量通常不如图着色，但对于 JIT 编译器或对编译速度敏感的场合，其速度优势是决定性的。

#### 为什么需要寄存器分配

理想情况下，编译器可以为每个值分配一个唯一的存储位置（无限寄存器模型）。但物理硬件只有有限个寄存器（NVIDIA GPU 每个 SM 约有 65536 个 32-bit 寄存器，每个 thread 最多可用 255 个）。寄存器是最快的存储器件（一个时钟周期即可访问），如果值被溢出到内存（spill），每次访问将花费数百个时钟周期。因此，好的寄存器分配直接决定程序的性能。

### 11.2.2 算法背景

#### 活跃范围计算的复杂度

活跃变量分析的标准实现需要对每个基本块计算 $USE[n]$ 和 $DEF[n]$，然后执行不动点迭代。对于 $N$ 个基本块的程序，最坏情况下不动点迭代需要 $O(N^2)$ 次方程应用（每次迭代至少增加一个变量到至少一个集合中），但实际程序通常在 2-4 次迭代内收敛（得益于后序遍历的顺序优化）。

#### 图着色的 NP 完全性与启发式策略

一般的图着色问题是 NP 完全的，这意味着不存在多项式时间的精确算法。但寄存器分配中的图着色有一个可利用的性质：在 simplify 阶段，我们可以安全地移除度数小于 $K$ 的节点（因为无论其邻居取什么颜色，它总有一种可用颜色）。当不存在这样的节点时，必须选择 spill——而选择哪个节点溢出（spill heuristic）是一个经验性决策，通常考虑节点的度数和其 loop nesting depth。

---

## 11.3 Triton 设计思想与哲学

### What：Triton 内存管理分层策略

Triton 编译器的内存管理策略可以概括为：**寄存器分配委托 LLVM，共享内存分配自己负责，全球存 scratch 兜底**。

具体来说：

| 内存层次 | 管理者 | 机制 |
|---------|--------|------|
| 寄存器（Register） | LLVM/NVPTX 后端 | 传统图着色/线性扫描寄存器分配 |
| 共享内存（Shared Memory） | Triton `Allocation` 分析 | buffer 级活化范围分析 + first-fit 图着色 |
| 全局 scratch 内存（Global Scratch） | `GlobalScratchMemoryAllocation` pass | 简单的顺序分配（offset 递进） |

### How：实现手法

Triton 的共享内存分配采用了一种**buffer 级**的分配策略，而非传统的**指令级**寄存器分配：

1. **Buffer 粒度：** 以 TTGIR 的 `ttg.local_alloc` 操作（显式 buffer）和需要中间存储的操作（如 `ttg.convert_layout`）为分配单位，而非单条指令的虚拟寄存器
2. **MLIR Liveness 集成：** 直接使用 MLIR 内置的 `Liveness` 分析获取每个 buffer 的活跃操作集合，转换为操作 ID 区间
3. **大块优先排序：** buffer 按大小降序排列再分配，减少碎片
4. **迭代图着色：** `computeOffsets()` 中的核心循环——先计算初始偏移、构建干涉图、着色分配、再重建干涉图——反复迭代直到所有干涉消除（不动点迭代）
5. **分区内存支持：** 针对某些 NVIDIA 架构的 partitioned shared memory，buffer 可被分割到不同物理分区，分区间通过 `neighbors` 关系确保不重叠

而在寄存器层面，Triton **不进行自己的寄存器分配**。`TritonGPUToLLVM` 转换将 TTGIR 操作转换为 LLVM IR 后，LLVM 的 NVPTX 后端负责将 LLVM 虚拟寄存器映射到物理寄存器。这利用了 LLVM 成熟的后端基础设施。

### Why：设计哲学与权衡

**为什么不由 Triton 自己分配寄存器？**

寄存器分配是现代编译器后端中最复杂的优化之一，LLVM 在这方面的实现已经非常成熟（支持 SSA-based register allocation、coalescing、spill code insertion、register class handling 等）。重复实现只会引入 bug 和维护负担。Triton 的设计哲学是**专注于自己擅长的领域（tile-level optimization）**，将底层指令级优化委托给 LLVM。

**为什么共享内存分配要自己实现？**

因为共享内存的语义是 Triton 特有的——它与 layout 系统深度耦合。例如，一个 `ttg.convert_layout` 操作可能需要一块共享内存作为 scratch pad 来完成 layout 转换（分布式寄存器 -> 共享内存 -> 不同布局的分布式寄存器）。这个 scratch 的大小由 source/destination layout、数据类型位宽、bank 数量共同决定（参见 `defaultAllocationAnalysisScratchSizeFn` 函数）。LLVM 不理解 Triton 的 layout 系统，因此无法确定这些 buffer 的大小。

**关键不变性（Invariants）：**

1. **同一 buffer 只被分配一次：** 每个 `ttg.local_alloc` 对应唯一的 buffer ID
2. **干涉的 buffer 不共享内存空间：** 如果两个 buffer 的活跃范围和 offset 范围同时重叠，它们必须占据不同的内存位置
3. **分区 buffer 位于不同物理分区：** 对于 partitioned shared memory，同一 tensor 的不同 partition buffer 必须位于不同物理分区（通过 `neighbors` 干涉强制分离）

---

## 11.4 数据结构设计剖析

### 11.4.1 Allocation 分析核心数据结构

```mermaid
classDiagram
    class Interval~T~ {
        -T Start
        -T End
        +start() T
        +end() T
        +size() T
        +intersects(Interval) bool
        +contains(T) bool
    }

    class BufferT {
        +BufferKind kind
        +BufferId id
        +Operation* owner
        +size_t size
        +size_t alignment
        +size_t offset
        +SmallVector~BufferT*~ neighbors
        +setOffsetAligned(size_t) size_t
    }

    class BufferKind {
        <<enumeration>>
        Explicit
        Scratch
        Virtual
    }

    class Allocation {
        -Operation* operation
        -OpScratchMapT opScratch
        -OpScratchMapT opVirtual
        -ValueBufferMapT valueBuffer
        -AliasBufferMapT aliasBuffer
        -BufferSetT bufferSet
        -size_t sharedMemorySize
        -size_t bufferIdCounter
        +run(FuncAllocMapT&, ScratchSizeFn, size_t)
        +getOffset(BufferId) size_t
        +getAllocatedSize(BufferId) size_t
        +getBufferIds(Value) SmallVector~BufferId~
        +getBufferId(Operation*) BufferId
        +getSharedMemorySize() size_t
        +getLiveBuffers() map
    }

    class ModuleAllocation {
        -FuncOffsetMapT sharedMemoryValue
        +getSharedMemorySize() size_t
        +getFunctionSharedMemoryBase(FunctionOpInterface) Value
    }

    class AllocationAnalysis {
        -Operation* operation
        -BufferRangeMapT bufferRange
        -Allocation* allocation
        +run()
        -getValuesAndSizes()
        -resolveLiveness()
        -computeOffsets()
        -calculateStarts()
        -buildInterferenceGraph()
        -allocate()
    }

    BufferT *-- BufferKind
    Allocation *-- BufferT
    AllocationAnalysis --> Allocation : modifies
    ModuleAllocation --> Allocation : per-function
    AllocationAnalysis ..> Interval : uses
```

**BufferT 三种类型的语义：**

| 类型 | 产生方式 | 示例 |
|------|---------|------|
| `Explicit` | `ttg.local_alloc` 操作 | 用户显式声明的共享内存 buffer |
| `Scratch` | 需要中间存储的操作 | `ttg.convert_layout`（layout 转换的暂存空间）、`ttg.reduce`（归约的共享内存工作区）、`ttg.scan`、`ttg.gather`、`ttg.histogram` |
| `Virtual` | 函数调用（`triton.call`） | 被调用函数需要的共享内存作为一个整体 buffer 在调用点分配 |

### 11.4.2 Allocation 分析算法全景

`AllocationAnalysis` 是整个共享内存分配的核心。它的 `run()` 方法包含三个主要阶段：

```mermaid
flowchart TD
    A["getValuesAndSizes()"] --> B["resolveLiveness()"]
    B --> C["computeOffsets()"]

    A --> A1["遍历所有 op<br/>getExplicitValueSize()"]
    A --> A2["遍历所有 op<br/>getScratchValueSize()"]
    A --> A3["运行 SharedMemoryAliasAnalysis<br/>建立 alias 关系"]

    B --> B1["Post-order 遍历<br/>为每个 op 分配 ID"]
    B --> B2["MLIR Liveness 分析<br/>获取每个值的活跃 op 集合"]
    B --> B3["resolveExplicitBufferLiveness()"]
    B --> B4["resolveAliasBufferLiveness()"]
    B --> B5["resolveScratchBufferLiveness()"]

    C --> C1["calculateStarts()<br/>按大小降序，首次尝试分配偏移"]
    C --> C2["buildInterferenceGraph()<br/>构建干涉图"]
    C --> C3["allocate()<br/>first-fit 图着色"]
    C --> C4{"干涉图为空？"}
    C4 -->|否| C2
    C4 -->|是| C5["完成"]

    style A fill:#e1f5fe
    style B fill:#fff3e0
    style C fill:#e8f5e9
```

**阶段一：收集 buffer 信息（`getValuesAndSizes`）**

源码位置：`triton/lib/Analysis/Allocation.cpp` 第 300-321 行。

首先遍历所有操作，对 `ttg.local_alloc` 显式分配（`getExplicitValueSize`），计算其元素数量和字节大小。对于 partitioned shared memory encoding，会创建多个 partition buffer 并通过 `neighbors` 链确保它们落在不同物理分区中。

然后对需要 scratch memory 的操作（`getScratchValueSize`），调用 `defaultAllocationAnalysisScratchSizeFn` 计算 scratch 大小。例如：

- **`ttg.convert_layout`**：如果源和目标 layout 之间的转换需要共享内存中转（`cvtNeedsSharedMemory` 返回 true），则通过 `getNumScratchElemsSwizzledCvt` 计算所需元素数——该函数考虑 swizzling 模式、bank 数量、CTA 拆分等因素
- **`ttg.reduce`**：通过 `ReduceOpHelper::getScratchSizeInBytes()` 计算
- **`ttg.scan`**：通过 `ScanLoweringHelper::getScratchSizeInBytes()` 计算
- **`triton.call`**：递归计算被调用函数的总共享内存需求

最后运行 `SharedMemoryAliasAnalysis`（见 11.4.3 节）建立 alias 关系。

**阶段二：计算活跃范围（`resolveLiveness`）**

源码位置：`triton/lib/Analysis/Allocation.cpp` 第 398-438 行。

1. 对整个函数的操作执行 **post-order 遍历**，为每个操作分配一个单调递增的 ID。关键设计：父操作的 ID 大于所有子操作（因为父操作在 post-order 中后出现），这确保了在子区域中使用的值，其活跃范围能正确延伸到父操作之后。

2. 调用 MLIR 的 `Liveness::resolveLiveness(value)` 获取使用该值的所有操作的集合。

3. 将操作集合转换为 `Interval<size_t>`（最小操作 ID 到最大操作 ID+1 的区间）。

对于 alias 关系（如通过 `scf.for` 的 block argument 传递的值），alias buffer 的活跃范围是原 buffer 和所有 alias buffer 活跃范围的并集（`resolveAliasBufferLiveness` 中的 `std::min`/`std::max` 操作）。

**阶段三：计算偏移（`computeOffsets`）**

源码位置：`triton/lib/Analysis/Allocation.cpp` 第 481-512 行。

这是一个**迭代不点动**过程：

1. **`calculateStarts`：** 将所有 buffer 按大小降序排列后，使用 multimap-based 算法。算法维护一个"可用起始点 -> 该起始点对应的活跃范围"的 multimap。对于每个 buffer，找到满足以下条件的可用起始点：该 buffer 的活跃范围与起始点对应的活跃范围相交，且不与 multimap 中任何其他条目的活跃范围相交。大 buffer 优先分配，减少碎片。

2. **`buildInterferenceGraph`：** 构建干涉图。两 buffer 干涉的条件：
   - 它们的 offset 范围（[offset, offset+size)）有重叠 AND
   - 它们的活跃范围有重叠 OR
   - 它们位于可能同时执行的异步区域中且 offset 重叠
   - 对于 partitioned buffer：它们在同一物理分区中（`getPartitionIndex` 相同）

3. **`allocate`：** first-fit 图着色。为 buffer 分配颜色，颜色 0 的 buffer 保持原偏移，颜色 k 的 buffer 偏移需要移位到不与颜色 <k 的干涉节点重叠的位置。对于 partition neighbor 干涉，不是移位到邻居结束之后，而是跳转到下一个物理分区边界。

4. 着色后再重建干涉图，因为偏移移位可能导致新的干涉。反复迭代直到干涉图为空（不点动收敛）。

### 11.4.3 Alias 分析

源码位置：`triton/lib/Analysis/Alias.cpp`。

`SharedMemoryAliasAnalysis` 是一个 MLIR **稀疏前向数据流分析**（`SparseForwardDataFlowAnalysis`），它传播 `AliasInfo`——每个共享内存值所关联的 `ttg.local_alloc` 操作的集合。

**核心规则（`visitOperation`）：**

| Op 类型 | Alias 行为 |
|---------|-----------|
| `ttg.local_alloc` | 创建新的 allocation，alias set = {自身} |
| `MemDescViewTrait` 操作（如 `ttg.memdesc_trans`、`ttg.memdesc_reshape`） | 直接继承操作数的 alias set |
| `arith.select`（返回值是 memdesc 类型） | 取两个操作数 alias set 的并集 |
| `ub.poison` | 空 alias set |

**在 Allocation 中的使用：** 当两个值共享 alias 关系时（例如通过 `scf.for` 的 block argument），它们的活跃范围会被合并（`resolveAliasBufferLiveness`），确保底层 buffer 在整个 alias 链活跃期间都不被复用。

```mermaid
flowchart LR
    alloc1["alloc v1<br/>buffer B1"] --> yield1["scf.yield v6<br/>alias: {v1}"]
    alloc2["alloc v2<br/>buffer B2"] --> yield2["..."]
    yield1 --> ba["scf.for 的<br/>block arg v3<br/>alias: {v1}"]

    subgraph "AliasInfo propagation"
        direction TB
        A["v1: {v1}"] --> C["v3: {v1}"]
        B["v2: {v2}"]
    end

    style alloc1 fill:#e1f5fe
    style alloc2 fill:#fff3e0
```

> 含义：v1 的 liveness 需要扩展到 v3 和 v6 的使用点，因为它们在 alias 意义上引用同一块共享内存。

### 11.4.4 BufferRegion 分析

源码位置：`triton/lib/Analysis/BufferRegion.cpp`。

`BufferRegionAnalysis` 是另一个 MLIR 稀疏前向数据流分析，它追踪每个 memdesc 类型的值的**细化子区域**信息（支持 shared memory 和 tensor memory 两种内存空间）。当代码通过 `ttg.memdesc_subslice` 只访问一个大的共享内存 buffer 的某一部分时，BufferRegion 分析确定具体的 `[baseOffset, length]` 访问范围。

**核心规则（`visitOperation`）：**

| Op | 行为 |
|----|------|
| `ttg.local_alloc` | 插入 `{offset, total_size}` 的 region |
| `ttg.memdesc_index` | 根据 index 将 region 拆分为多个子 buffer region |
| `ttg.memdesc_subslice` | 通过 LinearLayout 反演计算 subslice 在 buffer 中的字节偏移 |
| `ttg.memdesc_trans` / `ttg.memdesc_reshape` | 继承操作数的 region（passthrough） |
| `ttg.local_load` / `ttg.local_store` / `ttg.async_copy_global_to_local` | 记录为 memory access operation，收集其访问的 region |

**用途：** BufferRegion 分析主要用于 **ConcurrencySanitizer**（并发安全检查）——确定两个操作是否访问了共享内存的相同或重叠区域，以检测潜在的数据竞争。它运行在 `AllocateSharedMemory` pass 之后（需要读取 `allocation.offset` 属性）。

### 11.4.5 AllocateSharedMemory pass

源码位置：`triton/lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp`。

这是将 Allocation 分析结果"落地"为 IR 属性的 pass：

```cpp
struct AllocateSharedMemory : public AllocateSharedMemoryBase<AllocateSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);
    attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
```

`ModuleAllocation` 对整个 module 的调用图进行 post-order 遍历，为每个函数独立计算共享内存分配。然后 `attachAllocationSizeAndOffsetAttr` 执行两个任务：

1. **设置模块级属性：** `ttg.shared` = 总共享内存字节数
2. **设置操作级属性：** 对每个产生或消费共享内存的操作（`ttg.local_alloc`、`ttg.convert_layout` 等），添加 `allocation.offset` 属性（对于 partitioned tensor，添加 offset 数组）

这些属性在后续的 `TritonGPUToLLVM` 代码生成阶段被读取，用于生成正确的共享内存地址。

### 11.4.6 Global Scratch Memory Allocation

源码位置：`triton/lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp`。

当某些操作（如 `ttg.global_scratch_alloc`）需要在全局内存中分配临时空间时，`TritonGPUGlobalScratchAllocationPass` 负责分配偏移量。

**算法（`allocateGMem`）：**

1. 递归处理调用图中所有被调用函数（确保 callee 的 scratch 分配已完成）
2. 对函数内的所有 `ttg.global_scratch_alloc` 操作，按 post-order 顺序分配偏移量（简单递进，不考虑活跃范围复用——代码中注释 `TODO: Use a real algorithm`）
3. 对 `triton.call` 操作，同样为其 callee 所需的 scratch 分配偏移量
4. 设置 `ttg.global_scratch_memory_size` / `ttg.global_scratch_memory_offset` 等属性

此外还支持 **profile scratch memory**（用于 profiling 工具的第三方分配），通过 `third_party_allocation` 属性区分。

### 11.4.7 Python 侧运行时分配接口

源码位置：`triton/python/triton/runtime/_allocation.py`。

Triton 在 Python 侧暴露了 `set_allocator` 和 `set_profile_allocator` 两个接口，允许用户在 kernel launch 时提供自定义的全局内存分配器（`Allocator` 是一个接受 `(size, alignment, stream)` 并返回 `Buffer` 的 callable）。这使得运行时系统能够在 kernel 需要全局 scratch 内存时动态分配，而非在编译期将所有地址固定。

---

## 11.5 Triton 生态与整体设计哲学

### 11.5.1 两级 IR 与内存管理的分工

Triton 的两级 IR 设计在本章有清晰的体现：

- **TTIR 层：** 不涉及任何内存布局——操作使用 `tt.ptr<T>` 和 `tt.tensor<shape x T>`，没有内存空间的概念
- **TTGIR 层：** 引入 `ttg.local_alloc` / `ttg.local_load` / `ttg.local_store`、memory space（`global` / `shared` / `register`）、以及 layout encoding。**本章讨论的所有内存管理（Allocation、Alias、BufferRegion）都发生在 TTGIR 层**

这种分离使得内存管理完全在 GPU 感知的 IR 层解决，保证了 TTIR 的硬件无关性。

### 11.5.2 与 MLIR 标准方言的关系

Triton 没有使用 MLIR 标准方言中的 `memref` 来表示共享内存，而是自定义了 `MemDescType` 体系。原因在于：

1. **Layout 系统集成：** Triton 的共享内存 layout（`SharedEncoding`、`PaddedSharedEncoding`、`PartitionedSharedEncoding` 等）是 Triton 特有的概念，`memref` 无法表达
2. **Swizzling 感知：** `getNumScratchElemsSwizzledCvt` 等函数在计算 scratch 大小时需要考虑 swizzling 模式（不同的 swizzling 策略影响 bank conflict 和访问模式）
3. **Buffer 粒度：** Triton 的 buffer 以整个 "tile" 为粒度（而非单个元素），这与 tile-first 编程模型一致

### 11.5.3 与 CUDA 编程模型的对应关系

从 CUDA 程序员的角度看：

| CUDA 概念 | Triton 对应 |
|-----------|------------|
| `extern __shared__ float s_data[]` | `ttg.local_alloc` + `allocation.offset` 属性 |
| 手动管理共享内存偏移（`s_data + offset`） | Allocation 分析自动计算偏移 |
| `__syncthreads()` 确保共享内存一致性 | `ttg.async_wait` / barrier（第 10 章讨论） |
| 寄存器自动分配（编译器负责） | 委托 LLVM/NVPTX 后端 |
| 每个 thread 最多 255 个寄存器 | `num_warps` 参数间接控制（更多 warps = 每 thread 更少寄存器） |

Triton 隐藏了手动共享内存偏移计算——程序员只需用 `ttg.local_alloc` 声明 buffer，编译器自动分配和复用内存。

### 11.5.4 寄存器压力与 tile 参数的关系

在 Triton 的编程模型中，程序员不直接管理寄存器。但以下参数的组合间接决定了寄存器压力：

1. **`BLOCK_SIZE`（tile 大小）：** 更大的 tile 意味着每个 thread 处理更多元素，更多值需要存储在寄存器中
2. **`num_warps`：** 更多 warps 分到同一 block 中，共享同一块 register file。每个 SM 的寄存器总量固定（如 NVIDIA A100 每个 SM 有 65536 个 32-bit 寄存器），num_warps 越大，每 warp（每 thread）可用的寄存器越少
3. **Layout encoding：** 不同的 layout（blocked、mma、dot_op）决定了数据在 warps 和 threads 之间的分布方式，影响每个 thread 持有的寄存器数量
4. **Autotuning 中的寄存器约束：** `OutOfResources` 错误会在寄存器不足或共享内存不足时触发（参见 `compiler.py` 第 434-468 行），autotuner 据此剪枝搜索空间

```python
# 示例：展示 tile 大小和 num_warps 对寄存器压力的影响
import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,  # tile 大小直接影响每线程寄存器使用
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 加载 A tile —— BLOCK_M * BLOCK_K 个元素分布在 warps 间
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_k = tl.arange(0, BLOCK_K)[None, :]
    a = tl.load(a_ptr + offs_m * K + offs_k)  # 每线程持有的元素数 = BLOCK_M*BLOCK_K / (num_warps*32)

    # 加载 B tile
    offs_k = tl.arange(0, BLOCK_K)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    b = tl.load(b_ptr + offs_k * N + offs_n)

    # 计算 —— 累加器也在寄存器中
    c = tl.dot(a, b)  # MMA 指令使用特殊 layout，寄存器消耗更多

    # 存储结果
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(c_ptr + offs_m * N + offs_n, c)

# 运行示例：
# a = torch.randn(256, 128, device='cuda', dtype=torch.float16)
# b = torch.randn(128, 512, device='cuda', dtype=torch.float16)
# c = torch.empty(256, 512, device='cuda', dtype=torch.float16)
# grid = (triton.cdiv(256, BLOCK_M), triton.cdiv(512, BLOCK_N))
# matmul_kernel[grid](a, b, c, 256, 512, 128, BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, num_warps=4)
```

在上面的例子中，如果设置 `BLOCK_M=128, BLOCK_N=256, num_warps=4`，每个 thread 持有的寄存器可能超出硬件限制，导致编译失败或性能下降（强制 spill 到 local memory）。

---

## 11.6 章节小结

### 关键要点回顾

1. **寄存器分配是编译器后端的基础问题：** 活跃范围分析通过后向数据流方程确定值的生命周期；干涉图编码了值的共存约束；图着色算法（Chaitin-Briggs）和线性扫描是两种主要的启发式求解方案。

2. **Triton 分层管理内存：** 寄存器分配委托 LLVM/NVPTX 后端（使用成熟的图着色/线性扫描算法）；共享内存由 Triton 自己的 `Allocation` 分析管理（buffer 级，first-fit 图着色）；全局 scratch 内存作为溢出空间（简单顺序分配）。

3. **Allocation 分析的核心是迭代不点动：** 先按降序计算初始偏移（大块优先），再反复构建干涉图 -> 图着色 -> 检查收敛。干涉条件综合考虑了活跃范围重叠、偏移范围和异步区域并发。

4. **Alias 分析和 BufferRegion 分析服务于内存正确性：** Alias 分析扩展了 buffer 的活跃范围（通过控制流中的值传递）；BufferRegion 分析追踪共享内存的细化访问模式，支撑并发安全检查。

5. **寄存器压力由 tile 大小、num_warps 和 layout encoding 共同决定：** autotuner 在搜索空间中检测 `OutOfResources` 以剪枝不可行的配置组合。

### 与下一章的逻辑衔接

本章完成后，共享内存的布局已经确定（每个 buffer 的偏移量作为 IR 属性附加到了操作上）。第 12 章将讨论后端代码发射的最后一个环节——LLVM IR 如何被翻译为 PTX 汇编，再由 PTX 汇编器生成最终的 CUBIN 可执行文件。共享内存分配的结果（`allocation.offset` 属性和 `ttg.shared` 属性）将在 PTX 生成阶段被转换为 `.shared` 段声明和偏移量计算。

### 推荐深入阅读材料

- *Engineering a Compiler*, Keith D. Cooper & Linda Torczon, 3rd Edition, Chapter 12 (Register Allocation) -- 完整的寄存器分配理论，包括 Chaitin-Briggs 算法的详细步骤
- *Algorithms for Compile-Time Memory Optimization*, B. Ramakrishna Rau, et al., ACM 1993 -- `computeOffsets()` 所参考的经典论文，关于编译期内存复用优化
- *Register Allocation via Graph Coloring*, G. J. Chaitin, et al., 1981 -- 图着色寄存器分配的原始论文
- MLIR Liveness Analysis 文档: `mlir/Analysis/Liveness.h` -- Triton 直接使用的 MLIR 活跃分析基础设施
- Triton 源码中 `AllocationAnalysis` 类的实现注释 -- `triton/lib/Analysis/Allocation.cpp` 第 514-573 行的 `calculateStarts` 方法包含详细的算法图示注释

---

## 正确性校验报告

### 通过的验证项

1. **源码验证：** 所有引用的文件路径和函数名均已在 workspace 中确认存在：
   - `triton/lib/Analysis/Allocation.cpp`：确认 `AllocationAnalysis` 类的 `run()`、`getValuesAndSizes()`、`resolveLiveness()`、`computeOffsets()`、`calculateStarts()`、`buildInterferenceGraph()`、`allocate()` 方法均存在
   - `triton/include/triton/Analysis/Allocation.h`：确认 `BufferT` 结构体（含 `BufferKind` 枚举）、`Interval<T>`、`Allocation` 类、`ModuleAllocation` 类定义均与描述一致
   - `triton/lib/Analysis/Alias.cpp`：确认 `SharedMemoryAliasAnalysis::visitOperation` 的实现逻辑（`LocalAllocOp` 创建新 alloc，`MemDescViewTrait` 继承 alias，`SelectOp` union alias）
   - `triton/lib/Analysis/BufferRegion.cpp`：确认 `BufferRegionAnalysis::visitOperation` 中对 `LocalAllocOp`、`MemDescIndexOp`、`MemDescSubsliceOp`、`MemDescTransOp` 等操作的处理逻辑
   - `triton/lib/Conversion/TritonGPUToLLVM/AllocateSharedMemory.cpp`：确认 pass 结构为简单的 `ModuleAllocation` + `attachAllocationSizeAndOffsetAttr`
   - `triton/lib/Conversion/TritonGPUToLLVM/GlobalScratchMemoryAllocation.cpp`：确认 `allocateGMem` 递归处理和简单顺序分配策略
   - `triton/python/triton/runtime/_allocation.py`：确认 `set_allocator` 和 `set_profile_allocator` 接口

2. **教材交叉验证：** 活跃变量分析的数据流方程（backward analysis, USE/DEF sets）、干涉图定义、图着色算法的 build-coalesce-spillCost-simplify-spill-select 流程均与 EaC 第 12 章一致

3. **MLIR 文档验证：** `Liveness` 类（`mlir/Analysis/Liveness.h`）和 `SparseForwardDataFlowAnalysis` 的使用模式确认与 MLIR 官方文档一致

4. **TableGen 定义验证：** `AllocateSharedMemory` pass 在 `Passes.td` 中定义（第 6-15 行），`TritonGPUGlobalScratchAllocationPass` 在 `Passes.td` 中定义（第 17-27 行），描述与实现一致

### 无法确认的描述

- Global Scratch Memory Allocation 中提到的 `profile_scratch_memory` 的具体使用场景（profiling 工具集成）未在源码中找到明确的说明文档，标注为"基于属性命名推断"
- Triton 是否在某些情况下完全绕过 LLVM 后端直接生成 PTX 内联汇编（inline assembly）——在 `ConvertLayoutOpToLLVM` 中观察到部分共享内存操作通过 LLVM 内联汇编实现，但这不是"绕过 LLVM"，仍是利用 LLVM 的 inline asm 机制

### 发现并修正的错误

- 原始任务描述中提到 `triton/python/triton/runtime/_allocation.py` 的"Python-side allocation utilities"——实际代码中，该文件只定义了 `Allocator` / `Buffer` Protocol 和全局变量 `_allocator` / `_profile_allocator`，是一个运行时 allocator 接口而非完整的 allocation utility。已在正文中按实际内容描述。
