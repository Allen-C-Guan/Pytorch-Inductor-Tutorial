# 第六章：调度层 —— 从 IR 到执行计划

## 本章导读

在第四章中，我们认识了 Inductor 的中间表示（IR）—— 以 `Buffer` 和 `IRNode` 为核心的数据结构。在第五章中，我们了解了 `GraphLowering` 如何将 FX Graph 翻译为 IR。现在，我们面临一个关键问题：**有了 IR 之后，如何决定哪些操作合并为一个 kernel？以什么顺序执行？分配多少资源？**

这正是调度层（Scheduling Layer）的职责。如果说 Lowering 层是编译器的 "前端到中端" 的桥梁，那么调度层就是编译器的 **指令调度（Instruction Scheduling）+ 循环优化（Loop Optimization）+ 寄存器分配（Register Allocation）** 的三位一体。

本章将深入剖析 Inductor 调度层的三大核心组件：

1. **`Scheduler`** — 全局调度器，编译中期的总编排者
2. **`SchedulerNode` 继承树** — 调度节点的类型体系
3. **`BaseScheduling` 继承树** — 后端调度策略的多态实现

最后，我们将通过一个完整的追踪示例，展示 `[matmul -> relu -> add]` 序列如何经过调度层的全部决策过程。

---

## 6.1 Scheduler — 全局调度器

### 6.1.1 定位与角色

**源文件**：`torch/_inductor/scheduler.py`（约 L2015）

`Scheduler` 是一个独立类，不继承自任何基类，也没有子类。它是 Inductor 编译中期唯一的编排者——接收 IR 操作列表，输出有序的、经过融合优化的调度节点序列。

用一个编译器的类比来理解：

| 编译器阶段 | 传统编译器（如 LLVM） | Inductor |
|-----------|---------------------|----------|
| 前端 | Clang（C/C++ -> LLVM IR） | GraphLowering（FX Graph -> Inductor IR） |
| **中端优化 + 指令调度** | **Pass Manager + Scheduler** | **Scheduler** |
| 后端代码生成 | SelectionDAG -> MachineInstr | BaseScheduling -> Kernel codegen |

`Scheduler` 同时承担了传统编译器中三个独立的 pass：

- **指令调度（Instruction Scheduling）**：确定操作的执行顺序（拓扑排序）
- **循环优化（Loop Optimization）**：决定哪些循环可以融合（算子融合）
- **寄存器分配（Register Allocation）**：决定中间结果的存储策略（缓冲区生命周期）

### 6.1.2 核心职责

```
Scheduler 的四大职责
═══════════════════════

  ① 拓扑排序 ────── 按数据依赖排列 IR 节点的执行顺序
                    编译器类比：基本块内的指令排序

  ② 算子融合 ────── 将多个 IR 节点合并为一个 kernel
                    编译器类比：循环融合（Loop Fusion）

  ③ 内存规划 ────── 分析缓冲区生命周期，决定分配和释放时机
                    编译器类比：寄存器分配（Register Allocation）

  ④ 后端分派 ────── 按设备类型选择后端调度器（Triton/Cpp/Metal）
                    编译器类比：目标机器的指令选择
```

### 6.1.3 输入与输出

**输入**：`V.graph`（`GraphLowering` 的产物）—— 一组 IR Buffer 和 Operation 的集合

**输出**：有序的 `BaseSchedulerNode` 列表 —— 每个节点代表一个待生成的 kernel 或外部调用

数据流简图：

```
                    GraphLowering 阶段                      Scheduler 阶段
                    ════════════════                        ════════════════

  FX Graph          Lowering 规则         IR Buffer          融合 + 排序          调度节点
  ────────      ─────────────────►     ────────────      ────────────────►     ──────────────
  placeholder    → InputBuffer         buf_x (Input)
  matmul         → ComputedBuffer      buf_matmul (mm)
  relu           → ComputedBuffer      buf_relu (relu)
  add            → ComputedBuffer      buf_add (add)      ┌─ SchedulerNode(matmul) ───┐
  output         → 收集输出                                │  拓扑排序                  │
                                                          │  融合决策                  │
                                                          │  内存规划                  │
                                                          └─ FusedNode(relu + add) ──┘
```

### 6.1.4 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `self.nodes` | `list[BaseSchedulerNode]` | 经过拓扑排序和融合后的调度节点列表 |
| `self.backends` | `dict[device, BaseScheduling]` | 懒加载的后端调度器字典 |
| `self.scheduler_nodes_map` | `dict[Buffer, BaseSchedulerNode]` | IR Buffer 到 SchedulerNode 的映射 |
| `self.name_to_fused_node` | `dict[str, FusedSchedulerNode]` | 已融合节点的名称索引 |

**关键设计**：`self.backends` 是懒加载的。只有在代码生成阶段首次遇到某设备的节点时，才会创建对应的 `BaseScheduling` 实例。如果图里只有 CPU 节点，GPU 调度器永远不会被创建。

### 6.1.5 `Scheduler.__init__()` — 构建与初始化

构造函数的核心工作是：收集所有 IR Buffer，为每个 Buffer 创建调度节点，然后构建依赖图。

```
Scheduler.__init__(self, nodes: list[ir.Operation])
│
├── 1. 收集 IR Buffer
│     遍历所有 ir.Operation，收集:
│       - 输入 buffer（InputBuffer）
│       - 计算buffer（ComputedBuffer）
│       - 常量buffer（ConstantBuffer）
│     去重后得到 buffer 列表
│
├── 2. 创建 SchedulerNode
│     对每个 buffer:
│       - InputBuffer       → 不创建节点（纯数据源）
│       - ComputedBuffer    → SchedulerNode(buffer)
│       - ExternKernel      → ExternKernelSchedulerNode
│       - 模板匹配的节点    → TemplateSchedulerNode
│
├── 3. 依赖分析
│     使用 _RecordLoadStoreInner handler 执行每个 IR 闭包:
│       - 记录每个节点读取了哪些 buffer（MemoryDep 对象）
│       - 记录每个节点写入了哪些 buffer
│     构建依赖图：
│       node_A 依赖 node_B ⟺ node_A 读取了 node_B 的输出 buffer
│
├── 4. 算子融合
│     调用后端的 can_fuse_vertical / can_fuse_horizontal:
│       - 可融合的节点合并为 FusedSchedulerNode
│       - 更新依赖关系
│
└── 5. 拓扑排序
      基于 DAG 的拓扑排序，保证：
        - 被依赖的节点先于依赖者执行
        - 融合后的节点作为一个整体参与排序
```

### 6.1.6 `Scheduler.compile()` — 编译主流程

```
Scheduler.codegen()
│
├── 1. 遍历 self.nodes（已排好序的调度节点）
│     对每个 node:
│       │
│       ├── 判断节点类型
│       │     ├── ExternKernelSchedulerNode → 外部 kernel 调用
│       │     ├── TemplateSchedulerNode     → 模板 kernel
│       │     └── 普通 / FusedSchedulerNode → 后端 codegen
│       │
│       ├── 选择后端调度器
│       │     get_backend(node.device):
│       │       首次调用时懒创建:
│       │         CPU  → CppScheduling(self)
│       │         CUDA → CUDACombinedScheduling(self)
│       │         MPS  → MetalScheduling(self)
│       │
│       ├── 分派代码生成
│       │     backend.codegen_node(node)    # 普通节点
│       │     或 backend.codegen_template(node)  # 模板节点
│       │
│       └── 刷新判断
│             if backend.ready_to_flush():
│                 backend.flush()  # 将积攒的代码刷入 wrapper_code
│
├── 2. 最终 flush
│     对每个已创建的后端，调用 flush() 刷新所有未决代码
│
└── 3. 返回
      所有 kernel 代码已写入 V.graph.wrapper_code
```

### 6.1.7 依赖分析：`_RecordLoadStoreInner`

在调度开始之前，Scheduler 必须精确理解节点之间的数据依赖关系。这是通过 `_RecordLoadStoreInner` handler 实现的。

`_RecordLoadStoreInner` 是第三章介绍的 "分析型 Handler 族" 的一员。它的核心思路是：**重新执行每个 IR 节点的闭包（loop body callable），但把 `V.ops` 替换为一个只记录不计算的 handler**。

```
_RecordLoadStoreInner 的工作原理
═══════════════════════════════════

  正常 codegen 时:
    V.ops = TritonOverrides()
    loop_body()  →  产生 "tl.load()" / "tl.store()" 代码

  依赖分析时:
    V.ops = _RecordLoadStoreInner()
    loop_body()  →  记录 MemoryDep("buf_matmul", index=[0, 128])
                     记录 MemoryDep("buf_relu", index=[0, 128])
                     不产生任何代码，只记录依赖

  输出：一组 (reads, writes) 集合
    reads  = {MemoryDep("buf_x", ...), MemoryDep("buf_matmul", ...)}
    writes = {MemoryDep("buf_add", ...)}
```

编译器类比：这等价于传统编译器中对 SSA 形式 IR 进行 **use-def 链（使用-定义链）** 分析——确定每个值的定义点和所有使用点。

### 6.1.8 生命周期

```
时间线 ──────────────────────────────────────────────────────────►

  GraphLowering.run()
  │
  │  IR 构建完成
  │     │
  │     ▼
  │  GraphLowering._update_scheduler()
  │     │
  │     ├── scheduler = Scheduler(operations)    ◄── 创建
  │     │     └── __init__: 融合 + 排序 + 依赖分析
  │     │
  │     └── V.graph.scheduler = scheduler         ◄── 注册到全局
  │
  │  GraphLowering.codegen()
  │     │
  │     ├── scheduler.codegen()                    ◄── 使用
  │     │     └── 遍历 nodes → 分派 codegen
  │     │
  │     └── scheduler 不再被活跃使用
  │
  │  GraphLowering 实例被 GC 回收
  │     └── scheduler 随之回收                     ◄── 销毁
```

**创建者**：`GraphLowering._update_scheduler()` 调用 `Scheduler(self.operations)`

**全局注册**：`V.graph.scheduler`

**存活范围**：Phase 2 codegen 阶段

---

## 6.2 SchedulerNode 继承树

### 6.2.1 类层次总览

```
BaseSchedulerNode (scheduler.py, 抽象基类)
│
├── SchedulerNode              最常见的节点类型
│     一个 IR ComputedBuffer → 一个 SchedulerNode
│
├── FusedSchedulerNode         融合节点
│     多个 SchedulerNode 合并 → 一个 FusedSchedulerNode → 一个 kernel
│
├── ExternKernelSchedulerNode  外部 kernel 节点
│     调用外部库（cuBLAS, cuDNN, ATen fallback）
│
└── TemplateSchedulerNode      模板 kernel 节点
      预定义的优化 kernel（flash_attention, cutlass_gemm）
```

### 6.2.2 BaseSchedulerNode — 抽象基类

**源文件**：`torch/_inductor/scheduler.py`

`BaseSchedulerNode` 是所有调度节点的抽象基类，定义了调度节点的公共接口和属性。它不继承自任何外部基类。

**核心属性**：

| 属性 | 类型 | 说明 |
|------|------|------|
| `node` | `ir.Buffer` | 对应的底层 IR buffer |
| `dependencies` | `set[BaseSchedulerNode]` | 此节点依赖的前驱节点集合 |
| `users` | `set[BaseSchedulerNode]` | 依赖此节点的后继节点集合 |
| `group` | `(device, group_key)` | 设备和调度分组键 |
| `last_usage` | `set[str]` | 此节点最后一次使用哪些 buffer（用于内存释放决策） |

**编译器类比**：

如果把 Scheduler 看作一个指令调度器，那么 `BaseSchedulerNode` 就是一条 **调度单元（Scheduling Unit）**——类似于 LLVM 中的 `SDNode`（SelectionDAG Node）或 `MachineInstr`。它封装了一条"指令"的所有调度信息：依赖、资源需求、执行设备。

### 6.2.3 SchedulerNode — 标准调度节点

`SchedulerNode` 是最常见、最基本的调度节点类型。在 `Scheduler.__init__()` 的初始阶段，几乎每个 `ComputedBuffer` 都会被包装为一个 `SchedulerNode`。

**输入**：一个 `ComputedBuffer`（IR 层的计算结果）

**输出**：调度后的执行位置 + 代码生成指令

**生命周期**：

```
Scheduler.__init__() 阶段:
  ComputedBuffer("buf_relu") → SchedulerNode(buf_relu)
    │
    │  融合分析阶段:
    │  如果 can_fuse(relu_node, add_node) == True:
    │    relu_node 被吸收进 FusedSchedulerNode([relu_node, add_node])
    │    原始 relu_node 标记为 "已融合"
    │
    │  如果没有被融合:
    │    relu_node 独立存在，最终生成一个单独的 kernel
    │
    ▼
  代码生成阶段:
    独立节点 → backend.codegen_node(relu_node) → 一个 kernel
```

**关键追踪信息**：

| 信息 | 用途 |
|------|------|
| 读取的 buffer 列表 | 决定输入依赖 |
| 写入的 buffer | 决定输出 |
| buffer 大小（numel） | 估计计算量 |
| 迭代域（iteration domain） | 判断融合兼容性 |
| reduction 维度 | 判断是否可以与 pointwise 融合 |

### 6.2.4 FusedSchedulerNode — 融合节点

`FusedSchedulerNode` 是算子融合的直接产物。当调度器判定多个 SchedulerNode 可以合并执行时，它们被包装为一个 `FusedSchedulerNode`。

**输入**：多个 `SchedulerNode`

**输出**：一个 kernel（融合后的所有操作在一个 kernel 中完成）

**编译器类比**：**循环融合（Loop Fusion）**。在传统编译器中，两个独立的循环：

```c
// 融合前：两个循环，两次遍历数组
for (int i = 0; i < N; i++) { b[i] = relu(a[i]); }
for (int i = 0; i < N; i++) { c[i] = b[i] + d[i]; }

// 融合后：一个循环，一次遍历
for (int i = 0; i < N; i++) {
    tmp = relu(a[i]);
    c[i] = tmp + d[i];  // tmp 不需要写入内存
}
```

Inductor 的融合决策与此完全相同：如果两个操作的迭代域（iteration domain）相同，且中间结果可以被 "传递" 而不需要写回内存，就融合它们。

**FusedSchedulerNode 的结构**：

```
FusedSchedulerNode
│
├── constituents: list[SchedulerNode]
│     ├── [0] relu_node   ← 被融合的第一个节点
│     ├── [1] add_node    ← 被融合的第二个节点
│     └── ...             ← 可能更多
│
├── group: (device, group_key)
│     所有被融合节点的公共迭代域
│
├── dependencies: 并集
│     所有 constituents 的前驱依赖的并集
│     （去除内部的相互依赖）
│
└── users: 并集
      所有 constituents 的后继用户的并集
```

**重要约束**：被融合的节点必须满足：

1. **同一设备**：不能跨设备融合
2. **兼容的迭代域**：相同的元素数量（numel）或可融合的 reduction
3. **可接受的寄存器压力**：融合太多节点会导致 GPU 寄存器溢出
4. **非外部 kernel 边界**：cuBLAS 等外部调用的输出必须物化到内存

### 6.2.5 ExternKernelSchedulerNode — 外部 kernel 节点

`ExternKernelSchedulerNode` 代表一次对外部库函数的调用——那些不由 Inductor 自身生成代码，而是委托给已有的高性能库的操作。

**常见触发场景**：

| 操作 | 外部库 | 原因 |
|------|--------|------|
| `torch.matmul` | cuBLAS | 高度优化的 GEMM 实现 |
| `torch.nn.functional.conv2d` | cuDNN | 多种卷积算法的 autotuning |
| `torch.nn.functional.batch_norm` | ATen fallback | 复杂的状态管理 |
| 未实现 lowering 的算子 | ATen fallback | 兜底机制 |

**关键特征**：

- **不能与其他节点融合**：外部库的输入/输出必须是完整的 tensor，不能是中间的寄存器值
- **形成调度边界**：外部调用之前的所有依赖必须完成，之后的所有消费者必须等待
- **产生物化的 buffer**：结果必须写回内存（不能留在寄存器中传递给下一个操作）

```
调度边界的示意：

  ... → pointwise_op → pointwise_op → [cuBLAS matmul] → relu → add → ...
                                                  ↑
                                            调度边界
                                     结果必须物化到内存
                                     不能与前后操作融合
```

### 6.2.6 TemplateSchedulerNode — 模板 kernel 节点

`TemplateSchedulerNode` 代表一个 **预定义的优化 kernel 模板**。Inductor 维护了一组针对特定计算模式手工编写的 Triton kernel，当 IR 中的操作序列匹配到这些模式时，直接使用模板而非通用代码生成路径。

**常见模板**：

| 模板名称 | 匹配模式 | 优势 |
|----------|---------|------|
| `flash_attention` | `scaled_dot_product_attention` | IO 复杂度从 O(N^2) 降到 O(N) |
| `cutlass_gemm` | 大规模 matmul + epilogue | CUTLASS 的流水线化 GEMM |
| `persistent_reduction` | 特定 reduction 模式 | 减少 global memory 访问 |

**与普通节点的区别**：

- **不走 SIMDScheduling 的通用代码生成流程**
- **自包含**：模板自身定义了完整的 kernel 代码结构
- **不与其他节点融合**（除非支持 epilogue fusion，如 CUTLASS）

### 6.2.7 节点间的数据依赖关系

以一个简单的计算图为例，展示调度节点间的依赖关系：

```
计算图：z = relu(x @ y) + w

IR Buffer:
  buf_x       = InputBuffer("x", [M, K])
  buf_y       = InputBuffer("y", [K, N])
  buf_matmul  = ComputedBuffer("matmul", [M, N])   ← matmul(x, y)
  buf_relu    = ComputedBuffer("relu", [M, N])      ← relu(matmul)
  buf_w       = InputBuffer("w", [M, N])
  buf_add     = ComputedBuffer("add", [M, N])       ← relu + w

调度节点:
  sn_matmul = ExternKernelSchedulerNode(matmul)  ← cuBLAS
  sn_relu   = SchedulerNode(relu)
  sn_add    = SchedulerNode(add)

依赖图 (DAG):

  sn_matmul ◄── sn_x (InputBuffer, 不创建节点)
       │     ◄── sn_y (InputBuffer, 不创建节点)
       │
       ▼
  sn_relu
       │
       ├──► sn_w (InputBuffer)
       │
       ▼
  sn_add

  拓扑排序结果: [sn_matmul, sn_relu, sn_add]
```

---

## 6.3 BaseScheduling 继承树 — 后端调度策略

### 6.3.1 为什么需要后端调度策略？

Inductor 支持多种硬件后端（CPU、CUDA、MPS 等），每种后端对融合、分块（tiling）、代码生成有着截然不同的策略。例如：

- **Triton (GPU)**：适合大规模并行，可以通过 autotuning 搜索最优 block size
- **C++ (CPU)**：适合 OpenMP 并行 + SIMD 向量化，融合策略需要考虑 cache locality
- **Metal (Apple GPU)**：与 Triton 共享融合逻辑，但 kernel 的编译和注册方式不同

`BaseScheduling` 继承树采用 **策略模式（Strategy Pattern）** 来解决这个问题：定义统一的调度接口，每个后端提供自己的实现。

### 6.3.2 完整继承树

```
BaseScheduling (scheduler.py:4887, 抽象协议)
│
├── CppScheduling (codegen/cpp.py:4453)
│     CPU C++/OpenMP 后端
│     完整独立实现，自给自足
│
├── SIMDScheduling (codegen/simd.py:1114, 模板方法基类)
│     GPU/SIMD 后端的通用框架
│     提供完整的融合引擎 + tiling 分析 + codegen 管线
│     │
│     ├── TritonScheduling (codegen/triton.py:4099)
│     │     Triton GPU 后端
│     │     增加 autotuning + 异步编译 + GPU benchmarking
│     │
│     ├── MetalScheduling (codegen/mps.py:951)
│     │     Apple Metal 后端
│     │     极轻量，仅重写 define_kernel
│     │
│     └── HalideScheduling (codegen/halide.py:1656)
│           Halide 后端
│           轻量，重写 define_kernel + get_backend_features
│
├── CUDACombinedScheduling (codegen/cuda_combined_scheduling.py:32)
│     委托模式：持有 TritonScheduling + CUDACPPScheduling + ROCmCPPScheduling
│     不实现业务逻辑，仅按节点类型路由
│     （注意：直接继承 BaseScheduling，不是 SIMDScheduling 的子类）
│
├── CUDACPPScheduling (codegen/cuda/cuda_cpp_scheduling.py:39)
│     CUDA CUTLASS 模板后端
│     处理 CUTLASS GEMM/Conv 的 codegen 和 epilogue 融合
│
└── ROCmCPPScheduling (codegen/rocm/rocm_cpp_scheduling.py:18)
      ROCm 模板后端
      最简实现，不支持 epilogue 融合
```

### 6.3.3 设计模式分析

这棵继承树同时使用了三种经典设计模式：

**模式一：模板方法（Template Method）— SIMDScheduling**

`SIMDScheduling` 定义了 GPU/SIMD 后端的完整算法骨架——融合决策、tiling 选择、kernel 创建、代码生成。子类只需重写特定的 "钩子方法"（如 `define_kernel`、`create_kernel_choices`）来适配不同后端。

```
SIMDScheduling 的模板方法骨架:

  codegen_node(node):
    ├── 1. 提取节点信息              ← 所有后端通用
    ├── 2. 计算迭代范围              ← 所有后端通用
    ├── 3. 生成节点调度顺序           ← 所有后端通用
    ├── 4. 选择 tiling 策略          ← 所有后端通用
    ├── 5. 创建 kernel 实例           ← 子类可重写 create_kernel_choices()
    │     默认: [self.kernel_type(...)]
    │     Triton: 多个 autotuning 候选
    ├── 6. 执行 kernel codegen        ← 所有后端通用
    └── 7. 注册 kernel               ← 子类必须重写 define_kernel()
          Triton: async_compile.triton(...)
          Metal:  wrapper.define_kernel(...)
          Halide: async_compile.halide(...)
```

**模式二：策略（Strategy）— BaseScheduling 接口**

`BaseScheduling` 定义了统一的策略接口（`can_fuse_vertical`, `codegen_node`, `flush` 等），每个后端是一个独立的策略实现。`Scheduler` 通过 `self.backends` 字典在运行时选择策略。

```
策略模式的运行时选择:

  Scheduler
    │
    ├── get_backend(cpu_device)
    │     └── 返回 CppScheduling 实例
    │           策略：OpenMP + SIMD 向量化
    │
    ├── get_backend(cuda_device)
    │     └── 返回 CUDACombinedScheduling 实例
    │           策略：委托到 TritonScheduling（autotuning + Triton kernel）
    │
    └── get_backend(mps_device)
          └── 返回 MetalScheduling 实例
                策略：SIMDScheduling 骨架 + Metal kernel 注册
```

**模式三：委托（Delegation）— CUDACombinedScheduling**

`CUDACombinedScheduling` 不实现任何调度逻辑，而是持有三个内部 Scheduling 实例，按节点类型路由到对应后端。这是一种 **组合优于继承** 的设计。

```
CUDACombinedScheduling 的委托路由:

  收到节点
    │
    ├── 节点是 CUTLASS 模板？
    │     └── YES → _cuda_cpp_scheduling.codegen_template()
    │
    ├── 节点是 ROCm 模板？
    │     └── YES → _rocm_cpp_scheduling.codegen_template()
    │
    └── 其他（绝大多数情况）
          └── → _triton_scheduling.codegen_node()
```

### 6.3.4 BaseScheduling — 抽象协议

**源文件**：`torch/_inductor/scheduler.py`（约 L4887）

`BaseScheduling` 定义了所有后端调度器必须实现的接口。它的方法全部默认抛出 `NotImplementedError`，是一份纯粹的 **接口合同**。

**核心接口方法**：

| 方法 | 签名 | 职责 |
|------|------|------|
| `can_fuse_vertical` | `(node1, node2) -> bool` | 判断两个节点是否可以垂直融合（producer-consumer） |
| `can_fuse_horizontal` | `(node1, node2) -> bool` | 判断两个节点是否可以水平融合（sibling） |
| `group_fn` | `(sizes) -> group_key` | 将迭代维度映射为调度分组键 |
| `codegen_node` | `(node) -> None` | 为一个节点生成代码 |
| `codegen_template` | `(node) -> None` | 为一个模板节点生成代码 |
| `define_kernel` | `(name, code) -> None` | 注册一个 kernel 定义 |
| `flush` | `() -> None` | 将积攒的代码刷入 wrapper |
| `codegen_sync` | `() -> None` | 生成设备同步代码 |
| `ready_to_flush` | `() -> bool` | 判断是否准备好刷新（默认返回 False） |
| `fuse` | `(node1, node2) -> FusedSchedulerNode` | 执行融合操作（有默认实现） |

**编译器类比**：`BaseScheduling` 相当于 LLVM 的 `TargetLowering` 接口——它定义了 "目标机器必须回答的问题"，每个后端（X86、ARM、GPU）给出各自的答案。

### 6.3.5 SIMDScheduling — GPU/SIMD 模板方法基类

**源文件**：`torch/_inductor/codegen/simd.py`（约 L1114）

`SIMDScheduling` 是整个调度层最复杂的类，也是理解 Triton 后端的关键。它提供了一个 **完整的融合引擎 + tiling 分析 + 代码生成管线**，子类只需要替换特定环节。

**核心设计思想**：

```
SIMDScheduling 的 "流水线设备" 类比:

  ┌─────────────────────────────────────────────────────┐
  │              SIMDScheduling（流水线框架）              │
  │                                                       │
  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐  │
  │  │ 融合引擎 │─→│Tiling 分析│─→│Kernel 创建│─→│Codegen│  │
  │  │(通用逻辑)│  │(通用逻辑) │  │(可替换)   │  │(通用) │  │
  │  └─────────┘  └──────────┘  └────┬─────┘  └──┬───┘  │
  │                                  │             │       │
  │  ┌───────────────────────────────┘             │       │
  │  │  子类替换 "加工头":                          │       │
  │  │    TritonScheduling → TritonKernel          │       │
  │  │    MetalScheduling  → MetalKernel           │       │
  │  │    HalideScheduling → HalideKernel          │       │
  │  │                                             │       │
  │  │  子类替换 "包装机":                          │       │
  │  │    TritonScheduling → async_compile.triton() │       │
  │  │    MetalScheduling  → Metal kernel 注册     │       │
  │  │    HalideScheduling → async_compile.halide() │       │
  │  └─────────────────────────────────────────────┘       │
  └─────────────────────────────────────────────────────┘
```

**关键方法详解**：

| 方法 | 行号 | 职责 | 子类是否重写 |
|------|------|------|-------------|
| `group_fn` | L1122 | 将多维迭代空间扁平化为一维 group key | 否 |
| `can_fuse_vertical` | L1125 | 约 130 行的融合启发式：检查 numel 兼容、reduction 规则、tiling 兼容 | 否 |
| `can_fuse_horizontal` | L1125 | 水平融合检查：相同的迭代域 + 兼容的数据类型 | 否 |
| `codegen_node` | L1353 | 入口方法：提取节点、计算范围、生成调度、执行 codegen | 否 |
| `generate_node_schedule` | L1257 | 编排节点在 kernel 内的执行顺序 | 否 |
| `codegen_node_schedule` | L1405 | 核心 orchestrator：选择 tiling、创建 kernel、生成代码 | 否 |
| `create_kernel_choices` | L1478 | 创建 kernel 实例列表（默认 `[self.kernel_type(...)]`） | **是** |
| `define_kernel` | L2444 | 注册 kernel 到 wrapper code | **是** |
| `candidate_tilings` | L1731 | 计算 tiling 候选方案（带 LRU 缓存） | 否 |
| `select_tiling` | L2265 | 选择最佳 tiling 策略 | 否 |

**融合启发式（简化版）**：

```
can_fuse_vertical(producer, consumer):
  │
  ├── 基本检查
  │     ├── 设备相同？        → 不同设备不可融合
  │     ├── 不是外部 kernel？  → 外部 kernel 不可融合
  │     └── 不是模板节点？     → 模板节点不可融合（除非支持 epilogue）
  │
  ├── 迭代域兼容
  │     ├── producer 的输出 numel == consumer 的输入 numel？
  │     │     → 相同迭代域，可以 pointwise-to-pointwise 融合
  │     ├── consumer 是 reduction？
  │     │     → reduction 的外层维度与 producer 兼容即可
  │     └── tiling 兼容？
  │           → 两个节点的 tiling 策略不冲突
  │
  ├── 寄存器压力检查
  │     └── 融合后的总寄存器需求不超过阈值？
  │
  └── 返回 True / False

can_fuse_horizontal(node1, node2):
  │
  ├── 迭代域相同？   → 必须是同一形状
  ├── 设备相同？     → 不可跨设备
  ├── 数据类型兼容？  → 不混合不兼容的类型
  └── 不是外部/kernel 节点
```

### 6.3.6 TritonScheduling — Triton GPU 后端

**源文件**：`torch/_inductor/codegen/triton.py`（约 L4099）

`TritonScheduling` 是 `SIMDScheduling` 最重要、功能最丰富的子类。它继承了融合引擎和 tiling 分析的完整骨架，增加了 Triton 特有的 autotuning 和异步编译能力。

**核心类属性**：

```
kernel_type = TritonKernel
backend_features = {
    FOREACH,              # 支持 foreach 操作
    BUCKETIZE,            # 支持 bucketize
    INPLACE_BUFFERS,      # 支持原地缓冲区操作
    MASKED_SCATTER_WITH_INDEX,
    SCAN,                 # 支持 scan 操作
    SORT,                 # 支持 sort 操作
    TRITON_TEMPLATES,     # 支持 Triton 模板
    TUPLE_REDUCTION,      # 支持元组归约
}
```

**重写的方法**：

| 方法 | 行号 | 与 SIMDScheduling 的差异 |
|------|------|------------------------|
| `create_kernel_choices` | L4314 | 默认只创建一个 kernel 实例；Triton 创建多个 autotuning 候选（persistent / non-persistent、cooperative / non-cooperative） |
| `define_kernel` | L4159 | Triton 特有的 kernel 命名（`triton_<category>_<fused_name>_<suffix>`）、源码去重、异步编译 |
| `benchmark_fused_nodes` | L4217 | GPU benchmarking：生成 kernel → PyCodeCache 加载 → 执行测量 |
| `codegen_comment` | L4133 | 在 wrapper 中写入 Triton 调试信息 |

**Autotuning 流程**：

```
TritonScheduling 的 autotuning:

  create_kernel_choices(node):
    │
    ├── 候选 1: TritonKernel(persistent=False, cooperative=False)
    │     标准配置
    │
    ├── 候选 2: TritonKernel(persistent=True, cooperative=False)
    │     持久化 kernel（适合 reduction）
    │
    ├── 候选 3: TritonKernel(persistent=False, cooperative=True)
    │     协作式 kernel（多个 block 协作）
    │
    └── 候选 4: ...（根据节点类型可能有更多候选）

  然后:
    对每个候选 → benchmark_fused_nodes() → 测量执行时间
    选择最快的候选 → 用其配置生成最终 kernel
```

**Kernel 定义流程**：

```
define_kernel(name, kernel_code):
  │
  ├── 1. 命名: triton_<category>_<fused_name>_<suffix>
  │     category: "pointwise" / "reduction" / ...
  │     fused_name: 被融合节点名称的拼接
  │     suffix: 自增编号
  │
  ├── 2. 去重: 检查是否已存在相同源码的 kernel
  │     避免重复编译
  │
  ├── 3. 异步编译: async_compile.triton(name, kernel_code)
  │     将 kernel 提交到编译队列，不阻塞主线程
  │
  └── 4. 注册到 wrapper: V.graph.wrapper_code.define_kernel(...)
```

### 6.3.7 CppScheduling — CPU C++/OpenMP 后端

**源文件**：`torch/_inductor/codegen/cpp.py`（约 L4453）

`CppScheduling` 是一个 **完全独立** 的实现——不使用 `SIMDScheduling` 的模板方法框架，自己实现了全部 `BaseScheduling` 接口。

**为什么 CppScheduling 不用 SIMDScheduling？**

因为 CPU 和 GPU 的调度策略差异太大：

| 维度 | GPU (SIMDScheduling) | CPU (CppScheduling) |
|------|---------------------|---------------------|
| 并行模型 | SIMT（大量线程） | MIMD（少量核心 + SIMD 向量） |
| 融合策略 | 减少 global memory 访问 | 提高 cache locality |
| tiling | 按 block 划分 GPU grid | 按 SIMD 宽度向量化 |
| 代码生成 | Triton kernel（.ptx） | C++ for 循环（.so） |

**核心组件**：

| 组件 | 位置 | 职责 |
|------|------|------|
| `CppKernelProxy` | cpp.py:3800 | kernel 代码生成代理，管理标量 vs 向量化路径选择 |
| `KernelGroup` | cpp.py:5240 | 代码积攒容器，持有 loops_code、args、WorkSharing(OMP) |

**关键方法**：

| 方法 | 行号 | 职责 |
|------|------|------|
| `fuse` | L4488 | CPU 特有融合：foreach / template / 兼容范围 / 外层循环 |
| `can_fuse_vertical` | L4744 | 支持模板 epilogue、水平融合、外层循环融合 |
| `codegen_node` | L5062 | 核心：通过 CppKernelProxy 把节点翻译为 C++ kernel |
| `flush` | L5229 | 把积攒的代码刷出为完整 C++ 函数 |
| `define_kernel` | L5185 | 注册 C++ kernel 到 wrapper code |
| `try_loop_split` | L4757 | 循环分裂优化（消除索引中的除法） |

**KernelGroup 的生命周期**：

```
KernelGroup 是 CppScheduling 特有的 "代码积攒器":

  codegen_node(node1) → finalize_kernel → KernelGroup 积攒代码
  codegen_node(node2) → finalize_kernel → KernelGroup 继续积攒
  flush()             → 代码写出 → reset_kernel_group() → 新的空 KernelGroup
  codegen_node(node3) → finalize_kernel → 新 KernelGroup 开始积攒

  生命周期: [创建] → [积攒多个节点] → [flush] → [销毁] → [新的 KernelGroup]
```

### 6.3.8 MetalScheduling — Apple Metal 后端

**源文件**：`torch/_inductor/codegen/mps.py`（约 L951）

`MetalScheduling` 是 `SIMDScheduling` 的 **最轻量子类**。它只重写了 `define_kernel` 一个方法，其他全部继承父类。

```python
# MetalScheduling 的核心重写（简化）
class MetalScheduling(SIMDScheduling):
    kernel_type = MetalKernel

    def define_kernel(self, name, kernel_code):
        # Metal 特有的 kernel 命名和注册
        wrapper = V.graph.wrapper_code
        wrapper.define_kernel(
            name, kernel_code,
            gpu=False  # Metal 不走 CUDA 的编译路径
        )
```

**设计含义**：Metal 后端与 Triton 共享了全部的融合逻辑、tiling 分析和代码生成骨架，仅在 kernel 注册环节有所不同。这是模板方法模式的精髓——最大化代码复用，最小化后端适配成本。

### 6.3.9 CUDACombinedScheduling — 委托模式

**源文件**：`torch/_inductor/codegen/cuda_combined_scheduling.py`（约 L32）

`CUDACombinedScheduling` 是调度层最有趣的类——它 **不做任何实际工作**，只持有三个内部 Scheduling 实例并按节点类型路由。

```
CUDACombinedScheduling 的内部结构:

  ┌─────────────────────────────────────────────┐
  │         CUDACombinedScheduling               │
  │                                               │
  │  _triton_scheduling = TritonScheduling(self) │  ← 处理 95%+ 的节点
  │  _cuda_cpp_scheduling = CUDACPPScheduling    │  ← 处理 CUTLASS 模板
  │  _rocm_cpp_scheduling = ROCmCPPScheduling    │  ← 处理 ROCm 模板
  │                                               │
  │  所有 BaseScheduling 方法都是简单委托:         │
  │    codegen_node(node)  → _triton_scheduling   │
  │    codegen_template(n) → 按模板类型路由        │
  │    flush()             → _triton_scheduling   │
  │    can_fuse_vertical() → 按节点类型路由        │
  └─────────────────────────────────────────────┘
```

**编译器类比**：这相当于 GCC 的 **多 ABI 后端**——同一个编译器可以生成 32 位和 64 位代码，前端不关心目标 ABI，由一个中间路由层分派到对应的代码生成器。

**路由逻辑**（`choose_node_backend` 方法）：

```
choose_node_backend(node):
  │
  ├── node 是 CUDA C++ 模板（CUTLASS）？
  │     └── YES → _cuda_cpp_scheduling
  │
  ├── node 是 ROCm C++ 模板？
  │     └── YES → _rocm_cpp_scheduling
  │
  └── 其他（绝大多数 pointwise/reduction 操作）
        └── → _triton_scheduling
```

### 6.3.10 设备到后端调度器的映射

| 设备 | 后端调度器 | Kernel 类型 | 配置项 |
|------|-----------|-------------|--------|
| CPU | `CppScheduling` | `CppKernel` / `CppVecKernel` | `config.cpu_backend`（"cpp"/"halide"/"triton"） |
| CUDA | `CUDACombinedScheduling` → `TritonScheduling` | `TritonKernel` | `config.cuda_backend`（"triton"/"halide"） |
| CUDA (模板) | `CUDACombinedScheduling` → `CUDACPPScheduling` | `CUDATemplateKernel` | — |
| MPS | `MetalScheduling` | `MetalKernel` | — |
| XPU | `TritonScheduling` | `TritonKernel` | — |

注册逻辑位于 `codegen/common.py:init_backend_registration()`。

---

## 6.4 融合策略

算子融合（Operator Fusion）是 Inductor 最核心的优化手段。本节详细分析两种融合策略及其决策过程。

### 6.4.1 垂直融合（Vertical Fusion）— Producer-Consumer

**定义**：将一个生产者节点和它的消费者节点合并为一个 kernel。

**收益**：消除中间结果的内存写入和读取。生产者的输出可以直接留在寄存器中，传递给消费者使用，不需要写回 global memory。

**工厂类比**：

```
融合前（不融合）:
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ 工位 A    │     │ 仓库      │     │ 工位 B    │
  │ 生产零件  │─→│ 存入仓库  │─→│ 取出加工  │
  └──────────┘     └──────────┘     └──────────┘
  问题：仓库（内存）读写两次，浪费时间

融合后（垂直融合）:
  ┌────────────────────────────┐
  │          流水线工位          │
  │  工位 A 生产 → 直接传给工位 B │
  │  不经过仓库（内存）           │
  └────────────────────────────┘
  收益：消除两次内存访问
```

**典型例子**：

```
融合前:
  kernel_1: for i in range(N): temp[i] = relu(matmul_out[i])    # 写 temp 到 global memory
  kernel_2: for i in range(N): result[i] = temp[i] + bias[i]   # 读 temp 从 global memory

融合后:
  fused_kernel: for i in range(N):
      temp = relu(matmul_out[i])     # temp 在寄存器中
      result[i] = temp + bias[i]     # 直接使用寄存器中的 temp
  收益: 消除了 temp 数组的 global memory 写入 + 读取
```

**垂直融合的条件**：

```
can_fuse_vertical(producer, consumer):
  │
  ├── 1. 中间结果只使用一次？
  │     如果 producer 的输出被多个 consumer 使用，
  │     融合后仍需为其他 consumer 保留一份物化的 buffer
  │
  ├── 2. 迭代域兼容？
  │     pointwise + pointwise → numel 必须相同
  │     pointwise + reduction → reduction 的外层维度匹配
  │     reduction + pointwise → 通常不融合（反转方向不自然）
  │
  ├── 3. 寄存器压力可接受？
  │     融合后，中间结果必须能放入寄存器
  │     过大的中间 tensor 会导致 register spill
  │
  ├── 4. 后端支持？
  │     外部 kernel（cuBLAS）不可融合
  │     模板节点通常不可融合（除非支持 epilogue）
  │
  └── 5. 设备相同？
        跨设备操作不可融合
```

### 6.4.2 水平融合（Horizontal Fusion）— Sibling

**定义**：将具有相同父节点的兄弟节点合并为一个 kernel。

**收益**：更好地利用内存带宽——输入数据只需读取一次，即可同时计算多个输出。

**典型例子**：

```
融合前:
  kernel_1: for i in range(N): out1[i] = relu(x[i])       # 读 x
  kernel_2: for i in range(N): out2[i] = sigmoid(x[i])    # 读 x（再次）

融合后:
  fused_kernel: for i in range(N):
      val = x[i]               # 只读一次 x
      out1[i] = relu(val)      # 计算 relu
      out2[i] = sigmoid(val)   # 计算 sigmoid
  收益: x 只从 global memory 读取一次
```

**水平融合的条件**：

```
can_fuse_horizontal(node1, node2):
  │
  ├── 1. 迭代域完全相同？
  │     两个节点必须遍历相同数量的元素
  │
  ├── 2. 数据类型兼容？
  │     输入和输出的数据类型可以统一处理
  │
  ├── 3. 设备相同？
  │
  └── 4. 内存访问模式兼容？
        不会因为融合引入不规则的访存模式
```

### 6.4.3 融合决策流程

Scheduler 在初始化阶段对每个节点进行融合决策：

```
融合决策算法（简化版）:

  for node in topological_order:
    │
    ├── 检查垂直融合（与 producer 融合）
    │     for producer in node.producers:
    │       │
    │       ├── 后端支持？
    │       │     backend.can_fuse_vertical(producer, node)
    │       │
    │       ├── 融合有利？
    │       │     计算强度（compute intensity）> 阈值
    │       │     中间结果的大小 < 寄存器压力阈值
    │       │
    │       └── 如果可以 → 创建 FusedSchedulerNode([producer, node])
    │
    ├── 检查水平融合（与 siblings 融合）
    │     for sibling in node.siblings:
    │       │
    │       ├── 相同迭代域？
    │       ├── backend.can_fuse_horizontal(node, sibling)？
    │       │
    │       └── 如果可以 → 合并到同一个 FusedSchedulerNode
    │
    └── 如果不能融合 → node 保持独立的 SchedulerNode
```

### 6.4.4 什么情况下不能融合

理解 "什么阻止了融合" 与理解 "什么使融合生效" 同样重要：

```
阻止融合的因素
═══════════════

  ┌─ 跨设备边界 ─────────────────────────────────────────────┐
  │  CPU 节点不能与 GPU 节点融合。数据必须通过 PCI-e 传输，   │
  │  这比任何融合节省的时间都大得多。                         │
  └────────────────────────────────────────────────────────────┘

  ┌─ 不同的迭代域 ───────────────────────────────────────────┐
  │  reduction(x, dim=1) 遍历 N 个元素                       │
  │  pointwise(y) 遍历 M*N 个元素                            │
  │  迭代次数不同 → 无法在同一个循环中执行                     │
  └────────────────────────────────────────────────────────────┘

  ┌─ 外部 kernel 边界 ───────────────────────────────────────┐
  │  cuBLAS matmul 的输出必须完整地写回 global memory。       │
  │  后续操作不能 "内联" 到 cuBLAS kernel 中。               │
  └────────────────────────────────────────────────────────────┘

  ┌─ 寄存器压力过大 ─────────────────────────────────────────┐
  │  融合太多节点导致 GPU 寄存器不足，发生 register spill。   │
  │  Spill 到 local memory 反而比不融合更慢。                 │
  └────────────────────────────────────────────────────────────┘

  ┌─ 复杂的内存访问模式 ─────────────────────────────────────┐
  │  scatter / gather 操作的数据访问不连续，                   │
  │  融合后可能破坏后端的向量化假设。                          │
  └────────────────────────────────────────────────────────────┘

  ┌─ 多消费者 ───────────────────────────────────────────────┐
  │  如果 producer 的输出被 >1 个 consumer 使用，              │
  │  垂直融合只能消除一个方向，另一个 consumer 仍需物化结果。  │
  │  此时不一定值得融合。                                     │
  └────────────────────────────────────────────────────────────┘
```

### 6.4.5 融合收益量化

从编译器的视角，融合的核心收益可以用一个公式量化：

```
融合收益 = 节省的内存访问时间 - 融合增加的计算开销

  节省的内存访问:
    未融合:  producer 写中间结果 (N bytes) + consumer 读中间结果 (N bytes)
    融合后:  中间结果留在寄存器 (0 bytes global memory 访问)
    节省:    2 * N / bandwidth_global_memory

  融合增加的开销:
    - 可能增加寄存器压力 → register spill
    - 可能限制 GPU occupancy（每个 SM 能运行的 block 数减少）
    - 可能阻止某些单独优化（如 producer 的 tiling 策略被迫改变）

  决策: 节省 > 开销 → 融合；否则 → 不融合
```

---

## 6.5 追踪示例：`[matmul -> relu -> add]` 的完整调度过程

本节通过一个完整的端到端追踪，展示调度层如何处理一个典型的深度学习计算子图。

### 6.5.1 问题描述

```python
import torch

@torch.compile
def f(x, y, z):
    return torch.relu(x @ y) + z

# x: [M, K], y: [K, N], z: [M, N]
# 等价于: result = relu(matmul(x, y)) + z
```

### 6.5.2 输入：IR Buffer 集合

GraphLowering 阶段完成后，产生以下 IR buffer：

| Buffer 名称 | 类型 | 形状 | 描述 |
|-------------|------|------|------|
| `buf_x` | `InputBuffer` | [M, K] | 输入 x |
| `buf_y` | `InputBuffer` | [K, N] | 输入 y |
| `buf_matmul` | `ComputedBuffer` | [M, N] | x @ y 的结果 |
| `buf_relu` | `ComputedBuffer` | [M, N] | relu(matmul) 的结果 |
| `buf_z` | `InputBuffer` | [M, N] | 输入 z |
| `buf_out` | `ComputedBuffer` | [M, N] | 最终输出 relu + z |

每个 `ComputedBuffer` 关联一个 IR 闭包（loop body callable）：

```
buf_matmul 的闭包:
  → 调用 FallbackKernel(aten.mm, buf_x, buf_y)
  → matmul 不走 Inductor 通用路径，委托给 cuBLAS

buf_relu 的闭包:
  → Pointwise IR: lambda: maximum(0, load("buf_matmul", index))

buf_out 的闭包:
  → Pointwise IR: lambda: load("buf_relu", index) + load("buf_z", index)
```

### 6.5.3 Step 1：创建 SchedulerNode

`Scheduler.__init__()` 遍历所有 IR buffer，为每个 `ComputedBuffer` 创建调度节点：

```
创建过程:

  buf_x       → InputBuffer → 不创建节点（纯数据源）
  buf_y       → InputBuffer → 不创建节点
  buf_z       → InputBuffer → 不创建节点

  buf_matmul  → ComputedBuffer, 内含 FallbackKernel(aten.mm)
               → 检测到外部 kernel
               → 创建 ExternKernelSchedulerNode(buf_matmul)
               → 记为 sn_matmul

  buf_relu    → ComputedBuffer, 内含 Pointwise IR
               → 创建 SchedulerNode(buf_relu)
               → 记为 sn_relu

  buf_out     → ComputedBuffer, 内含 Pointwise IR
               → 创建 SchedulerNode(buf_out)
               → 记为 sn_out

创建结果:
  sn_matmul = ExternKernelSchedulerNode  ← 外部 kernel
  sn_relu   = SchedulerNode              ← 标准 pointwise
  sn_out    = SchedulerNode              ← 标准 pointwise
```

### 6.5.4 Step 2：依赖分析

使用 `_RecordLoadStoreInner` handler 执行每个节点的 IR 闭包，记录读写依赖：

```
依赖分析过程:

  对 sn_matmul 执行 IR 闭包:
    V.ops = _RecordLoadStoreInner()
    closure() → 记录:
      reads:  {MemoryDep("buf_x", ...), MemoryDep("buf_y", ...)}
      writes: {MemoryDep("buf_matmul", ...)}
    依赖: sn_matmul 不依赖其他调度节点（buf_x, buf_y 是输入，不是 ComputedBuffer）

  对 sn_relu 执行 IR 闭包:
    V.ops = _RecordLoadStoreInner()
    closure() → 记录:
      reads:  {MemoryDep("buf_matmul", ...)}
      writes: {MemoryDep("buf_relu", ...)}
    依赖: sn_relu 依赖 sn_matmul（因为读取了 buf_matmul 的输出）

  对 sn_out 执行 IR 闭包:
    V.ops = _RecordLoadStoreInner()
    closure() → 记录:
      reads:  {MemoryDep("buf_relu", ...), MemoryDep("buf_z", ...)}
      writes: {MemoryDep("buf_out", ...)}
    依赖: sn_out 依赖 sn_relu（因为读取了 buf_relu 的输出）

依赖图 (DAG):

  sn_matmul
       │
       ▼
  sn_relu ──────── (sn_z 是 InputBuffer，不在 DAG 中)
       │
       ▼
  sn_out
```

### 6.5.5 Step 3：拓扑排序

基于 DAG 的拓扑排序：

```
入度分析:
  sn_matmul: 入度 = 0 （不依赖任何调度节点）
  sn_relu:   入度 = 1 （依赖 sn_matmul）
  sn_out:    入度 = 1 （依赖 sn_relu）

Kahn's 算法:
  Round 1: 入度 0 的节点 → sn_matmul
           移除 sn_matmul → sn_relu 入度降为 0

  Round 2: 入度 0 的节点 → sn_relu
           移除 sn_relu → sn_out 入度降为 0

  Round 3: 入度 0 的节点 → sn_out

拓扑排序结果: [sn_matmul, sn_relu, sn_out]
```

### 6.5.6 Step 4：融合决策

对拓扑排序后的节点序列，逐个尝试融合。假设后端为 CUDA（`TritonScheduling`）：

```
融合决策过程:

  ┌─ 处理 sn_matmul ──────────────────────────────────────────┐
  │                                                             │
  │  节点类型: ExternKernelSchedulerNode                         │
  │  融合决策: 外部 kernel → 不可融合                             │
  │                                                             │
  │  can_fuse_vertical(sn_matmul, sn_relu)?                     │
  │    → sn_matmul 是 ExternKernel → 不能融合                    │
  │    → 原因: cuBLAS 的输出必须物化到 global memory              │
  │                                                             │
  │  结果: sn_matmul 独立存在，形成一个调度边界                     │
  └─────────────────────────────────────────────────────────────┘

  ┌─ 处理 sn_relu ────────────────────────────────────────────┐
  │                                                             │
  │  can_fuse_vertical(sn_matmul, sn_relu)?                     │
  │    → sn_matmul 是 ExternKernel → 不能融合                    │
  │                                                             │
  │  can_fuse_vertical(sn_relu, sn_out)?                        │
  │    → 检查条件:                                               │
  │      1. 两个节点都是 SchedulerNode?       → YES             │
  │      2. 设备相同?                          → YES (都是 CUDA) │
  │      3. 迭代域兼容?                        → YES (都是 M*N)  │
  │      4. 后端支持?                          → YES             │
  │      5. 寄存器压力可接受?                   → YES (只多一个值) │
  │    → 融合！                                                 │
  │                                                             │
  │  创建: FusedSchedulerNode([sn_relu, sn_out])                │
  │  记为 fused_relu_add                                        │
  └─────────────────────────────────────────────────────────────┘

  ┌─ 处理 sn_out ─────────────────────────────────────────────┐
  │                                                             │
  │  sn_out 已被融合到 fused_relu_add 中                         │
  │  跳过                                                       │
  └─────────────────────────────────────────────────────────────┘

融合后的节点序列:
  [sn_matmul (ExternKernel), fused_relu_add (FusedSchedulerNode)]
```

### 6.5.7 Step 5：内存规划

分析 buffer 的生命周期，确定分配和释放时机：

```
Buffer 生命周期分析:

  buf_x       ─── 全程存活（输入）
  buf_y       ─── 全程存活（输入）
  buf_z       ─── 全程存活（输入）
  buf_matmul  ─── sn_matmul 写入 → sn_relu 读取 → 之后不再使用 → 可释放
  buf_relu    ─── sn_relu 写入 → sn_out 读取 → 之后不再使用 → 可释放
  buf_out     ─── sn_out 写入 → 最终输出 → 全程存活

  融合后:
  buf_matmul  ─── sn_matmul 写入 → fused_relu_add 读取 → 可释放
  buf_relu    ─── 不再需要物化！因为 relu 的结果直接在寄存器中传递给 add
                  （融合的收益：buf_relu 可以完全消除）
  buf_out     ─── 最终输出

释放计划:
  sn_matmul 执行后: 无释放（buf_matmul 还要被 fused_relu_add 使用）
  fused_relu_add 执行后: 释放 buf_matmul
  最终输出: buf_out

  节省: buf_relu 完全不分配内存 → 节省 M * N * 4 bytes
```

### 6.5.8 Step 6：后端选择与代码生成分派

```
代码生成分派:

  Scheduler._codegen() 遍历融合后的节点:

  ┌─ 节点 1: sn_matmul (ExternKernelSchedulerNode) ──────────┐
  │                                                            │
  │  节点类型: ExternKernel                                     │
  │  后端: CUDA → get_backend("cuda")                          │
  │       → 首次调用，懒创建 CUDACombinedScheduling             │
  │       → 内部创建 TritonScheduling + CUDACPPScheduling      │
  │                                                            │
  │  分派: backend.codegen_node(sn_matmul)                      │
  │       → 但 ExternKernel 不走 codegen_node                   │
  │       → 走特殊路径：生成 cuBLAS 调用代码                     │
  │                                                            │
  │  生成的 wrapper 代码（简化）:                                │
  │    buf_matmul = torch.empty([M, N], device='cuda')          │
  │    torch.mm(buf_x, buf_y, out=buf_matmul)                  │
  │    # cuBLAS matmul 调用                                     │
  └────────────────────────────────────────────────────────────┘

  ┌─ 节点 2: fused_relu_add (FusedSchedulerNode) ────────────┐
  │                                                            │
  │  节点类型: FusedSchedulerNode（包含 relu + add）             │
  │  后端: 复用已有的 CUDACombinedScheduling                    │
  │       → 委托到 _triton_scheduling                          │
  │                                                            │
  │  分派: TritonScheduling.codegen_node(fused_relu_add)        │
  │                                                            │
  │  内部流程:                                                  │
  │    1. 提取节点信息                                          │
  │       - 迭代域: M * N 个元素                                │
  │       - 输入: buf_matmul, buf_z                             │
  │       - 输出: buf_out                                      │
  │                                                            │
  │    2. 生成节点调度顺序                                      │
  │       - sn_relu 在前（生产者）                               │
  │       - sn_out 在后（消费者）                                │
  │                                                            │
  │    3. 选择 tiling 策略                                      │
  │       - 1D pointwise: grid = (M*N + BLOCK_SIZE - 1) // ... │
  │                                                            │
  │    4. 创建 TritonKernel 实例                                 │
  │       - create_kernel_choices() → 多个 autotuning 候选     │
  │       - benchmark 选择最优配置                               │
  │                                                            │
  │    5. 执行 kernel codegen                                   │
  │       - 设置 V.kernel = TritonKernel                       │
  │       - 设置 V.ops = CSEProxy(TritonOverrides)             │
  │       - 执行 IR 闭包 → 生成 Triton 代码                     │
  │                                                            │
  │    6. define_kernel()                                       │
  │       - 命名: triton_pointwise_relu_add_0                   │
  │       - 异步编译                                            │
  └────────────────────────────────────────────────────────────┘
```

### 6.5.9 生成的 Triton Kernel 代码

以下是 `fused_relu_add` 最终生成的 Triton kernel（简化版）：

```python
@triton.jit
def triton_pointwise_relu_add_0(
    buf_matmul_ptr,  # matmul 结果
    buf_z_ptr,       # 输入 z
    buf_out_ptr,     # 输出
    N_elements,      # 总元素数 M * N
    BLOCK_SIZE: tl.constexpr,  # autotuning 选择的 block 大小
):
    # 1. 计算当前 program 处理的元素偏移
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_elements

    # 2. 加载输入
    matmul_val = tl.load(buf_matmul_ptr + offsets, mask=mask)
    z_val = tl.load(buf_z_ptr + offsets, mask=mask)

    # 3. 计算（融合了 relu + add 两个操作）
    relu_val = tl.where(matmul_val > 0, matmul_val, 0.0)
    result = relu_val + z_val

    # 4. 存储输出
    tl.store(buf_out_ptr + offsets, result, mask=mask)
```

**关键观察**：

- `relu_val` 是中间结果，**完全在寄存器中传递**，不需要写入 global memory
- 如果不融合，`relu_val` 需要先写入一个临时 buffer，然后第二个 kernel 再读取它
- 融合后，`buf_relu` 这个临时 buffer **完全不存在**——节省了一次 global memory 的写和一次读

### 6.5.10 生成的 Wrapper 代码

Wrapper 代码将两个 kernel 调用串联起来：

```python
def compiled_fn(buf_x, buf_y, buf_z):
    # 1. 分配 buf_matmul（cuBLAS 的输出 buffer）
    buf_matmul = torch.empty([M, N], device='cuda', dtype=torch.float32)

    # 2. 外部 kernel: cuBLAS matmul
    torch.mm(buf_x, buf_y, out=buf_matmul)

    # 3. 分配输出 buffer
    buf_out = torch.empty([M, N], device='cuda', dtype=torch.float32)

    # 4. 融合 kernel: relu + add
    triton_pointwise_relu_add_0[
        (M * N + BLOCK_SIZE - 1) // BLOCK_SIZE,  # grid size
    ](
        buf_matmul, buf_z, buf_out,
        M * N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # 5. 释放 buf_matmul（不再使用）
    del buf_matmul

    return buf_out
```

### 6.5.11 完整调度时间线回顾

```
时间 ──────────────────────────────────────────────────────────────────►

  GraphLowering 完成
  │  输出: {buf_x, buf_y, buf_z, buf_matmul, buf_relu, buf_out}
  │
  ▼
  Scheduler.__init__()
  │
  ├── 创建节点:
  │   sn_matmul = ExternKernelSchedulerNode
  │   sn_relu   = SchedulerNode
  │   sn_out    = SchedulerNode
  │
  ├── 依赖分析:
  │   sn_relu depends on sn_matmul
  │   sn_out depends on sn_relu
  │
  ├── 拓扑排序:
  │   [sn_matmul, sn_relu, sn_out]
  │
  └── 融合决策:
      sn_matmul: 不可融合（外部 kernel）
      sn_relu + sn_out: 融合为 fused_relu_add
      │
      ▼
  融合结果: [sn_matmul, fused_relu_add]
  │
  ▼
  Scheduler.codegen()
  │
  ├── sn_matmul → cuBLAS 调用
  │     buf_x, buf_y → torch.mm() → buf_matmul
  │
  ├── fused_relu_add → TritonScheduling.codegen_node()
  │     → TritonKernel → triton_pointwise_relu_add_0
  │     buf_matmul, buf_z → fused kernel → buf_out
  │
  └── flush() → 所有代码写入 wrapper
      │
      ▼
  PythonWrapperCodegen.generate()
  │  组装完整 wrapper 函数
  │
  ▼
  编译完成: 返回可执行的 compiled_fn
```

### 6.5.12 调度决策总结

对本例的调度决策做一个完整的总结：

| 决策 | 选择 | 原因 |
|------|------|------|
| matmul 的节点类型 | `ExternKernelSchedulerNode` | matmul 委托给 cuBLAS，不走 Inductor 代码生成 |
| matmul 是否融合 | 否 | 外部 kernel 边界，输出必须物化 |
| relu + add 是否融合 | 是 | 相同迭代域、同设备、寄存器压力可接受 |
| buf_relu 是否分配 | 否 | 融合后中间结果留在寄存器中 |
| relu + add 的 kernel 类型 | `TritonKernel` (pointwise) | 1D pointwise 操作，CUDA 后端 |
| matmul 的执行方式 | `torch.mm()` | 直接调用 cuBLAS |

**性能收益**：

```
不融合的执行:
  kernel_1 (cuBLAS): x @ y → buf_matmul     [写 buf_matmul 到 global memory]
  kernel_2 (Triton): relu(buf_matmul) → buf_relu  [读 buf_matmul, 写 buf_relu]
  kernel_3 (Triton): buf_relu + z → buf_out       [读 buf_relu, 读 z, 写 buf_out]
  总 global memory 访问: 读 3 次 + 写 3 次 = 6 次 M*N*4 bytes

融合后的执行:
  kernel_1 (cuBLAS): x @ y → buf_matmul     [写 buf_matmul]
  kernel_2 (Triton): relu(buf_matmul) + z → buf_out  [读 buf_matmul, 读 z, 写 buf_out]
  总 global memory 访问: 读 2 次 + 写 2 次 = 4 次 M*N*4 bytes

  节省: 2 次 M*N*4 bytes 的 global memory 访问
  buf_relu 完全消除 → 节省 M*N*4 bytes 内存
```

---

## 6.6 本章小结

调度层是 Inductor 编译管线的 "大脑"——它将 Lowering 层产生的 IR 转化为有序的、经过优化的执行计划。本章我们学习了三个核心组件：

**Scheduler** — 全局调度器，独立类，承担拓扑排序、算子融合、内存规划、后端分派四大职责。它是编译中期唯一的编排者，决定了整个计算图的最优执行策略。

**SchedulerNode 继承树** — 调度节点的类型体系。四种节点类型对应不同的执行策略：
- `SchedulerNode`：标准节点，一个 IR buffer 对应一个 kernel
- `FusedSchedulerNode`：融合节点，多个 buffer 合并为一个 kernel
- `ExternKernelSchedulerNode`：外部调用节点，形成调度边界
- `TemplateSchedulerNode`：模板节点，使用预定义的高性能 kernel

**BaseScheduling 继承树** — 后端调度策略的多态实现。三种设计模式的交织：
- **模板方法**：`SIMDScheduling` 提供算法骨架，`TritonScheduling` / `MetalScheduling` 填充细节
- **策略**：`BaseScheduling` 定义接口，每个后端是独立的策略实现
- **委托**：`CUDACombinedScheduling` 按节点类型路由到不同子调度器

在下一章中，我们将深入代码生成层——看看 `TritonScheduling.codegen_node()` 内部是如何将融合后的 IR 闭包转化为实际的 Triton kernel 代码的。
