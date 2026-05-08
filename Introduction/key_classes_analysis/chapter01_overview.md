# 第一章：全局概览 —— Inductor 编译全流程与类家族地图

> "如果你要在森林中导航，首先要做的不是研究每一棵树的纹理，而是获得一张完整的地形图。"
>
> 本章的目标正是提供这张地形图。我们将从编译器理论的经典框架出发，建立对 Inductor 整体架构的认知坐标系。后续章节将逐一深入每个类家族的内部细节，而本章给出的是它们之间的位置关系与协作方式。

---

## 1.1 编译器视角：Inductor 在 PyTorch 编译栈中的位置

### 1.1.1 PyTorch 编译栈的三阶段流水线

`torch.compile()` 背后并非单一的编译器，而是一条由三个独立子系统组成的三阶段流水线。理解这条流水线，是理解 Inductor 的前提。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     torch.compile(model)                                │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │ TorchDynamo  │───>│ AOTAutograd  │───>│      Inductor            │  │
│  │              │    │              │    │                          │  │
│  │ Python字节码 │    │ 前向/反向分离 │    │ IR优化 + 融合 + 代码生成 │  │
│  │ 捕获与图构建 │    │ ATen算子分解  │    │ Triton / C++ / CUDA     │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                                         │
│  Python源码  ──>  FX Graph (联合图)  ──>  FX Graph (后向分离)  ──>  可执行代码  │
└─────────────────────────────────────────────────────────────────────────┘
```

**第一阶段：TorchDynamo —— 从动态 Python 到静态计算图**

TorchDynamo（`torch/_dynamo/`）是编译栈的前端入口。它通过 CPython 的 PEP 659 字节码监控机制（Frame Evaluation API），在 Python 函数首次执行时捕获其执行轨迹，并将动态的 Python 控制流（if/else、循环、函数调用）转换为静态的 FX Graph（`torch.fx.Graph`）。

从编译器视角看，TorchDynamo 扮演的是**前端解析器**的角色——将高层语言翻译为编译器中间表示。类比 GCC，TorchDynamo 相当于将 C 源码解析为 GENERIC 树；类比 LLVM，相当于 Clang 将源码转化为 LLVM IR 的前端阶段。

TorchDynamo 的输出是一个"联合图"（Joint Graph），其中前向计算与梯度计算仍然交织在一起。

**第二阶段：AOTAutograd —— 前向/反向分离与算子分解**

AOTAutograd（`torch/_functorch/aot_autograd.py`）接受 TorchDynamo 输出的联合图，执行两项关键任务：

1. **前向/反向图分离**：将联合图分解为独立的前向图（forward graph）和反向图（backward graph），实现 Ahead-of-Time 梯度计算图的编译。
2. **ATen 算子分解**：将高层 PyTorch 操作（如 `torch.nn.functional.cross_entropy`）分解为 ATen 级别的基础操作（如 `aten.log_softmax`、`aten.nll_loss`），形成 `aten` 命名空间下的标准算子集合。

AOTAutograd 的输出是两个分离的 FX Graph，每个图仅包含 `torch.ops.aten` 和 `torch.ops.prims` 命名空间的算子。这正是 Inductor 的输入格式。

**第三阶段：Inductor —— 编译与代码生成**

Inductor（`torch/_inductor/`）接收 AOTAutograd 输出的 FX Graph，经过自身的 Lowering、优化、调度与代码生成流水线，最终产出可直接执行的目标平台代码。这就是本书的主题。

### 1.1.2 Inductor 内部的三段式划分

借鉴经典编译器的三段式架构（前端 - 中端 - 后端），Inductor 自身也可以划分为三层：

```
Inductor 内部架构

┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  ┌────────────────┐                                          │
│  │   前端 (Lowering)         │                              │
│  │   GraphLowering           │                              │
│  │   FX Graph → Inductor IR  │                              │
│  └─────────┬──────┘                                          │
│            │                                                 │
│            ▼                                                 │
│  ┌────────────────┐                                          │
│  │   中端 (IR + Optimization) │                             │
│  │   TensorBox / Buffer / Pointwise / Reduction              │
│  │   常量折叠 / 死代码消除 / 缓冲区复用                       │
│  └─────────┬──────┘                                          │
│            │                                                 │
│            ▼                                                 │
│  ┌────────────────┐                                          │
│  │   后端 (Scheduling + Codegen) │                          │
│  │   Scheduler → 融合决策                                      │
│  │   Kernel Codegen → Triton / C++ / CUDA                   │
│  │   Wrapper Codegen → 可执行模块                             │
│  └────────────────┘                                          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

| 编译器层 | Inductor 对应 | 经典编译器类比 (LLVM) | 核心职责 |
|---------|-------------|-------------------|--------|
| 前端 (Frontend) | `GraphLowering` + `lowering.py` | Clang → LLVM IR | 将外部 IR 转换为内部 IR |
| 中端 (Middle-end) | IR 层（`TensorBox` / `Buffer` / `Pointwise`）+ IR Passes | LLVM IR + Optimization Passes | IR 表示、变换与优化 |
| 后端 (Backend) | `Scheduler` + `Kernel` Codegen + `WrapperCodegen` | LLVM 后端（SelectionDAG / GlobalISel）| 调度、融合与目标平台代码生成 |

需要特别指出的是，Inductor 的"中端"并非传统的 SSA 形式 IR，而是基于 Python 闭包的 Define-by-Run IR（详见第四章）。这一设计选择深刻影响了 Inductor 的 Handler 协议体系（第三章）和代码生成策略（第七章）。

---

## 1.2 完整编译生命周期：从 `compile_fx_inner()` 到可执行模块

Inductor 的编译入口函数是 `torch/_inductor/compile_fx.py` 中的 `compile_fx_inner()`。整个编译过程可以划分为五个阶段（Phase），每个阶段完成明确的转换任务。

### 1.2.1 调用链总览

```
torch.compile(model)
    │
    ▼
TorchDynamo: 捕获 FX Graph
    │
    ▼
AOTAutograd: 前向/反向分离
    │
    ▼
compile_fx_inner()                          ← Inductor 入口
    │
    ▼
_compile_fx_inner()
    │
    ├─── Phase 0: 常量折叠子图编译
    │         const_graph = GraphLowering(const_gm, ...)
    │         const_graph.run(*inputs)
    │         const_graph.compile_to_module()
    │
    ├─── Phase 1: 主图 GraphLowering
    │         graph = GraphLowering(gm, ...)
    │         graph.run(*example_inputs)
    │
    ├─── Phase 2: Scheduler
    │         scheduler = Scheduler(graph.buffers)
    │         scheduler.fuse_nodes()
    │
    ├─── Phase 3: Backend Scheduling + Kernel Codegen
    │         backend_codegen(node)  ← 每个融合节点
    │         → TritonKernel / CppKernel
    │
    └─── Phase 4: Wrapper Codegen
              graph.compile_to_module()
              → PythonWrapperCodegen / CppWrapperCodegen
              → 最终可执行模块
```

### 1.2.2 Phase 0：常量折叠子图编译

**源文件**：`torch/_inductor/compile_fx.py`

并非所有编译都需要 Phase 0。当 FX Graph 中包含可静态确定的常量子图（如只依赖参数和常量的计算），Inductor 会将其提取为独立的子图，提前编译执行。

```python
# compile_fx.py 中的核心逻辑（简化）
if const_gm is not None:
    const_graph = GraphLowering(const_gm, ...)
    const_graph.run(*const_inputs)
    const_graph.compile_to_module()   # 生成常量折叠后的结果
```

Phase 0 创建一个临时的 `GraphLowering` 实例（`const_graph`），其生命周期短暂——完成常量求值后即被销毁。这与 Phase 1 中贯穿编译全生命周期的主 `GraphLowering` 实例形成对比。

**编译器类比**：类似于 GCC 的常量传播优化（`-fconstprop`），但 Inductor 选择将其实现为独立的子编译流程。

### 1.2.3 Phase 1：GraphLowering —— FX Graph 到 Inductor IR

**源文件**：`torch/_inductor/graph.py`（`GraphLowering` 类）、`torch/_inductor/lowering.py`（lowering 注册表）

这是 Inductor 前端的核心阶段。`GraphLowering` 继承自 `torch.fx.Interpreter`，逐节点解释执行 FX Graph，将每个 ATen 算子节点转换为 Inductor 内部的 IR 节点（`TensorBox` / `StorageBox` / `Buffer`）。

```python
# Phase 1 的核心逻辑（简化）
graph = GraphLowering(gm, ...)            # 创建主 GraphLowering 实例
graph.run(*example_inputs)                 # 逐节点解释执行 FX Graph
# run() 内部：遍历 FX Graph 的每个节点，
#   查找 lowering.py 中的注册表（~150+ 显式 lowering + ~150+ fallback），
#   调用对应的 lowering 函数，
#   生成 TensorBox(Pointwise(...)) 或 TensorBox(ComputedBuffer(Reduction(...))) 等 IR 节点
```

`GraphLowering.run()` 执行完毕后，FX Graph 中的所有节点都被转换为以 `Buffer` 为根的 IR 对象集合，存储在 `graph.buffers` 中。

关键设计点：
- **注册表模式**：`lowering.py` 维护一个从 ATen 算子到 lowering 函数的注册表（`@register_lowering` 装饰器）。每个 lowering 函数接收上游的 `TensorBox`，返回新的 `TensorBox`。
- **闭包即 IR**：每个 IR 节点的计算逻辑以 Python 闭包（`inner_fn`）的形式存储，而非传统的 AST 或 SSA IR 节点。例如 `Pointwise` 的 `inner_fn(index)` 接收多维索引，返回该位置的标量值。
- **Fallback 机制**：对于 Inductor 尚未显式支持的算子（约 150+ 个），通过 `make_fallback()` 注册为 `FallbackKernel`，回退到 eager 模式执行。

**编译器类比**：Phase 1 相当于 LLVM 中从 Clang AST 到 LLVM IR 的转换。`GraphLowering` 是"翻译器"，`lowering.py` 是"翻译规则表"。

### 1.2.4 Phase 2：Scheduler —— IR 到调度计划

**源文件**：`torch/_inductor/scheduler.py`（`Scheduler` 类）

Scheduler 是 Inductor 中端的优化核心。它接收 `graph.buffers` 中的 IR 节点集合，执行拓扑排序、算子融合与重排序，最终产出有序的调度节点序列。

```python
# Phase 2 的核心逻辑（简化）
scheduler = Scheduler(graph.buffers)
# _init(): 为每个 Buffer 创建 SchedulerNode，建立依赖关系
# fuse_nodes(): 贪婪融合算法
#   1. 以共享 buffer 为线索初筛可融合节点对
#   2. 调用后端的 can_fuse_vertical() / can_fuse_horizontal() 验证
#   3. 按 score_fusion() 多级排序（模板融合优先 > 同类融合 > 内存收益 > 拓扑距离）
#   4. 按优先级依次融合
```

融合是 Scheduler 的核心任务。Inductor 采用贪婪融合策略，支持两种融合方向：

| 融合方向 | 含义 | 示例 |
|---------|-----|------|
| 垂直融合 (Vertical Fusion) | 生产者-消费者融合：将前一个算子的输出直接馈入后一个算子，避免中间结果的内存写回 | `x + y` → `relu(x + y)` |
| 水平融合 (Horizontal Fusion) | 兄弟节点融合：共享同一输入的多个算子合并为一个 kernel | `sin(x) + cos(x)` |

Scheduler 的输出是一个有序的 `SchedulerNode` 列表，其中融合后的节点以 `FusedSchedulerNode` 表示。

**编译器类比**：Scheduler 相当于 LLVM 中的 Pass Manager，协调多个优化 Pass 的执行。融合策略类似于 LLVM 的循环融合（Loop Fusion）优化。

### 1.2.5 Phase 3：Backend Scheduling + Kernel Codegen —— 调度计划到内核代码

**源文件**：`torch/_inductor/scheduler.py`（`BaseScheduling` 及其子类）、`torch/_inductor/codegen/` 目录

Phase 3 是后端的代码生成阶段。对调度节点列表中的每个节点，后端调度器（`BaseScheduling` 子类）决定具体的代码生成策略，然后由对应的 Kernel 类完成目标平台代码的生成。

```python
# Phase 3 的核心逻辑（简化）
for node in scheduler.nodes:
    backend = get_backend(node.get_device())
    backend.codegen_node(node)    # 后端调度器分派到具体 Kernel
```

后端调度器与 Kernel 类的对应关系：

| 后端 | 调度器（`BaseScheduling` 子类） | Kernel 类 | 目标语言 |
|-----|------------------------|----------|--------|
| Triton GPU | `TritonScheduling` | `TritonKernel` | Triton (`@triton.jit`) |
| CPU C++ | `CppScheduling` | `CppKernel` / `CppVecKernel` | C++ / OpenMP |
| CUDA Template | `CUDACombinedScheduling` | `CUDATemplateKernel` | CUTLASS + Triton |
| Metal GPU | `MetalScheduling` | `MetalKernel` | Metal Shading Language |

每个 Kernel 类遵循统一的三段式代码组织模式：**loads**（输入读取）→ **compute**（计算）→ **stores**（输出写入）。以 `TritonKernel` 为例，它将 IR 闭包（`inner_fn`）中的 `ops.load()`、`ops.floor()`、`ops.store()` 等调用，分别转换为 `tl.load()`、`tl.math.floor()`、`tl.store()` 等 Triton API 调用，最终生成完整的 `@triton.jit` 装饰的 Python 函数源码。

这一转换过程的核心机制是 **Handler 协议**（详见第三章）：IR 闭包中的 `ops.xxx()` 调用被不同 Handler 拦截，分析型 Handler（如 `MockHandler`、`DtypePropagationOpsHandler`）完成类型推断和依赖分析，代码生成型 Handler（如 `TritonOverrides`、`CppOverrides`）生成目标语言代码字符串。

**编译器类比**：Phase 3 相当于 LLVM 后端的指令选择（Instruction Selection）和寄存器分配（Register Allocation）。`TritonOverrides` / `CppOverrides` 等价于 LLVM 的目标描述文件（`.td` 文件）。

### 1.2.6 Phase 4：Wrapper Codegen —— 从 Kernel 到可执行模块

**源文件**：`torch/_inductor/codegen/wrapper.py`（`PythonWrapperCodegen`）、`torch/_inductor/codegen/cpp_wrapper/`（`CppWrapperGpu` 等）

Phase 1-3 生成的是一个个独立的 Kernel 函数（如 Triton kernel）。Phase 4 的任务是将这些分散的 Kernel 组装为一个完整的、可被 PyTorch 直接调用的模块。

```python
# Phase 4 的核心逻辑（简化）
graph.compile_to_module()
# 内部：
#   1. 收集所有已编译的 Kernel 函数
#   2. 生成 wrapper 函数：处理输入缓冲区的内存布局转换、Kernel 的调用顺序、输出缓冲区的回收
#   3. 组装为可导入的 Python 模块或动态链接库
```

Wrapper 有两条路径：

| 路径 | Wrapper 类 | 输出 | 适用场景 |
|-----|-----------|-----|---------|
| Python Wrapper | `PythonWrapperCodegen` | Python 源码（`.py`） | 默认路径，调试友好 |
| C++ Wrapper | `CppWrapperGpu` / `CppWrapperCpu` | C++ 源码或 AOT 编译产物 | 高性能部署，减少 Python 开销 |

**编译器类比**：Phase 4 相当于编译器后端的代码发射（Code Emission）和链接（Linking）阶段——将各个独立编译的模块组装为最终的可执行文件。

### 1.2.7 五阶段生命周期时序图

```
时间 ──────────────────────────────────────────────────────────────────────>

                    Phase 0            Phase 1           Phase 2          Phase 3           Phase 4

V.ops 上下文   ┌─ NullHandler ──┐ ┌ MockHandler ─┐ ┌──────────────────────────────────────────┐
               │ (常量折叠)      │ │ (IR 调试)     │ │  分析型 Handler 链                         │
               └────────────────┘ └───────────────┘ │  Mock → 依赖分析 → 值域分析 → 代码生成     │
                                                  └──────────────────────────────────────────┘
V.graph 上下文          ┌──── const_graph ────┐  ┌──────── graph (贯穿全生命周期) ──────────────────────┐
                        └─────────────────────┘  └──────────────────────────────────────────────────┘
V.kernel 上下文                                                   ┌─ TritonKernel ─┐  ┌── Wrapper ──┐
                                                                 │ (代码生成)      │  │ (模块组装)   │
                                                                 └────────────────┘  └─────────────┘

核心类实例     GraphLowering×1     GraphLowering×1   Scheduler×1       Kernel×N           WrapperCodegen×1
               (短生命周期)        (贯穿全生命周期)   (贯穿后半周期)     (每个融合组一个)     (贯穿末期)

数据形态       FX Graph            IR (Buffer集合)    SchedulerNode     Kernel源码字符串     Python/C++模块
               → IR               → (带依赖信息)      列表             → 目标语言代码        → 可执行模块
```

---

## 1.3 关键类家族聚类与职责矩阵

Inductor 的核心类可以聚类为六大功能家族。这六大家族覆盖了从输入到输出的完整编译管线，每个家族在特定的编译阶段活跃，承担明确的职责。

### 1.3.1 六大家族概览

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        Inductor 核心类家族地图                                     │
│                                                                                  │
│  ┌─────────────────┐                                                              │
│  │ 家族1: 全局基础设施 │ ── Virtualized<T>, _V, NullHandler                        │
│  │ (运行时上下文引擎)  │    编译全阶段的隐式协作基础设施                               │
│  └─────────────────┘                                                              │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐                                      │
│  │ 家族2: Lowering层  │────│ 家族5: Handler协议族 │                                │
│  │ (FX → IR 翻译)     │    │ (抽象域框架)         │                                │
│  │ GraphLowering      │    │ OpsHandler[T]        │                                │
│  │ lowering.py注册表   │    │ 分析型 / 代码生成型  │                                │
│  └────────┬────────┘     └────────┬────────┘                                      │
│           │                       │                                               │
│           ▼                       │                                               │
│  ┌─────────────────┐              │                                               │
│  │ 家族3: IR表示层    │◄────────────┘                                              │
│  │ (中间表示)         │                                                            │
│  │ TensorBox / Buffer │                                                            │
│  │ Pointwise / Reduction │                                                         │
│  └────────┬────────┘                                                              │
│           │                                                                      │
│           ▼                                                                      │
│  ┌─────────────────┐     ┌─────────────────┐                                      │
│  │ 家族4: 调度层      │────│ 家族6: 代码生成层     │                                │
│  │ (融合与调度)        │    │ (Kernel + Wrapper)    │                                │
│  │ Scheduler          │    │ TritonKernel           │                                │
│  │ BaseScheduling     │    │ CppKernel              │                                │
│  │ SchedulerNode继承树 │    │ WrapperCodegen         │                                │
│  └─────────────────┘     └─────────────────┘                                      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### 1.3.2 家族1：全局基础设施

**核心职责**：提供编译全阶段的动态作用域机制，使各类实例可以通过全局入口协作，而无需显式传递引用。

| 属性 | 说明 |
|-----|------|
| 源文件 | `torch/_inductor/virtualized.py` |
| 核心类 | `Virtualized<T>`、`_V`、`NullHandler`、`NullKernelHandler` |
| 活跃阶段 | 编译全阶段（隐式运行） |
| 输入/输出 | 无显式数据流；通过线程局部状态读写协作 |
| 类比 | Lisp 的 `fluid-let` / 多 pass 编译器的全局符号表 |

`_V` 是全局门面（Facade），暴露 14 个动态作用域上下文，其中最关键的包括：

| 上下文名 | 类型 | 安装时机 | 用途 |
|---------|-----|---------|-----|
| `V.ops` | `OpsHandler[T]` 子类 | Phase 1 (MockHandler) / Phase 3 (TritonOverrides 等) | IR 闭包执行时的操作拦截器 |
| `V.graph` | `GraphLowering` | Phase 0 / Phase 1 开始 | 当前编译图的全局引用 |
| `V.kernel` | `Kernel` 子类 | Phase 3 每个 Kernel 编译时 | 当前 Kernel 的上下文 |
| `V.fake_mode` | `FakeTensorMode` | 编译开始时 | 形状/类型推断的虚拟执行模式 |
| `V.choices` | 列表 | Phase 3 autotuning 时 | 收集候选 Kernel 配置 |

> 详细解析见第二章。

### 1.3.3 家族2：Lowering 层

**核心职责**：将 FX Graph 中的 ATen 算子节点逐个翻译为 Inductor IR 节点，是前端到中端的桥梁。

| 属性 | 说明 |
|-----|------|
| 源文件 | `torch/_inductor/graph.py`（`GraphLowering`）、`torch/_inductor/lowering.py`（注册表） |
| 核心类 | `GraphLowering`、`FallbackKernel` |
| 核心函数 | `make_pointwise()`、`make_reduction()`、`@register_lowering()`、`make_fallback()` |
| 活跃阶段 | Phase 0（常量折叠）、Phase 1（主图 lowering） |
| 输入 | FX Graph（`torch.fx.Graph`），包含 ATen 算子节点 |
| 输出 | IR 节点集合（`graph.buffers`），以 `Buffer` 为根的 IR 对象 |
| 类比 | LLVM 中 Clang AST → LLVM IR 的翻译器 |

`GraphLowering` 继承 `torch.fx.Interpreter`，通过重写 `placeholder()`、`call_function()`、`output()` 等方法，在逐节点遍历 FX Graph 时，查找 `lowering.py` 中的注册表并调用对应的 lowering 函数。

```
FX Graph Node                  lowering.py 注册表                   IR 节点
─────────────                  ────────────────                   ────────

aten.floor.default  ──────>  lowerings[aten.floor] = ceil()  ──>  TensorBox(
                                                                          StorageBox(
                                                                            Pointwise(inner_fn, ...)
                                                                          )
                                                                        )

aten.sum.dim_IntList ──────> lowerings[aten.sum] = sum_()    ──>  TensorBox(
                                                                          StorageBox(
                                                                            ComputedBuffer(
                                                                              Reduction(inner_fn, ...)
                                                                            )
                                                                          )
                                                                        )

aten.cholesky_inverse ─────> fallbacks[aten.cholesky_inverse] ──>  FallbackKernel(...)
                             (make_fallback 注册)
```

> 详细解析见第五章。

### 1.3.4 家族3：IR 表示层

**核心职责**：定义 Inductor 的中间表示数据结构，承载从 Lowering 到 Codegen 之间的所有计算语义。

| 属性 | 说明 |
|-----|------|
| 源文件 | `torch/_inductor/ir.py` |
| 核心类 | `TensorBox`、`StorageBox`、`Buffer`（及子类）、`Pointwise`、`Reduction`、`Scatter` 等 |
| 活跃阶段 | Phase 1 生成 → Phase 2 消费/变换 → Phase 3 读取闭包 |
| 输入 | 由 Lowering 层构建 |
| 输出 | 被 Scheduler 消费，Kernel Codegen 读取其闭包 |
| 类比 | LLVM IR 的 `Value` / `Instruction` / `BasicBlock` 层次结构 |

IR 层的类继承树如下（简化）：

```
IRNode
  ├── TensorBox               ← 引用语义包装器（指向 StorageBox）
  │     └── StorageBox        ← 存储层抽象（指向底层 Buffer）
  │           └── Buffer      ← 缓冲区基类
  │                 ├── InputBuffer         ← 模型输入（placeholder）
  │                 ├── ConstantBuffer      ← 编译期常量
  │                 ├── ComputedBuffer      ← 计算结果（持有 data 描述）
  │                 ├── ReinterpretView     ← 零拷贝视图（view/reshape）
  │                 └── SliceView           ← 切片视图
  │
  └── LooseRangedOps          ← 操作 IR 节点基类
        ├── Pointwise         ← 逐元素操作（index → value）
        ├── Reduction         ← 规约操作（index + rindex → value）
        ├── Scatter           ← 散射写入操作
        └── ...
```

关键设计特征：

- **闭包即 IR**：`Pointwise` 和 `Reduction` 的计算逻辑以 Python 闭包（`inner_fn`）形式存储，而非传统的 AST 或 SSA IR 指令。例如：
  ```python
  # Pointwise 的 inner_fn：接收索引，返回该位置的标量值
  def inner_fn(index):
      i0, i1, i2 = index
      tmp0 = ops.load(arg0_1, i2 + 1024 * i1 + 524288 * i0)  # 加载输入
      tmp1 = ops.floor(tmp0)                                    # 计算 floor
      return tmp1
  ```

- **三层包装**：`TensorBox` → `StorageBox` → `Buffer`，实现"张量值"与"存储方式"与"计算逻辑"的三层解耦。

> 详细解析见第四章。

### 1.3.5 家族4：调度层

**核心职责**：接收 IR 节点集合，执行拓扑排序、算子融合与重排序，产出有序的调度节点序列。

| 属性 | 说明 |
|-----|------|
| 源文件 | `torch/_inductor/scheduler.py` |
| 核心类 | `Scheduler`、`BaseSchedulerNode`（及子类）、`BaseScheduling`（及子类） |
| 活跃阶段 | Phase 2（融合决策）→ Phase 3（后端调度策略） |
| 输入 | IR 节点集合（`graph.buffers`） |
| 输出 | 有序的 `SchedulerNode` 列表（部分已融合为 `FusedSchedulerNode`） |
| 类比 | LLVM 的 Pass Manager + 循环融合优化 |

调度层包含两棵继承树：

**调度节点继承树**（数据结构视角）：

```
BaseSchedulerNode
  ├── SchedulerNode              ← 基础调度节点（对应一个 IR buffer）
  ├── FusedSchedulerNode         ← 融合节点（多个 node 融合为一个 kernel）
  ├── ExternKernelSchedulerNode  ← 外部 kernel（直接调用高性能库）
  ├── TemplateSchedulerNode      ← 模板 kernel（如 flash attention）
  ├── NopKernelSchedulerNode     ← 空操作节点
  ├── ForeachKernelSchedulerNode ← foreach 批量操作节点
  └── GroupedSchedulerNode       ← 分组节点
```

**后端调度策略继承树**（策略模式视角）：

```
BaseScheduling                    ← 抽象协议（can_fuse_vertical / can_fuse_horizontal / codegen_node）
  ├── SIMDScheduling             ← GPU/SIMD 模板方法基类
  │     ├── TritonScheduling     ← Triton GPU 后端（autotuning + 异步编译）
  │     └── MetalScheduling      ← Metal GPU 后端
  ├── CppScheduling              ← CPU C++/OpenMP 后端
  ├── HalideScheduling           ← Halide 后端
  └── CUDACombinedScheduling     ← 委托模式（Triton + CUTLASS 路由）
```

> 详细解析见第六章。

### 1.3.6 家族5：Handler 协议族

**核心职责**：定义 IR 闭包执行时的操作拦截协议，使同一份 IR 闭包能够被不同语义域（调试、分析、代码生成）解释执行。

| 属性 | 说明 |
|-----|------|
| 源文件 | `torch/_inductor/ops_handler.py`（协议定义）、`torch/_inductor/codegen/`（代码生成 Handler） |
| 核心类 | `OpsHandler[T]`（协议）、`DefaultHandler`、`MockHandler`、`CSEProxy`、`TritonOverrides`、`CppOverrides` 等 |
| 活跃阶段 | Phase 1（类型推断）→ Phase 3（代码生成） |
| 输入 | IR 闭包（`inner_fn` callable）+ 语义域参数 |
| 输出 | T 类型结果（T 取决于具体 Handler：字符串、类型信息、值域、代码片段等） |
| 类比 | 编译器中的抽象解释（Abstract Interpretation）多域分析 |

Handler 协议族是 Inductor 架构中最精巧的设计之一。它的核心思想是：**同一份 IR 闭包（`inner_fn`），在不同的 Handler 下执行，会产生不同语义域的结果。**

```
                        ┌── MockHandler         → 字符串表示（IR 可视化/调试）
                        │
IR 闭包 (inner_fn) ────┼── DtypePropagation    → 类型信息（torch.dtype 传播）
                        │
(同一个 callable)       ├── ValueRangeAnalysis  → 值域信息（ValueRanges[Expr]）
                        │
                        ├── CSEProxy            → 公共子表达式消除 + 代码生成
                        │     └── TritonOverrides → Triton 代码字符串
                        │
                        └── CppOverrides        → C++ 代码字符串
```

典型的工作链路（Phase 3 中对一个融合节点）：

```
CSEProxy(
  SimplifyIndexing(
    TritonOverrides()
  )
)
```

同一 IR 闭包被**执行 4 次**，每次安装不同的 Handler：
1. `MockHandler` — 生成 IR 的字符串表示（调试用）
2. `_RecordLoadStoreInner` — 分析内存依赖关系
3. `ValueRangeAnalysis` — 推断值的范围
4. `TritonOverrides`（经由 `CSEProxy` 包装）— 生成 Triton 代码字符串

> 详细解析见第三章。

### 1.3.7 家族6：代码生成层

**核心职责**：将融合后的调度节点转换为目标平台代码，并组装为可执行模块。

| 属性 | 说明 |
|-----|------|
| 源文件 | `torch/_inductor/codegen/` 目录下各文件 |
| 核心类 | `Kernel`（基类）、`TritonKernel`、`CppKernel`、`WrapperCodegen`（基类）、`PythonWrapperCodegen`、`CppWrapperGpu` 等 |
| 活跃阶段 | Phase 3（Kernel 代码生成）→ Phase 4（Wrapper 组装） |
| 输入 | 融合后的 `SchedulerNode`（内含 IR 闭包） |
| 输出 | 目标平台可执行代码（Triton kernel 源码 / C++ 源码 / Python wrapper 模块） |
| 类比 | LLVM 后端的指令发射（Code Emission）+ 链接器（Linker） |

代码生成层包含两个子族：

**Kernel 代码生成子族**（Phase 3）：

```
Kernel                           ← 基类，三段式（loads / compute / stores）
  ├── SIMDKernel                ← GPU/SIMD 基类（展平索引、range_trees）
  │     ├── TritonKernel        ← Triton 后端（生成 @triton.jit kernel）
  │     └── TritonSplitScanKernel ← 分裂扫描模式
  ├── CppKernel                 ← C++ 循环嵌套
  │     ├── CppVecKernel        ← SIMD 向量化
  │     └── CppTile2DKernel     ← 2D 分块向量化
  ├── TritonTemplateKernel      ← Triton 模板（如 flash attention）
  └── CUDATemplateKernel        ← CUDA 模板
```

**Wrapper 代码生成子族**（Phase 4）：

```
WrapperCodegen                  ← 基类
  ├── PythonWrapperCodegen      ← Python 源码（.py）
  │     └── SubgraphPythonWrapperCodegen ← 嵌套子图
  ├── WrapperFxCodegen          ← FX IR 形式
  ├── CppWrapperCpu             ← C++ 源码（CPU）
  ├── CppWrapperGpu             ← C++ 源码（GPU，含 CUDA 流管理）
  └── CppWrapperMps             ← C++ 源码（Metal）
```

> Kernel 代码生成详见第七章，Wrapper 代码生成详见第八章。

### 1.3.8 六大家族职责矩阵总表

| 家族 | 核心类 | 源文件 | 活跃阶段 | 输入 | 输出 | 编译阶段角色 |
|-----|-------|-------|---------|------|------|-----------|
| 1. 全局基础设施 | `Virtualized<T>`, `_V`, `NullHandler` | `virtualized.py` | 全阶段 | 无（隐式状态） | 无（隐式协作） | 运行时基础设施 |
| 2. Lowering 层 | `GraphLowering`, `FallbackKernel` | `graph.py`, `lowering.py` | Phase 0-1 | FX Graph | IR 节点集合 | 前端翻译 |
| 3. IR 表示层 | `TensorBox`, `Buffer`, `Pointwise`, `Reduction` | `ir.py` | Phase 1-3 | Lowering 层构建 | Scheduler/Codegen 消费 | 中端 IR |
| 4. 调度层 | `Scheduler`, `SchedulerNode`族, `BaseScheduling`族 | `scheduler.py` | Phase 2-3 | IR 节点集合 | 有序调度节点列表 | 中端优化 |
| 5. Handler 协议族 | `OpsHandler[T]`, `MockHandler`, `CSEProxy`, `TritonOverrides` | `ops_handler.py`, `codegen/*.py` | Phase 1, 3 | IR 闭包 | 多语义域结果 | 分析 + 后端桥接 |
| 6. 代码生成层 | `TritonKernel`, `CppKernel`, `PythonWrapperCodegen` | `codegen/*.py` | Phase 3-4 | 调度节点 | 目标平台可执行代码 | 后端代码生成 |

---

## 1.4 数据流全景图

### 1.4.1 追踪目标：`f(x, y) = relu(x + y)`

我们以一个最简单的逐元素计算为例，跟踪数据从 Python 函数定义到最终可执行模块的完整流转过程。

```python
import torch

@torch.compile
def f(x, y):
    return torch.relu(x + y)

# 调用
x = torch.randn(32, 64, device="cuda")
y = torch.randn(32, 64, device="cuda")
result = f(x, y)
```

### 1.4.2 完整数据流追踪

```
═══════════════════════════════════════════════════════════════════════════
              Inductor 编译数据流：f(x, y) = relu(x + y)
═══════════════════════════════════════════════════════════════════════════

阶段 0: Python 函数 → TorchDynamo → FX Graph
───────────────────────────────────────────────

    def f(x, y):                     TorchDynamo 字节码捕获
        return torch.relu(x + y)          │
                                          ▼
                              FX Graph (联合图):
                              ┌─────────────────────────┐
                              │ placeholder: x           │
                              │ placeholder: y           │
                              │ add = aten.add(x, y)     │
                              │ relu = aten.relu(add)    │
                              │ output: (relu,)          │
                              └─────────────────────────┘

阶段 1: AOTAutograd → 分离后的 FX Graph（Inductor 的输入）
───────────────────────────────────────────────

    前向图 (无梯度场景下直接使用):
    ┌──────────────────────────────────────────────────────┐
    │ def forward(self, x: "f32[32,64]cuda",               │
    │                 y: "f32[32,64]cuda"):                 │
    │   add   = aten.add.Tensor(x, y)                      │
    │   relu  = aten.relu.default(add)                     │
    │   return (relu,)                                     │
    └──────────────────────────────────────────────────────┘

    ↓ compile_fx_inner()

阶段 2: Phase 1 — GraphLowering (FX Graph → Inductor IR)
───────────────────────────────────────────────

    GraphLowering 逐节点解释执行:

    Node: x (placeholder)
    ┌──────────────────────────────────────────┐
    │ TensorBox(StorageBox(                     │
    │   InputBuffer(                            │
    │     name='x',                             │
    │     layout=FixedLayout('cuda:0',          │
    │       torch.float32,                      │
    │       size=[32, 64], stride=[64, 1])      │
    │   )                                       │
    │ ))                                        │
    └──────────────────────────────────────────┘

    Node: aten.add(x, y) ──→ lowering 注册表查找 ──→ make_pointwise(add_fn)
    ┌──────────────────────────────────────────┐
    │ TensorBox(StorageBox(                     │
    │   Pointwise(                              │
    │     device='cuda:0',                      │
    │     dtype=torch.float32,                  │
    │     inner_fn=lambda index:                │
    │       ops.load(x, idx(index))             │
    │       + ops.load(y, idx(index)),          │
    │     ranges=[32, 64],                      │
    │     origins={add}                         │
    │   )                                       │
    │ ))                                        │
    └──────────────────────────────────────────┘

    Node: aten.relu(add) ──→ lowering 注册表查找 ──→ make_pointwise(relu_fn)
    ┌──────────────────────────────────────────┐
    │ TensorBox(StorageBox(                     │
    │   Pointwise(                              │
    │     device='cuda:0',                      │
    │     dtype=torch.float32,                  │
    │     inner_fn=lambda index:                │
    │       ops.relu(ops.load(add_buf, idx)),   │
    │     ranges=[32, 64],                      │
    │     origins={relu}                        │
    │   )                                       │
    │ ))                                        │
    └──────────────────────────────────────────┘

    ↓ graph.buffers 收集所有 IR 节点

阶段 3: Phase 2 — Scheduler (IR → 调度节点 + 融合)
───────────────────────────────────────────────

    Scheduler._init(): 为每个 Buffer 创建 SchedulerNode

    SchedulerNode(add_node)    ←── add 对应的 Buffer
    SchedulerNode(relu_node)   ←── relu 对应的 Buffer

    fuse_nodes(): 贪婪融合

    检查: add_node 和 relu_node 能否垂直融合？
    ✓ 同设备 (cuda:0)
    ✓ relu_node 读取 add_node 的输出（垂直依赖）
    ✓ 均为逐元素 Pointwise 操作
    ✓ 融合内存收益 > 0（避免 add 中间结果的写回/重读）

    → 融合为 FusedSchedulerNode(add + relu)

    融合后的 IR（新增 Reduction 位置为空，仅 Pointwise 融合）:
    ┌──────────────────────────────────────────┐
    │ ComputedBuffer(                           │
    │   name='buf0',                            │
    │   data=Pointwise(                         │
    │     inner_fn=lambda index:                │
    │       tmp0 = ops.load(x, idx(index))      │
    │       tmp1 = ops.load(y, idx(index))      │
    │       tmp2 = tmp0 + tmp1                  │  ← add 内联
    │       tmp3 = ops.relu(tmp2)               │  ← relu 内联
    │       return tmp3                         │
    │     ,                                     │
    │     ranges=[32, 64],                      │
    │     origins={add, relu}   ← 融合溯源      │
    │   )                                       │
    │ )                                         │
    └──────────────────────────────────────────┘

    ↓ 有序的 SchedulerNode 列表

阶段 4: Phase 3 — TritonScheduling + TritonKernel (Kernel Codegen)
───────────────────────────────────────────────

    TritonScheduling.codegen_node(fused_node)

    步骤 1: 创建 TritonKernel 实例，安装 V.kernel 和 V.ops
    步骤 2: IR 闭包在 Handler 链下执行 4 次:

    ┌─ 第 1 次: MockHandler ─────────────────────────────────┐
    │ 输入: inner_fn callable                                  │
    │ 输出: IR 字符串（调试用）                                  │
    │ 作用: 确认 IR 闭包可正确执行                               │
    └──────────────────────────────────────────────────────────┘

    ┌─ 第 2 次: _RecordLoadStoreInner ──────────────────────┐
    │ 输入: inner_fn callable                                  │
    │ 输出: 内存依赖关系 (MemoryDep)                            │
    │ 作用: 分析 load/store 的依赖，指导调度                     │
    └──────────────────────────────────────────────────────────┘

    ┌─ 第 3 次: ValueRangeAnalysis ─────────────────────────┐
    │ 输入: inner_fn callable                                  │
    │ 输出: 每个临时变量的值域 (ValueRanges)                     │
    │ 作用: 辅助 Triton 的边界检查优化                           │
    └──────────────────────────────────────────────────────────┘

    ┌─ 第 4 次: CSEProxy(SimplifyIndexing(TritonOverrides())) ─────┐
    │ 输入: inner_fn callable                                       │
    │ 输出: Triton 代码字符串                                        │
    │                                                                │
    │ V.ops = CSEProxy(...)                                         │
    │ 执行 inner_fn 时:                                              │
    │   ops.load(x, idx) → CSE 查表 → 命中? 返回已有变量名            │
    │                              未命中? TritonOverrides.load()     │
    │                              → "tl.load(x_ptr + offset, ...)"   │
    │   tmp0 + tmp1  → TritonOverrides.add()                         │
    │                 → "tmp0 + tmp1"                                 │
    │   ops.relu(v) → TritonOverrides.relu()                         │
    │                → "tl.where(v > 0, v, 0)"                       │
    │   ops.store(...) → TritonOverrides.store()                     │
    │                   → "tl.store(out_ptr + offset, val)"           │
    └─────────────────────────────────────────────────────────────────┘

    步骤 3: 生成完整的 Triton Kernel 源码:

    ┌─────────────────────────────────────────────────────────┐
    │ @triton.jit                                             │
    │ def kernel_fused_add_relu(                              │
    │     x_ptr, y_ptr, out_ptr,                             │
    │     numel,                                              │
    │     BLOCK_SIZE: tl.constexpr,                           │
    │ ):                                                      │
    │     pid = tl.program_id(0)                              │
    │     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)│
    │     mask = offsets < numel                              │
    │                                                          │
    │     # loads                                              │
    │     x = tl.load(x_ptr + offsets, mask=mask)             │
    │     y = tl.load(y_ptr + offsets, mask=mask)             │
    │                                                          │
    │     # compute                                            │
    │     add_result = x + y                                   │
    │     result = tl.where(add_result > 0, add_result, 0)    │
    │                                                          │
    │     # stores                                             │
    │     tl.store(out_ptr + offsets, result, mask=mask)      │
    └─────────────────────────────────────────────────────────┘

    ↓ 多个 Kernel 源码字符串

阶段 5: Phase 4 — PythonWrapperCodegen (Wrapper 组装)
───────────────────────────────────────────────

    PythonWrapperCodegen 将所有 Kernel 组装为可调用模块:

    ┌─────────────────────────────────────────────────────────┐
    │ class CompiledModule:                                    │
    │     def __init__(self):                                  │
    │         self.kernel_fused_add_relu = kernel_fused_add_relu│
    │                                                          │
    │     def forward(self, x, y):                             │
    │         out = torch.empty_like(x)                        │
    │         numel = x.numel()                                │
    │         grid = lambda meta: (triton.cdiv(numel,          │
    │                              meta['BLOCK_SIZE']),)       │
    │         kernel_fused_add_relu[grid](                     │
    │             x, y, out, numel,                            │
    │             BLOCK_SIZE=1024                              │
    │         )                                                │
    │         return (out,)                                    │
    └─────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

### 1.4.3 类家族角色类比——工厂模式

如果将 Inductor 编译过程类比为一条工厂流水线，各类家族扮演的角色如下：

```
┌────────────────────────────────────────────────────────────────────┐
│                    Inductor 工厂流水线类比                           │
│                                                                    │
│   原料  ──> │ 前处理车间 │ ──> │ 加工车间 │ ──> │ 后处理车间 │ ──> 产品│
│             │ (Lowering)  │     │ (Scheduler)│   │ (Codegen)  │       │
│                                                                    │
│   家族1: Virtualized ── 工厂的中央空调/供电系统（全局基础设施）         │
│     ↳ _V.ops       ── 生产线上的工具插槽（可更换不同工具头）            │
│     ↳ _V.graph     ── 车间主任的全局对讲机（全局图引用）                │
│     ↳ _V.kernel    ── 当前工位的操作手册（当前 Kernel 上下文）          │
│                                                                    │
│   家族2: Lowering    ── 前处理车间（原料分类与初加工）                  │
│     ↳ GraphLowering ── 车间主管（逐件审核原料，分配到对应产线）          │
│     ↳ lowering.py   ── 加工工艺手册（每种原料的加工方法注册表）          │
│                                                                    │
│   家族3: IR 表示层   ── 半成品仓库（标准化存储加工中间件）               │
│     ↳ TensorBox     ── 标准化容器（统一的存取接口）                    │
│     ↳ Pointwise     ── 零件图纸（包含加工逻辑的闭包）                  │
│     ↳ Buffer        ── 存储货架（管理内存布局与生命周期）               │
│                                                                    │
│   家族4: 调度层      ── 生产调度中心（合并工序、优化排产）              │
│     ↳ Scheduler     ── 调度主任（统筹全局，决策哪些工序合并）          │
│     ↳ BaseScheduling── 各产线的工艺规范（Triton/C++ 不同的合并规则）   │
│                                                                    │
│   家族5: Handler 族  ── 可更换工具头（同一台机器，不同加工模式）        │
│     ↳ MockHandler   ── 调试探针（只看不加工，输出诊断信息）            │
│     ↳ 分析型Handler ── 质检仪器（检测零件规格：类型/值域/依赖）        │
│     ↳ TritonOverrides── Triton 专用工具头（生成 Triton 代码）         │
│                                                                    │
│   家族6: 代码生成层  ── 成品组装车间（产出最终可执行代码）              │
│     ↳ TritonKernel  ── 成品加工机（将图纸变为 Triton 代码）           │
│     ↳ WrapperCodegen── 总装线（将各成品组装为完整模块）                │
└────────────────────────────────────────────────────────────────────┘
```

### 1.4.4 数据形态演变总结

在整个编译流水线中，数据经历了六次形态转换：

```
Python 函数
    │ TorchDynamo (字节码捕获)
    ▼
FX Graph (联合图)
    │ AOTAutograd (前向/反向分离 + ATen 分解)
    ▼
FX Graph (后向分离，仅 ATen/Prims 算子)
    │ Phase 1: GraphLowering (逐节点 lowering)
    ▼
Inductor IR (TensorBox / Buffer / Pointwise 集合)
    │ Phase 2: Scheduler (拓扑排序 + 融合)
    ▼
调度计划 (有序 SchedulerNode 列表，部分已融合)
    │ Phase 3: Kernel Codegen (Handler 链 + 目标语言生成)
    ▼
Kernel 源码字符串 (Triton / C++ 函数)
    │ Phase 4: Wrapper Codegen (模块组装)
    ▼
可执行模块 (Python 模块 / C++ 动态链接库)
```

每次转换都有明确的执行者（特定的类家族）、明确的输入输出（特定的数据结构）、以及明确的设计动机（特定的优化目标）。这正是 Inductor 架构清晰性的体现。

---

## 本章小结

本章从三个维度建立了 Inductor 的全局认知框架：

1. **纵向位置**（1.1 节）：Inductor 在 PyTorch 编译栈中处于第三阶段，接收 AOTAutograd 输出的 ATen 级别 FX Graph，自身再划分为 Lowering（前端）、IR + Pass（中端）、Scheduling + Codegen（后端）三层。

2. **时间线**（1.2 节）：从 `compile_fx_inner()` 到可执行模块的五阶段编译生命周期——Phase 0 常量折叠、Phase 1 GraphLowering、Phase 2 Scheduler、Phase 3 Backend Codegen、Phase 4 Wrapper 组装。

3. **空间结构**（1.3-1.4 节）：六大家族（全局基础设施、Lowering 层、IR 表示层、调度层、Handler 协议族、代码生成层）的职责划分与协作方式，以及数据在各类家族间的流转轨迹。

接下来，我们将从最底层的全局基础设施开始，逐章深入每个类家族的内部设计。第二章聚焦 `Virtualized` 动态作用域引擎——它是所有类家族协作的隐式纽带。
