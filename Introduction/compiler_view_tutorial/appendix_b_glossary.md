# 附录 B：术语表

> 本术语表收录全书涉及的专业术语，按类别组织，提供中英文对照、一句话定义和章节引用。术语按英文字母排序。

---

## 1. 编译器理论术语 (Compiler Theory Terms)

### A

**Abstract Syntax Tree (AST)** — 抽象语法树

源代码的树形结构表示，其中每个节点代表一个语法构造。在传统编译器中由语法分析器生成。在 PyTorch 编译栈中，FX Graph 扮演类似角色。

*参见：第 2 章*

### B

**Basic Block** — 基本块

一段连续的指令序列，只有一个入口点（第一条指令）和一个出口点（最后一条指令），中间没有分支跳转。是控制流分析的基本单位。Inductor 的 LoopBody 内部结构可以类比基本块。

*参见：第 3 章、第 6 章*

**Backend** — 后端

编译器中负责将中间表示翻译为目标机器代码的阶段，包含指令选择、寄存器分配和指令调度三个子阶段。在 PyTorch 编译栈中，Inductor 扮演后端角色。

*参见：第 1 章、第 8 章*

**Bytecode** — 字节码

源代码编译后产生的低级指令序列。CPython 将 Python 源码编译为字节码后由虚拟机执行。Dynamo 通过拦截 CPython 字节码来追踪程序行为。

*参见：第 2 章*

### C

**CFG (Control Flow Graph)** — 控制流图

以基本块为节点、以控制转移为边的有向图，表示程序中所有可能的执行路径。依赖分析中使用的依赖图是 CFG 的一种扩展形式。

*参见：第 3 章、第 6 章*

**Code Generation** — 代码生成

编译器后端的最终阶段，将优化后的中间表示翻译为目标语言（如机器码、C++、Triton）。Inductor 支持两种目标：Triton（GPU）和 C++（CPU）。

*参见：第 8 章*

**Compiler** — 编译器

将一种语言的程序翻译为另一种语言的程序，同时保持语义不变的工具程序。经典模型分为前端、优化器和后端三个阶段。

*参见：第 1 章*

**Constant Folding** — 常量折叠

一种编译优化，在编译时计算所有操作数均为常量的表达式，用结果替换原表达式。Inductor 在 `constant_folding.py` 中实现。

*参见：第 5 章*

**Constant Propagation** — 常量传播

一种编译优化，跟踪常量值在程序中的传播路径，将引用常量的变量直接替换为常量值。

*参见：第 5 章*

**CSE (Common Subexpression Elimination)** — 公共子表达式消除

识别并消除重复计算的相同表达式，用之前计算的结果替换后续相同表达式。Inductor 在 IR 层面实现 CSE。

*参见：第 5 章*

### D

**DAG (Directed Acyclic Graph)** — 有向无环图

一种没有有向环的有向图。Inductor 的计算图是 DAG，保证存在合法的拓扑排序（执行顺序）。

*参见：第 1 章、第 3 章*

**DCE (Dead Code Elimination)** — 死代码消除

删除对程序输出没有影响的代码（死代码），减少不必要的计算和内存分配。

*参见：第 5 章*

**Dispatch** — 分发/派发

在运行时或编译时根据操作数的类型、设备等属性选择具体实现的机制。PyTorch 使用 dispatch key 机制路由算子调用到不同后端。

*参见：第 1 章*

**Dominance** — 支配关系

在控制流图中，如果从入口到节点 B 的所有路径都必须经过节点 A，则称 A 支配 B。用于依赖分析和调度决策。

*参见：第 6 章*

### E

**Emit** — 发射/生成

编译器代码生成阶段将中间表示转换为目标代码的过程。例如，Inductor 的 Triton codegen "emit" 一个 Triton kernel。

*参见：第 8 章*

### F

**Frontend** — 前端

编译器中负责理解源语言的阶段，包含词法分析、语法分析和语义分析，将源代码转换为中间表示。在 PyTorch 编译栈中，Dynamo 扮演前端角色。

*参见：第 1 章、第 2 章*

### G

**Guard** — 守卫条件

Dynamo 在编译时记录的假设条件（如输入张量的 shape、dtype、device 等），运行时进行检查。如果 Guard 条件不满足，则触发重新编译或回退到 eager mode。

*参见：第 2 章*

### I

**IR (Intermediate Representation)** — 中间表示

编译器内部用来表示程序的数据结构，介于源语言和目标语言之间。Inductor 使用两层 IR：FX Graph（高层）和 Inductor IR（低层）。

*参见：第 1 章、第 3 章*

**Instruction Selection** — 指令选择

编译器后端的子阶段，将 IR 操作映射到目标机器的指令。Inductor 将 IR 操作映射为 Triton 操作（GPU）或 C++ 表达式（CPU）。

*参见：第 8 章*

**Instruction Scheduling** — 指令调度

编译器后端的子阶段，确定操作的执行顺序以优化性能。Inductor 的 Scheduler 模块确定融合组的执行顺序。

*参见：第 10 章*

### J

**JIT (Just-In-Time) Compilation** — 即时编译

在程序运行时进行编译的技术。与 AOT（Ahead-of-Time）编译不同，JIT 可以利用运行时信息进行优化。PyTorch Inductor 是一个 JIT 编译器。

*参见：第 1 章*

### L

**Liveness Analysis** — 活跃性分析

确定程序中每个变量在哪些程序点上"活跃"（即后续还会被使用）的数据流分析。是寄存器分配和内存复用的基础。

*参见：第 9 章*

**Loop Fusion** — 循环融合

将两个或多个循环合并为一个循环的优化，减少循环开销和中间结果的内存访问。是 Inductor 最核心的优化策略之一。

*参见：第 7 章*

**Loop Tiling / Blocking** — 循环分块

将循环的迭代空间划分为较小的块（tile），使每个块的数据能放入高速缓存或共享内存，提高数据局部性。Inductor 在 Triton kernel 中使用 tiling 策略。

*参见：第 7 章*

**Loop Unrolling** — 循环展开

将循环体复制多份以减少循环控制开销、增加指令级并行的优化。

*参见：第 7 章*

**Lowering** — 翻译/降低

将高层次的中间表示转换为低层次的中间表示的过程。在 Inductor 中，Lowering 指将 FX Graph 翻译为 Inductor IR。

*参见：第 4 章*

### O

**Optimizer** — 优化器

编译器中负责改进中间表示以提高执行效率的阶段，在不改变语义的前提下应用各种优化变换。

*参见：第 1 章、第 5 章*

### P

**Pass** — 遍/趟

编译器中对整个程序（或其 IR）进行一次遍历并执行特定优化或变换的过程。Inductor 的编译过程由多个 pass 组成。

*参见：第 5 章*

**Parsing** — 语法分析

编译器前端将词法单元流组织成语法结构（通常是 AST 或类似树形结构）的过程。

*参见：第 2 章*

### R

**Register Allocation** — 寄存器分配

编译器后端的子阶段，将无限多个虚拟寄存器映射到有限个物理寄存器。Inductor 中对应的机制是 buffer 内存分配与复用。

*参见：第 9 章*

**Rewrite Rule** — 重写规则

定义如何将一种形式的表达式替换为等价但更优形式的规则。Inductor 的 lowering 和优化中大量使用重写规则。

*参见：第 4 章、第 5 章*

### S

**Scanning / Lexical Analysis** — 词法分析

编译器前端的第一个阶段，将字符流分解为词法单元 (Token) 流。

*参见：第 2 章*

**Semantic Analysis** — 语义分析

编译器前端中检查程序语义正确性（如类型检查、变量先声明后使用等）的阶段。

*参见：第 2 章*

**SSA (Static Single Assignment)** — 静态单赋值

一种 IR 形式，要求每个变量只被赋值一次。Inductor 的 TensorBox 提供了类似 SSA 的不可变语义。

*参见：第 3 章*

**Symbol Table** — 符号表

编译器前端用于记录标识符（变量名、函数名等）及其属性的数据结构。Dynamo 的 Guard 管理器可视为一种符号表。

*参见：第 2 章*

**Syntax-Directed Translation** — 语法导向翻译

在语法分析过程中同时进行翻译的技术，每个语法规则关联一个翻译动作（语义动作）。Inductor 的 lowering 机制遵循这种模式。

*参见：第 4 章*

### T

**Three-Phase Model** — 三阶段模型

经典编译器架构，将编译过程分为前端（理解源语言）、优化器（改进 IR）和后端（生成目标代码）三个逻辑阶段。

*参见：第 1 章*

**Token** — 词法单元

词法分析器从字符流中识别出的最小有意义单位，如关键字、标识符、运算符等。

*参见：第 2 章*

**Topological Sort** — 拓扑排序

对 DAG 节点的线性排列，使得所有边的方向一致。Inductor 的调度器在合法的拓扑排序中选择执行效率最高的一种。

*参见：第 1 章、第 10 章*

### V

**Value Numbering** — 值编号

一种优化技术，为每个计算出的值分配唯一编号，通过编号判断两个表达式是否计算了相同的值。是 CSE 的理论基础。

*参见：第 3 章、第 5 章*

---

## 2. 优化术语 (Optimization Terms)

### A

**Autotuning** — 自动调优

通过在实际硬件上测试不同参数配置来寻找最优实现参数的技术。Inductor 的 Triton kernel 使用 autotuning 选择最优的 BLOCK_SIZE 等参数。

*参见：第 7 章、第 8 章*

### B

**Buffer Reuse** — 缓冲区复用

将不再需要的 buffer 的内存分配给新的 buffer，减少总内存使用量。类比寄存器分配中的寄存器合并。

*参见：第 9 章*

**Branch Elimination** — 分支消除

在编译时确定分支条件的结果，删除不可能执行的分支代码。

*参见：第 5 章*

### C

**Compute-Bound** — 计算密集型

性能瓶颈在计算（而非内存访问）的操作类型。矩阵乘法是典型的计算密集型操作。Inductor 对计算密集型和内存密集型操作采用不同的融合策略。

*参见：第 7 章*

**Cost Model** — 代价模型

用于估计不同实现方案性能的分析模型。Inductor 的调度器使用代价模型来指导融合决策和调度策略选择。

*参见：第 7 章、第 10 章*

### D

**Data Locality** — 数据局部性

程序在短时间内反复访问相同或相邻内存位置的特性。tiling 优化通过提高数据局部性来提升性能。

*参见：第 7 章*

**Decomposition** — 算子分解

将复杂算子分解为更基本的算子组合。例如，将 `softplus` 分解为 `log(1 + exp(x))`。在 `decomposition.py` 中实现。

*参见：第 4 章*

**Operator Fusion** — 算子融合

将多个独立的算子合并为一个更大的算子（kernel），减少中间结果的内存读写和 kernel launch 开销。是 Inductor 最核心的优化。

*参见：第 7 章*

### E

**Expression Simplification** — 表达式简化

用更简单的等价表达式替换复杂表达式的优化。Inductor 使用 sympy 对符号表达式进行简化。

*参见：第 5 章*

### F

**Fission** — 循环分裂

将一个循环拆分为多个循环的变换。与融合相反，在某些场景下（如提高并行性）分裂更优。

*参见：第 7 章*

**Fusion Group** — 融合组

被决定融合在一起执行的一组 SchedulerNode。在代码生成时，一个融合组产生一个 kernel。

*参见：第 7 章*

### I

**In-Place Operation** — 就地操作

直接在输入缓冲区上修改数据而不分配新缓冲区的操作。Inductor 的 reinplace pass 尝试将 out-of-place 操作转换为 in-place 操作。

*参见：第 5 章、第 9 章*

**Indexing Optimization** — 索引优化

简化或优化数组索引表达式的变换，减少索引计算开销。在 `optimize_indexing.py` 中实现。

*参见：第 5 章*

### K

**Kernel Launch Overhead** — Kernel 启动开销

GPU 上启动一个 kernel 所需的固定时间开销。算子融合通过减少 kernel 数量来降低总启动开销。

*参见：第 7 章*

### L

**LICM (Loop-Invariant Code Motion)** — 循环不变量外提

将循环体内不依赖循环变量的计算移到循环外面的优化。

*参见：第 7 章*

### M

**Memory Bandwidth** — 内存带宽

单位时间内从内存读取或写入的数据量。许多 ML 操作是内存带宽密集型的，融合优化的核心收益来自减少内存访问次数。

*参见：第 7 章*

**Memory-Bound** — 内存密集型

性能瓶颈在内存访问（而非计算）的操作类型。逐元素操作（如 ReLU、add）是典型的内存密集型操作。

*参见：第 7 章*

### N

**Node Merging** — 节点合并

将图中多个节点合并为一个节点的变换。在融合决策中，多个 SchedulerNode 合并为一个 FusedSchedulerNode。

*参见：第 7 章*

### O

**Operator Fusion** — 算子融合

见 **Operator Fusion** 条目。

*参见：第 7 章*

**Optimization Pass** — 优化遍

见 **Pass** 条目。

*参见：第 5 章*

### P

**Padding** — 填充

在数据边界添加额外元素以满足对齐或形状要求的操作。

*参见：第 7 章*

**Peephole Optimization** — 窥孔优化

一种局部优化技术，检查少量相邻指令并替换为更高效的等价序列。

*参见：第 5 章*

### R

**Reduction** — 归约

将一组值通过二元操作（如求和、求最大值）合并为一个值的计算模式。Inductor IR 中有专门的 `Reduction` 节点类型。

*参见：第 3 章、第 7 章*

### S

**Strength Reduction** — 强度削减

将高代价操作替换为低代价等价操作的优化。例如，用移位替换乘以 2 的幂次的乘法。

*参见：第 5 章*

### T

**Tiling Strategy** — 分块策略

确定如何将迭代空间划分为块的具体方案，包括块大小、遍历顺序等。Triton kernel 的 BLOCK_SIZE 是 tiling 策略的关键参数。

*参见：第 7 章*

### V

**Vectorization** — 向量化

将标量操作转换为 SIMD（单指令多数据）操作，一条指令同时处理多个数据元素。Inductor 的 CPU codegen 使用 AVX/AVX-512 指令进行向量化。

*参见：第 7 章、第 8 章*

---

## 3. PyTorch 特有术语 (PyTorch-Specific Terms)

### A

**AOT Autograd** — AOT 自动微分

PyTorch 的提前计算自动微分机制，在编译前构建包含前向和反向传播的联合图（joint graph），使得 Inductor 可以同时优化前向和反向传播。

*参见：第 4 章、第 11 章*

**AOTI (AOTInductor)** — 提前编译 Inductor

Inductor 的提前编译模式，将模型编译为可独立部署的 `.so` 文件，不需要 Python 运行时。

*参见：第 11 章、第 12 章*

**ATen** — PyTorch C++ 张量库

PyTorch 底层的 C++ 张量操作库，提供所有基本算子的实现。FX Graph 中的算子调用最终对应到 ATen 操作。

*参见：第 1 章*

**Autograd** — 自动微分引擎

PyTorch 的自动微分系统，通过追踪张量操作构建计算图，支持自动反向传播。

*参见：第 1 章、第 4 章*

### D

**Decomposition** — 算子分解表

PyTorch 定义的一组规则，将高层算子（如 `torch.nn.functional.softmax`）分解为基本 ATen 算子。分解发生在 Dynamo 追踪后、Inductor lowering 前。

*参见：第 4 章*

**Dispatch Key** — 分发键

PyTorch 用于路由算子调用到具体实现的标识符，如 `CPU`、`CUDA`、`Autograd`、`CompositeExplicitAutograd` 等。

*参见：第 1 章*

**DTensor (Distributed Tensor)** — 分布式张量

PyTorch 的分布式张量抽象，描述张量在多个设备上的分布方式。Inductor 需要理解 DTensor 的分片语义。

*参见：第 12 章*

**Dynamo** — 动态编译器

`torch._dynamo` 模块，PyTorch 编译栈的前端。通过拦截 CPython 字节码执行来追踪 Python 函数的 tensor 操作，生成 FX Graph。

*参见：第 1 章、第 2 章*

### E

**Eager Mode** — 即时执行模式

PyTorch 的默认执行模式，每个操作立即执行，不做全局优化。用户可以直接 `print()` 中间结果，使用 Python debugger 调试。

*参见：第 1 章*

**ExportedProgram** — 导出程序

`torch.export` 产出的模型表示，包含计算图和元信息（输入签名、动态 shape 约束等）。可用于脱离原始 Python 代码的模型部署。

*参见：第 12 章*

### F

**Fake Tensor** — 虚拟张量

不带实际数据的张量对象，只记录 shape、dtype、device 等元信息。Dynamo 和 Inductor 使用 Fake Tensor 进行 shape 推导，避免实际计算。

*参见：第 2 章、第 4 章*

**FX Graph** — FX 计算图

`torch.fx` 模块产出的有向无环图 IR。节点类型包括 `placeholder`（输入）、`call_function`（函数调用）、`call_module`（模块调用）和 `output`（输出）。

*参见：第 2 章、第 3 章*

**FX GraphModule** — FX 图模块

将 FX Graph 包装为可执行的 `nn.Module`，是 Dynamo 传给 Inductor 的核心数据结构。

*参见：第 2 章*

### G

**Graph Break** — 图打断

Dynamo 在遇到无法编译的 Python 代码（如调用 C 扩展、使用 `id()` 等）时，将编译图分为多段，无法编译的部分交给 eager mode 执行。

*参见：第 2 章*

**Guard** — 守卫条件

Dynamo 在编译时记录的假设条件。运行时检查这些条件是否满足——如果满足，使用缓存的编译结果；如果不满足，触发重新编译或回退。

*参见：第 2 章*

### H

**Higher-Order Operator** — 高阶算子

接受函数作为参数的算子（如 `torch.utils.checkpoint`、`torch.cond`）。在编译时需要特殊处理以正确展开。

*参见：第 2 章*

### M

**Meta Tensor** — 元张量

类似 Fake Tensor，不带实际数据的张量。用于在不需要实际数据的情况下进行 shape 和 dtype 推导。

*参见：第 4 章*

### N

**Native Function** — 原生函数

在 `native_functions.yaml` 中定义的 PyTorch 算子，是 code generation pipeline 的源头。

*参见：第 1 章*

### P

**Proxy** — 代理对象

FX 追踪期间用于替代实际张量的对象，记录操作而不执行计算。每个 Proxy 对应 FX Graph 中的一个 Node。

*参见：第 2 章*

### S

**Subgraph** — 子图

一个完整的计算图被 graph break 分割后的一个片段。每个子图独立编译。

*参见：第 2 章*

**Symbolic Shape** — 符号形状

使用符号变量（如 `s0`、`s1`）表示的张量维度，使编译结果可以适应多种输入形状。

*参见：第 1 章、第 3 章*

### T

**torch.compile()** — torch.compile 接口

PyTorch 2.0 引入的编译 API。一行代码将模型包装为编译版本，首次调用时触发 JIT 编译。

*参见：第 1 章*

**torch.export** — torch.export 导出

PyTorch 的模型导出工具，将模型转换为 `ExportedProgram`，用于部署场景。

*参见：第 12 章*

### V

**Vmap** — 向量化映射

`torch.vmap` 函数，将一个函数自动向量化，在 batch 维度上并行执行。来自 `torch._functorch`。

*参见：第 12 章*

---

## 4. Inductor 特有术语 (Inductor-Specific Terms)

### B

**Buffer** — 缓冲区

Inductor IR 中表示已实现（realized）计算结果的核心数据结构。每个 Buffer 对应一个需要分配内存的张量，包含名称、内存布局和计算逻辑。

*参见：第 3 章、第 9 章*

**Buffer Name** — 缓冲区名称

每个 Buffer 的唯一字符串标识符，在代码生成时作为变量名使用。由 `utils.py` 中的命名函数生成。

*参见：第 3 章*

### C

**Compile FX** — 编译入口

`compile_fx.py` 中的入口函数，接收 Dynamo 传来的 FX GraphModule，协调 GraphLowering、AOT Autograd 和编译缓存。

*参见：第 1 章、第 11 章*

### D

**Dep (Dependency)** — 依赖关系

Inductor 中两个 IR 节点之间的依赖描述。具体类型包括 `MemoryDep`（内存依赖）、`StarDep`（全局依赖）和 `WeakDep`（弱依赖）。

*参见：第 6 章*

### E

**Extern Kernel** — 外部核函数

不由 Inductor 生成代码，而是调用外部库（如 cuBLAS、cuDNN、CUTLASS）的 kernel。`ExternKernel` 是 `ir.py` 中的一个类。

*参见：第 3 章、第 8 章*

### F

**FusedSchedulerNode** — 融合调度节点

多个 SchedulerNode 融合后的组合节点，在代码生成时产生一个 kernel。

*参见：第 7 章*

### G

**GraphLowering** — 图级翻译器

`graph.py` 中的核心类，接收 FX GraphModule，遍历每个 FX Node，调用 lowering 规则将其翻译为 Inductor IR，并管理 buffer 的生命周期。

*参见：第 4 章*

### I

**IR Node** — IR 节点

Inductor IR 的基本元素，包括 `TensorBox`、`StorageBox`、`Buffer`、`Pointwise`、`Reduction`、`Scan` 等。

*参见：第 3 章*

### L

**Layout** — 布局

描述 Buffer 的内存布局信息，包括形状 (shape)、步长 (stride) 和设备 (device)。

*参见：第 3 章*

**LoopBody** — 循环体

Inductor IR 中循环的内部表示，包含一组指令（表达式），描述循环每个迭代的计算逻辑。

*参见：第 3 章*

**Lowering Registry** — 翻译注册表

`lowering.py` 中维护的全局映射表（`lowerings` 字典），将每个 ATen 算子映射到对应的 IR 构建函数。

*参见：第 4 章*

### M

**MemoryDep** — 内存依赖

描述两个节点之间因访问同一 buffer 而产生的依赖关系，包含 buffer 名称、索引范围和访问大小。是 `dependencies.py` 中最常见的依赖类型。

*参见：第 6 章*

**Memory Donation** — 内存捐赠

当一个 buffer 不再需要时，将其内存"捐赠"给后续 buffer 使用的优化机制。在 `memory.py` 中实现。

*参见：第 9 章*

### N

**Nop (No-op)** — 空操作

不执行任何计算的操作节点，在 IR 中用于保持数据流的结构完整性。

*参见：第 3 章*

### P

**Pointwise** — 逐元素操作

Inductor IR 中表示逐元素计算的节点类型。输出张量的每个元素只依赖于输入张量中相同位置的元素（或广播位置）。如 `add`、`relu`、`mul` 等。

*参见：第 3 章*

### R

**Realize** — 实现

将延迟计算的 IR 节点强制计算并存储为 Buffer 的操作。未 realize 的节点表示"按需计算"，realize 后表示"已计算并存储"。

*参见：第 3 章、第 9 章*

**Reduction (IR Node)** — 归约 IR 节点

Inductor IR 中表示归约计算的节点类型。包含正常迭代范围和归约迭代范围，如 `sum`、`max`、`mean` 等操作。

*参见：第 3 章*

### S

**Scheduler** — 调度器

`scheduler.py` 中的核心模块，负责依赖分析、融合决策和节点排序，是连接 IR 和 codegen 的桥梁。

*参见：第 6 章、第 10 章*

**SchedulerNode** — 调度节点

调度器的基本工作单元，封装一个 Buffer 及其依赖信息。多个 SchedulerNode 可以融合为一个 FusedSchedulerNode。

*参见：第 6 章、第 7 章*

**SizeVar** — 大小变量

符号化的维度变量，用于表示动态 shape。由 `sizevars.py` 管理。

*参见：第 3 章*

**StarDep** — 全局依赖

表示一个节点对某个 buffer 的全局依赖（不限于特定索引范围），通常用于归约操作的输出。

*参见：第 6 章*

**StorageBox** — 存储盒子

包裹底层 IR 数据的中间层，管理 buffer 的生命周期。一个 StorageBox 可能被多个 TensorBox 共享（view 关系）。

*参见：第 3 章*

### T

**TensorBox** — 张量盒子

Inductor IR 中张量的最外层包装，提供类似 SSA 的不可变接口。跟踪 view 操作（如 reshape、permute），内部指向 StorageBox。

*参见：第 3 章*

**Tiling Utils** — 分块工具

`tiling_utils.py` 提供的分块策略计算工具，帮助调度器确定最优的 tile 大小。

*参见：第 7 章*

### V

**Virtualized** — 虚拟化

`virtualized.py` 提供的机制，允许 IR 节点在不绑定具体循环变量索引的情况下表达计算逻辑，实现延迟绑定。

*参见：第 3 章、第 8 章*

### W

**WeakDep** — 弱依赖

一种非强制的依赖关系，表示语义上的顺序约束但不涉及内存冲突。用于保持操作的原始顺序。

*参见：第 6 章*

**Wrapper CodeGen** — 包装器代码生成

将生成的 kernel 代码包装为可调用的 Python/C++ 函数，处理输入输出传递和 buffer 管理。

*参见：第 8 章、第 11 章*

---

## 5. GPU/硬件术语 (GPU/Hardware Terms)

### A

**AVX (Advanced Vector Extensions)** — 高级向量扩展

x86 CPU 上的 SIMD 指令集扩展。Inductor 的 CPU codegen 使用 AVX2 和 AVX-512 指令进行向量化。

*参见：第 7 章、第 8 章*

### B

**Bank Conflict** — bank 冲突

GPU 共享内存中多个线程同时访问同一 bank 导致的性能下降现象。

*参见：第 8 章*

**Block (Thread Block)** — 线程块

GPU 编程模型中，一组在同一个流多处理器 (SM) 上执行的线程集合。线程块内的线程可以通过共享内存通信并同步。在 Triton 中对应一个 `program` instance。

*参见：第 8 章*

**Block Size** — 块大小

Triton kernel 中每个 program instance 处理的数据元素数量，是 tiling 策略的关键参数。通过 autotuning 选择最优值。

*参见：第 7 章、第 8 章*

### C

**CUDA** — 统一计算设备架构

NVIDIA GPU 的并行计算平台和编程模型。Inductor 不直接生成 CUDA 代码，而是生成 Triton 代码，再由 Triton 编译器编译为 CUDA。

*参见：第 1 章、第 8 章*

**CUDA Graph** — CUDA 图

NVIDIA 提供的机制，将多个 CUDA kernel 调用录制为一个图，减少 CPU 端的 launch 开销。Inductor 支持自动 CUDA Graph 集成。

*参见：第 11 章*

**cuBLAS** — CUDA 基础线性代数子程序

NVIDIA 提供的高性能线性代数库。Inductor 可以调用 cuBLAS 执行矩阵乘法等操作，作为外部 kernel。

*参见：第 8 章*

**CUTLASS** — CUDA 张量核库

NVIDIA 提供的基于 CUDA 的矩阵乘法和卷积模板库。Inductor 可以使用 CUTLASS 生成高性能 GEMM kernel。

*参见：第 8 章*

### G

**Global Memory** — 全局内存

GPU 上容量最大但延迟最高的内存。所有线程都可以访问。减少全局内存访问次数是算子融合的核心动机。

*参见：第 7 章、第 8 章*

**Grid** — 线程网格

GPU 编程模型中，一次 kernel launch 的所有线程块的集合。Grid 的大小决定了总并行度。

*参见：第 8 章*

### K

**Kernel** — 核函数

在 GPU 上并行执行的函数。每个 kernel 由多个线程块组成，每个线程块由多个线程组成。Inductor 为每个融合组生成一个 kernel。

*参见：第 7 章、第 8 章*

### L

**Local Memory** — 本地内存

GPU 上每个线程私有的内存空间，通常映射到全局内存。当寄存器不足时使用。

*参见：第 8 章*

### O

**Occupancy** — 占用率

GPU 上实际活跃的 warp 数量与 SM 支持的最大 warp 数量之比。影响 GPU 的利用效率。

*参见：第 8 章*

### R

**Register** — 寄存器

GPU 上每个线程私有的最快存储。数量有限（典型 GPU 每线程最多 255 个寄存器），是性能的关键约束。

*参见：第 8 章、第 9 章*

### S

**Shared Memory** — 共享内存

GPU 上线程块内所有线程共享的高速内存。容量有限（典型 GPU 每个 SM 48-164 KB），但延迟远低于全局内存。Triton 自动管理共享内存的使用。

*参见：第 7 章、第 8 章*

**SIMD (Single Instruction, Multiple Data)** — 单指令多数据

一种并行计算方式，一条指令同时处理多个数据元素。GPU 的基本执行模型。CPU 上通过 AVX/SSE 指令实现 SIMD。

*参见：第 7 章、第 8 章*

**SM (Streaming Multiprocessor)** — 流多处理器

GPU 上的基本计算单元。每个 SM 包含多个 CUDA 核心，可以并发执行多个线程块。

*参见：第 8 章*

### T

**Tensor Core** — 张量核心

NVIDIA GPU（Volta 架构起）上的专用矩阵计算单元，可以在一个时钟周期内完成小矩阵乘法。Inductor 的 Triton kernel 利用 Tensor Core 加速矩阵运算。

*参见：第 8 章*

**Thread** — 线程

GPU 编程模型中最小的执行单元。每个线程有自己的寄存器和程序计数器。在 Triton 中，每个线程处理 BLOCK_SIZE 内的一个数据元素。

*参见：第 8 章*

**Triton** — Triton 语言/编译器

OpenAI 开发的高级 GPU 编程语言和编译器。提供比 CUDA 更高层次的抽象（自动管理共享内存、线程同步），是 Inductor GPU codegen 的目标语言。

*参见：第 1 章、第 8 章*

### W

**Warp** — 线程束

NVIDIA GPU 上 32 个线程组成的执行单元。同一 warp 内的线程以锁步 (lockstep) 方式执行相同指令。

*参见：第 8 章*

---

## 术语索引（按拼音排序）

| 中文术语 | 英文术语 | 类别 | 章节 |
|---------|---------|------|------|
| 包装器代码生成 | Wrapper CodeGen | Inductor | 第 8、11 章 |
| 保留 | Guard | PyTorch | 第 2 章 |
| 被动依赖 | WeakDep | Inductor | 第 6 章 |
| 常量传播 | Constant Propagation | 优化 | 第 5 章 |
| 常量折叠 | Constant Folding | 优化 | 第 5 章 |
| 抽象语法树 | Abstract Syntax Tree (AST) | 编译器 | 第 2 章 |
| 处理器核心 | Tensor Core | GPU | 第 8 章 |
| 词汇分析 | Scanning / Lexical Analysis | 编译器 | 第 2 章 |
| 词法单元 | Token | 编译器 | 第 2 章 |
| 代价模型 | Cost Model | 优化 | 第 7、10 章 |
| 代数简化 | Strength Reduction | 优化 | 第 5 章 |
| 单指令多数据 | SIMD | GPU | 第 7、8 章 |
| 调度器 | Scheduler | Inductor | 第 6、10 章 |
| 调度节点 | SchedulerNode | Inductor | 第 6、7 章 |
| 动态张量 | DTensor | PyTorch | 第 12 章 |
| 翻译注册表 | Lowering Registry | Inductor | 第 4 章 |
| 分布式张量 | DTensor | PyTorch | 第 12 章 |
| 分发键 | Dispatch Key | PyTorch | 第 1 章 |
| 分支消除 | Branch Elimination | 优化 | 第 5 章 |
| 高级向量扩展 | AVX | GPU | 第 7、8 章 |
| 高阶算子 | Higher-Order Operator | PyTorch | 第 2 章 |
| 公共子表达式消除 | CSE | 优化 | 第 5 章 |
| 共享内存 | Shared Memory | GPU | 第 7、8 章 |
| 关键路径 | Critical Path | 编译器 | 第 10 章 |
| 归约 IR 节点 | Reduction (IR Node) | Inductor | 第 3 章 |
| 归约操作 | Reduction | 优化 | 第 3、7 章 |
| 全局内存 | Global Memory | GPU | 第 7、8 章 |
| 全局依赖 | StarDep | Inductor | 第 6 章 |
| 即时执行模式 | Eager Mode | PyTorch | 第 1 章 |
| 寄存器 | Register | GPU | 第 8、9 章 |
| 寄存器分配 | Register Allocation | 编译器 | 第 9 章 |
| 计算密集型 | Compute-Bound | 优化 | 第 7 章 |
| 假张量 | Fake Tensor | PyTorch | 第 2、4 章 |
| 剪枝/窥孔优化 | Peephole Optimization | 优化 | 第 5 章 |
| 就地操作 | In-Place Operation | 优化 | 第 5、9 章 |
| 聚合操作 | Reduction | 优化 | 第 3、7 章 |
| 块大小 | Block Size | GPU | 第 7、8 章 |
| 联合图 | Joint Graph | PyTorch | 第 4 章 |
| 列表调度 | List Scheduling | 编译器 | 第 10 章 |
| 流多处理器 | SM | GPU | 第 8 章 |
| 循环不变量外提 | LICM | 优化 | 第 7 章 |
| 循环分块 | Loop Tiling / Blocking | 优化 | 第 7 章 |
| 循环分裂 | Loop Fission | 优化 | 第 7 章 |
| 循环融合 | Loop Fusion | 优化 | 第 7 章 |
| 循环展开 | Loop Unrolling | 优化 | 第 7 章 |
| 虚拟张量 | Meta Tensor | PyTorch | 第 4 章 |
| 虚拟化 | Virtualized | Inductor | 第 3、8 章 |
| 图打断 | Graph Break | PyTorch | 第 2 章 |
| 图级翻译器 | GraphLowering | Inductor | 第 4 章 |
| 外部核函数 | Extern Kernel | Inductor | 第 3、8 章 |
| 向量化 | Vectorization | 优化 | 第 7、8 章 |
| 向量化映射 | Vmap | PyTorch | 第 12 章 |
| 消除公共子表达式 | CSE | 优化 | 第 5 章 |
| 线程 | Thread | GPU | 第 8 章 |
| 线程块 | Block (Thread Block) | GPU | 第 8 章 |
| 线程束 | Warp | GPU | 第 8 章 |
| 线程网格 | Grid | GPU | 第 8 章 |
| 消除死代码 | DCE | 优化 | 第 5 章 |
| 语法分析 | Parsing | 编译器 | 第 2 章 |
| 语法导向翻译 | Syntax-Directed Translation | 编译器 | 第 4 章 |
| 语义分析 | Semantic Analysis | 编译器 | 第 2 章 |
| 张量盒子 | TensorBox | Inductor | 第 3 章 |
| 张量核心 | Tensor Core | GPU | 第 8 章 |
| 指令调度 | Instruction Scheduling | 编译器 | 第 10 章 |
| 指令选择 | Instruction Selection | 编译器 | 第 8 章 |
| 中间表示 | Intermediate Representation (IR) | 编译器 | 第 1、3 章 |
| 逐元素操作 | Pointwise | Inductor | 第 3 章 |
| 子图 | Subgraph | PyTorch | 第 2 章 |
| 自动调优 | Autotuning | 优化 | 第 7、8 章 |
| 自动微分引擎 | Autograd | PyTorch | 第 1、4 章 |
| 就地编译 | JIT Compilation | 编译器 | 第 1 章 |
| 核函数 | Kernel | GPU | 第 7、8 章 |
| 存储盒子 | StorageBox | Inductor | 第 3 章 |
| 缓冲区 | Buffer | Inductor | 第 3、9 章 |
| 缓冲区复用 | Buffer Reuse | 优化 | 第 9 章 |
| 缓冲区名称 | Buffer Name | Inductor | 第 3 章 |
| 内存带宽 | Memory Bandwidth | 优化 | 第 7 章 |
| 内存密集型 | Memory-Bound | 优化 | 第 7 章 |
| 内存捐赠 | Memory Donation | Inductor | 第 9 章 |
| 内存依赖 | MemoryDep | Inductor | 第 6 章 |
| 内存规划 | Memory Planning | Inductor | 第 9 章 |
| 数据局部性 | Data Locality | 优化 | 第 7 章 |
| 死代码消除 | DCE | 优化 | 第 5 章 |
| 算子分解 | Decomposition | PyTorch | 第 4 章 |
| 算子融合 | Operator Fusion | 优化 | 第 7 章 |
| 索引优化 | Indexing Optimization | 优化 | 第 5 章 |
| 三阶段模型 | Three-Phase Model | 编译器 | 第 1 章 |
| 符号表 | Symbol Table | 编译器 | 第 2 章 |
| 符号形状 | Symbolic Shape | PyTorch | 第 1、3 章 |
| 分发 | Dispatch | 编译器 | 第 1 章 |
| 分块策略 | Tiling Strategy | 优化 | 第 7 章 |
| 融合调度节点 | FusedSchedulerNode | Inductor | 第 7 章 |
| 融合组 | Fusion Group | 优化 | 第 7 章 |
| 实现 | Realize | Inductor | 第 3、9 章 |
| 守卫条件 | Guard | PyTorch | 第 2 章 |
| 输出程序 | ExportedProgram | PyTorch | 第 12 章 |
| CUDA 图 | CUDA Graph | GPU | 第 11 章 |
| 大小变量 | SizeVar | Inductor | 第 3 章 |
| 代理对象 | Proxy | PyTorch | 第 2 章 |
| 动态编译器 | Dynamo | PyTorch | 第 1、2 章 |
| 独立编译 Inductor | AOTI | PyTorch | 第 11、12 章 |
| 翻译/降低 | Lowering | 编译器 | 第 4 章 |
| 发射/生成 | Emit | 编译器 | 第 8 章 |
| 翻译 | Lowering | 编译器 | 第 4 章 |
| 遍/趟 | Pass | 编译器 | 第 5 章 |
| 控制流图 | CFG | 编译器 | 第 3、6 章 |
| FX 计算图 | FX Graph | PyTorch | 第 2、3 章 |
| FX 图模块 | FX GraphModule | PyTorch | 第 2 章 |
| 活跃性分析 | Liveness Analysis | 编译器 | 第 9 章 |
| 即时编译 | JIT Compilation | 编译器 | 第 1 章 |
| 编译器 | Compiler | 编译器 | 第 1 章 |
| 基本块 | Basic Block | 编译器 | 第 3、6 章 |
| 节点合并 | Node Merging | 优化 | 第 7 章 |
| 空操作 | Nop | Inductor | 第 3 章 |
| 循环体 | LoopBody | Inductor | 第 3 章 |
| 支配关系 | Dominance | 编译器 | 第 6 章 |
| 布局 | Layout | Inductor | 第 3 章 |
| IR 节点 | IR Node | Inductor | 第 3 章 |
| 编译入口 | Compile FX | Inductor | 第 1、11 章 |
| AOT 自动微分 | AOT Autograd | PyTorch | 第 4、11 章 |
| bank 冲突 | Bank Conflict | GPU | 第 8 章 |
| cuBLAS | cuBLAS | GPU | 第 8 章 |
| CUTLASS | CUTLASS | GPU | 第 8 章 |
| Triton | Triton | GPU | 第 1、8 章 |
| torch.compile | torch.compile() | PyTorch | 第 1 章 |
| torch.export | torch.export | PyTorch | 第 12 章 |
| 重写规则 | Rewrite Rule | 编译器 | 第 4、5 章 |
| 后端 | Backend | 编译器 | 第 1、8 章 |
| 前端 | Frontend | 编译器 | 第 1、2 章 |
| 优化器 | Optimizer | 编译器 | 第 1、5 章 |
| 有向无环图 | DAG | 编译器 | 第 1、3 章 |
| 拓扑排序 | Topological Sort | 编译器 | 第 1、10 章 |
| 值编号 | Value Numbering | 编译器 | 第 3、5 章 |
| 静态单赋值 | SSA | 编译器 | 第 3 章 |
| 字节码 | Bytecode | 编译器 | 第 2 章 |
| ATen | ATen | PyTorch | 第 1 章 |
| 原生函数 | Native Function | PyTorch | 第 1 章 |
| 占用率 | Occupancy | GPU | 第 8 章 |
| 本地内存 | Local Memory | GPU | 第 8 章 |
| 表达式简化 | Expression Simplification | 优化 | 第 5 章 |
| 强度削减 | Strength Reduction | 优化 | 第 5 章 |
| 填充 | Padding | 优化 | 第 7 章 |
| 依赖关系 | Dep (Dependency) | Inductor | 第 6 章 |
