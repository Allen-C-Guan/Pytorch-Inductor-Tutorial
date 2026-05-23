# Triton 编译器设计：从 DSL 到机器码

## 副标题：以编译器设计视角系统解构 Triton 编译架构

---

## 本书简介

本书是一本面向计算机专业本科生的编译器教学书籍，以 **Triton 编译器** 为贯穿全书的真实案例，系统性地讲解现代领域特定编译器（Domain-Specific Compiler）的设计与实现。Triton 是当前深度学习编译栈中最活跃的 GPU 内核编译器之一，它是 PyTorch Inductor 的默认代码生成后端，也是理解 ML 编译器全栈技术的绝佳入口。

本书按照编译器的自然管线（pipeline）组织内容：**前端（Triton DSL + TTIR） → 中间层（TTGIR 设计与优化）→ 后端（指令选择、流水线、寄存器分配、代码发射）→ 系统集成（JIT、缓存、Autotuning）**。每一章对应一个编译器阶段，先讲编译器理论（以 *Engineering a Compiler* 第3版为理论基础），再剖析 Triton 在该阶段的真实设计与实现，最后总结设计权衡与替代方案。

与传统的编译器教材不同，本书不满足于"描述代码做了什么"，而是持续追问 **What-How-Why** 三层：**What**（什么功能）、**How**（如何实现）、**Why**（为什么这样设计）。读者将不仅学会使用 Triton 编写 GPU 内核，更能理解其 MLIR-based 两级 IR 的设计哲学、Layout 系统贯穿全管线的设计精巧之处，以及软件流水线和 Warp Specialization 等高级优化背后的编译器理论。

---

## 目标读者与先修知识

### 目标读者

- 计算机科学与技术、软件工程、人工智能等专业的高年级本科生
- 对编译器、GPU 编程、深度学习系统感兴趣的硕士/博士研究生
- 希望深入理解 PyTorch Inductor 和 Triton 编译器内部原理的 ML 系统工程师
- 正在学习 MLIR 编译器基础设施并希望有一个完整案例研究的开发者

### 先修知识

| 知识领域 | 最低要求 | 书中对应处理方式 |
|----------|---------|----------------|
| **编程基础** | 熟练掌握一门编程语言（Python 优先） | 第 2 章从 Python DSL 出发，语法自然 |
| **数据结构与算法** | 本科课程水平（图、树、哈希表、DP） | 涉及算法时补充背景知识（第 7、11 章） |
| **计算机体系结构** | 了解 CPU 基本组成（寄存器、缓存、内存） | 第 1、8 章补充 GPU 体系结构基础 |
| **线性代数** | 矩阵乘法、张量的基本概念 | 仅在 MM/MHA 等算子示例中涉及 |
| **编译器** | 无要求，从零开始 | 第 1 章从编译器的基本概念讲起 |
| **GPU 编程（CUDA）** | 无要求，从零开始 | 第 1、2 章介绍 GPU 编程模型（SIMT、warp 等） |
| **MLIR / LLVM** | 无要求，从零开始 | 第 3 章系统讲解 MLIR 基础设施 |

如果你对编译器和 GPU 完全没有概念，不用担心——这正是本书要帮你建立的。如果你已有 CUDA 编程经验，可以跳过基础部分，直接从第二部分（第 3 章起）阅读。

---

## 如何使用本书

### 阅读路线

本书不是一本适合"从第一章读到最后一章"的单一路线教材。根据不同读者的背景和目标，推荐以下阅读路线：

#### 路线A：编译器初学者路线（推荐大多数读者）

```
第1章 → 第2章 → 第3章 → 第4章 → 第6章 → 第9章 → 第12章 → 第13章 → 第15章
```

先建立 Triton 编译管线的整体认知（前端 → 中间层 → 后端 → 系统集成），再回头深入感兴趣的优化章节（第5、7、8、10、11、14章）。这条路线假设读者没有编译器基础，从头建立完整的编译器世界观。

#### 路线B：系统工程师路线（希望快速上手实践）

```
第1章 → 第2章 → 第13章 → 第14章 → 第15章
```

先理解 Triton 的编程模型和 JIT/Autotuning 机制，能够编写和调试 Triton 内核后，再深入学习编译器内部。这条路线适合偏工程、希望快速上手的读者。

#### 路线C：编译器极客路线（对编译器设计本身感兴趣）

```
第1章 → 第3章 → 第4章 → 第5章 → 第6章 → 第7章 → 第8章 → 第9章 → 第10章 → 第11章 → 第12章
```

从编译器设计的角度，系统性遍历 IR 设计、类型系统、Lowering、优化、指令选择、寄存器分配的完整链条。这条路线适合希望深入理解编译器构造的读者，建议配合 *Engineering a Compiler* 教材同步阅读。

#### 路线D：专题研究路线（对特定话题感兴趣）

| 感兴趣的话题 | 推荐阅读章节 |
|-------------|-------------|
| Triton 如何表示和操作 GPU 内存 | 第4章（Layout 系统）、第8章（内存优化）、第11章（寄存器分配） |
| Triton 如何做代码生成 | 第9章（指令选择）、第12章（PTX/CUBIN 发射） |
| Triton 的高级优化技术 | 第7章（循环优化）、第10章（流水线/Warp Spec） |
| Inductor 如何调用 Triton | 第1章（生态定位）、第13章（JIT）、第14章（Autotuning） |
| Triton 的跨硬件可移植性 | 第4章（Layout 抽象）、第6章（Lowering）、第12章（多后端） |

### 配套阅读建议

- **编译器理论**：建议同步阅读 *Engineering a Compiler* (Cooper & Torczon, 3rd Ed.)，每章标注了对应的 EaC 章节号。
- **MLIR 学习**：建议先浏览 MLIR 官方 Toy Tutorial（https://mlir.llvm.org/docs/Tutorials/Toy/），第3章会从那里过渡到 TTIR。
- **GPU 体系结构**：建议参考 *Programming Massively Parallel Processors* (Kirk & Hwu, 4th Ed.)，第8章和第10章会大量引用。
- **源码阅读**：本书标注了所有关键源码的路径（相对本仓库根目录），建议打开源码边读边学。

---

## 全书目录

### 第一部分：基础与全景

#### [第 1 章：编译器设计导论与 Triton 全景](ch01_introduction.md)

本章从零开始介绍编译器的基本概念——前端、中间表示（IR）、优化与后端，帮助读者建立编译器的世界观。随后介绍 GPU 编程模型（SIMT、内存层次、warp 调度）以及 ML 编译器生态地图（XLA、TVM、Halide、Triton 的定位）。最后给出 Triton 在 PyTorch 编译栈中的定位（`torch.compile -> Inductor -> Triton`）、其基于 tile 的核心编程哲学以及两级 IR（TTIR + TTGIR）的设计动机。

---

### 第二部分：前端——Triton DSL 与 TTIR

#### [第 2 章：Triton 编程语言设计](ch02_triton_dsl.md)

本章深入 Triton DSL（`triton.language`）的设计。从 program/kernel/block/tile 的编程模型出发，逐一讲解 `tl.program_id`、`tl.arange`、`tl.load`、`tl.store` 等核心原语的语法与语义。分析 Python 作为 DSL 宿主语言的设计考量（嵌入式 DSL 的编译器理论），并与 CUDA C++、OpenCL 进行编程模型对比，揭示 Triton 以 tile（而非 thread）为基本编程单元的设计哲学。

#### [第 3 章：MLIR 基础设施与 Triton IR（TTIR）设计](ch03_mlir_ttir.md)

本章首先系统讲解 MLIR 的核心概念——Operation、Dialect、Region、Block——建立 MLIR 作为可扩展编译器基础设施的认知框架。随后聚焦 TTIR 方言设计，深度剖析 `tt.make_range`、`tt.load`、`tt.store`、`tt.reduce`、`tt.dot` 等核心 Op 的语义定义（TableGen `.td` 文件）、SSA 结构与数据流图，并与 FX Graph、StableHLO 进行对比，阐明 Triton 为何选择自定义 MLIR 方言而非直接使用现有标准方言。

---

### 第三部分：中间层——TTGIR 与优化

#### [第 4 章：TritonGPU IR（TTGIR）—— GPU 专属 IR 设计](ch04_ttgir_design.md)

本章是本部分的基石章节，全面剖析 Triton 的第二级 IR——TTGIR。核心内容包括：（1）**Layout 系统**——`blocked`、`mma`、`dot_op`、`slice`、`shared`、`nvidia_mma`、`warp_specialized` 等 encoding 类型，以及它们如何编码数据在 GPU 线程和内存上的分布；（2）**Memory Space**——`global`、`shared`、`register` 的编译器语义；（3）**Layout 转换**——`ConvertLayoutOp` 的插入逻辑与正确性条件。本章揭示了 Triton 最核心的设计：用 Layout 抽象将硬件细节从数据流中解耦。

#### [第 5 章：语义分析与类型系统](ch05_type_system.md)

本章从编译器前端的类型检查与类型推断理论（EaC Ch.5）出发，深入剖析 Triton DSL 的类型系统实现。包括 `triton.language.semantic` 的语义检查逻辑、`dtype` 体系与 `constexpr` 机制、标量/指针/张量的类型层次、运算符重载与类型推导规则、Python 函数 inline 展开为 TTIR 操作序列的过程，以及编译期常量折叠（`constexpr`）在设计上的巧妙之处。

#### [第 6 章：Lowering—— TTIR → TTGIR 的方言转换](ch06_lowering_ttir_ttgir.md)

本章聚焦 TTIR 到 TTGIR 的 lowering——这是 Triton 编译管线中最重要的方言转换。在 MLIR DialectConversion 框架的理论基础上，详细剖析 `TritonToTritonGPUPass` 的核心策略：为每个操作分配 Layout（数据 Layout 传播算法的形式化描述——前向传播与后向传播的不动点迭代）；当相邻操作的 Layout 不兼容时自动插入 `ConvertLayoutOp`；将硬件无关的 `tt.load`/`tt.store` 转换为硬件感知的 `ttg.local_load`/`ttg.local_store` + `ttg.async_copy`。

#### [第 7 章：循环优化—— Tiling、Peeling 与展开](ch07_loop_optimization.md)

本章系统讲解循环优化理论（EaC Ch.9）在 Triton 中的应用。涵盖循环分块（tiling/strip-mining）与 Triton tile-based 编程模型的天然亲和性、循环剥离（loop peeling）——`LoopPeeling` pass 的实现与决策逻辑、循环展开（unrolling）在 TTGIR 层的 warp-level 展开策略，以及依赖分析（loop-carried vs. loop-independent dependence）在决定优化可行性中的角色。

#### [第 8 章：内存优化—— Coalescing、Layout 与 Shared Memory](ch08_memory_optimization.md)

本章从 GPU 内存层次（HBM → L2 → L1/Shared Memory → Register File）的理论基础出发，深入 Triton 的内存优化体系。涵盖合并访问（coalescing）的地址模式分析与 `CoalesceUtils` 优化 pass、Layout 系统与内存访问模式的对应关系、Shared Memory 分配（`AllocateSharedMemory` pass）、异步拷贝管线（`ttg.async_copy_global_to_local`）与数据预取策略，以及 Barrier/Membar 同步的插入逻辑。

---

### 第四部分：后端——代码生成与运行时

#### [第 9 章：指令选择—— TTGIR → LLVM IR](ch09_instruction_selection.md)

本章深入 Triton 编译管线中最核心的代码生成环节——将 TTGIR 转换为 LLVM IR。在指令选择理论（EaC Ch.10：树模式匹配、DAG 覆盖）的基础上，逐一剖析 `ElementwiseOpToLLVM`（逐元素操作映射）、`ReduceOpToLLVM`（warp shuffle + shared memory 归约）、`DotOpToLLVM`（Tensor Core MMA 指令选择）、`LoadOpToLLVM`/`StoreOpToLLVM`（地址计算与 masking）、`ConvertLayoutOpToLLVM`（shared memory 中转的 layout 转换）以及控制流和扫描操作的 LLVM 映射。

#### [第 10 章：软件流水线（Pipelining）与 Warp Specialization](ch10_pipeline_warp_spec.md)

本章讲解 Triton 最精巧的性能优化技术。在指令调度理论（EaC Ch.11）框架下，剖析 Triton 的软件流水线系统：`PipelineExpander`（异步拷贝+等待的流水线展开）、`PipeliningUtility`（阶段划分与 barrier 插入）、乒乓 buffer 双缓冲策略。随后深入 Warp Specialization——`WarpSpecialization` pass 将 warp 划分为生产者/消费者角色，`WarpSpecializeUtility` 管理 warp group 分配与同步，以及 MMAv5 引入的 `MMAv5PipelineUtility` 管线。

#### [第 11 章：寄存器分配与内存管理](ch11_register_allocation.md)

本章从寄存器分配理论（EaC Ch.12：活跃范围分析、干涉图、图着色 Chaitin-Briggs 算法、线性扫描）出发，剖析 Triton 的内存管理层。涵盖 `Allocation` 分析（buffer 活跃范围、内存复用）、`BufferRegion` 分析（buffer 区域划分）、`AllocateSharedMemory`（scratch memory 分配）、`GlobalScratchMemoryAllocation`、以及 `Alias` 分析（指针别名关系与内存依赖）。最后讨论 Triton 如何将寄存器分配委托给 LLVM/NVPTX 后端的设计考量。

#### [第 12 章：后端代码发射—— LLVM → PTX → CUBIN](ch12_backend_code_emission.md)

本章追踪代码生成的最后一公里：LLVM IR → NVPTX 后端 → PTX 汇编 → CUBIN（SASS）的完整翻译过程。在 LLVM 后端流水线简介（EaC Ch.13）的基础上，讲解 NVPTX 后端的翻译机制、PTX 汇编器的优化策略，以及 Triton 的多后端支持架构——NVIDIA（CUDA）、AMD（ROCm/HIP）、Ascend（昇腾 NPU）后端的插件接口与可移植性设计。

---

### 第五部分：集成、调优与展望

#### [第 13 章：JIT 编译系统与缓存管理](ch13_jit_cache.md)

本章聚焦 Triton 的编译执行模型。讲解 JIT（Just-In-Time）编译的完整工作流程——从 `ASTSource` 到 `compile()` 到 IR passes 到 PTX/CUBIN 的全链路；编译缓存系统——`CacheManager` 的 hash-based 缓存键生成（源码 hash + 编译选项 + GPU 架构）、缓存失效策略、环境变量变更检测；异步编译（`_async_compile.py`，编译与执行重叠）；以及 AOT（Ahead-Of-Time）编译模式的实现。

#### [第 14 章：Autotuning 系统](ch14_autotuning.md)

本章深入 Triton 的自动调优系统。在自动调优的编译器理论基础（经验搜索 vs. 模型驱动优化）上，剖析 `autotuner.py` 的架构——Config 空间定义（`BLOCK_SIZE`、`num_warps`、`num_stages`、`num_ctas`）、搜索策略（网格搜索、遗传算法、贝叶斯优化）、`OutOfResources` 检测与剪枝、`Heuristic` 模式与 `Autotuner` 模式的分工，以及与 PyTorch Inductor 的调优协同关系。

#### [第 15 章：端到端编译流程回顾与展望](ch15_end_to_end.md)

本章将全书所学串联为一条完整的编译管线：`@triton.jit → AST 构建 → TTIR 生成 → 类型推断 → TTIR passes → TTIR→TTGIR lowering → TTGIR passes (coalescing, pipelining, warp spec 等) → TTGIR→LLVM conversion → LLVM opt → NVPTX → PTX → CUBIN → kernel launch`。回顾各阶段间的数据接口与 IR 形态变化，总结 Triton 编译器设计中的关键权衡（两级 IR、Layout 抽象、Python-First、MLIR-Based），并展望未来方向——动态 shape、sparsity、flash attention 专用优化、MLIR-based 统一生态。

---

## 附录

| 附录 | 内容 |
|------|------|
| **[附录 A](appendix_a_eac_mapping.md)** | *Engineering a Compiler* 章节完整映射——本书每小节与 EaC 教材的对应关系 |
| **[附录 B](appendix_b_mlir_primer.md)** | MLIR 核心概念速查——Operation、Dialect、Region、Block、Pass、Pattern Rewrite 等关键概念的定义与用法 |
| **[附录 C](appendix_c_gpu_architecture.md)** | GPU 体系结构速查——NVIDIA（Ampere/Hopper）、AMD（CDNA）、Ascend（达芬奇）的硬件参数与编程模型对比 |
| **[附录 D](appendix_d_tablegen_reference.md)** | 关键 TableGen 定义索引——TTIR/TTGIR 所有 Op、Type、Attribute 的完整 `.td` 定义路径索引 |
| **[附录 E](appendix_e_glossary.md)** | 术语表——全书涉及的所有编译器、GPU、MLIR 术语的中英对照与释义 |

---

## 参考资源

### 核心教材

| 教材 | 版本 | 覆盖内容 |
|------|------|---------|
| *Engineering a Compiler* | Keith D. Cooper & Linda Torczon, 3rd Edition (2022) | 编译器理论基础：扫描、解析、语义分析、IR 设计、优化、指令选择、寄存器分配、指令调度 |
| *Programming Massively Parallel Processors* | David B. Kirk & Wen-mei W. Hwu, 4th Edition (2022) | GPU 体系结构与 CUDA 编程：SIMT、内存层次、warp 调度、Tensor Core、coalescing、bank conflict |
| *MLIR Tutorial* | LLVM 官方文档 (https://mlir.llvm.org/docs/Tutorials/) | MLIR 基础设施：Toy Tutorial、Dialect 定义、Pass 框架、Pattern Rewrite、DialectConversion |

### 核心论文

| 论文 | 出处 | 关键贡献 |
|------|------|---------|
| Tillet, P., et al. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." | MAPS@PLDI, 2019 | Triton 编译器的原始设计：tile-based 编程模型、两级 IR、blocked layout |
| Lattner, C., et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." | CGO, 2021 | MLIR 的设计哲学：Dialect、Operation、可扩展的编译器基础设施 |
| Ragan-Kelley, J., et al. "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines." | PLDI, 2013 | Halide 的算法与调度分离设计——Triton tile-based 编程模型的思想来源之一 |
| Chen, T., et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." | OSDI, 2018 | TVM 的端到端优化编译器——与 Triton 的定位对比 |

### 关键源码索引

本书所有源码引用均基于本 workspace 中的以下源码目录：

| 模块 | 路径（相对于 workspace 根目录） | 内容 |
|------|-------------------------------|------|
| Triton Python 前端 | `triton/python/triton/language/` | Triton DSL（`triton.language`），Python 层语义分析与代码生成 |
| Triton 编译器编排 | `triton/python/triton/compiler/` | 编译驱动（`compiler.py`）、代码生成器（`code_generator.py`） |
| Triton 运行时 | `triton/python/triton/runtime/` | JIT 编译、缓存管理、autotuner、kernel launch |
| Triton 后端接口 | `triton/python/triton/backends/` | 后端抽象接口（`compiler.py`、`driver.py`） |
| TTIR 方言定义 | `triton/include/triton/Dialect/Triton/IR/` | TTIR Op（`TritonOps.td`）、Type（`TritonTypes.td`）、Attr（`TritonAttrDefs.td`） |
| TTGIR 方言定义 | `triton/include/triton/Dialect/TritonGPU/IR/` | TTGIR Op/Type/Attr、Layout 编码定义 |
| TTIR→TTGIR Lowering | `triton/lib/Conversion/TritonToTritonGPU/` | TTIR 到 TTGIR 的 lowering pass |
| TTGIR→LLVM Conversion | `triton/lib/Conversion/TritonGPUToLLVM/` | TTGIR 到 LLVM IR 的代码生成（Elementwise、Reduce、Load/Store、Dot 等） |
| Triton 分析模块 | `triton/lib/Analysis/` | Alias 分析、Allocation 分析、AxisInfo 分析、Membar 分析 |
| TTIR Transform | `triton/include/triton/Dialect/Triton/Transforms/` | TTIR 层优化 pass（loop peeling 等） |
| TTGIR Transform | `triton/include/triton/Dialect/TritonGPU/Transforms/` | TTGIR 层优化 pass（coalescing、pipelining、prefetch、warp specialization 等） |
| Triton C++ 核心 | `triton/lib/Dialect/Triton/` 和 `triton/lib/Dialect/TritonGPU/` | 方言实现、pass 注册、verifier |
| Triton 解释器 | `triton/python/triton/runtime/interpreter.py` | Triton kernel 解释执行器（调试用） |
| Triton Autotuner | `triton/python/triton/runtime/autotuner.py` | 自动调优系统实现 |
| libdevice 数学库 | `triton/python/triton/language/extra/libdevice.py` | 数学库函数映射（对标 CUDA libdevice） |

---

## 本书规范

### 写作规范

- **语言**：正文使用简体中文。技术术语（如 Dialect、Layout、Warp、Coalescing、Lowering）在首次出现时标注英文原文，后续统一使用中文术语或英文缩写。
- **What-How-Why 三层逻辑**：全书所有阐述遵循"是什么 → 如何实现 → 为什么这样设计"的递进结构。不满足于描述代码的功能，必须追问设计决策背后的推理、权衡与替代方案分析。
- **渐进深入**：每章从直觉/类比出发 → 形式化描述 → 映射到源码，确保不同基础的读者都能跟上。
- **源码为本**：所有源码路径、行号、函数签名、TableGen 定义必须经过实际验证，不得编造。每章末尾附"正确性校验报告"。

### 术语规范

全书使用统一的术语体系，以下是核心术语的中英对照：

| 中文术语 | 英文原文 | 说明 |
|---------|---------|------|
| 方言 | Dialect | MLIR 的核心扩展机制，一组相关的 Operation、Type、Attribute 的集合 |
| 操作 | Operation / Op | MLIR 的基本计算单元，相当于 LLVM IR 的 Instruction |
| 类型 | Type | MLIR 的类型系统基本单元 |
| 属性 | Attribute | 编译期常量元数据 |
| 通道 | Pass | MLIR 的 IR 转换/优化单元 |
| 降级 | Lowering | 将高层 IR 转换为低层 IR 的过程 |
| 布局 | Layout / Encoding | TTGIR 中描述数据在 GPU 线程和内存上分布方式的元数据 |
| 线程束 | Warp | NVIDIA GPU 中 32 个线程为一组的调度和执行单元 |
| 线程块 | Thread Block / CTA | GPU 编程中线程的基本组织单元 |
| 合并访问 | Coalescing | GPU 全局内存访问优化技术，多个线程访问连续地址以最大化带宽 |
| 软件流水线 | Software Pipelining / Pipelining | 通过重叠计算与数据传输来隐藏延迟的编译优化技术 |
| 线程束特化 | Warp Specialization | 将 warp 划分为不同角色（生产者/消费者）以提升并行度的技术 |
| 瓦片 | Tile | Triton 的基本编程单元，对应一个线程块处理的数据子块 |
| 即时编译 | JIT (Just-In-Time) | 运行时编译模式，Triton 的默认编译执行方式 |
| 自动调优 | Autotuning | 通过搜索不同配置参数组合来找到最优性能的过程 |

### 代码示例规范

- 代码示例以 Python 为主（Triton DSL），关键 IR dump 结果亦作为代码块展示
- 每个代码示例必须包含注释说明其演示的概念及对应章节
- 可运行的示例标注运行方式和预期输出
- IR dump 示例标注生成命令（如 `TRITON_DUMP_IR=1`）

```python
# 示例：Triton tile-based 编程模型演示（对应第 2 章）
# 运行：python example.py
# 查看 TTIR：TRITON_DUMP_IR=1 python example.py

import triton
import triton.language as tl
import torch

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

# 调用：
# x = torch.randn(1024, device='cuda')
# y = torch.randn(1024, device='cuda')
# output = torch.empty_like(x)
# vector_add_kernel[(32,)](x, y, output, 1024, BLOCK_SIZE=32)
```

### 图示规范

全书使用 Mermaid 语法绘制图表，统一风格：

| 图表类型 | Mermaid 语法 | 用途 |
|---------|-------------|------|
| 类图 | `classDiagram` | TTIR/TTGIR Op 继承层次、Layout 类型层次 |
| 序列图 | `sequenceDiagram` | 编译 pipeline 各阶段的调用序列和数据流 |
| 流程图 | `flowchart TD/LR` | 编译管线概览、优化决策流程 |
| 状态图 | `stateDiagram-v2` | Layout 传播状态转换、autotuner 状态机 |

每张图必须有标题和简短说明。

### 两级 IR 的标注约定

全书始终严格区分 TTIR（硬件无关的数据流 IR）和 TTGIR（GPU 硬件感知的并行化 IR），不可混淆两者的职责边界：

- 讨论 TTIR 时，用 `tt.` 前缀表示 Op（如 `tt.load`、`tt.reduce`）
- 讨论 TTGIR 时，用 `ttg.` 前缀表示 Op（如 `ttg.local_load`、`ttg.async_copy_global_to_local`）
- Layout 系统仅存在于 TTGIR 层，TTIR 层仅有 `tt.encoding` 作为占位符

---

## 致谢

本书的撰写得益于以下资源与社区的启发：

- **PyTorch 团队**开发的 Inductor 编译器，为本书提供了最重要的编译栈上下文
- **Triton 社区**的持续贡献，特别是 OpenTriton 项目的开源协同
- *Engineering a Compiler* 的作者 Keith D. Cooper 和 Linda Torczon，他们的教材为编译器教学建立了黄金标准
- MLIR 社区和 Chris Lattner 等人的开创性工作，使可扩展编译器基础设施成为现实

---

*本书内容由 Claude 协助撰写，所有源码引用均基于本 workspace 中的真实代码。如有错误或遗漏，欢迎指出。*
