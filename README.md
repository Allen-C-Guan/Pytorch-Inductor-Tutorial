# PyTorch 编译器技术栈 + 基础算子教学（inductor / operator / triton / MLIR）（持续补充中）

系统性的 PyTorch 编译器技术栈源码教学材料，按 **Inductor → operator → Triton → MLIR** 的顺序组织四大核心：**Inductor 编译器**、**基础算子（ATen Ops）**、**Triton 编译器** 与 **MLIR 编译器基础设施**，面向准备从事 ML 编译器开发的工程师，以专家视角提供从全局架构到关键代码的完整学习路径。

## 教学目标

- **Inductor**：理解 PyTorch 2 编译管线的完整架构——Dynamo 追踪 → FX 图优化 → Lowering → 调度与融合 → 代码生成
- **operator（基础算子）**：理解被 Inductor 编译的对象——ATen 基础算子到底在做什么（数学语义 / 操作逻辑 / 在模型里的角色），尤其是难懂的「操作类算子」（as_strided / gather / scatter …），以及每个算子在 Inductor 中走哪条降级路径（pointwise / reduction / template / fallback）
- **Triton**：理解 Triton 编译器的全栈设计——从 DSL 前端（TTIR）到 GPU 后端（TTGIR → LLVM IR → PTX → CUBIN）
- **MLIR**：理解 MLIR 编译器基础设施的设计哲学——先从全局俯瞰（编译器/IR 演进史、Dialect 体系、渐进式递降），再以编译器学科视角逐层拆解其工作机制（IR 表示、def-use chain、重写范式）
- 掌握每个阶段的设计动机、编译器理论基础与源码实现细节

## 权威参考

- PyTorch 2 论文 (ASPLOS 2024): *"PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"*
- Triton 论文 (MAPS@PLDI 2019): *"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"*
- MLIR 论文 (CGO 2021): *"MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"*
- MLIR Dialect 教程 (Lei.Chat): 由 MLIR/SPIR-V 核心开发者撰写的系列博客，涵盖 Linalg/Vector Dialect 与变换管线
- TorchInductor 设计帖: [dev-discuss.pytorch.org](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- 编译器理论教材: *Engineering a Compiler* (Keith D. Cooper & Linda Torczon, 3rd Edition)

---

## 目录结构

```
my pytorch tutorial/
├── README.md                            # 本文件
│
├── Inductor Introduction/               # Inductor 编译器教程
│   ├── README.md                        # Inductor 教程导航
│   ├── compiler_view_tutorial/          # 【核心】编译器设计视角（12章 + 2附录）
│   ├── skeleton_tutorial/               # 管线阶段视角（5个 phase）
│   └── key_classes_analysis/            # OOP/类设计视角（9章 + 附录）
│
├── operator/                            # 基础算子（ATen Ops）教程：以 Inductor/编译器视角看算子
│   ├── README.md                        # 算子书导航（含 161 算子速查）
│   ├── 00-…~16-…                        # 17 章：张量基底 + 数学类 + 操作类 + 模型层原语
│   ├── stride-essence.md                # stride 本质深度专题
│   └── appendix-b/c                     # 复合→基础分解 cookbook、Inductor 降级速查
│
├── Triton Introduction/                 # Triton 编译器教程
│   ├── README.md                        # Triton 教程导航
│   ├── High level Introduction of triton/  # 高层导论（3章 + 1附录）
│   └── triton_compiler_view_tutorial/   # 编译器设计视角（15章 + 5附录）
│
├── MLIR/                                # MLIR 编译器基础设施教程
│   ├── README.md                        # MLIR 教程导航
│   ├── MLIR high level overview/        # 第一阶段：高层概述（4章，含 Linalg/Vector Dialect 深度解析）
│   └── compiler-view-of-MLIR/           # 第二阶段：编译器视角教材（Ch0-Ch13）
│
└── code/
    └── src/
        ├── data_flow_panorama.py        # 数据流加工全景演示脚本
        ├── torch_compile_debug/         # torch.compile 调试输出
        └── phase1/                      # Phase 1 相关代码
```

---

## 四大教程体系

### Inductor 编译器教程 → [Inductor Introduction/](Inductor%20Introduction/)

三套并行教程从不同角度覆盖 Inductor 编译器：

| 教程 | 视角 | 篇幅 | 适合场景 |
|------|------|------|---------|
| **compiler_view_tutorial**（核心） | 编译器理论，映射 EaC 教材 | 12章 + 2附录 | 建立整体认知框架 |
| **skeleton_tutorial** | 编译管线阶段，12节标准化模板 | 5个 phase | 快速了解各阶段边界与数据流 |
| **key_classes_analysis** | OOP/类设计，类继承树与协作关系 | 9章 + 附录 | 深入源码理解"为什么这样设计" |

### operator 基础算子教程 → [operator/](operator/)

> **定位**：在理解「Inductor 怎么编译」之前，先看懂「被编译的算子在干什么」。它是三大编译器教程的**算子前置知识**——Inductor 的输入就是一张由 `aten.*` 算子组成的计算图。

以「位置保持 vs 位置变化」为分类轴，系统讲解 Core ATen 全 161 个基础算子的功能：数学公式 / 操作逻辑 / 实现复杂度 / 在模型里的角色，并标注每个算子在 Inductor 中走哪条降级路径。重心放在**操作类算子**（as_strided / gather / scatter / sort … 这些「不在算数学、而在搬运/重排/索引」的算子）。

**详细目录见 [operator/README.md](operator/README.md)**

| 部分 | 章节 | 核心内容 |
|------|------|---------|
| 前置 | [第 0 章 张量基底](operator/00-tensor-substrate.md) · [stride 本质](operator/stride-essence.md) | shape/dtype/stride/广播/类型提升/SSA 契约；stride 的数学本质与「能做/不能做」边界 |
| Part I 数学语义类 | [第 1 章](operator/01-elementwise-arithmetic.md) · [第 2 章](operator/02-transcendental.md) · [第 3 章](operator/03-comparison-boolean.md) · [第 4 章](operator/04-reductions.md) · [第 5 章](operator/05-linear-algebra-core.md) · [第 6 章](operator/06-activations.md) | 逐元素算术、三角与超越、比较/布尔/位运算、规约、线性代数核心、激活函数 |
| Part II 张量操作类（重心） | [第 7 章](operator/07-shape-and-view.md) · [第 8 章](operator/08-concat-split.md) · [第 9 章](operator/09-indexing-family.md) · [第 10 章](operator/10-sort-topk.md) · [第 11 章](operator/11-creation-filling.md) · [第 12 章](operator/12-memory-layout-dtype.md) | 形状与视图、拼接与切分、索引家族（最难）、排序与选取、创建与填充、内存布局与类型转换 |
| Part III 模型层原语 | [第 13 章](operator/13-convolution.md) · [第 14 章](operator/14-pooling.md) · [第 15 章](operator/15-normalization.md) · [第 16 章](operator/16-padding-upsampling-distance.md) | 卷积、池化、归一化、填充/上采样/采样 |
| 附录 | [附录 B 复合→基础分解](operator/appendix-b-decompositions.md) · [附录 C Inductor 降级速查](operator/appendix-c-inductor-cheatsheet.md) | silu/gelu/softmax… 如何分解；每类算子走 pointwise/reduction/template/fallback 哪条路 |

### Triton 编译器教程 → [Triton Introduction/](Triton%20Introduction/)

| 教程 | 视角 | 篇幅 | 适合场景 |
|------|------|------|---------|
| **High level Introduction of triton** | 工程实践，高层全景 | 3章 + 1附录 | 快速建立 Triton 编译器整体认知 |
| **triton_compiler_view_tutorial** | 编译器理论，映射 EaC + MLIR + GPU 教材 | 15章 + 5附录 | 系统性掌握 Triton 编译器全栈设计 |

### MLIR 编译器基础设施教程 → [MLIR/](MLIR/)

MLIR 教程采用**两阶段**结构，先建立全景再深入机制：

| 教程 | 视角 | 篇幅 | 适合场景 |
|------|------|------|---------|
| **MLIR high level overview**（第一阶段） | 编译器/IR 演进史 + Dialect 体系 + 变换管线 | 4 章 | 理解 MLIR 设计哲学与结构化代码生成的宏观图景 |
| **compiler-view-of-MLIR**（第二阶段） | 编译器学科镜头下的 MLIR 工作机制 | Ch0–Ch13 | 逐层拆解 IR 表示、def-use chain、图重写范式等核心机制 |

第一阶段从 LLVM IR → SPIR-V → MLIR 的演进逻辑出发，讲清 Linalg/Vector Dialect 如何协同完成渐进式递降（Progressive Lowering）；由 MLIR/SPIR-V 核心开发者 Lei Zhang 撰写。第二阶段则以编译器学科谱系为镜头，把 MLIR 当作编译器教材的一员来审视——它回答了哪些经典问题、用了什么新抽象、相比 LLVM/GCC/Sea of Nodes 做了哪些取舍。两阶段从"宏观图景"递进到"机制内幕"，与 Triton 编译器教程形成互补（Triton 编译器本身基于 MLIR 构建）。

---

## 阅读路线

### Inductor 编译器学习路线

#### 第一阶段：建立全局观

1. **[inductor_overview.md](Inductor%20Introduction/skeleton_tutorial/inductor_overview.md)** — 从文件结构角度建立全局地图，理解 `torch/_inductor/` 下各文件的职责与协作关系
2. **[phase1_global_view.md](Inductor%20Introduction/skeleton_tutorial/phase1_global_view.md)** — 从数据流角度理解编译管线：Dynamo → FX Graph → Lowering → Scheduling → Codegen
3. **运行 [data_flow_panorama.py](code/src/data_flow_panorama.py)** — 通过实际代码运行观察数据在各阶段的变化

#### 第二阶段：编译器设计视角（推荐）

以 *Engineering a Compiler* 为骨架的系统性 12 章教材，每章包含编译器理论与 Inductor 源码实现的映射分析。

**详细目录见 [compiler_view_tutorial/README.md](Inductor%20Introduction/compiler_view_tutorial/README.md)**

| 部分 | 章节 | 核心内容 |
|------|------|---------|
| 基础与全景 | [第 1 章](Inductor%20Introduction/compiler_view_tutorial/ch01_introduction.md) · [第 2 章](Inductor%20Introduction/compiler_view_tutorial/ch02_fx_graph.md) | 编译器设计导论、Dynamo 字节码追踪与 FX Graph |
| 前端 | [第 3 章](Inductor%20Introduction/compiler_view_tutorial/ch03_ir_design.md) | Inductor IR 设计 |
| 中间层 | [第 4 章](Inductor%20Introduction/compiler_view_tutorial/ch04_lowering.md) · [第 5 章](Inductor%20Introduction/compiler_view_tutorial/ch05_optimization.md) | Lowering、图优化 |
| 后端 | [第 6 章](Inductor%20Introduction/compiler_view_tutorial/ch06_dependency.md) · [第 7 章](Inductor%20Introduction/compiler_view_tutorial/ch07_fusion.md) · [第 8 章](Inductor%20Introduction/compiler_view_tutorial/ch08_codegen.md) · [第 9 章](Inductor%20Introduction/compiler_view_tutorial/ch09_memory.md) · [第 10 章](Inductor%20Introduction/compiler_view_tutorial/ch10_scheduling.md) | 依赖分析、融合、代码生成、内存管理、指令调度 |
| 集成与展望 | [第 11 章](Inductor%20Introduction/compiler_view_tutorial/ch11_e2e_pipeline.md) · [第 12 章](Inductor%20Introduction/compiler_view_tutorial/ch12_ecosystem.md) | 端到端流程回顾、PyTorch 生态协同 |

#### 第三阶段：核心类设计

从面向对象设计视角，以类继承树为核心线索逐一剖析关键类的设计动机、接口语义和协作方式。

**详细目录见 [key_classes_analysis/README.md](Inductor%20Introduction/key_classes_analysis/README.md)**

| 章节 | 核心内容 |
|------|---------|
| [第 1 章](Inductor%20Introduction/key_classes_analysis/chapter01_overview.md) | 全局架构概览 |
| [第 2 章](Inductor%20Introduction/key_classes_analysis/chapter02_virtualized.md) | Virtualized 基础设施 |
| [第 3 章](Inductor%20Introduction/key_classes_analysis/chapter03_handler_protocol.md) | Handler 协议 |
| [第 4 章](Inductor%20Introduction/key_classes_analysis/chapter04_ir_representation.md) | IR 表示体系 |
| [第 5 章](Inductor%20Introduction/key_classes_analysis/chapter05_lowering.md) | Lowering 相关类 |
| [第 6 章](Inductor%20Introduction/key_classes_analysis/chapter06_scheduling.md) | 调度相关类 |
| [第 7 章](Inductor%20Introduction/key_classes_analysis/chapter07_kernel_codegen.md) | Kernel 代码生成 |
| [第 8 章](Inductor%20Introduction/key_classes_analysis/chapter08_wrapper_codegen.md) | Wrapper 代码生成 |
| [第 9 章](Inductor%20Introduction/key_classes_analysis/chapter09_end_to_end.md) | 端到端流程串联 |

### operator 基础算子学习路线

读懂 Inductor 编译的对象。建议在通读 Inductor 教程后、深入各算子 lowering 前切入。

1. **[第 0 章 张量基底](operator/00-tensor-substrate.md)** — shape/dtype/stride/contiguity/广播/类型提升，建立全书「位置保持 vs 位置变化」分类轴
2. **[stride 的本质](operator/stride-essence.md)** — 一块一维内存如何假装成任意多维张量，讲透 stride「能做 / 不能做」的精确边界
3. **按轴阅读**：先 Part I（数学语义类，第 1–6 章）建立直觉，再进入 Part II（张量操作类，第 7–12 章，**全书重心**），最后 Part III（模型层原语，第 13–16 章）
4. **[附录 C Inductor 降级速查](operator/appendix-c-inductor-cheatsheet.md)** — 查每个算子走 pointwise / reduction / template / fallback 的哪条路（torch 2.7.1）

### Triton 编译器学习路线

Triton 是 PyTorch Inductor 的默认代码生成后端，也是理解 GPU 编译器全栈技术的绝佳入口。

**入门推荐**：先读 [High level Introduction of triton](Triton%20Introduction/High%20level%20Introduction%20of%20triton/)（3 章），快速建立 Triton kernel 全生命周期、编译器架构、Dialect/Pass Pipeline 的整体认知，再进入下方系统性的编译器理论教材。

**详细目录见 [triton_compiler_view_tutorial/README.md](Triton%20Introduction/triton_compiler_view_tutorial/README.md)**

| 部分 | 章节 | 核心内容 |
|------|------|---------|
| 基础与全景 | [第 1 章](Triton%20Introduction/triton_compiler_view_tutorial/ch01_introduction.md) | 编译器设计导论与 Triton 全景 |
| 前端 | [第 2 章](Triton%20Introduction/triton_compiler_view_tutorial/ch02_triton_dsl.md) · [第 3 章](Triton%20Introduction/triton_compiler_view_tutorial/ch03_mlir_ttir.md) | Triton DSL 设计、MLIR 与 TTIR |
| 中间层 | [第 4 章](Triton%20Introduction/triton_compiler_view_tutorial/ch04_ttgir_design.md) · [第 5 章](Triton%20Introduction/triton_compiler_view_tutorial/ch05_type_system.md) · [第 6 章](Triton%20Introduction/triton_compiler_view_tutorial/ch06_lowering_ttir_ttgir.md) · [第 7 章](Triton%20Introduction/triton_compiler_view_tutorial/ch07_loop_optimization.md) · [第 8 章](Triton%20Introduction/triton_compiler_view_tutorial/ch08_memory_optimization.md) | TTGIR 设计、类型系统、Lowering、循环优化、内存优化 |
| 后端 | [第 9 章](Triton%20Introduction/triton_compiler_view_tutorial/ch09_instruction_selection.md) · [第 10 章](Triton%20Introduction/triton_compiler_view_tutorial/ch10_pipeline_warp_spec.md) · [第 11 章](Triton%20Introduction/triton_compiler_view_tutorial/ch11_register_allocation.md) · [第 12 章](Triton%20Introduction/triton_compiler_view_tutorial/ch12_backend_code_emission.md) | 指令选择、流水线/Warp Spec、寄存器分配、PTX/CUBIN 发射 |
| 集成与展望 | [第 13 章](Triton%20Introduction/triton_compiler_view_tutorial/ch13_jit_cache.md) · [第 14 章](Triton%20Introduction/triton_compiler_view_tutorial/ch14_autotuning.md) · [第 15 章](Triton%20Introduction/triton_compiler_view_tutorial/ch15_end_to_end.md) | JIT 与缓存、Autotuning、端到端回顾 |

| 附录 | 内容 |
|------|------|
| [附录 A](Triton%20Introduction/triton_compiler_view_tutorial/appendix_a_eac_mapping.md) | EaC 章节完整映射 |
| [附录 B](Triton%20Introduction/triton_compiler_view_tutorial/appendix_b_mlir_primer.md) | MLIR 核心概念速查 |
| [附录 C](Triton%20Introduction/triton_compiler_view_tutorial/appendix_c_gpu_architecture.md) | GPU 体系结构速查 |
| [附录 D](Triton%20Introduction/triton_compiler_view_tutorial/appendix_d_tablegen_reference.md) | TableGen 定义索引 |
| [附录 E](Triton%20Introduction/triton_compiler_view_tutorial/appendix_e_glossary.md) | 术语表 |

### MLIR 学习路线

MLIR 学习分两阶段推进：**先建立全景，再深入机制**。

#### 第一阶段：MLIR high level overview（宏观图景）

源自 Lei.Chat 博客系列文章，由 MLIR/SPIR-V 核心开发者撰写，共 **4 章**，建立编译器/IR 演进史到 MLIR Dialect 体系再到核心变换管线的整体认知。

| 章节 | 核心内容 | 适合场景 |
|------|---------|---------|
| [第 1 章](MLIR/MLIR%20high%20level%20overview/ch1-%E7%BC%96%E8%AF%91%E5%99%A8%E4%B8%8E%E4%B8%AD%E9%97%B4%E8%A1%A8%E7%A4%BA%3A%20LLVM%20IR%2C%20SPIR-V%2C%20%E4%BB%A5%E5%8F%8A%20MLIR.md) | 编译器/IR 演进：LLVM IR → SPIR-V → MLIR | 建立历史视角，理解 MLIR 设计动机 |
| [第 2 章](MLIR/MLIR%20high%20level%20overview/ch2-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3%20MLIR%20Dialect.md) | MLIR Dialect 体系：Linalg/Vector/Tensor/Memref/SCF/CF | 理解沙漏型架构和 Dialect 协作关系 |
| [第 3 章](MLIR/MLIR%20high%20level%20overview/ch3-MLIR%20Vector%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) | Vector Dialect 深度：三级分层 + 完整变换管线 | 掌握向量层的渐进式递降 |
| [第 4 章](MLIR/MLIR%20high%20level%20overview/ch4-MLIR%20Linalg%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) | Linalg Dialect 深度：Tiling/Fusion/Distribution/Vectorization | 理解结构化代码生成入口 |

#### 第二阶段：compiler-view-of-MLIR（机制内幕）

以编译器学科为镜头逐章拆解 MLIR 的工作机制——IR 如何被表示、构造、重写、销毁，共 **Ch0–Ch13**。

**详细目录见 [compiler-view-of-MLIR/README.md](MLIR/compiler-view-of-MLIR/README.md)**

| 部分 | 章节 | 核心内容 |
|------|------|---------|
| 导论 | [Ch0](MLIR/compiler-view-of-MLIR/00-MLIR在编译器学科中的位置.md) · [Ch1](MLIR/compiler-view-of-MLIR/01-MLIRContext-对象的所有权与唯一化.md) | MLIR 在编译器学科中的位置、MLIRContext 的所有权与唯一化 |
| IR 逻辑结构 | [Ch2](MLIR/compiler-view-of-MLIR/02-Operation-一切皆操作及其部件语义.md) · [Ch3](MLIR/compiler-view-of-MLIR/03-Block-Region与控制流结构.md) | Operation 部件语义、Block/Region 与控制流结构 |
| 数据流边 | [Ch4](MLIR/compiler-view-of-MLIR/04-def-use-chain-从数据流分析到use-list实现.md) | def-use chain 从数据流分析到 use-list 实现 |
| IR 遍历 | [Ch5](MLIR/compiler-view-of-MLIR/05-遍历IR-walk-predecessor与变更契约.md) | walk、predecessor 与变更契约 |
| IR 构建 | [Ch6](MLIR/compiler-view-of-MLIR/06-从AST到IR-OpBuilder与两类挂接.md) · [Ch7](MLIR/compiler-view-of-MLIR/07-Operation-create的完整路径与尾分配.md) · [Ch8](MLIR/compiler-view-of-MLIR/08-use-def链如何自动形成.md) | OpBuilder、Operation::create 与尾分配、use-def 链自动形成 |
| IR 重写 | [Ch9](MLIR/compiler-view-of-MLIR/09-matchAndRewrite-图重写范式与全生命周期.md) · [Ch10](MLIR/compiler-view-of-MLIR/10-replaceOp-RAUW与erase-重写的原子动作.md) · [Ch11](MLIR/compiler-view-of-MLIR/11-驱动-工作表与监听者.md) | matchAndRewrite 图重写范式、RAUW/erase 原子动作、驱动/工作表/监听者 |
| 专题与综合 | [Ch12](MLIR/compiler-view-of-MLIR/12-专题-type的parse-print往返.md) · [Ch13](MLIR/compiler-view-of-MLIR/13-综合-BufferCastOpFold闭环.md) | type 的 parse/print 往返、BufferCastOpFold 端到端闭环 |

---

## 编译器理论教材映射

三大编译器教程（Inductor / Triton / MLIR）均以 *Engineering a Compiler* (3rd Edition) 为理论骨架，以下是关键映射。

> **operator 教程不在此列**：它不是以 EaC 为骨架的编译器教材，而是以「算子分类轴」（位置保持 vs 位置变化）组织的**算子功能手册**，定位为三大编译器教程的算子前置知识——EaC 阶段表里的任何一行，落到 Inductor 源码里的具体处理对象，都可在 [operator/](operator/) 找到对应算子的功能与降级路径讲解。

| EaC 章节 | 主题 | Inductor 对应模块 | Triton 对应模块 | MLIR 对应内容 |
|----------|------|------------------|----------------|-------------|
| Ch.1-3 | 编译器概览、词法/语法分析 | Dynamo 字节码追踪 | Triton DSL 前端 | 编译器/IR 演进史（第一阶段·第 1 章）；MLIR 在编译器学科中的位置（第二阶段·Ch0） |
| Ch.4 | 中间表示 (IR) | FX Graph IR、Inductor IR | TTIR + TTGIR (MLIR) | MLIR Dialect 体系（第一阶段·第 2 章）；Operation/Block/Region（第二阶段·Ch2-3） |
| Ch.5 | 语法导向翻译 | Lowering: FX Graph → Inductor IR | TTIR → TTGIR Lowering | Progressive Lowering / Dialect Conversion；从 AST 到 IR（第二阶段·Ch6） |
| Ch.8 | 优化简介 (CSE, DCE) | 图优化 passes | TTIR/TTGIR 优化 passes | Canonicalization patterns；matchAndRewrite 图重写（第二阶段·Ch9-11） |
| Ch.9 | 循环优化 (fusion, tiling) | Scheduler 融合决策 | Tiling, Peeling, Coalescing | Linalg Tiling/Fusion（第一阶段·第 4 章） |
| Ch.10 | 指令选择 | Codegen: IR → Triton/C++ | TTGIR → LLVM IR Conversion | Vector Lowering / Unrolling（第一阶段·第 3 章） |
| Ch.11 | 指令调度 | Scheduler 节点排序 | 软件流水线、Warp Specialization | Distribution（第一阶段·第 4 章） |
| Ch.12 | 寄存器分配 | Buffer 内存分配 | Shared Memory 分配、Alias 分析 | Bufferization / 寄存器资源分配 |
| Ch.13 | 后端编译总结 | Inductor 端到端回顾 | LLVM → PTX → CUBIN 发射 | LLVM/SPIR-V Dialect 导出；BufferCastOpFold 闭环（第二阶段·Ch13） |

---

## 运行环境

教学脚本依赖 PyTorch 开发环境（含 `torch.compile` 支持）和 Triton：

```bash
# 激活环境
source .venv/bin/activate

# 运行 Inductor 数据流全景演示
python "my pytorch tutorial/code/src/data_flow_panorama.py"

# 查看详细编译日志
TORCH_LOGS="+inductor" python "my pytorch tutorial/code/src/data_flow_panorama.py"

# 查看 Triton 编译中间产物
TRITON_DUMP_IR=1 python your_triton_kernel.py
```

`code/src/torch_compile_debug/` 目录下保存了多次运行的 `torch.compile` 调试输出（FX graph 可读版、转换版、Inductor IR、生成的 kernel 代码），可直接查阅作为学习参考。
