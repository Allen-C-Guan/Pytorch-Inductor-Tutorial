# PyTorch（inductor/triton）编译器技术栈教学（持续补充中）

系统性的 PyTorch 编译器技术栈源码教学材料，覆盖 **Inductor 编译器**、**Triton 编译器** 与 **MLIR 编译器基础设施** 三大核心，面向准备从事 ML 编译器开发的工程师，以专家视角提供从全局架构到关键代码的完整学习路径。

## 教学目标

- **Inductor**：理解 PyTorch 2 编译管线的完整架构——Dynamo 追踪 → FX 图优化 → Lowering → 调度与融合 → 代码生成
- **Triton**：理解 Triton 编译器的全栈设计——从 DSL 前端（TTIR）到 GPU 后端（TTGIR → LLVM IR → PTX → CUBIN）
- **MLIR**：理解 MLIR 编译器基础设施的设计哲学——从 LLVM IR/SPIR-V 演进史到 Dialect 体系、渐进式递降（Progressive Lowering）和结构化代码生成
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
│   ├── compiler_view_tutorial/          # 编译器设计视角（12章 + 2附录）
│   ├── skeleton_tutorial/               # 管线阶段视角（5个 phase）
│   ├── key_classes_analysis/            # OOP/类设计视角（9章 + 附录）
│   ├── key_code_tutorial/               # 调试驱动的核心函数精讲
│   ├── my notes/                        # 个人阅读笔记
│   └── Engineering_a_Compiler/          # EaC 教材章节总结
│
├── Triton Introduction/                 # Triton 编译器教程
│   ├── README.md                        # Triton 教程导航
│   ├── High level Introduction of triton/  # 高层导论（3章 + 1附录）
│   └── triton_compiler_view_tutorial/   # 编译器设计视角（15章 + 5附录）
│
├── MLIR/                                # MLIR 编译器基础设施教程
│   ├── README.md                        # MLIR 教程导航
│   └── MLIR high level overview/        # 高层概述（4章，含 Linalg/Vector Dialect 深度解析）
│
└── code/
    └── src/
        ├── data_flow_panorama.py        # 数据流加工全景演示脚本
        ├── torch_compile_debug/         # torch.compile 调试输出
        └── phase1/                      # Phase 1 相关代码
```

---

## 三大教程体系

### Inductor 编译器教程 → [Inductor Introduction/](Inductor%20Introduction/)

三套并行教程从不同角度覆盖 Inductor 编译器：

| 教程 | 视角 | 篇幅 | 适合场景 |
|------|------|------|---------|
| **compiler_view_tutorial** | 编译器理论，映射 EaC 教材 | 12章 + 2附录 | 建立整体认知框架 |
| **skeleton_tutorial** | 编译管线阶段，12节标准化模板 | 5个 phase | 快速了解各阶段边界与数据流 |
| **key_classes_analysis** | OOP/类设计，类继承树与协作关系 | 9章 + 附录 | 深入源码理解"为什么这样设计" |

### Triton 编译器教程 → [Triton Introduction/](Triton%20Introduction/)

| 教程 | 视角 | 篇幅 | 适合场景 |
|------|------|------|---------|
| **High level Introduction of triton** | 工程实践，高层全景 | 3章 + 1附录 | 快速建立 Triton 编译器整体认知 |
| **triton_compiler_view_tutorial** | 编译器理论，映射 EaC + MLIR + GPU 教材 | 15章 + 5附录 | 系统性掌握 Triton 编译器全栈设计 |

### MLIR 编译器基础设施教程 → [MLIR/](MLIR/)

| 教程 | 视角 | 篇幅 | 适合场景 |
|------|------|------|---------|
| **MLIR high level overview** | 编译器/IR 演进史 + Dialect 体系 + 变换管线 | 4章 | 理解 MLIR 设计哲学与结构化代码生成核心机制 |

MLIR 教程填补了编译器基础设施层的理论空白——从 LLVM IR → SPIR-V → MLIR 的演进逻辑，到 Linalg/Vector Dialect 如何协同完成渐进式递降（Progressive Lowering）。由 MLIR/SPIR-V 核心开发者 Lei Zhang 撰写，源自一线编译器开发经验，与 Triton 编译器教程形成互补（Triton 编译器本身基于 MLIR 构建）。

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

MLIR 教程以 Lei.Chat 博客系列文章为核心，从编译器/IR 演进史到 MLIR Dialect 体系再到核心变换管线，逐层深入。

#### 入门推荐

**[MLIR high level overview](MLIR/MLIR%20high%20level%20overview/)（4 章）**

| 章节 | 核心内容 | 适合场景 |
|------|---------|---------|
| [第 1 章](MLIR/MLIR%20high%20level%20overview/ch1-%E7%BC%96%E8%AF%91%E5%99%A8%E4%B8%8E%E4%B8%AD%E9%97%B4%E8%A1%A8%E7%A4%BA%3A%20LLVM%20IR%2C%20SPIR-V%2C%20%E4%BB%A5%E5%8F%8A%20MLIR.md) | 编译器/IR 演进：LLVM IR → SPIR-V → MLIR | 建立历史视角，理解 MLIR 设计动机 |
| [第 2 章](MLIR/MLIR%20high%20level%20overview/ch2-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3%20MLIR%20Dialect.md) | MLIR Dialect 体系：Linalg/Vector/Tensor/Memref/SCF/CF | 理解沙漏型架构和 Dialect 协作关系 |
| [第 3 章](MLIR/MLIR%20high%20level%20overview/ch3-MLIR%20Vector%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) | Vector Dialect 深度：三级分层 + 完整变换管线 | 掌握向量层的渐进式递降 |
| [第 4 章](MLIR/MLIR%20high%20level%20overview/ch4-MLIR%20Linalg%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) | Linalg Dialect 深度：Tiling/Fusion/Distribution/Vectorization | 理解结构化代码生成入口 |

---

## 编译器理论教材映射

三个编译器教程均以 *Engineering a Compiler* (3rd Edition) 为理论骨架，以下是关键映射：

| EaC 章节 | 主题 | Inductor 对应模块 | Triton 对应模块 | MLIR 对应内容 |
|----------|------|------------------|----------------|-------------|
| Ch.1-3 | 编译器概览、词法/语法分析 | Dynamo 字节码追踪 | Triton DSL 前端 | 编译器/IR 演进史（第 1 章） |
| Ch.4 | 中间表示 (IR) | FX Graph IR、Inductor IR | TTIR + TTGIR (MLIR) | MLIR Dialect 体系（第 2 章） |
| Ch.5 | 语法导向翻译 | Lowering: FX Graph → Inductor IR | TTIR → TTGIR Lowering | Progressive Lowering / Dialect Conversion |
| Ch.8 | 优化简介 (CSE, DCE) | 图优化 passes | TTIR/TTGIR 优化 passes | Canonicalization patterns |
| Ch.9 | 循环优化 (fusion, tiling) | Scheduler 融合决策 | Tiling, Peeling, Coalescing | Linalg Tiling/Fusion（第 4 章） |
| Ch.10 | 指令选择 | Codegen: IR → Triton/C++ | TTGIR → LLVM IR Conversion | Vector Lowering / Unrolling（第 3 章） |
| Ch.11 | 指令调度 | Scheduler 节点排序 | 软件流水线、Warp Specialization | Distribution（第 4 章） |
| Ch.12 | 寄存器分配 | Buffer 内存分配 | Shared Memory 分配、Alias 分析 | Bufferization / 寄存器资源分配 |
| Ch.13 | 后端编译总结 | Inductor 端到端回顾 | LLVM → PTX → CUBIN 发射 | LLVM/SPIR-V Dialect 导出 |

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
