# PyTorch 编译器技术栈教学（持续补充中）

系统性的 PyTorch 编译器技术栈源码教学材料，覆盖 **Inductor 编译器** 与 **Triton 编译器** 两大核心，面向准备从事 ML 编译器开发的工程师，以专家视角提供从全局架构到关键代码的完整学习路径。

## 教学目标

- **Inductor**：理解 PyTorch 2 编译管线的完整架构——Dynamo 追踪 → FX 图优化 → Lowering → 调度与融合 → 代码生成
- **Triton**：理解 Triton 编译器的全栈设计——从 DSL 前端（TTIR）到 GPU 后端（TTGIR → LLVM IR → PTX → CUBIN）
- 掌握每个阶段的设计动机、编译器理论基础与源码实现细节

## 权威参考

- PyTorch 2 论文 (ASPLOS 2024): *"PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"*
- Triton 论文 (MAPS@PLDI 2019): *"Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"*
- MLIR 论文 (CGO 2021): *"MLIR: Scaling Compiler Infrastructure for Domain Specific Computation"*
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
│   └── triton_compiler_view_tutorial/   # 编译器设计视角（15章 + 5附录）
│
└── code/
    └── src/
        ├── data_flow_panorama.py        # 数据流加工全景演示脚本
        ├── torch_compile_debug/         # torch.compile 调试输出
        └── phase1/                      # Phase 1 相关代码
```

---

## 两大教程体系

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
| **triton_compiler_view_tutorial** | 编译器理论，映射 EaC + MLIR + GPU 教材 | 15章 + 5附录 | 系统性掌握 Triton 编译器全栈设计 |

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

---

## 编译器理论教材映射

两个编译器教程均以 *Engineering a Compiler* (3rd Edition) 为理论骨架，以下是关键映射：

| EaC 章节 | 主题 | Inductor 对应模块 | Triton 对应模块 |
|----------|------|------------------|----------------|
| Ch.1-3 | 编译器概览、词法/语法分析 | Dynamo 字节码追踪 | Triton DSL 前端 |
| Ch.4 | 中间表示 (IR) | FX Graph IR、Inductor IR | TTIR + TTGIR (MLIR) |
| Ch.5 | 语法导向翻译 | Lowering: FX Graph → Inductor IR | TTIR → TTGIR Lowering |
| Ch.8 | 优化简介 (CSE, DCE) | 图优化 passes | TTIR/TTGIR 优化 passes |
| Ch.9 | 循环优化 (fusion, tiling) | Scheduler 融合决策 | Tiling, Peeling, Coalescing |
| Ch.10 | 指令选择 | Codegen: IR → Triton/C++ | TTGIR → LLVM IR Conversion |
| Ch.11 | 指令调度 | Scheduler 节点排序 | 软件流水线、Warp Specialization |
| Ch.12 | 寄存器分配 | Buffer 内存分配 | Shared Memory 分配、Alias 分析 |
| Ch.13 | 后端编译总结 | Inductor 端到端回顾 | LLVM → PTX → CUBIN 发射 |

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
