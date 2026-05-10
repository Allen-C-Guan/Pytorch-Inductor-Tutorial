# PyTorch Inductor 源码教学（持续补充中）

系统性的 PyTorch Inductor 编译器源码教学材料，面向准备从事 Inductor 后端开发的程序员，以专家视角提供从全局架构到关键代码的完整学习路径。

## 教学目标

理解 PyTorch 2 编译管线的完整架构：Dynamo 追踪 → FX 图优化 → Lowering → 调度与融合 → 代码生成，掌握每个阶段的设计动机、编译器理论基础与源码实现细节。

## 权威参考

- PyTorch 2 论文 (ASPLOS 2024): *"PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"*
- TorchInductor 设计帖: [dev-discuss.pytorch.org](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- 编译器理论教材: *Engineering a Compiler* (Keith D. Cooper & Linda Torczon, 3rd Edition)

---

## 目录结构

```
my pytorch tutorial/
├── Introduction/
│   ├── compiler_view_tutorial/      # 编译器设计视角（12章 + 2附录）
│   ├── skeleton_tutorial/           # 管线阶段视角（5个 phase）
│   ├── key_classes_analysis/        # OOP/类设计视角（9章 + 附录）
│   ├── key_code_tutorial/           # 调试驱动的核心函数精讲
│   ├── my notes/                    # 个人阅读笔记
│   └── Engineering_a_Compiler/      # EaC 教材章节总结
├── code/
│   └── src/
│       ├── data_flow_panorama.py    # 数据流加工全景演示脚本
│       ├── torch_compile_debug/     # torch.compile 调试输出
│       └── phase1/                  # Phase 1 相关代码
└── README.md
```

三套并行教程从不同角度覆盖相同内容：

| 教程 | 视角 | 篇幅 | 适合场景 |
|------|------|------|---------|
| **compiler_view_tutorial** | 编译器理论，映射 EaC 教材 | 12章 + 2附录 | 建立整体认知框架 |
| **skeleton_tutorial** | 编译管线阶段，12节标准化模板 | 5个 phase | 快速了解各阶段边界与数据流 |
| **key_classes_analysis** | OOP/类设计，类继承树与协作关系 | 9章 + 附录 | 深入源码理解"为什么这样设计" |

---

## 阅读路线

### 第一阶段：建立全局观

1. **[inductor_overview.md](Introduction/skeleton_tutorial/inductor_overview.md)** — 从文件结构角度建立全局地图，理解 `torch/_inductor/` 下各文件的职责与协作关系
2. **[phase1_global_view.md](Introduction/skeleton_tutorial/phase1_global_view.md)** — 从数据流角度理解编译管线：Dynamo → FX Graph → Lowering → Scheduling → Codegen
3. **运行 [data_flow_panorama.py](code/src/data_flow_panorama.py)** — 通过实际代码运行观察数据在各阶段的变化

### 第二阶段：编译器设计视角（推荐）

以 *Engineering a Compiler* 为骨架的系统性 12 章教材，每章包含编译器理论与 Inductor 源码实现的映射分析。

**详细目录见 [compiler_view_tutorial/README.md](Introduction/compiler_view_tutorial/README.md)**

| 部分 | 章节 | 核心内容 |
|------|------|---------|
| 基础与全景 | [第 1 章](Introduction/compiler_view_tutorial/ch01_introduction.md) · [第 2 章](Introduction/compiler_view_tutorial/ch02_fx_graph.md) | 编译器设计导论、Dynamo 字节码追踪与 FX Graph |
| 前端 | [第 3 章](Introduction/compiler_view_tutorial/ch03_ir_design.md) | Inductor IR 设计 |
| 中间层 | [第 4 章](Introduction/compiler_view_tutorial/ch04_lowering.md) · [第 5 章](Introduction/compiler_view_tutorial/ch05_optimization.md) | Lowering、图优化 |
| 后端 | [第 6 章](Introduction/compiler_view_tutorial/ch06_dependency.md) · [第 7 章](Introduction/compiler_view_tutorial/ch07_fusion.md) · [第 8 章](Introduction/compiler_view_tutorial/ch08_codegen.md) · [第 9 章](Introduction/compiler_view_tutorial/ch09_memory.md) · [第 10 章](Introduction/compiler_view_tutorial/ch10_scheduling.md) | 依赖分析、融合、代码生成、内存管理、指令调度 |
| 集成与展望 | [第 11 章](Introduction/compiler_view_tutorial/ch11_e2e_pipeline.md) · [第 12 章](Introduction/compiler_view_tutorial/ch12_ecosystem.md) | 端到端流程回顾、PyTorch 生态协同 |

### 第三阶段：核心类设计

从面向对象设计视角，以类继承树为核心线索逐一剖析关键类的设计动机、接口语义和协作方式。

| 章节 | 核心内容 |
|------|---------|
| [第 1 章](Introduction/key_classes_analysis/chapter01_overview.md) | 全局架构概览 |
| [第 2 章](Introduction/key_classes_analysis/chapter02_virtualized.md) | Virtualized 基础设施 |
| [第 3 章](Introduction/key_classes_analysis/chapter03_handler_protocol.md) | Handler 协议 |
| [第 4 章](Introduction/key_classes_analysis/chapter04_ir_representation.md) | IR 表示体系 |
| [第 5 章](Introduction/key_classes_analysis/chapter05_lowering.md) | Lowering 相关类 |
| [第 6 章](Introduction/key_classes_analysis/chapter06_scheduling.md) | 调度相关类 |
| [第 7 章](Introduction/key_classes_analysis/chapter07_kernel_codegen.md) | Kernel 代码生成 |
| [第 8 章](Introduction/key_classes_analysis/chapter08_wrapper_codegen.md) | Wrapper 代码生成 |
| [第 9 章](Introduction/key_classes_analysis/chapter09_end_to_end.md) | 端到端流程串联 |

### 第四阶段：骨架教程——逐阶段学习

按编译管线的执行顺序，每个文档包含 12 个标准化章节。

| 文档 | 核心内容 | 编译器理论重点 |
|------|---------|--------------|
| [phase2_fx_optimization.md](Introduction/skeleton_tutorial/phase2_fx_optimization.md) | FX 图优化：模式匹配、算子分解、CSE | 图优化、算子融合、指令选择 |
| [phase3_lowering.md](Introduction/skeleton_tutorial/phase3_lowering.md) | Lowering：FX Graph → Inductor IR | IR 转换、SSA、Define-by-Run |
| [phase4_scheduling_fusion.md](Introduction/skeleton_tutorial/phase4_scheduling_fusion.md) | 调度与融合：依赖分析、算子融合 | 指令调度、循环融合、Tiling |
| [phase5_codegen.md](Introduction/skeleton_tutorial/phase5_codegen.md) | 代码生成：Triton/C++ kernel 生成 | 代码生成、向量化、循环展开 |

### 第五阶段：关键代码精讲

- **[compute_dependencies_debug_guide.md](Introduction/key_code_tutorial/compute_dependencies_debug_guide.md)** — `Scheduler.compute_dependencies()`，依赖 DAG 的构建过程
- **[scheduler_init_debug_guide.md](Introduction/key_code_tutorial/scheduler_init_debug_guide.md)** — `Scheduler._init`，从 IR 节点列表到优化调度图的完整流程

---

## 编译器理论教材映射

每个教学文档中的"编译器背景知识"章节均参考 *Engineering a Compiler* (3rd Edition)：

| Inductor 阶段 | 编译器理论 | 教材章节 |
|---------------|-----------|---------|
| FX 优化 | 图优化、CSE、DCE | Ch.8 Introduction to Optimization |
| FX 优化 | 算子分解、模式匹配 | Ch.10 Instruction Selection |
| Lowering | IR 设计（SSA、CFG） | Ch.4 Intermediate Representations |
| Lowering | 指令选择、类型推导 | Ch.4, Ch.5, Ch.10 |
| 调度融合 | 循环融合、Tiling | Ch.9 Loop Optimizations |
| 调度融合 | 依赖分析、指令调度 | Ch.9, Ch.11 Instruction Scheduling |
| 代码生成 | 代码生成、向量化、循环展开 | Ch.9, Ch.10, Ch.12 |

---

## 运行环境

教学脚本依赖 PyTorch 开发环境（含 `torch.compile` 支持）：

```bash
# 运行数据流全景演示
python "my pytorch tutorial/code/src/data_flow_panorama.py"

# 查看详细编译日志
TORCH_LOGS="+inductor" python "my pytorch tutorial/code/src/data_flow_panorama.py"
```

`code/src/torch_compile_debug/` 目录下保存了多次运行的 `torch.compile` 调试输出（FX graph 可读版、转换版、Inductor IR、生成的 kernel 代码），可直接查阅作为学习参考。
