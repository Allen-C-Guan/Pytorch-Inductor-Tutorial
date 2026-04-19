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
│   ├── skeleton_tutorial/          # 骨架教程：按编译管线阶段组织的系统性教学文档
│   │   ├── inductor_overview.md    #   全景地图：Inductor 文件结构与功能指南
│   │   ├── phase1_global_view.md   #   Phase 1：全局观 — 编译管线数据流全景
│   │   ├── phase2_fx_optimization.md  # Phase 2：FX 图优化
│   │   ├── phase3_lowering.md      #   Phase 3：Lowering（FX Graph → Inductor IR）
│   │   ├── phase4_scheduling_fusion.md # Phase 4：调度与融合
│   │   └── phase5_codegen.md       #   Phase 5：代码生成
│   ├── key_code_tutorial/          # 关键代码精讲：以调试视角逐行剖析核心函数
│   │   ├── compute_dependencies_debug_guide.md  # Scheduler.compute_dependencies()
│   │   └── scheduler_init_debug_guide.md         # Scheduler._init
│   └── my notes/                   # 个人阅读笔记
│       └── phase1代码阅读笔记.md
├── code/
│   └── src/
│       ├── data_flow_panorama.py   # 教学用例：数据流加工全景演示脚本
│       └── torch_compile_debug/    # torch.compile 调试输出（FX graph、IR、生成代码等）
└── README.md
```

---

## 阅读路线

### 第一阶段：建立全局观

1. **[inductor_overview.md](Introduction/skeleton_tutorial/inductor_overview.md)** — 从文件结构角度建立全局地图，理解 `torch/_inductor/` 下各文件的职责与协作关系
2. **[phase1_global_view.md](Introduction/skeleton_tutorial/phase1_global_view.md)** — 从数据流角度理解编译管线：Dynamo → FX Graph → Lowering → Scheduling → Codegen，每个阶段的输入/输出是什么
3. **运行 [data_flow_panorama.py](code/src/data_flow_panorama.py)** — 通过实际代码运行观察数据在各阶段的变化

### 第二阶段：逐阶段深入学习

按编译管线的执行顺序，每个文档包含 12 个标准化章节：设计哲学、编译器背景知识、核心调用栈、主体流程、UML 架构、关键思想代码、关键源码、核心技术、Debug 路线、数据流加工、交叉校验。

| 文档 | 核心内容 | 编译器理论重点 |
|------|---------|--------------|
| [phase2_fx_optimization.md](Introduction/skeleton_tutorial/phase2_fx_optimization.md) | FX 图优化：模式匹配、算子分解、CSE | 图优化、算子融合、指令选择 |
| [phase3_lowering.md](Introduction/skeleton_tutorial/phase3_lowering.md) | Lowering：FX Graph → Inductor IR | IR 转换、SSA、Define-by-Run、指令选择 |
| [phase4_scheduling_fusion.md](Introduction/skeleton_tutorial/phase4_scheduling_fusion.md) | 调度与融合：依赖分析、算子融合 | 指令调度、循环融合、Tiling、依赖分析 |
| [phase5_codegen.md](Introduction/skeleton_tutorial/phase5_codegen.md) | 代码生成：Triton/C++ kernel 生成 | 代码生成、向量化、循环展开、寄存器分配 |

### 第三阶段：关键代码精讲

- **[compute_dependencies_debug_guide.md](Introduction/key_code_tutorial/compute_dependencies_debug_guide.md)** — 深入 `Scheduler.compute_dependencies()`，理解依赖 DAG 的构建过程
- **[scheduler_init_debug_guide.md](Introduction/key_code_tutorial/scheduler_init_debug_guide.md)** — 深入 `Scheduler._init`，理解从 IR 节点列表到优化调度图的完整流程

---

## 编译器理论教材映射

每个教学文档中的"编译器背景知识"章节均参考 *Engineering a Compiler* (3rd Edition)，按以下映射关系组织：

| Inductor 阶段 | 编译器理论 | 教材章节 |
|---------------|-----------|---------|
| FX 优化 | 图优化、CSE、DCE | Ch.8 Introduction to Optimization |
| FX 优化 | 算子分解、模式匹配 | Ch.10 Instruction Selection |
| Lowering | IR 设计（SSA、CFG） | Ch.4 Intermediate Representations |
| Lowering | 指令选择、类型推导 | Ch.4, Ch.5, Ch.10 |
| 调度融合 | 循环融合、Tiling | Ch.9 Loop Optimizations |
| 调度融合 | 依赖分析、指令调度 | Ch.9, Ch.11 Instruction Scheduling |
| 调度融合 | 寄存器压力 | Ch.11, Ch.12 Register Allocation |
| 代码生成 | 代码生成、向量化、循环展开 | Ch.9, Ch.10, Ch.12 |

---

## 运行环境

教学脚本依赖 PyTorch 开发环境（含 `torch.compile` 支持）。运行示例：

```bash
# 运行数据流全景演示
python "my pytorch tutorial/code/src/data_flow_panorama.py"

# 查看详细编译日志
TORCH_LOGS="+inductor" python "my pytorch tutorial/code/src/data_flow_panorama.py"
```

`code/src/torch_compile_debug/` 目录下保存了多次运行的 `torch.compile` 调试输出（FX graph 可读版、转换版、Inductor IR、生成的 kernel 代码），可直接查阅作为学习参考。
