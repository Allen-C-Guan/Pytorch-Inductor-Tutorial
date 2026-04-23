# PyTorch Inductor：从编译器设计视角的系统性分析

> *An Architect's Guide to PyTorch Inductor — A Compiler Design Perspective*

## 关于本书

本书面向有一定编程基础但编译器知识有限的计算机本科生，从编译器设计的经典视角系统性分析 PyTorch Inductor 的架构与实现。全书以 *Engineering a Compiler* (Keith D. Cooper & Linda Torczon) 的编译器设计流程为骨架，将 Inductor 的各个模块映射到编译器的经典阶段。

## 章节映射总表

| Engineering a Compiler 章节 | 主题 | Inductor 对应模块 | 本书章节 |
|---|---|---|---|
| Ch.1-3 | 编译器概览、词法分析、语法分析 | `torch._dynamo` 字节码追踪与 FX Graph 构建 | 第 1-2 章 |
| Ch.4 | 中间表示 (IR) | FX Graph IR、Inductor IR (ir.py)、SchedulerNode | 第 3 章 |
| Ch.5 | 语法导向翻译 | Lowering: FX Graph → Inductor IR | 第 4 章 |
| Ch.6 | 编译器前端综述 | `torch._dynamo` 的 Python 字节码分析 | 第 2 章 |
| Ch.7 | 后端综述 | Inductor 后端架构总览 | 第 1 章 |
| Ch.8 | 优化简介 (CSE, DCE, 常量传播, 值编号) | Inductor 图优化 passes | 第 5 章 |
| Ch.9 | 循环优化 (fusion, tiling, unrolling, vectorization) | Scheduler 融合决策、tiling 策略、vectorization | 第 7 章 |
| Ch.10 | 指令选择 | Codegen: IR → Triton/C++ kernel 代码生成 | 第 8 章 |
| Ch.11 | 指令调度 | Scheduler 节点排序、依赖分析、关键路径调度 | 第 6、10 章 |
| Ch.12 | 寄存器分配 | Buffer 内存分配、内存复用、memory donation | 第 9 章 |
| Ch.13 | 后端编译总结 | Inductor 编译 pipeline 端到端回顾 | 第 11 章 |

## 目录

### 第一部分：基础与全景

- [第 1 章：编译器设计导论与 Inductor 全景](ch01_introduction.md)
- [第 2 章：Python 字节码追踪与 FX Graph 构建](ch02_fx_graph.md)

### 第二部分：前端——图构建与中间表示

- [第 3 章：Inductor 中间表示设计](ch03_ir_design.md)

### 第三部分：中间层——翻译与优化

- [第 4 章：Lowering——从 FX Graph 到 Inductor IR](ch04_lowering.md)
- [第 5 章：图优化](ch05_optimization.md)

### 第四部分：后端——调度与代码生成

- [第 6 章：依赖分析与调度前置](ch06_dependency.md)
- [第 7 章：融合策略与循环优化](ch07_fusion.md)
- [第 8 章：指令选择与代码生成](ch08_codegen.md)
- [第 9 章：内存管理与缓冲区分配](ch09_memory.md)
- [第 10 章：指令调度](ch10_scheduling.md)

### 第五部分：集成与展望

- [第 11 章：端到端编译流程回顾](ch11_e2e_pipeline.md)
- [第 12 章：与 PyTorch 生态的协同设计](ch12_ecosystem.md)

### 附录

- [附录 A：Engineering a Compiler 章节完整映射](appendix_a_eac_mapping.md)
- [附录 B：术语表](appendix_b_glossary.md)

## 阅读路线图

```
┌──────────────────────────────────────────────┐
│        第 1 章：编译器设计导论与全景          │
│        (了解全貌，建立整体认知)               │
└──────────────┬───────────────────────────────┘
               │
       ┌───────┴───────┐
       v               v
┌──────────────┐ ┌──────────────────────────────┐
│  第 2 章     │ │  第 3 章                      │
│  FX Graph    │ │  Inductor IR 设计             │
│  (前端捕获)  │ │  (中间表示，编译器核心数据结构) │
└──────┬───────┘ └──────────────┬───────────────┘
       │                        │
       v                        v
┌──────────────────────────────────────────────┐
│  第 4 章：Lowering (前端→后端的翻译桥梁)      │
└──────────────┬───────────────────────────────┘
               │
       ┌───────┴───────┐
       v               v
┌──────────────┐ ┌──────────────────────────────┐
│  第 5 章     │ │  第 6 章                      │
│  图优化      │ │  依赖分析与调度前置            │
│  (中层优化)  │ │  (后端准备)                    │
└──────┬───────┘ └──────────────┬───────────────┘
       │                        │
       v                        v
┌──────────────────────────────────────────────┐
│  第 7 章：融合策略与循环优化                   │
│  (后端核心优化)                               │
└──────────────┬───────────────────────────────┘
               │
       ┌───────┴───────┐
       v               v
┌──────────────┐ ┌──────────────────────────────┐
│  第 8 章     │ │  第 9 章                      │
│  代码生成    │ │  内存管理                      │
│  (Triton/C++)│ │  (缓冲区分配)                  │
└──────┬───────┘ └──────────────┬───────────────┘
       │                        │
       v                        v
┌──────────────────────────────────────────────┐
│  第 10 章：指令调度 (节点执行顺序)            │
└──────────────┬───────────────────────────────┘
               │
               v
┌──────────────────────────────────────────────┐
│  第 11 章：端到端流程回顾                     │
│  第 12 章：PyTorch 生态协同设计               │
└──────────────────────────────────────────────┘
```

## 先修知识

- Python 编程基础
- PyTorch 基本使用（张量操作、模型定义）
- 基本的数据结构（图、树、链表）
- 线性代数基础（矩阵运算）

## 参考资料

1. Keith D. Cooper & Linda Torczon, *Engineering a Compiler*, 3rd Edition
2. PyTorch 2.0 论文: TorchInductor: A PyTorch Native Compiler (MLSys 2023)
3. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations (MAPL 2022)
4. PyTorch 官方文档: https://pytorch.org/docs/stable/torch.compiler.html
