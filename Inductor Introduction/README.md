# Introduction —— PyTorch Inductor 入门学习材料集

> 本目录包含从不同维度学习 PyTorch Inductor 的系统性教程。其中 **compiler_view_tutorial 是核心教程**，另外两套教程从管线阶段与类设计角度提供互补视角。

## 内容导航

### [compiler_view_tutorial/](compiler_view_tutorial/) — 编译器设计视角【核心教程】

以 *Engineering a Compiler* 为骨架，将 Inductor 的各个模块映射到编译器的经典阶段。从编译器理论的宏观视角理解 Inductor 的架构设计，共 12 章 + 2 附录。

**适合**：希望建立 Inductor 整体认知框架，了解"每个模块在编译器中对应什么角色"。

### [key_classes_analysis/](key_classes_analysis/) — 核心类剖析

从面向对象设计的视角切入，以类继承树为核心线索，逐一剖析每个关键类的设计动机、接口语义和协作方式。共 9 章 + 附录，涵盖 Virtualized 基础设施、Handler 协议、IR 表示、Lowering、调度、代码生成等核心模块。

**适合**：希望深入源码，理解"这个类为什么这样设计"和"这些类如何协同工作"。

### [skeleton_tutorial/](skeleton_tutorial/) — 编译阶段骨架分析

按 Inductor 编译管线的主要阶段（常量折叠 → Lowering → 调度 → Kernel Codegen → Wrapper Codegen）逐步拆解，聚焦每个阶段的输入输出与核心逻辑。

**适合**：希望快速了解编译管线的各阶段边界和数据流转。

## 推荐阅读顺序

```
首次学习路线：

compiler_view_tutorial (Ch.1-2)     建立 Inductor 全局认知（核心入口）
        │
        v
key_classes_analysis (Ch.1-9)       深入源码与类设计
        │
        v
skeleton_tutorial                    对照编译管线各阶段
```

> 也可以根据当前需要直接跳转到对应模块——每个子目录都是自包含的。
