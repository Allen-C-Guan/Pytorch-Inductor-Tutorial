# PyTorch Inductor 核心类剖析

> *Core Classes Analysis — A Deep Dive into Inductor's Class Hierarchy and Design*

## 关于本系列

本系列从**面向对象设计**的视角切入 PyTorch Inductor，以类继承树为核心线索，逐一剖析每个关键类的设计动机、接口语义和协作方式。与 [compiler_view_tutorial](../compiler_view_tutorial/) 的编译器理论视角不同，本系列更贴近源码，侧重于回答"这个类为什么这样设计"和"这些类如何协同工作"。

核心分析维度：

- **继承体系**：每个类在继承树中的位置与角色
- **设计模式**：Virtualized 动态作用域、Handler 解释器模式、策略模式等
- **数据流**：数据在类实例之间的流转路径
- **源码定位**：精确到文件和关键行号，方便对照阅读

## 目录

| 章节 | 文件 | 主题 | 核心类 |
|------|------|------|--------|
| 第 1 章 | [chapter01_overview.md](chapter01_overview.md) | 全局概览与类家族地图 | Inductor 三阶段架构全貌 |
| 第 2 章 | [chapter02_virtualized.md](chapter02_virtualized.md) | Virtualized 动态作用域引擎 | `Virtualized<T>`, `OpsValue`, `OpsWrapper` |
| 第 3 章 | [chapter03_handler_protocol.md](chapter03_handler_protocol.md) | Handler 协议体系 | `OpsHandler[T]`, `DefaultHandler`, `OpOverrides` |
| 第 4 章 | [chapter04_ir_representation.md](chapter04_ir_representation.md) | IR 表示层 | `IRNode`, `Buffer`, `TensorBox`, `Pointwise`, `Reduction` |
| 第 5 章 | [chapter05_lowering.md](chapter05_lowering.md) | Lowering 层 | `GraphLowering`, lowering 子系统 |
| 第 6 章 | [chapter06_scheduling.md](chapter06_scheduling.md) | 调度层 | `Scheduler`, `SchedulerNode`, `BaseScheduling` |
| 第 7 章 | [chapter07_kernel_codegen.md](chapter07_kernel_codegen.md) | 内核代码生成 | `Kernel`, `TritonKernel`, `CppKernel` |
| 第 8 章 | [chapter08_wrapper_codegen.md](chapter08_wrapper_codegen.md) | Wrapper 代码生成 | `PythonWrapperCodegen`, `CppWrapperCpu/Gpu` |
| 第 9 章 | [chapter09_end_to_end.md](chapter09_end_to_end.md) | 端到端综合追踪 | 全流程类交互实战 |
| 附录 | [appendix.md](appendix.md) | 完整类名索引与源码速查 | 全书涉及的所有核心类 |

## 阅读路线图

```
┌─────────────────────────────────────────────────┐
│  第 1 章：全局概览（建立整体坐标系）              │
└─────────────────┬───────────────────────────────┘
                  │
          ┌───────┴───────┐
          v               v
┌──────────────┐  ┌────────────────────────────────┐
│  第 2 章     │  │  第 3 章                        │
│  Virtualized │  │  Handler 协议                    │
│  (基础设施)  │  │  (抽象域框架)                    │
└──────┬───────┘  └───────────┬────────────────────┘
       │                      │
       v                      v
┌──────────────────────────────────────────────────┐
│  第 4 章：IR 表示层 (核心数据结构)                │
└─────────────────┬────────────────────────────────┘
                  │
                  v
┌──────────────────────────────────────────────────┐
│  第 5 章：Lowering (前端→中端翻译)               │
└─────────────────┬────────────────────────────────┘
                  │
                  v
┌──────────────────────────────────────────────────┐
│  第 6 章：调度层 (融合决策 + 执行规划)           │
└─────────┬────────────────┬───────────────────────┘
          │                │
          v                v
┌──────────────┐  ┌────────────────┐
│  第 7 章     │  │  第 8 章        │
│  Kernel Codegen │  Wrapper Codegen │
│  (内核生成)  │  │  (模块组装)     │
└──────┬───────┘  └───────┬────────┘
       │                  │
       v                  v
┌──────────────────────────────────────────────────┐
│  第 9 章：端到端综合追踪 (全流程串联)            │
└──────────────────────────────────────────────────┘
```

## 章节间依赖关系

- **第 2、3 章**（Virtualized + Handler）是后续所有章节的前置知识
- **第 4 章**（IR）依赖第 2、3 章的基础设施概念
- **第 5 章**（Lowering）依赖第 4 章的 IR 数据结构
- **第 6 章**（Scheduling）依赖第 4、5 章的 IR 和 Lowering 产出
- **第 7、8 章**（Codegen）依赖第 6 章的调度计划
- **第 9 章**（端到端）综合前 8 章的所有知识点
