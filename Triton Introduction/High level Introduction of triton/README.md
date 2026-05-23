# Triton 高层导论

> 面向 Triton 初学者的高层全景介绍，从 kernel 全生命周期、编译器架构到 Dialect 与 Pass Pipeline，建立 Triton 编译器的整体认知。

## 定位

本系列与 [triton_compiler_view_tutorial](../triton_compiler_view_tutorial/) 互补：

| 维度 | High level Introduction of triton | triton_compiler_view_tutorial |
|------|-----------------------------------|-------------------------------|
| **视角** | 高层全景，工程实践导向 | 编译器理论，映射 EaC + MLIR + GPU 三本教材 |
| **深度** | 入门级，适合快速建立认知框架 | 系统性，适合深入掌握全栈设计 |
| **篇幅** | 3 章 + 1 附录 | 15 章 + 5 附录 |
| **适合** | 刚接触 Triton，希望了解"这是什么、怎么运作" | 希望系统性理解 Triton 编译器内部设计与实现 |

## 目录

### [第 1 章：Triton Kernel 全流程揭秘——从编译到运行](chapter1.%20Triton-Kernel-全流程揭秘-从编译到运行.md)

从 Python 源码到 GPU 执行，系统性拆解 Triton kernel 的全生命周期。涵盖 JIT 编译的五阶段管线（Python AST → TTIR → TTGIR → LLVM IR → PTX → CUBIN）、launch 阶段 program 到硬件线程的映射机制，以及编译期与运行时的协同关系。

→ [第 1 章附录：编译各阶段 IR 产物详解](appendix-chapter1.md)

### [第 2 章：Triton 编译器架构全景与多后端适配](chapter2.%20Triton-编译器架构全景与多后端适配.md)

介绍 Triton 编译器在整个大模型编译栈中的定位——介于图级优化与底层指令生成之间的中间层角色。剖析其基于 MLIR 的多层 IR 架构、多后端支持机制（NVIDIA CUDA / AMD ROCm / Ascend 昇腾），以及 Triton 如何通过渐进式 Lowering 实现算法表达与硬件能力的解耦。

### [第 3 章：Triton Compiler Core——Dialect 与 Pass Pipeline](chapter3.%20Triton-Compiler-Core-Dialect与Pass-Pipeline.md)

深入 Triton 编译器的核心——方言（Dialect）设计与 Pass Pipeline 架构。讲解 Triton 方言如何抽象 AI 算子的计算逻辑与数据流动，以及从前端解析到中间优化再到后端代码生成的完整 Pass 管线。末尾通过一个自定义方言与优化 Pass 的案例展现 Triton 的扩展活力。

## 推荐阅读顺序

```
第 1 章（kernel 全流程）
    │
    ├── 附录：各阶段 IR 产物速查
    │
    v
第 2 章（编译器架构全景）
    │
    v
第 3 章（Dialect 与 Pass Pipeline）
```

三章按"从现象到本质"组织：先看一个 kernel 从编译到运行的完整路径（第 1 章），建立感性认识；再展开编译器架构的全景地图（第 2 章）；最后深入最核心的 Dialect 与 Pass 设计（第 3 章）。

## 后续学习

完成本系列后，建议进入 [triton_compiler_view_tutorial](../triton_compiler_view_tutorial/) 进行系统性的编译器理论深度学习，或对照学习 [Inductor 教程](../../Inductor%20Introduction/) 理解 Inductor → Triton 的完整技术栈。
