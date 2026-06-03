# MLIR Introduction——MLIR 编译器基础设施学习材料

> 本目录包含 MLIR 编译器基础设施的系统性学习资料，涵盖从编译器/中间表示演进历史到 MLIR 核心 Dialect（Linalg、Vector）的深度解析。

## 内容导航

### [MLIR high level overview/](MLIR%20high%20level%20overview/)——MLIR 高层概述

源自 Lei.Chat 博客的系列文章，由 MLIR/SPIR-V 核心开发者撰写，从编译器与中间表示的演进历史出发，逐步深入到 MLIR 代码生成相关的核心 Dialect 体系和变换流程，共 **4 章**。

**适合**：希望理解编译器/IR 演进逻辑、MLIR 设计哲学，以及 Linalg/Vector Dialect 如何协同完成结构化代码生成。

## 系列文章概览

| 章节 | 标题 | 核心内容 |
|------|------|---------|
| **第 1 章** | [编译器与中间表示: LLVM IR, SPIR-V, 以及 MLIR](MLIR%20high%20level%20overview/ch1-%E7%BC%96%E8%AF%91%E5%99%A8%E4%B8%8E%E4%B8%AD%E9%97%B4%E8%A1%A8%E7%A4%BA%3A%20LLVM%20IR%2C%20SPIR-V%2C%20%E4%BB%A5%E5%8F%8A%20MLIR.md) | 编译器与 IR 演进史：抽象与语义、正确与优化、LLVM IR 的解绑与模块化、SPIR-V 的标准化与兼容性、MLIR 的基础设施化与进一步解耦 |
| **第 2 章** | [机器学习编译器代码生成相关 MLIR Dialect](MLIR%20high%20level%20overview/ch2-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3%20MLIR%20Dialect.md) | MLIR Dialect 体系全景：内嵌结构操作、类型系统、Dialect 作为建模粒度、Linalg/Vector/Tensor/Memref/SCF/CF 等核心 Dialect 的层次关系与协作方式 |
| **第 3 章** | [MLIR Vector Dialect 以及 Patterns](MLIR%20high%20level%20overview/ch3-MLIR%20Vector%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) | Vector Dialect 深度剖析：三级操作分层（硬件无关/硬件相关/基础操作）、向量化→展开→Hoisting→递降的完整变换管线，含 matmul/conv 实例演练 |
| **第 4 章** | [MLIR Linalg Dialect 以及 Patterns](MLIR%20high%20level%20overview/ch4-MLIR%20Linalg%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) | Linalg Dialect 设计详解：结构化 Op 的统一 IR 结构、分块(Tiling)→融合(Fusion)→分配(Distribution)→向量化(Vectorization)的变换体系 |

### 教程特色

| 维度 | 说明 |
|------|------|
| **一手经验** | 作者为 MLIR/SPIR-V 核心开发者（Lei Zhang），内容源自实际编译器开发的一线经验 |
| **知其所以然** | 每篇文章追问设计折中——为什么这样设计？相比其他方案的优势和代价是什么？ |
| **实例驱动** | 第 3 章以 matmul 和 convolution 为例，完整展示从 mhlo dialect → linalg → vector → 最终形态的变换全过程 |
| **图表辅助** | 配有 Dialect 层次关系图和代码生成流程图（SVG 格式），直观展示各 Dialect 的定位与转换关系 |

## 推荐阅读顺序

```
首次学习路线：

第 1 章                           建立编译器/IR 演进的全局认知
    │                             LLVM IR → SPIR-V → MLIR，理解"为什么需要 MLIR"
    │
    v
第 2 章                           掌握 MLIR Dialect 体系全景
    │                             理解各 Dialect 的层次关系与沙漏型编译器架构
    │
    v
第 4 章（Linalg）                  理解结构化代码生成的入口抽象
    │                             分块、融合、分配、向量化四大变换
    │
    v
第 3 章（Vector）                  深入向量层的渐进式递降
                                 向量化→展开→清理→Hoisting→递降的完整 pipeline
```

> 每章是自包含的，但第 3-4 章依赖第 2 章建立的 Dialect 体系概念，第 1 章提供了理解 MLIR 设计动机的历史背景。

## 快速跳转

| 想了解... | 直接读 |
|-----------|--------|
| 编译器/IR 为什么从 LLVM 演进到 MLIR | [第 1 章](MLIR%20high%20level%20overview/ch1-%E7%BC%96%E8%AF%91%E5%99%A8%E4%B8%8E%E4%B8%AD%E9%97%B4%E8%A1%A8%E7%A4%BA%3A%20LLVM%20IR%2C%20SPIR-V%2C%20%E4%BB%A5%E5%8F%8A%20MLIR.md) |
| MLIR 的 Dialect 体系架构 | [第 2 章](MLIR%20high%20level%20overview/ch2-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3%20MLIR%20Dialect.md) |
| Vector Dialect 三级操作分层 | [第 3 章](MLIR%20high%20level%20overview/ch3-MLIR%20Vector%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) |
| Vector 完整变换管线 (matmul 实例) | [第 3 章 §变换](MLIR%20high%20level%20overview/ch3-MLIR%20Vector%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) |
| Linalg Op 的统一 IR 结构 | [第 4 章 §设计考虑](MLIR%20high%20level%20overview/ch4-MLIR%20Linalg%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) |
| Linalg 分块(Tiling)详解 | [第 4 章 §分块](MLIR%20high%20level%20overview/ch4-MLIR%20Linalg%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) |
| Linalg 融合(Fusion)机制 | [第 4 章 §融合](MLIR%20high%20level%20overview/ch4-MLIR%20Linalg%20Dialect%20%E4%BB%A5%E5%8F%8A%20Patterns.md) |
| 什么是渐进式递降(Progressive Lowering) | [第 1 章](MLIR%20high%20level%20overview/ch1-%E7%BC%96%E8%AF%91%E5%99%A8%E4%B8%8E%E4%B8%AD%E9%97%B4%E8%A1%A8%E7%A4%BA%3A%20LLVM%20IR%2C%20SPIR-V%2C%20%E4%BB%A5%E5%8F%8A%20MLIR.md) + [第 2 章](MLIR%20high%20level%20overview/ch2-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%BC%96%E8%AF%91%E5%99%A8%E4%BB%A3%E7%A0%81%E7%94%9F%E6%88%90%E7%9B%B8%E5%85%B3%20MLIR%20Dialect.md) |

## 与 Inductor/Triton 教程的关系

MLIR 是理解现代编译器基础设施的基石，与 Inductor 和 Triton 教程形成互补：

- **Triton 编译器基于 MLIR 构建**：Triton 的 TTIR 和 TTGIR 均以 MLIR Dialect 形式实现。先学 MLIR 再学 Triton，可以更深入理解 Triton 的 IR 设计、Pass Pipeline 和 Dialect Conversion 机制。
- **Inductor → Triton 的桥梁**：Inductor 生成 Triton 代码，Triton 内部通过 MLIR 基础设施编译为 GPU 可执行文件。MLIR 教程填补了"Triton 编译器如何基于 MLIR 工作"的理论空白。
- **建议学习顺序**：MLIR 基础（本文）→ Triton 编译器教程 → Inductor 编译器教程，形成从底层基础设施到上层编译栈的完整知识链。

Triton 教程入口：[../Triton%20Introduction/](../Triton%20Introduction/)
Inductor 教程入口：[../Inductor%20Introduction/](../Inductor%20Introduction/)

---

> 本系列文章版权归原作者 Lei Zhang 所有，遵循 CC 4.0 BY-SA 版权协议。
> 原文链接见各章节末尾。
