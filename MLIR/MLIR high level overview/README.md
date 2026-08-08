# MLIR 高层概述

> 返回：[← MLIR 教程导航](../README.md)
>
> 由 MLIR/SPIR-V 核心开发者 **Lei Zhang** 撰写的系列博客，从编译器与中间表示的演进历史出发，逐步深入到 MLIR 代码生成相关的核心 Dialect 体系和变换流程。

## 目录

| 章节 | 标题 | 核心内容 |
|------|------|---------|
| [第 1 章](ch1-编译器与中间表示:%20LLVM%20IR,%20SPIR-V,%20以及%20MLIR.md) | 编译器与 IR 演进 | LLVM IR → SPIR-V → MLIR 的历史演进，理解 MLIR 的设计动机 |
| [第 2 章](ch2-机器学习编译器代码生成相关%20MLIR%20Dialect.md) | MLIR Dialect 体系 | Linalg/Vector/Tensor/Memref/SCF/CF 等核心 Dialect，理解沙漏型架构 |
| [第 4 章](ch4-MLIR%20Linalg%20Dialect%20以及%20Patterns.md) | Linalg Dialect 深度 | Tiling/Fusion/Distribution/Vectorization，结构化代码生成入口 |
| [第 3 章](ch3-MLIR%20Vector%20Dialect%20以及%20Patterns.md) | Vector Dialect 深度 | 三级操作分层 + 完整变换管线 |
| [附录](ch3-appendix-IR语法详解.md) | IR 语法详解 | 配合第 3 章的 IR 语法参考 |

## 推荐阅读顺序

第 1 章 → 第 2 章 → 第 4 章（Linalg）→ 第 3 章（Vector）。每章自包含，但第 3-4 章依赖第 2 章建立的 Dialect 体系概念。

## 教程特色

- **一手经验**：作者为 MLIR/SPIR-V 核心开发者，内容源自一线工程实践
- **知其所以然**：不只讲"有什么"，更追问为什么这样设计、相比其他方案做了什么取舍
- **实例驱动**：以 matmul 和 convolution 的完整变换过程为线索，贯穿 Dialect 和 Pass 的讲解

---
*系列文章版权归原作者 Lei Zhang 所有，遵循 CC 4.0 BY-SA 版权协议。*
