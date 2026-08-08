# Triton Introduction——Triton 编译器学习材料

> 返回：[← PyTorch 编译器技术栈总览](../README.md)
>
> 本目录包含从编译器设计视角系统学习 Triton 编译器的完整教程。

## 内容导航

### ⭐ [High level Introduction of triton/](High%20level%20Introduction%20of%20triton/)——高层导论【核心推荐】

以工程实践视角对 Triton 编译器进行全景介绍，是 Triton 入门的最佳起点，共 **3 章 + 1 附录**。

| 章节 | 标题 | 核心内容 |
|------|------|---------|
| 第 1 章 | [Triton Kernel 全流程揭秘](High%20level%20Introduction%20of%20triton/chapter1.%20Triton-Kernel-%E5%85%A8%E6%B5%81%E7%A8%8B%E6%8F%AD%E7%A7%98-%E4%BB%8E%E7%BC%96%E8%AF%91%E5%88%B0%E8%BF%90%E8%A1%8C.md) | kernel 全生命周期：从 Python DSL 到 GPU 可执行文件 |
| 第 2 章 | [Triton 编译器架构全景与多后端适配](High%20level%20Introduction%20of%20triton/chapter2.%20Triton-%E7%BC%96%E8%AF%91%E5%99%A8%E6%9E%B6%E6%9E%84%E5%85%A8%E6%99%AF%E4%B8%8E%E5%A4%9A%E5%90%8E%E7%AB%AF%E9%80%82%E9%85%8D.md) | 编译器架构、多后端（NVIDIA/AMD/Ascend）适配 |
| 第 3 章 | [Triton Compiler Core Dialect 与 Pass Pipeline](High%20level%20Introduction%20of%20triton/chapter3.%20Triton-Compiler-Core-Dialect%E4%B8%8EPass-Pipeline.md) | Dialect 体系与 Pass Pipeline 设计 |

**适合**：刚接触 Triton，希望快速建立对 Triton 编译器"是什么、怎么运作"的整体认知。

### [triton_compiler_view_tutorial/](triton_compiler_view_tutorial/)——编译器设计视角（深度学习）

以 *Engineering a Compiler* + *MLIR Tutorial* + *Programming Massively Parallel Processors* 三本教材为理论骨架，系统性拆解 Triton 编译器的全栈设计，共 **15 章 + 5 附录**。

**适合**：完成 High level Introduction 后，希望系统性深入 TTIR/TTGIR 两级 IR、Layout 抽象、软件流水线等核心设计。详细目录见 [triton_compiler_view_tutorial/README.md](triton_compiler_view_tutorial/README.md)。

### 教程特色

**High level Introduction of triton（核心推荐）：**

| 维度 | 说明 |
|------|------|
| **工程实践视角** | 以 kernel 全生命周期为线索，从现象到本质，快速建立"Triton 编译器怎么运作"的完整认知 |
| **多后端覆盖** | 覆盖 NVIDIA (CUDA)、AMD (ROCm)、Ascend (昇腾) 三套后端的架构差异与适配策略 |
| **快速全景** | 3 章即可掌握 Triton 编译器全貌，是入门的最佳起点 |

**triton_compiler_view_tutorial（深度学习）：**

| 维度 | 说明 |
|------|------|
| **编译器视角** | 按编译器 pipeline 组织：DSL 前端 → TTIR → TTGIR 优化 → LLVM IR 代码生成 → PTX/CUBIN 发射 |
| **What-How-Why** | 每章追问三层：功能是什么、如何实现、为什么这样设计 |
| **源码为本** | 所有 OP/Type/Pass 定义、文件路径、行号均交叉验证，每章附正确性校验报告 |
| **MLIR 贯穿** | 从 MLIR 基础概念讲到 TableGen 方言定义、DialectConversion、Pass 管线 |

## 推荐阅读顺序

```
首次学习路线：

⭐ High level Introduction of triton  第 1-3 章    【入门最佳起点】
        │                          kernel 全流程 → 编译器架构 → Dialect/Pipeline
        │                          3 章快速建立 Triton 编译器整体认知
        │
        v
triton_compiler_view_tutorial         第 1-15 章   【系统深度学习】
                                    编译器理论视角逐章拆解 Triton 全栈
                                    详细目录见 triton_compiler_view_tutorial/README.md
```

> 建议先通读 High level Introduction 建立全景认知，再根据需要进入 compiler_view_tutorial 的对应章节深入学习。每章是自包含的，可按需跳转。

## 快速跳转

| 想了解... | 直接读 |
|-----------|--------|
| Triton kernel 全生命周期（编译→运行） | [High level 第 1 章](High%20level%20Introduction%20of%20triton/chapter1.%20Triton-Kernel-%E5%85%A8%E6%B5%81%E7%A8%8B%E6%8F%AD%E7%A7%98-%E4%BB%8E%E7%BC%96%E8%AF%91%E5%88%B0%E8%BF%90%E8%A1%8C.md) |
| Triton 编译器架构与多后端适配 | [High level 第 2 章](High%20level%20Introduction%20of%20triton/chapter2.%20Triton-%E7%BC%96%E8%AF%91%E5%99%A8%E6%9E%B6%E6%9E%84%E5%85%A8%E6%99%AF%E4%B8%8E%E5%A4%9A%E5%90%8E%E7%AB%AF%E9%80%82%E9%85%8D.md) |
| Dialect 与 Pass Pipeline 设计 | [High level 第 3 章](High%20level%20Introduction%20of%20triton/chapter3.%20Triton-Compiler-Core-Dialect%E4%B8%8EPass-Pipeline.md) |
| 系统性编译器理论学习（15 章全栈） | [triton_compiler_view_tutorial/](triton_compiler_view_tutorial/README.md) |

## 与 Inductor / MLIR 教程的关系

Triton 是 PyTorch Inductor 的默认 GPU 代码生成后端。Inductor 生成 Triton 代码，Triton 编译器将其编译为 GPU 可执行文件。同时，Triton 编译器**本身基于 MLIR 构建**（TTIR、TTGIR 均为 MLIR Dialect），因此 MLIR 教程是理解 Triton 内部设计的前置知识。

- **MLIR 教程（前置推荐）**：Triton 的两级 IR、Dialect 定义、Pass Pipeline、DialectConversion 等设计均建立在 MLIR 之上。建议先读 MLIR 教程建立基础设施认知——第一阶段 [MLIR high level overview](../MLIR/MLIR%20high%20level%20overview/) 讲清 Dialect 体系与变换管线，第二阶段 [compiler-view-of-MLIR](../MLIR/compiler-view-of-MLIR/) 深入 IR 表示、def-use chain、图重写等机制。
- **Inductor 教程**：理解上层调度、融合决策后再深入底层代码生成；或反过来，理解 GPU kernel 编译原理后，更容易理解 Inductor 的 codegen 策略。
- **并行对照学习**：MLIR → Triton → Inductor 三套教程对照阅读，建立从底层基础设施到上层编译栈的完整技术链认知。

MLIR 教程入口：[../MLIR/](../MLIR/)
Inductor 教程入口：[../Inductor%20Introduction/](../Inductor%20Introduction/)
