# Triton Introduction——Triton 编译器学习材料

> 本目录包含从编译器设计视角系统学习 Triton 编译器的完整教程。

## 内容导航

### [High level Introduction of triton/](High%20level%20Introduction%20of%20triton/)——高层导论

以工程实践视角对 Triton 编译器进行全景介绍，覆盖 kernel 全生命周期（编译→运行）、编译器架构全景与多后端适配、Dialect 与 Pass Pipeline 设计，共 **3 章 + 1 附录**。

**适合**：刚接触 Triton，希望快速建立对 Triton 编译器"是什么、怎么运作"的整体认知。

### [triton_compiler_view_tutorial/](triton_compiler_view_tutorial/)——编译器设计视角

以 *Engineering a Compiler* + *MLIR Tutorial* + *Programming Massively Parallel Processors* 三本教材为理论骨架，将 Triton 编译器的各个模块映射到编译器的经典阶段（前端 → 中间层 → 后端 → 运行时），从编译器理论的宏观视角理解 Triton 的全栈架构设计，共 **15 章 + 5 附录**。

**适合**：希望系统性掌握 Triton 编译器全栈设计，理解 TTIR/TTGIR 两级 IR 体系、Layout 抽象、软件流水线等核心设计。

### 教程特色

| 维度 | 说明 |
|------|------|
| **编译器视角** | 按编译器 pipeline 组织：DSL 前端 → TTIR → TTGIR 优化 → LLVM IR 代码生成 → PTX/CUBIN 发射 |
| **What-How-Why** | 每章追问三层：功能是什么、如何实现、为什么这样设计 |
| **源码为本** | 所有 OP/Type/Pass 定义、文件路径、行号均交叉验证，每章附正确性校验报告 |
| **MLIR 贯穿** | 从 MLIR 基础概念讲到 TableGen 方言定义、DialectConversion、Pass 管线 |
| **GPU 硬件映射** | 覆盖 NVIDIA (CUDA)、AMD (ROCm)、Ascend (昇腾) 三套后端的可移植性设计 |

## 推荐阅读顺序

```
首次学习路线：

High level Introduction of triton  第 1-3 章   快速建立 Triton 整体认知
        │                          kernel 全流程 → 编译器架构 → Dialect/Pipeline
        │
        v
triton_compiler_view_tutorial      第 1 章      从编译器理论视角理解 Triton 全景
        │                         编译器设计导论与 Triton 全景
        │
        v
triton_compiler_view_tutorial      第 2-3 章    理解 Triton 编程模型和 TTIR
        │                         DSL 设计 / MLIR 与 TTIR
        │
        v
triton_compiler_view_tutorial      第 4-6 章    深入两级 IR 核心设计
        │                         TTGIR / 类型系统 / Lowering
        │
        v
triton_compiler_view_tutorial      第 7-12 章   掌握优化与代码生成
        │                         循环/内存优化 → 指令选择 → 流水线 → 寄存器分配 → PTX 发射
        │
        v
triton_compiler_view_tutorial      第 13-15 章  理解运行时系统
                                 JIT/缓存 → Autotuning → 端到端回顾
```

> 也可以根据当前需要直接跳转到对应章节——每章是自包含的，但会标注前置依赖。

## 快速跳转

| 想了解... | 直接读 |
|-----------|--------|
| Triton 在 PyTorch 编译栈中的位置 | [第 1 章](triton_compiler_view_tutorial/ch01_introduction.md) |
| 如何用 Triton DSL 编写 GPU kernel | [第 2 章](triton_compiler_view_tutorial/ch02_triton_dsl.md) |
| MLIR 是什么、TTIR 怎么定义的 | [第 3 章](triton_compiler_view_tutorial/ch03_mlir_ttir.md) |
| Triton 的 Layout 系统（最核心设计） | [第 4 章](triton_compiler_view_tutorial/ch04_ttgir_design.md) |
| TTIR 如何 lowering 到 TTGIR | [第 6 章](triton_compiler_view_tutorial/ch06_lowering_ttir_ttgir.md) |
| Triton 如何生成 PTX/CUBIN | [第 12 章](triton_compiler_view_tutorial/ch12_backend_code_emission.md) |
| Triton 的 JIT 和缓存机制 | [第 13 章](triton_compiler_view_tutorial/ch13_jit_cache.md) |
| 如何给 Triton kernel 做 autotune | [第 14 章](triton_compiler_view_tutorial/ch14_autotuning.md) |
| 完整编译流程回顾 | [第 15 章](triton_compiler_view_tutorial/ch15_end_to_end.md) |
| 术语速查 | [附录 E](triton_compiler_view_tutorial/appendix_e_glossary.md) |

## 与 Inductor / MLIR 教程的关系

Triton 是 PyTorch Inductor 的默认 GPU 代码生成后端。Inductor 生成 Triton 代码，Triton 编译器将其编译为 GPU 可执行文件。同时，Triton 编译器**本身基于 MLIR 构建**（TTIR、TTGIR 均为 MLIR Dialect），因此 MLIR 教程是理解 Triton 内部设计的前置知识。

- **MLIR 教程（前置推荐）**：Triton 的两级 IR、Dialect 定义、Pass Pipeline、DialectConversion 等设计均建立在 MLIR 之上。建议先读 MLIR 教程建立基础设施认知——第一阶段 [MLIR high level overview](../MLIR/MLIR%20high%20level%20overview/) 讲清 Dialect 体系与变换管线，第二阶段 [compiler-view-of-MLIR](../MLIR/compiler-view-of-MLIR/) 深入 IR 表示、def-use chain、图重写等机制。
- **Inductor 教程**：理解上层调度、融合决策后再深入底层代码生成；或反过来，理解 GPU kernel 编译原理后，更容易理解 Inductor 的 codegen 策略。
- **并行对照学习**：MLIR → Triton → Inductor 三套教程对照阅读，建立从底层基础设施到上层编译栈的完整技术链认知。

MLIR 教程入口：[../MLIR/](../MLIR/)
Inductor 教程入口：[../Inductor%20Introduction/](../Inductor%20Introduction/)
