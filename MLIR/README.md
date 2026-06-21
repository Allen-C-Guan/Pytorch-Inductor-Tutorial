# MLIR Introduction——MLIR 编译器基础设施学习材料

> 本目录包含 MLIR 编译器基础设施的系统性学习资料。学习路线分为**两阶段**：先建立全景（编译器/IR 演进史、Dialect 体系、变换管线），再深入机制（以编译器学科为镜头逐层拆解 MLIR 的工作机制）。

## 学习路线总览

```
第一阶段：建立全景（"MLIR 是什么、为什么需要它"）
   MLIR high level overview/        4 章博客系列
        │  由 MLIR/SPIR-V 核心开发者 Lei Zhang 撰写
        │  LLVM IR → SPIR-V → MLIR 的演进、Dialect 体系、Linalg/Vector 变换管线
        ▼
第二阶段：深入机制（"MLIR 内部怎么工作"）
   compiler-view-of-MLIR/           Ch0–Ch13 教材
        编译器学科镜头下逐章拆解
        IR 如何被表示、构造、重写、销毁
```

> 两阶段是递进关系：第一阶段回答"为什么有 MLIR、它解决什么问题"，第二阶段回答"这些机制在编译器学科里属于什么问题、MLIR 怎么落地"。第一阶段是第二阶段的认知前提——没有 Dialect 与 Progressive Lowering 的宏观图景，第二阶段的 Operation/def-use chain/图重写等机制就缺乏落点。

---

## 第一阶段：[MLIR high level overview/](MLIR%20high%20level%20overview/)——MLIR 高层概述

源自 Lei.Chat 博客的系列文章，由 MLIR/SPIR-V 核心开发者撰写，从编译器与中间表示的演进历史出发，逐步深入到 MLIR 代码生成相关的核心 Dialect 体系和变换流程，共 **4 章**。

**适合**：希望理解编译器/IR 演进逻辑、MLIR 设计哲学，以及 Linalg/Vector Dialect 如何协同完成结构化代码生成。

### 系列文章概览

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

### 第一阶段推荐阅读顺序

```
第 1 章                           建立编译器/IR 演进的全局认知
    │                             LLVM IR → SPIR-V → MLIR，理解"为什么需要 MLIR"
    v
第 2 章                           掌握 MLIR Dialect 体系全景
    │                             理解各 Dialect 的层次关系与沙漏型编译器架构
    v
第 4 章（Linalg）                  理解结构化代码生成的入口抽象
    │                             分块、融合、分配、向量化四大变换
    v
第 3 章（Vector）                  深入向量层的渐进式递降
                                 向量化→展开→清理→Hoisting→递降的完整 pipeline
```

> 每章是自包含的，但第 3-4 章依赖第 2 章建立的 Dialect 体系概念，第 1 章提供了理解 MLIR 设计动机的历史背景。

### 第一阶段快速跳转

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

---

## 第二阶段：[compiler-view-of-MLIR/](compiler-view-of-MLIR/)——以编译器视角理解 MLIR

> 一本写给研究生与编译器工程师的教材，以**编译器学科**为镜头，系统讲解 MLIR 的工作机制。不把 MLIR 当作一套 API 来罗列，而是把它当作编译器学科谱系中的一个成员来审视——它回答了哪些经典问题、用了什么新抽象、相比 LLVM/GCC/Sea of Nodes 做了哪些取舍。

**适合**：已建立 MLIR 全景认知，希望理解 Operation/Block/Region/def-use chain/matchAndRewrite 等机制背后的设计动机与编译原理出处。

### 全书阅读路线图

本书以**一个 IR 的完整生命周期**为骨架——IR 如何被表示、如何被构造、如何被重写、如何被安全销毁。这条主线天然对应编译器教科书的经典主题：

```text
第一部分·导论
   Ch0  MLIR 在编译器学科中的位置
   Ch1  MLIRContext：对象的所有权与唯一化
        │
        v  (认识 IR 的拥有者后，看 IR 的逻辑部件)
第二部分·IR 的逻辑结构（"长什么样"）
   Ch2  Operation：一切皆操作及其部件语义
   Ch3  Block、Region 与控制流结构
        │
        v  (认识 Value 后，引入数据流边)
第三部分·数据流边：def-use chain（"数据怎么流"）
   Ch4  def-use chain：从数据流分析到 use-list 实现
        │
        v  (有了结构与数据流边，就能遍历)
第四部分·IR 的遍历（"怎么看"）
   Ch5  遍历 IR：walk、predecessor 与变更契约
        │
        v  (会看了，就开始造)
第五部分·IR 的构建（"怎么造"）
   Ch6  从 AST 到 IR：OpBuilder 与两类挂接
   Ch7  Operation::create 的完整路径与尾分配内存模型
   Ch8  use-def 链如何自动形成
        │
        v  (造好了，就开始改)
第六部分·IR 的重写（"怎么改"——优化）
   Ch9  matchAndRewrite：图重写范式与全生命周期
   Ch10 replaceOp、RAUW 与 erase：重写的原子动作
   Ch11 驱动、工作表与监听者
        │
        v  (把前面所有机制串起来)
第七部分·专题与综合
   Ch12 专题：type 的 parse/print 往返
   Ch13 综合：BufferCastOpFold 闭环
```

### 各章速览

| 章 | 标题 | 编译原理切入 | 核心机制 |
|---|---|---|---|
| Ch0 | [MLIR 在编译器学科中的位置](compiler-view-of-MLIR/00-MLIR在编译器学科中的位置.md) | IR 表示三派（AST/三地址码/图IR）；可扩展 IR 哲学 | 五大基本概念、SSA 直觉 |
| Ch1 | [MLIRContext：对象的所有权与唯一化](compiler-view-of-MLIR/01-MLIRContext-对象的所有权与唯一化.md) | 符号表/类型表传统；hash-consing | pImpl、StorageUniquer、dialect 加载 |
| Ch2 | [Operation：一切皆操作及其部件语义](compiler-view-of-MLIR/02-Operation-一切皆操作及其部件语义.md) | IR 表示理论；一切皆操作 | Operation 部件语义、Operand vs Attribute 分野 |
| Ch3 | [Block、Region 与控制流结构](compiler-view-of-MLIR/03-Block-Region与控制流结构.md) | CFG 与基本块；φ 节点 | Block argument 取代 φ、Region 结构化作用域 |
| Ch4 | [def-use chain：从数据流分析到 use-list 实现](compiler-view-of-MLIR/04-def-use-chain-从数据流分析到use-list实现.md) | 数据流分析；def-use chain 物理实现 | use-list、`back` 指针、insertInto/removeFromCurrent |
| Ch5 | [遍历 IR：walk、predecessor 与变更契约](compiler-view-of-MLIR/05-遍历IR-walk-predecessor与变更契约.md) | 数据流分析迭代算法 | walk、predecessor、变更契约 |
| Ch6 | [从 AST 到 IR：OpBuilder 与两类挂接](compiler-view-of-MLIR/06-从AST到IR-OpBuilder与两类挂接.md) | 前端语义分析→IR 构造 | OpBuilder、两类挂接、OperationState |
| Ch7 | [Operation::create 与尾分配](compiler-view-of-MLIR/07-Operation-create的完整路径与尾分配.md) | 声明式 IR 定义（.td→.inc） | 三段式构建管线、尾分配内存模型 |
| Ch8 | [use-def 链如何自动形成](compiler-view-of-MLIR/08-use-def链如何自动形成.md) | SSA 不变量的自动维护 | OpOperand 构造即入链 |
| Ch9 | [matchAndRewrite：图重写范式与全生命周期](compiler-view-of-MLIR/09-matchAndRewrite-图重写范式与全生命周期.md) | 项重写系统、图重写 | OpRewritePattern、两个继承家族 |
| Ch10 | [replaceOp、RAUW 与 erase：重写的原子动作](compiler-view-of-MLIR/10-replaceOp-RAUW与erase-重写的原子动作.md) | SSA 不变量维护 | RAUW 三原子动作、erase 安全、DCE |
| Ch11 | [驱动、工作表与监听者](compiler-view-of-MLIR/11-驱动-工作表与监听者.md) | 不动点与单调框架 | worklist、listener、终止性 |
| Ch12 | [专题：type 的 parse/print 往返](compiler-view-of-MLIR/12-专题-type的parse-print往返.md) | 词法/语法/语义分析分层 | parse 作为反序列化器 |
| Ch13 | [综合：BufferCastOpFold 闭环](compiler-view-of-MLIR/13-综合-BufferCastOpFold闭环.md) | 端到端优化 pass | walk→match→RAUW→erase 全链路 |

> 详细的阅读路线、源码约定与原文素材对照见 [compiler-view-of-MLIR/README.md](compiler-view-of-MLIR/README.md)。

### 第二阶段怎么读

**第一遍（建立全景）**：顺序读 Ch0–Ch5。建立全书的元话语——MLIR 是什么、IR 长什么样、数据怎么流、怎么遍历。

**第二遍（深入机制）**：按需读 Ch6–Ch11。这是 IR 的动态生命周期——构建、重写。如果你的兴趣是写 lowering pass，Ch6–Ch11 是核心。

**第三遍（贯通）**：读 Ch12–Ch13。这两章是综合应用——Ch12 从前端序列化的视角回扣编译流水线，Ch13 把前十一章的机制串成一个真实的优化 pass。

---

## 与 Inductor/Triton 教程的关系

MLIR 是理解现代编译器基础设施的基石，与 Inductor 和 Triton 教程形成互补：

- **Triton 编译器基于 MLIR 构建**：Triton 的 TTIR 和 TTGIR 均以 MLIR Dialect 形式实现。先学 MLIR 再学 Triton，可以更深入理解 Triton 的 IR 设计、Pass Pipeline 和 Dialect Conversion 机制。
- **Inductor → Triton 的桥梁**：Inductor 生成 Triton 代码，Triton 内部通过 MLIR 基础设施编译为 GPU 可执行文件。MLIR 教程填补了"Triton 编译器如何基于 MLIR 工作"的理论空白。
- **建议学习顺序**：MLIR 基础（本文）→ Triton 编译器教程 → Inductor 编译器教程，形成从底层基础设施到上层编译栈的完整知识链。

Triton 教程入口：[../Triton%20Introduction/](../Triton%20Introduction/)
Inductor 教程入口：[../Inductor%20Introduction/](../Inductor%20Introduction/)

---

> 第一阶段系列文章版权归原作者 Lei Zhang 所有，遵循 CC 4.0 BY-SA 版权协议，原文链接见各章节末尾。
