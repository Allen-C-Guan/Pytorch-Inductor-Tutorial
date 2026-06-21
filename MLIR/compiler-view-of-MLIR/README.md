# 以编译器视角理解 MLIR：一本关于 MLIR 工作机制的教材

> **本书的定位**　这是一写给研究生与编译器工程师的教材，以**编译器学科**为镜头，系统讲解 MLIR 的工作机制。我们不把 MLIR 当作一套 API 来罗列，而是把它当作编译器学科谱系中的一个成员来审视——它回答了哪些经典问题、用了什么新抽象、相比 LLVM/GCC/Sea of Nodes 做了哪些取舍。
>
> **作者视角**　本书以大学编译器教授写教科书的口吻行文：先立论（这在编译器学科里是什么问题、前人怎么做的），后展开（MLIR 的抽象与实现）；原理与源码来回穿梭，而非"先一大段理论、再一大段代码"。
>
> **在 MLIR 学习路线中的位置**　本书是 MLIR 学习的**第二阶段**。第一阶段 [MLIR high level overview/](../MLIR%20high%20level%20overview/)（4 章博客）建立全景——编译器/IR 演进史、Dialect 体系、Linalg/Vector 变换管线，回答"为什么需要 MLIR、它解决什么问题"。本书承接全景，回答"这些机制在编译器学科里属于什么问题、MLIR 怎么落地"——IR 如何被表示、构造、重写、销毁。建议先完成第一阶段再进入本书；如已具备 MLIR 基础，可直接从本书开始。返回上层导航：[../README.md](../README.md)。

---

## 本书写给谁

- 学过编译原理（Dragon Book 级别的 SSA、数据流分析、基本块/控制流图），想看清楚 MLIR 是如何把这些经典概念落到工程里的研究生与高年级本科生；
- 用过或想用 MLIR 做 lowering / 方言设计 / Pass 开发，但不满足于"会用 API"，想理解 API 背后设计动机的工程师；
- 想把分散的 MLIR 知识系统化、并能讲清"为什么"的研究者与教师。

---

## 全书阅读路线图

本书以**一个 IR 的完整生命周期**为骨架——IR 如何被表示、如何被构造、如何被重写、如何被安全销毁。这条主线天然对应编译器教科书的经典主题：

```text
第一部分·导论
   Ch0  MLIR 在编译器学科中的位置
   Ch1  MLIRContext：对象的所有权与唯一化
        │
        ▼  (认识 IR 的拥有者后，看 IR 的逻辑部件)
第二部分·IR 的逻辑结构（"长什么样"）
   Ch2  Operation：一切皆操作及其部件语义
   Ch3  Block、Region 与控制流结构
        │
        ▼  (认识 Value 后，引入数据流边)
第三部分·数据流边：def-use chain（"数据怎么流"）
   Ch4  def-use chain：从数据流分析到 use-list 实现
        │
        ▼  (有了结构与数据流边，就能遍历)
第四部分·IR 的遍历（"怎么看"）
   Ch5  遍历 IR：walk、predecessor 与变更契约
        │
        ▼  (会看了，就开始造)
第五部分·IR 的构建（"怎么造"）
   Ch6  从 AST 到 IR：OpBuilder 与两类挂接
   Ch7  Operation::create 的完整路径与尾分配内存模型
   Ch8  use-def 链如何自动形成
        │
        ▼  (造好了，就开始改)
第六部分·IR 的重写（"怎么改"——优化）
   Ch9  matchAndRewrite：图重写范式与全生命周期
   Ch10 replaceOp、RAUW 与 erase：重写的原子动作
   Ch11 驱动、工作表与监听者
        │
        ▼  (把前面所有机制串起来)
第七部分·专题与综合
   Ch12 专题：type 的 parse/print 往返
   Ch13 综合：BufferCastOpFold 闭环
```

### 为什么这样排序（顺序友好性）

本书的章节顺序经过精心设计，保证**读者按顺序阅读时，任何一章都不依赖其后章节的知识**。每一章开头都标注了"前置依赖"，全部指向前面的章节。这与许多编译器教程"先堆一个抽象数据结构、再告诉你它为什么有用"的做法相反——本书遵循好教材的铁律：**按需引入，具体先于抽象**。

例如，use-list（def-use chain 的物理实现）没有放在全书最前面当"底层基础"，而是放在第 4 章——因为它的存在动机（OpOperand 是连接两个 op 的边）只有在读者认识了 Value（第 2 章）之后才能自然显现。再如，尾分配内存模型放在第 7 章与 `Operation::create` 同章，而不是孤零零地放在静态结构部分——因为它是 create 时的分配技巧，脱离 create 讲尾分配是空中楼阁。

---

## 各章速览

| 章 | 标题 | 编译原理切入 | 核心机制 |
|---|---|---|---|
| Ch0 | MLIR 在编译器学科中的位置 | IR 表示三派（AST/三地址码/图IR）；可扩展 IR 哲学 | 五大基本概念、SSA 直觉 |
| Ch1 | MLIRContext：对象的所有权与唯一化 | 符号表/类型表传统；hash-consing | pImpl、StorageUniquer、dialect 加载 |
| Ch2 | Operation：一切皆操作及其部件语义 | IR 表示理论；一切皆操作 | Operation 部件语义、Operand vs Attribute 分野 |
| Ch3 | Block、Region 与控制流结构 | CFG 与基本块；φ 节点 | Block argument 取代 φ、Region 结构化作用域 |
| Ch4 | def-use chain：从数据流分析到 use-list 实现 | 数据流分析；def-use chain 物理实现 | use-list、`back` 指针、insertInto/removeFromCurrent |
| Ch5 | 遍历 IR | 数据流分析迭代算法 | walk、predecessor、变更契约 |
| Ch6 | 从 AST 到 IR | 前端语义分析→IR 构造 | OpBuilder、两类挂接、OperationState |
| Ch7 | Operation::create 与尾分配 | 声明式 IR 定义（.td→.inc） | 三段式构建管线、尾分配内存模型 |
| Ch8 | use-def 链如何自动形成 | SSA 不变量的自动维护 | OpOperand 构造即入链 |
| Ch9 | matchAndRewrite | 项重写系统、图重写 | OpRewritePattern、两个继承家族 |
| Ch10 | RAUW 与 erase | SSA 不变量维护 | RAUW 三原子动作、erase 安全、DCE |
| Ch11 | 驱动、工作表与监听者 | 不动点与单调框架 | worklist、listener、终止性 |
| Ch12 | type 的 parse/print 往返 | 词法/语法/语义分析分层 | parse 作为反序列化器 |
| Ch13 | BufferCastOpFold 闭环 | 端到端优化 pass | walk→match→RAUW→erase 全链路 |

---

## 怎么读这本书

**第一遍（建立全景）**：顺序读 Ch0–Ch5。这六章建立了全书的元话语——MLIR 是什么、IR 长什么样、数据怎么流、怎么遍历。读完这六章，你能在脑中画出一棵 MLIR IR 树的完整结构。

**第二遍（深入机制）**：按需读 Ch6–Ch11。这六章是 IR 的动态生命周期——构建、重写。如果你的兴趣是写 lowering pass，Ch6–Ch11 是核心；如果只想理解 IR 表示，Ch0–Ch5 已够。

**第三遍（贯通）**：读 Ch12–Ch13。这两章是综合应用——Ch12 从前端序列化的视角回扣编译流水线，Ch13 把前十一章的机制串成一个真实的优化 pass。

---

## 源码约定

- 本书所有源码引用基于 **LLVM 19.1.7**，位于 `MLIR-Tutorial/third_party/llvm-project/mlir/`。
- 引用形式为可点击的相对链接：[`Operation.h:84`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L84)。
- 代码块标注语言（`cpp`/`mlir`/`text`/`tablegen`），不使用裸代码块。
- 图示一律用 ASCII，不引入外部图片。

## 文献引用

每章首次引入一个编译原理概念时，标注其经典文献出处（如 Dragon Book 章节号、Cytron 1991 等）。书末汇总完整参考文献。

---

## 本书与原文素材的对照

本书由 `docs/` 下若干篇学习笔记（"知识碎片"）系统化重构而成。原文保留不动，作为素材；本书的每一章都在章末"原文对照"小节标注其素材来源，便于读者回查原文。

| 本书章节 | 主要原文素材 |
|---|---|
| Ch0 | `the-abstract-of-MLIR.md` |
| Ch1 | `MLIR-IR-树的构建过程教程_精品.md` §1（MLIRContext） |
| Ch2, Ch3 | `MLIR-IR-Node组织与遍历插入删除教程.md` §0-1 |
| Ch4 | `MLIR-IR-Node组织...` §1.2/§2.2（use-list/back）+ `MLIR-RAUW-...机理详解.md` §1 + `operator-use-def的形成.md` §8.1 |
| Ch5 | `MLIR-IR-Node组织...` §3 |
| Ch6 | `MLIR-IR-树的构建过程教程_精品.md` §0,2-5 |
| Ch7 | `operator-use-def的形成.md` §3-7 + `MLIR-IR-Node组织...` §2.1（尾分配） |
| Ch8 | `operator-use-def的形成.md` §8 + `MLIR-IR-树...` §6 |
| Ch9 | `MLIR-matchAndRewrite-重写过程教程_精品.md` §0-3 |
| Ch10 | `MLIR-matchAndRewrite...` §4 + `MLIR-RAUW-...机理详解.md` 全文 + `MLIR-IR-Node组织...` §5（erase） |
| Ch11 | `MLIR-matchAndRewrite-重写过程教程_精品.md` §6-8 |
| Ch12 | `ns-tensor-parse-walkthrough.md` |
| Ch13 | `MLIR-IR-Node组织...` §6（BufferCastOpFold） |

---

*开始阅读：[Ch0 MLIR 在编译器学科中的位置](00-MLIR在编译器学科中的位置.md)*
