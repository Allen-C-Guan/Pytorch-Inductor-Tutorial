# 第 0 章 MLIR 在编译器学科中的位置

> **本章位置**　全书的起点。在讲解任何 MLIR 的具体机制之前，我们先回答一个更根本的问题：MLIR 在编译器这门学科里处于什么位置？它回应了哪些经典问题、又提出了哪些新抽象？这一章建立全书其余十三章赖以展开的元话语。
>
> **前置依赖**　无。本章假设读者学过编译原理（Dragon Book 级别），但对 MLIR 尚无先验知识。

---

## 编译原理切入：编译流水线与"中间表示"

### 0.1 什么是"中间表示"，为什么编译器需要它

任何一本编译器教材（如 Aho 等的 *Compilers: Principles, Techniques, and Tools*，即 Dragon Book [Aho 2006]）的开篇都会画出那条经典的编译流水线：

```text
源代码 → [词法分析] → [语法分析] → [语义分析] → [中间代码生成]
                                                       │
                                                       ▼
                  机器码 ← [代码生成] ← [优化] ← [中间表示优化]
```

这条流水线里反复出现一个词——**中间表示（Intermediate Representation, IR）**。从源代码到机器码之间，编译器几乎从不会一步翻译到位，而是在中间设置一个（或多个）IR，让前端（把源语言翻成 IR）与后端（把 IR 翻成机器码）解耦。Dragon Book 第 8 章用整整一章来讨论"中间代码生成"，正是因为 IR 是编译器架构的核心枢纽。

为什么要解耦？一个朴素但深刻的理由：**多前端、多后端**的可组合性。假设有 M 种源语言、N 种目标机器，没有 IR 时需要写 M×N 套编译器；有了统一的 IR，只需写 M 套前端 + N 套后端，复杂度从 M×N 降到 M+N。这个观察可以追溯到 20 世纪 60 年代 UNCOL 的设想，并在此后被 LLVM、.NET CLR、Java JVM 反复验证。

但 IR 的价值不止于"枢纽"。Dragon Book 第 9 章展示了 IR 真正的威力所在：**优化**。绝大多数优化——常量传播、死代码消除、公共子表达式消除、循环不变量外提——都是作用在 IR 上的。一个设计良好的 IR 让优化算法能够无视源语言与目标机器的细节，专注于"程序的计算结构"。因此，IR 的设计直接决定了编译器能做多少优化、做得有多好。

本章的核心问题由此而来：

> **在编译器学科的谱系里，IR 应该如何被表示？不同的表示流派各有什么取舍？MLIR 选择了哪一派，又为什么？**

### 0.2 IR 表示的三种流派

Dragon Book 第 8.4 节把中间表示大致分为"面向树"与"面向三地址"两类。我们把视野放宽，可以把编译器历史上的 IR 表示归为三大流派，每一派都对应一种把"程序"抽象成"数据结构"的哲学。

**流派一：抽象语法树（Abstract Syntax Tree, AST）**

AST 最接近源代码的结构。它是一棵树，内部节点是运算符，叶子是操作数。Clang 用 AST 表示 C/C++ 源码，Roslyn 用 AST 表示 C#。AST 的优势是"形如源码"——类型检查、作用域分析等前端工作天然适合在树上做。但 AST 不适合数据流优化：一条赋值 `a = b + c` 在 AST 里是一棵深度为 2 的子树，要做"哪个变量在哪里被使用"这种分析，需要在树上反复上下走，很不自然。

**流派二：三地址码（Three-Address Code, TAC）**

Dragon Book 第 8.3 节定义的三地址码把每条指令限制为"最多三个操作数、一个运算"，并把它拍平成线性序列：

```text
t1 = b + c
a  = t1
```

LLVM IR、GCC 的 GIMPLE 都是三地址码的代表。三地址码的扁平化结构让数据流分析变得自然——每条指令显式写出它的输入（操作数）与输出（结果），分析"值如何流动"就是在指令序列上跟踪。它的代价是失去了 AST 的层次结构，控制流需要用显式的跳转和基本块来表达。

**流派三：图 IR（Graph IR）**

图 IR 不满足于"线性序列"，而是显式地把数据依赖画成图。最经典的是 Cliff Click 的 **Sea of Nodes** [Click 1995]，被 Java 的 Graal 编译器采用。在 Sea of Nodes 里，每个值是一个节点，数据依赖与控制依赖都是边，编译器直接在一幅大图上做优化。图 IR 的好处是显式建模了依赖关系，自动暴露并行度；代价是数据结构与算法比线性序列复杂得多。

下表对照三派的特征：

| 流派 | 代表 | 结构 | 擅长 | 不擅长 |
|---|---|---|---|---|
| AST | Clang AST、Roslyn | 树（嵌套） | 前端分析、类型检查 | 数据流优化 |
| 三地址码 | LLVM IR、GIMPLE | 线性序列 + 基本块 | 数据流分析 | 表达层次结构 |
| 图 IR | Sea of Nodes (Graal) | 显式依赖图 | 暴露并行度、依赖追踪 | 数据结构复杂 |

### 0.3 MLIR 选择了什么：三地址码的血统，加上图 IR 的数据流边

MLIR（Multi-Level Intermediate Representation）由 Chris Lattner 等人在 2019 年提出 [Lattner 2020]，它给出的答案是**融合派**：以类三地址码为主体，同时显式建模数据流边。具体而言，MLIR 的 IR 是一个"**任意多元（N-ary）的 SSA 图**"——每个 Operation（MLIR 的指令）可以有任意多个输入和输出，输入输出之间的连接关系（谁的数据流给谁）由被称为 use-def 链的边显式维护。

这个选择值得追问。为什么不全用 AST？因为 MLIR 要服务的是从高层神经网络图到机器码的全栈编译，高层可以不"形如源代码"。为什么不全用 Sea of Nodes 那样的纯图？因为图 IR 的工程复杂度高，且 MLIR 还想保留层次结构（一个 Operation 可以嵌套包含一整个子 IR，详见 Ch3）。

MLIR 的真正创新不在"融合"本身，而在于它提出的一组让这种融合可行的抽象——其中最核心的就是"**一切皆 Operation**"和"**Dialect（方言）**"。这些抽象是第 2 章以后的主题，本章先给出直觉。

> **编译原理浸润点**　"三地址码"与"图 IR"不是 MLIR 的发明，而是编译器学科几十年的沉淀。MLIR 的贡献是把它们的优点工程化地融合进一个可扩展框架。理解这一点很重要：读 MLIR 的源码时，你会反复看到 Dragon Book 第 8、9 章的影子——基本块（Ch3）、数据流分析（Ch4/Ch5）、死代码消除（Ch10）。MLIR 没有重新发明编译原理，它是在已有的编译原理地基上盖了一座可扩展的大楼。

---

## MLIR 的五大基本概念

### 0.4 把 MLIR 拆成五个"积木"

理解任何一门 IR，关键是抓住它的基本积木。MLIR 把这些积木精简到了五个，可以分为**结构层级**与**元数据层级**两组。这一节我们先建立直觉，严格的源码级定义留给后续章节。

#### 结构层级：嵌套的骨架

三个元素构成 MLIR 的结构树（谁包含谁）：

**Operation（操作/算子）——"语义的绝对载体"**　Operation 是 MLIR 中最基础的执行单元。关键之处在于 MLIR 的"一切皆 Operation"哲学：在传统编译器里，指令、函数、模块是完全不同的概念，而在 MLIR 里，一个加法指令是 Operation，一个函数定义是 Operation，连最顶层的整个程序模块也是 Operation。这种统一简化了遍历与变换（因为只需要一套机制处理所有层次），代价是需要额外约定（trait、interface）来区分"这个 Operation 是函数吗"。

```mlir
%res = arith.addi %a, %b                  // 加法指令，是一个 Operation
func.func @my_func() { ... }              // 函数定义，也是一个 Operation
builtin.module { ... }                    // 整个模块，还是一个 Operation
```

**Region（区域）——"作用域与控制流的容器"**　Operation 内部可以挂载若干个 Region，用于表达闭包、作用域或层次化结构。Region 本身没有独立语义——它的语义由包裹它的 Operation 来解释。例如一个 `for` 循环（Operation）的循环体就是一个 Region；一个函数（Operation）的函数体也是一个 Region。

**Block（基本块）——"顺序执行的线性序列"**　Region 内部由一个或多个 Block 组成。Block 就是 Dragon Book 第 8 章定义的基本块：一段没有内部控制流分支的、顺序执行的 Operation 序列。MLIR 的一个标志性设计是：**Block 可以像函数一样带参数（Block Argument）**，这是它对传统 SSA φ 节点的替代（详见 Ch3）。

这三个元素相互嵌套：Operation 包含 Region，Region 包含 Block，Block 又包含 Operation，形成无限递归。

```text
Operation ──包含──► Region ──包含──► Block ──包含──► Operation ──► ...
```

#### 元数据层级：修饰与描述

两个元素不构成骨架，而是精确描述 Operation 与数据的状态：

**Attribute（属性）——"编译期已知的常量元数据"**　Attribute 是附加在 Operation 上的、在编译阶段就已完全确定的静态信息。它不是运行时流动的数据。例如一个函数的名字（`"my_func"`）、卷积的步长（`strides = [1, 1]`）、一整块静态权重，都是 Attribute。**Operand 与 Attribute 的严格分野是 MLIR 可分析性的根基**——这个论断贯穿第 2 章。

**Type（类型）——"运行时数据的形态契约"**　Type 定义了 Operation 在运行时处理的数据的形态：标量 `i32`、张量 `tensor<4x4xf32>`、内存引用 `memref<?x8xf32>`。MLIR 是强类型系统，每一根连接两个 Operation 的数据线（Value）都必须有明确的 Type。

### 0.5 用一个比喻把它们串起来

如果我们把一段 MLIR 代码比作一家公司：

- **Operation** 是部门或员工（执行具体任务）；
- **Region** 是办公室（划定边界，"研发部办公室"）；
- **Block** 是办公室里的流水线（按顺序执行工序）；
- **Attribute** 是挂在员工脖子上的工牌（编译期静态信息，说明他是谁、负责什么）；
- **Type** 是流水线上加工的产品的规格书（规定了输入材料和输出成品的标准）。

这个比喻的局限在于：它把 Operation/Region/Block 画成了"平铺的层次"，而 MLIR 真正的精妙在于它们是**无限递归嵌套**的。第 2、3 章会用源码精确刻画这种嵌套。

> **编译原理浸润点**　"基本块"这个概念来自 Dragon Book 第 8.3 节，"属性"对应编译期常量折叠所需的静态信息（Dragon Book 第 8.5 节）。MLIR 没有新造术语，而是沿用并强化了编译器教材的经典词汇。读 MLIR 文档时如果觉得某个词眼熟——它多半确实来自 Dragon Book。

---

## SSA 形式：MLIR 的数学地基

### 0.6 什么是 SSA

MLIR 的所有数据流都建立在 **SSA（Static Single Assignment，静态单赋值）** 形式之上。SSA 是现代编译器（LLVM、MLIR、Graal）的共同选择，它源于 Cytron 等人 1991 年的经典论文 [Cytron 1991]。

SSA 的定义只有一句话，但分量极重：

> **每个值在其作用域内只被定义一次**（这是 "single assignment"），并且**这个定义必须支配（dominate）所有使用它的地方**（这是 "static" 的真正含义）。

"支配"是一个精确的图论概念，我们在下一小节正式定义。先用一个例子建立直觉。下面这段非 SSA 代码里，变量 `x` 被赋值了两次：

```text
// 非 SSA：x 被赋值两次
if (cond) {
    x = 1;
} else {
    x = 2;
}
print(x);    // 这里的 x 是 1 还是 2？取决于控制流
```

转成 SSA 形式，每个赋值引入一个新的名字，并显式处理控制流汇合：

```text
if (cond) {
    x1 = 1;
} else {
    x2 = 2;
}
x3 = φ(x1, x2);   // φ 节点：根据来自哪个分支选择 x1 或 x2
print(x3);
```

注意每个名字（`x1`、`x2`、`x3`）只被定义一次。`φ` 是 SSA 引入的特殊函数，表示"控制流汇合处，根据来路选择哪个定义"。在 MLIR 里，φ 节点被 **Block argument** 替代（详见 Ch3），但 SSA 的核心不变量保持一致。

### 0.7 支配关系：SSA 的形式化基石

"定义支配使用"中的"支配"需要严格定义。给定控制流图（CFG），我们说节点 D **支配（dominate）** 节点 N，记作 `D dom N`，如果**从入口节点到 N 的每一条路径都经过 D**。每个节点支配自己。

支配关系构成一棵**支配树（dominator tree）**，可以用 Lengauer-Tarjan 算法高效构造 [Lengauer 1979]。SSA 形式要求：**一个值的定义点必须支配它的所有使用点**。这保证了"走到使用这个值的地方时，这个值一定已经被算出来了"——否则就是使用了未定义的值。

为什么 SSA 这么重要？它带来三个编译器写作者梦寐以求的性质：

1. **每个 use 的唯一定义立即可得**——O(1) 获取"这个操作数是谁算出来的"。
2. **每个 def 的所有用户可遍历**——O(users) 枚举"这个结果被谁用了"。
3. **死代码消除（DCE）变得自然**——没有使用者的 def 就是死的，可以直接删。

这三条性质是第 4 章（def-use chain 的物理实现）、第 10 章（erase 与 DCE）的理论依据。

### 0.8 MLIR 如何落地 SSA

MLIR 用一组数据结构来维护 SSA 的不变量。本章只给直觉，源码级的实现在第 4、8 章：

- 每个值的"定义"端是一个容器（OpResult 或 BlockArgument，详见 Ch2）；
- 这个容器持有一个使用链的头（`firstUse`），指向所有使用它的地方；
- 每个"使用"是一个 OpOperand，它既知道自己用的值（指向定义端），又串在使用链里。

```text
Value（定义端）
  └─ firstUse ──► [OpOperand A] ──► [OpOperand B] ──► nullptr
                     │                  │
                  owner=opX          owner=opY
                  (opX 的某个操作数引用了这个 Value)
```

这幅图就是 SSA 的 def-use chain（定义-使用链）的物理实现。第 4 章会逐字节打开它，解释一个看似简单的链表为什么用了"指针的指针"这种精巧设计。

> **编译原理浸润点**　SSA 不是 MLIR 的发明，而是 1991 年 Cytron 等人的工作 [Cytron 1991]。MLIR 选用 SSA 是因为它已经是现代编译器优化的标准地基——LLVM IR、GCC 的 GIMPLE（部分）、Graal 都基于 SSA。当你读 MLIR 源码里到处可见的 `use`、`def`、`getDefiningOp`，要知道这些概念背后是三十多年的编译器研究。本书从第 4 章开始会反复回到 SSA 的不变量——它是理解 MLIR 一切数据流机制（构建、重写、销毁）的钥匙。

---

## 可扩展性：MLIR 区别于 LLVM 的根本特征

### 0.9 为什么"可扩展"是一个编译器问题

到此为止，MLIR 看起来像是"又一个基于 SSA 的三地址码"，和 LLVM IR 差别不大。但 MLIR 真正的差异化创新在于一个词：**可扩展（extensible）**。

LLVM IR 有一个**固定的指令集**——`add`、`load`、`store`、`call` 等指令是 LLVM 项目预先定义好的，你不能给 LLVM IR 增加一条新指令（除非修改 LLVM 源码并发布新版本）。这就像一套固定的汇编语言。对于 LLVM 的定位（作为 C/C++/Rust 等语言的通用后端），固定指令集是合理的：LLVM IR 处于编译栈的较低层，需要的是稳定的、普适的指令。

但 MLIR 面对的是一个不同的世界。MLIR 想要覆盖的是**整个编译栈**——从最高层的神经网络计算图（TensorFlow 的图、PyTorch 的图），到中间的算子层，到循环与张量层（`scf`、`linalg`），再到 LLVM IR 层，最后到机器码。这些层次的"指令"千差万别：神经网络层有 `softmax`、`conv2d`，循环层有 `for`、`if`，机器层有 `add`、`mul`。一个固定的指令集根本装不下这么丰富的语义。

MLIR 的解法是**Dialect（方言）机制**：允许任何人（不只是 MLIR 项目本身）定义自己的"指令集"，称为一个 dialect。每个 dialect 提供一组 Operation、Type、Attribute，可以像插件一样加载到 MLIR 的上下文里。例如：

- `arith` dialect 提供 `addi`、`mulf` 等算术运算；
- `func` dialect 提供 `func.func`、`func.call` 等函数抽象；
- `llvm` dialect 把 LLVM IR 的指令包成 Operation；
- 用户可以定义自己的 dialect（如本系列教程的 `north_star` dialect）。

```mlir
%1 = arith.addi %a, %b : i32              // arith dialect 的 addi
%2 = "north_star.softmax"(%1) <{axis = 1 : i64}>   // north_star dialect 的 softmax
```

### 0.10 多层次与渐进式 Lowering

Dialect 机制带来一个深远的后果：MLIR 的编译过程不是"源 → 一个 IR → 机器码"，而是"**源 → 高层 dialect → 中层 dialect → ... → LLVM dialect → LLVM IR → 机器码**"。每一层是一个 dialect，相邻层之间通过 **Conversion（转换/lowering）** 把高层 op 翻译成低层 op。这个把高层语义逐步下沉的过程叫**渐进式 lowering（progressive lowering）**。

```text
神经网络图 dialect ──lowering──► 算子 dialect ──lowering──► 循环/张量 dialect
                                                                 │
                                                                 ▼
              机器码 ◄── LLVM IR ◄── LLVM dialect ◄───────────── ...
```

这个架构是 MLIR 论文 [Lattner 2020] 的核心论点：**编译器应该有多层、可组合的 IR，而不是单一的一层**。每一层 IR 只负责它那个抽象层次，层与层之间用 lowering 衔接。这比"一个 IR 通吃所有层次"要清晰得多，也让每一层的优化可以做得更精准。

本书主要关注 MLIR IR 本身的机制（表示、构造、重写、销毁），lowering 的完整实战留待后续。但 Ch9–Ch11 讲的重写（rewrite）机制，正是 lowering 的核心动词——lowering 本质上就是"用一组重写规则把高层 op 换成低层 op"。

---

## 本书要讲的四件事

基于上面的铺垫，本书围绕 MLIR IR 的**生命周期**展开，回答四个问题：

| 问题 | 对应编译器学科主题 | 本书部分 |
|---|---|---|
| **IR 长什么样？** | IR 表示理论 | 第二、三、四部分（Ch2–Ch5） |
| **IR 怎么造出来？** | 前端语义分析 → IR 构造 | 第五部分（Ch6–Ch8） |
| **IR 怎么改？** | 项重写系统、优化 | 第六部分（Ch9–Ch11） |
| **IR 怎么安全消亡？** | 不变量维护、死代码消除 | 融入第六部分（Ch10） |

贯穿这四件事的，是 SSA 不变量、def-use chain、use-list 这几个编译器学科的经典概念。你会在每一章都遇到它们——它们是 MLIR 的"血液"。

---

## 编译原理浸润点回顾

本章引入了五个贯穿全书的编译原理概念，后续章节会反复回到它们：

1. **IR 表示三派**（AST/三地址码/图 IR）：MLIR 是融合派。在第 2 章（Operation）、第 4 章（use-list 显式建模数据流边）、第 12 章（parse 跳过 AST）都会回扣。
2. **SSA 形式**：MLIR 的数学地基。第 4 章形式化其物理实现，第 8 章讲构造时如何自动维护，第 10 章讲重写与销毁时如何保持不变量。
3. **支配关系**：SSA 的形式化基石。第 3 章（Block argument 取代 φ）、第 8 章（构建顺序保支配）都会用到。
4. **数据流分析**：第 4、5 章的直接主题——use-def chain 是数据流分析的物理基础。
5. **可扩展/多层次 IR 哲学**：MLIR 区别于 LLVM 的根本。第 1 章（dialect 加载）、第 7 章（声明式 .td 定义）都建立在这个哲学上。

---

## 本章关键结论

1. **MLIR 是编译器学科谱系中的一员**，它不是凭空发明的——三地址码的血统（Dragon Book Ch.8）、SSA 的数学地基（Cytron 1991）、图 IR 的数据流边（Sea of Nodes）都是它的源头。
2. **MLIR 的核心创新是可扩展性与多层次**。Dialect 机制让任何人定义自己的"指令集"，渐进式 lowering 让编译栈从神经网络图一路下沉到机器码。这与 LLVM 的"固定指令集、单一 IR 层"形成对比。
3. **"一切皆 Operation"是 MLIR 的统一抽象**。指令、函数、模块都是 Operation，差异由 trait/interface 表达。这简化了遍历与变换。
4. **五大基本概念分两组**：结构层级（Operation/Region/Block）构成嵌套骨架，元数据层级（Attribute/Type）描述状态。Operand 与 Attribute 的分野是可分析性的根基（Ch2 详解）。
5. **SSA 形式是 MLIR 的数学地基**。"每个值只定义一次、定义支配使用"这一不变量，由 def-use chain 物理实现（Ch4），由构造自动维护（Ch8），由重写与销毁保持（Ch10）。

---

## 下一章预告

本章建立了"MLIR 是什么"的全景。但全景图里有一个角色我们一笔带过了——**MLIRContext**。所有的 Operation、Type、Attribute 都"活"在一个 context 里，context 析构时它们一起消失。这个"拥有者"是如何用数据结构把所有 IR 对象攥在手心的？为什么 Type 和 Attribute 要被"唯一化"（intern），而 Operation 不需要？这些问题的答案在编译器学科里对应"符号表/类型表的传统设计"。下一章，我们打开 `MLIRContext` 的源码。

---

## 原文对照

本章素材主要来自：
- `docs/the-abstract-of-MLIR.md`（五大基本概念的直觉性介绍，§0.4–0.5 大幅扩展）
- 编译流水线、IR 表示三派的内容为本书新增的编译原理铺垫
- SSA 与支配关系的形式化（§0.6–0.7）为本书补充的编译原理背景，对应 Dragon Book Ch.9 与 Cytron 1991

## 参考文献

- **[Aho 2006]** Aho, Lam, Sethi, Ullman. *Compilers: Principles, Techniques, and Tools* (2nd ed.). Addison-Wesley, 2006.（即 Dragon Book，第 8 章中间代码生成、第 9 章机器无关优化）
- **[Click 1995]** Cliff Click. "Combining Analyses, Combining Optimizations." PhD Thesis, Rice University, 1995.（Sea of Nodes 的提出）
- **[Cytron 1991]** Ron Cytron, Jeanne Ferrante, Barry Rosen, Mark Wegman, Kenneth Zadeck. "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph." *ACM TOPLAS*, 13(4), 1991.（SSA 形式的奠基论文）
- **[Lattner 2020]** Chris Lattner, Jacques Pienaar, Mehdi Amini, et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." CGO 2021. arXiv:2002.11054.
- **[Lengauer 1979]** Thomas Lengauer, Robert Tarjan. "A Fast Algorithm for Finding Dominators in a Flowgraph." *ACM TOPLAS*, 1(1), 1979.（支配树的经典构造算法）
