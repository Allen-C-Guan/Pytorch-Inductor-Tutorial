# 第 3 章 Block、Region 与控制流结构

> **本章位置**　第 2 章讲了 Operation 这个"统一节点"。但 Operation 不是孤立存在的——它挂在一个 Block 里，或者自己包含若干个 Region。本章打开 Block 与 Region 的源码，回答两个问题：Block 为什么就是 Dragon Book 定义的基本块？MLIR 用什么取代了传统 SSA 的 φ 节点？
>
> **前置依赖**　第 0 章（SSA 与支配）、第 2 章（Operation 可嵌套、Value 两面性）。
>
> **编译原理切入**　本章主题是**控制流图（CFG）与基本块**——Dragon Book 第 8.4 节的核心抽象。MLIR 的 Block 就是 CFG 基本块，Region 是结构化作用域容器。但本章有一个 MLIR 区别于 LLVM 的重要设计选择要立论：**Block argument 取代 φ 节点**。传统 SSA（Cytron 1991）用 φ 处理控制流汇合，MLIR 把参数直接挂在 block 上，沿 successor 显式传值——这是对 SSA 的工程简化，让控制流汇合处不再需要特殊的"伪指令"。

---

## 3.1 控制流图与基本块：编译器的经典抽象

在进入 MLIR 源码前，先从编译器学科立论。Dragon Book 第 8.4 节定义了**基本块（basic block）**：一段连续的指令序列，只有一个入口（第一条指令）和一个出口（最后一条指令），中间没有跳转进来或出去。把基本块作为节点、基本块之间的跳转作为边，就构成了**控制流图（Control Flow Graph, CFG）**。

```text
          ┌─────────────┐
          │  Block B1   │ ──┐
          │ (入口块)    │   │
          └─────────────┘   ▼
                      ┌─────────────┐
                      │  Block B2   │
                      │ (条件分支)  │
                      └──────┬──────┘
                         ┌───┴───┐
                         ▼       ▼
                    ┌────────┐ ┌────────┐
                    │ Block  │ │ Block  │
                    │  B3    │ │  B4    │
                    └────┬───┘ └────┬───┘
                         └────┬─────┘
                              ▼
                         ┌────────┐
                         │ Block  │
                         │  B5    │
                         └────────┘
```

CFG 是编译器做控制流分析（可达性、循环检测、支配关系）的基础数据结构。Dragon Book 第 9 章（数据流分析）几乎全部建立在 CFG 之上——到达定值、活跃变量、可用表达式，都是在 CFG 上迭代求解的。

MLIR 的 Block 直接对应这个抽象。一个 Block 是一段 Operation 序列，它的最后一个 Operation 通常是 terminator（终结指令），负责跳转到后继 Block。这一章我们看 Block 在 MLIR 里如何实现，以及它与传统基本块的关键差异。

## 3.2 Block 类的源码：身兼二职

[`Block`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h) 的声明（[Block.h:30-31](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h#L30)）揭示它身兼二职：

```cpp
// Block.h:30-31
class Block : public IRObjectWithUseList<BlockOperand>,
              public llvm::ilist_node_with_parent<Block, Region>
```

这两个基类对应 Block 的两个角色：

1. **`IRObjectWithUseList<BlockOperand>`**——Block 作为"被引用的容器"，持有前驱遍历的使用链头。`BlockOperand` 是 terminator op 指向这个 block 的边（控制流边）。Block 的 `firstUse` 指向所有跳转到它的 BlockOperand。这是**前驱遍历的物理基础**（详见 Ch5）。
2. **`ilist_node_with_parent<Block, Region>`**——Block 自己又是一个链表节点，挂在父 Region 的 `iplist<Block>` 上。这和 Operation 挂在 Block 的 iplist 上是同一套机制（侵入式链表）。

这两个角色合起来说：Block **既是被前驱引用的对象**（use-list 头），**又是 Region 内链表的一个节点**。

### 3.2.1 Block 的私有成员

Block 的三个私有成员（[Block.h:392-398](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h#L392)）：

```cpp
llvm::PointerIntPair<Region*, 1, bool> parentValidOpOrderPair;  // 父 Region + order 有效位
OpListType operations;      // = llvm::iplist<Operation>    ← 块内的 Operation 链表
std::vector<BlockArgument> arguments;  // ← 块参数（取代 φ）
```

- `operations`：这个 Block 包含的 Operation 序列，用 `iplist<Operation>`（LLVM 侵入式链表）组织。这就是"基本块内的指令序列"——和 Dragon Book 说的基本块内部结构一致。
- `arguments`：这个 Block 的参数列表。这是 MLIR 对 SSA φ 节点的替代，下一节详述。
- `parentValidOpOrderPair`：指向父 Region 的指针（同时打包了一个"操作顺序是否有效"的标志位）。

注意一个细节：Block 的 **`operations` 用 iplist，`arguments` 用 std::vector**。为什么这个差别？因为 Operation 要频繁插入/删除（遍历、重写时），iplist 的 O(1) 插入删除是必需的；而 BlockArgument 的增删相对少，用 vector 足够。这是 MLIR 根据访问模式选择数据结构的典型工程取舍。

### 3.2.2 iplist：侵入式链表（结构边）

这里要展开讲一下 `iplist`，因为它是 MLIR 结构边的核心。

```cpp
// Block.h:134
using OpListType = llvm::iplist<Operation>;
OpListType operations;

// Region.h:44
using BlockListType = llvm::iplist<Block>;
BlockListType blocks;
```

`iplist` 是 LLVM 的侵入式链表（intrusive list）。对比标准库的 `std::list`：

| 容器 | 插入/删除 | 节点存储 | 节点内嵌钩子 |
|---|---|---|---|
| `std::list<T>` | O(1) | 外部分配的 list_node | 否，节点被包装 |
| `llvm::iplist<T>` | O(1) | 节点**自己就是** list_node | 是，通过继承 `ilist_node_with_parent` |

关键区别在"节点自己就是 list_node"。`std::list<Operation>` 会为每个 Operation 外面再包一个 list_node（额外内存、额外指针解引用）；而 `iplist<Operation>` 要求 Operation 自己继承 `ilist_node`，链表的 prev/next 钩子就嵌在 Operation 内部——零额外开销。

这就是为什么 Operation 继承 `ilist_node_with_parent<Operation, Block>`（见 Ch2 §2.2）。这个继承不是装饰——它让 Operation 自己就具备了被 iplist 链接的能力。把 Operation 插入 Block 就是 O(1) 地改几个指针，删除同理。

> **编译原理浸润点：侵入式数据结构**　iplist 是 LLVM 的招牌数据结构，源自 Linux 内核的 `list_head` 传统。侵入式链表的优势是零额外分配 + O(1) 插删，代价是节点必须"事先同意"被链接（继承钩子）。这个权衡在编译器 IR 里极其合适——IR 节点数量大、频繁变换，零额外开销至关重要。第 4 章讲 use-list 时会再次遇到侵入式设计（OpOperand 自带 nextUse/back 钩子），第 5 章讲遍历时会看到侵入式结构如何让遍历高效。

## 3.3 Block Argument 取代 φ：SSA 的工程简化

现在到本章的核心论断。传统 SSA 用 **φ 节点（phi node）** 处理控制流汇合。回到第 0 章的例子：

```text
if (cond) { x = 1; } else { x = 2; }
print(x);
```

转成传统 SSA，汇合块开头要插一个 φ：

```text
if (cond) { x1 = 1; } else { x2 = 2; }
x3 = φ(x1, x2);   // 根据来自哪个分支，选 x1 或 x2
print(x3);
```

φ 是一个"伪指令"——它不在运行时真正计算什么，只是说"如果从分支 A 来，就取 A 的值；从 B 来就取 B 的值"。φ 的引入让 SSA 形式化得以严格（Cytron 1991），但它在工程上有几个不便：φ 的参数语义特殊（"按来路选择"）、删除/插入基本块时要维护 φ 的参数、文本表示不够直观。

MLIR 给出了一个不同的方案：**把参数直接挂在 Block 上**。汇合块不再是"无参块 + 开头的 φ"，而是"带参数的块，前驱跳转时显式传参"。

```mlir
// MLIR 风格：汇合块 b3 带一个参数 %x，前驱跳转时传值
^b1:
  cond_br %cond, ^b2, ^b3(%c1 : i32)    // 跳 b3，传 %c1
^b2:
  br ^b3(%c2 : i32)                       // 跳 b3，传 %c2
^b3(%x : i32):                            // b3 的参数，相当于传统 SSA 的 φ
  use %x
```

这里 `^b3(%x : i32)` 的 `%x` 就是 Block argument，它取代了传统 SSA 的 φ。前驱 `^b1` 跳转时传 `%c1`，`^b2` 跳转时传 `%c2`，汇合处 `%x` 的值就是"谁跳过来就传的谁"。

### 3.3.1 为什么 Block argument 比 φ 好

这个设计选择的收益：

1. **语义统一**。Block argument 和函数参数是一样的机制——"调用者传参，被调用者用参"。terminator 的 successor operand 就是"传参"。没有特殊的"伪指令"。
2. **文本表示直观**。读者看到 `^b3(%x : i32)` 立刻知道这个块带一个参数，比看到一堆 φ 直观。
3. **维护简单**。增删基本块时，只需调整前驱的 successor operand 和后继的 argument 列表，不需要解析/维护 φ 的特殊参数结构。
4. **支配性更易论证**。Block argument 天然支配块内所有 op（参数在块入口定义），这是 SSA 不变量的自然满足。

代价是：Block argument 把"控制流汇合"的信息分散到了所有前驱的 terminator 上（每个前驱都要显式传参），而 φ 把它集中在汇合块开头。但前者更符合"显式优于隐式"的工程美学，所以 MLIR（以及 Swift SIL 等现代 IR）都选了这条。

> **编译原理浸润点：φ 节点 vs Block argument**　这个对比直接来自 Cytron 1991 的讨论。φ 是 SSA 形式化的经典机制，但后续很多 IR（MLIR、Swift SIL、Cranelift 的 IR）都选择用 block argument 替代它。这是 SSA 理论落地工程时的一个普遍趋势——用更统一的机制（参数）取代特殊机制（伪指令）。读 MLIR 源码时如果看到 `BlockArgument` 类（它是 Value 的子类型之一，见 Ch2 §2.4），要知道它背后是 SSA 控制流汇合的工程实现。

### 3.3.2 BlockArgument 的存储

`BlockArgument` 存在 Block 的 `arguments` 向量里（`std::vector<BlockArgument>`）。它是 Value 的子类型之一（另一个是 OpResult）。从概念上，BlockArgument 就是"这个 Block 的入口参数"，它的定义者是 Block 本身（不是某个 Operation）。

这意味着 BlockArgument 的 `getDefiningOp()` 返回 nullptr（它没有定义 op），而它的 use-list 头部在 BlockArgument 自己身上。这个细节在 Ch4 讲 use-list 时会再次涉及——Value 的两种子类型（OpResult 和 BlockArgument）虽然都是 Value，但定义方式不同。

## 3.4 Region：结构化作用域容器

讲完 Block，看 Region。[`Region`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Region.h) 是个简单得多的结构（[Region.h:24-26](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Region.h#L24)）：

```cpp
// Region.h:24-26
class Region {
private:
  BlockListType blocks;             // = llvm::iplist<Block>   (:331)
  Operation *container = nullptr;   // 指向包含此 region 的 Op  (:334)
};
```

Region 就是"一个 Block 的链表 + 指回父 Operation 的指针"。它的核心特征是：

> **Region 自身不是一个 Value 或 IRObject——它没有 use-list、没有 Location、没有名字。它是一个纯的"块集合容器"。**

这一点值得强调。Operation、Value、Block 都是"一等公民"——有身份、有 use-list（或被 use-list 引用）、可以被查询。Region 不是。Region 是 Operation 的"内部容器"，它的语义完全由包裹它的 Operation 解释。

### 3.4.1 Region 的几个关键事实

- **`getContext()` / `getLoc()` 委托给 `container`**（父 Op）。如果 Region 还没 attach 到任何 Op（`container == nullptr`），调用这些会 crash（[Region.cpp:24-27](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Region.cpp#L24)）。这是常见的陷阱——游离的 Region 不能用。
- **`getRegionNumber()` 用指针减法 O(1) 求得**（[Region.cpp:65](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Region.cpp#L65)）。因为 Region 对象在 Operation 尾分配区中是**连续存储**的，所以 `this - &getParentOp()->getRegions()[0]` 就是它的序号。这是尾分配的副作用之一（Ch7 详解）。
- **`getArguments()` / `addArgument()` / `insertArgument()` / `eraseArgument()` 全部直接转发给 `front()`**——即入口块（entry block）的参数。这意味着"Region 的参数"实际是"Region 入口块的参数"，二者等价。

### 3.4.2 Region 表达的结构化作用域

Region 的核心语义是**词法作用域**。Region 内部定义的 Value 默认不能被外部直接访问——这是结构化编程的"作用域"在 IR 层面的体现。

哪些 Operation 会有 Region？凡是"有内部逻辑结构"的 Operation。例子：

- **`func.func`**：函数体是一个 Region（包含若干 Block 表达函数的控制流）。
- **`scf.for`**：循环体是一个 Region。
- **`scf.if`**：then 分支一个 Region，else 分支另一个 Region。
- **`builtin.module`**：整个模块是一个 Region。

```mlir
// func.func 的函数体是一个 Region
func.func @f() {
  // ┌── Region 的入口块 ──┐
  %c1 = arith.constant 1 : i32
  %sum = arith.addi %c1, %c1 : i32
  return %sum : i32
  // └─────────────────────┘
}

// scf.if 有两个 Region（then 和 else）
%r = scf.if %cond -> i32 {
  %a = ... 
  scf.yield %a : i32
} else {
  %b = ...
  scf.yield %b : i32
}
```

> **编译原理浸润点：结构化控制流与作用域**　Dragon Book 第 6 章讨论"结构化控制流"（structured control flow）——if/for/while 等有明确入口出口的控制结构。MLIR 的 Region 是结构化控制流的 IR 表达：一个 Region 对应一个结构化作用域。这与 LLVM IR 的"只有显式基本块 + 跳转"形成对比——MLIR 可以同时表达结构化（Region）和非结构化（Block 间的跳转）控制流。这个灵活性是 MLIR 服务多层 IR 的需要：高层 IR（如神经网络计算图）通常结构化，低层 IR（如 LLVM dialect）通常非结构化。

## 3.5 结构树全景：Operation-Region-Block-Operation 的递归

把第 2、3 章合起来，画出 MLIR IR 的结构树全景：

```text
Operation (ModuleOp)
 └─ Region #0
     └─ Block (entry)
         ├─ Operation (func.func @f)
         │   └─ Region #0
         │       └─ Block (entry)
         │           ├─ Operation (arith.constant)
         │           ├─ Operation (arith.addi)
         │           └─ Operation (return)   ← terminator
         ├─ Operation (func.func @g)
         │   └─ Region #0
         │       └─ Block (entry)
         │           └─ ...
         └─ ...
```

这是一个**无限递归**的结构：Operation 包含 Region，Region 包含 Block，Block 包含 Operation，Operation 又可以包含 Region……这种递归嵌套是 MLIR 表达层次结构（从模块到函数到指令）的方式。

注意这条嵌套链上的**两种边**：

1. **结构边（iplist）**：Block → Operation（Block 用 iplist 持有 Operation）、Region → Block（Region 用 iplist 持有 Block）。这些边维护"谁包含谁"的嵌套。
2. **指向父节点的指针**：Operation 有 `block` 指针指回所在 Block，Region 有 `container` 指针指回父 Operation。这些反向指针让"从子找父"成为 O(1)。

> **本章不讲的边：数据流边。**　结构边只表达"谁包含谁"，不表达"谁的数据流向谁"。后者是 operand/result 的 use-def 边，由 use-list 维护，是第 4 章的主题。本章结束后，你应该能在脑中画出 MLIR IR 的**结构树**，但还不知道**数据流图**怎么画——那是下一章的工作。这种"先结构、后数据流"的顺序是有意的：结构树是静态的（谁嵌套在谁里），数据流图是动态的（值如何流动），分开讲更清晰。

## 3.6 结构边的遍历预告

本章最后预告一下 Ch5 要讲的遍历。有了结构树，自然要问怎么遍历它：

- **遍历一个 Block 内的 Operation**：`for (auto &op : *block)`，沿 iplist 走。
- **遍历一个 Region 内的 Block**：`for (auto &b : region)`，沿 iplist 走。
- **递归遍历整棵树（walk）**：`op->walk(callback)`，深度优先遍历所有嵌套的 Region/Block/Operation。

这些遍历的源码细节（特别是"遍历时能不能改 IR"的变更契约）属于 Ch5。本章只要记住：**结构树用 iplist 维护，遍历就是沿 iplist 走**。

还有一个遍历本章先点到：**控制流遍历（predecessor/successor）**。给定一个 Block，怎么找它的前驱和后继？答案是：

- **后继（successor）**：取这个 Block 的 terminator，读它的 successor operand 列表。
- **前驱（predecessor）**：遍历这个 Block 自己的 BlockOperand use-list（这就是 Block 继承 `IRObjectWithUseList<BlockOperand>` 的用途）。

这个"前驱遍历靠 use-list 推导"的设计极其巧妙，Ch5 会展开。本章只要记住：**Block 既在 Region 的 iplist 里（结构边），又持有 BlockOperand 的 use-list（控制流边）**——这两个角色是 Block 身兼二职的体现。

---

## 编译原理浸润点回顾

1. **CFG 与基本块**：本章主题。Block 就是 Dragon Book 第 8.4 节的基本块。MLIR 的 Block 内部用 iplist 持有 Operation，对应基本块内的指令序列。
2. **φ 节点 vs Block argument**：本章核心论断。MLIR 用 block argument 取代传统 SSA 的 φ（Cytron 1991），是 SSA 落地工程的普遍趋势。
3. **结构化控制流与作用域**：Region 表达结构化作用域，对应 Dragon Book 第 6 章的结构化控制流。MLIR 同时支持结构化（Region）与非结构化（Block 跳转）控制流。
4. **侵入式数据结构（iplist）**：本章引入 iplist，作为 MLIR 结构边的物理实现。Ch4 会再次遇到（use-list 也是侵入式）。
5. **支配关系**：Block argument 天然支配块内所有 op（参数在块入口定义），这是 SSA 不变量的自然满足。

---

## 本章关键结论

1. **Block 就是 Dragon Book 的基本块**，内部用 iplist 持有 Operation 序列。它身兼二职：被前驱引用的容器（use-list 头）+ Region 内链表的节点。
2. **Block argument 取代 φ 节点**。这是 MLIR 对传统 SSA 的工程简化——用统一的"参数"机制取代特殊的"伪指令"。前驱跳转时显式传参，汇合处用 block argument。
3. **Region 是结构化作用域容器**。它自身不是一等公民（无 use-list/location/name），语义由父 Operation 解释。`func.func`、`scf.for`、`scf.if` 等都用 Region 表达内部结构。
4. **MLIR IR 是无限递归的结构树**：Operation ⊃ Region ⊃ Block ⊃ Operation。结构边用 iplist（正向）+ 指回父的指针（反向）维护。
5. **本章只讲了结构边**，数据流边（use-def）留 Ch4。结构树是静态嵌套，数据流图是值的流动，分开讲更清晰。

---

## 下一章预告

至此，第二部分（IR 的逻辑结构）结束。我们认识了 Operation、Block、Region——MLIR IR 的结构骨架。但骨架上的"血肉"——数据如何从一个 op 流到另一个 op——还没讲。第 4 章打开数据流边：**def-use chain 的物理实现**。这一章会回答：为什么 MLIR 用一个看似简单的单向链表维护 use-list？那个被很多人称为"全文最难懂"的 `back` 指针（指针的指针）到底解决什么问题？这是全书最底层、也最精巧的一章。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-IR-Node组织与遍历插入删除教程.md` §1.3（Block）、§1.4（Region）——**保留并重组**
- Block argument 取代 φ 的论述综合了原文与 Cytron 1991
- 编译原理铺垫（CFG/基本块/结构化控制流）为本书新增，对应 Dragon Book Ch.6、Ch.8.4

## 参考文献

- **[Aho 2006]** Dragon Book，第 8.4 节（基本块与流图）、第 6 章（结构化控制流）。
- **[Cytron 1991]** Cytron et al. φ 节点的提出；block argument 替代 φ 的工程趋势。
- **[Lattner 2020]** Lattner et al. "MLIR"，Region 与 block argument 的设计论述。
