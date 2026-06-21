# 第 5 章 遍历 IR：walk、predecessor 与变更契约

> **本章位置**　第 2–4 章认识了 IR 的结构树与数据流边。有了这些静态结构，自然要问：怎么**遍历**它们？本章讲三类遍历——结构遍历（walk，深度优先走结构树）、数据流遍历（沿 use-list 走）、控制流遍历（predecessor/successor，沿 Block 间跳转）。其中有一个编译器写作者必知的契约——**遍历期间能不能改 IR？什么时候能 erase 当前 op？**——这个"变更契约"决定了优化 pass 能不能正确地"边遍历边改"。
>
> **前置依赖**　第 2 章（Operation）、第 3 章（Block/Region 结构边）、第 4 章（use-list 数据流边）。
>
> **编译原理切入**　本章主题是**数据流分析的迭代算法**。Dragon Book 第 9 章的数据流分析（到达定值、活跃变量、可用表达式）都是"在 CFG 上反复迭代，直到结果不再变化"。这些迭代的物理基础就是本章讲的遍历——沿着结构树或数据流边走。而"迭代时能不能改 IR"对应编译器的一个经典难题：**变换过程中的迭代器失效**。Dragon Book 不深究这个工程问题（它假设分析阶段不改 IR），但任何真实的优化 pass 都必须面对——MLIR 用 `make_early_inc_range` 给出了精巧的解法。

---

## 5.1 遍历为什么是编译器的基础设施

在打开源码前，先从编译器学科立论：遍历为什么这么重要？

编译器的绝大多数工作都是某种形式的遍历：

- **类型检查 / 语义分析**：遍历 AST 或 IR，检查每个节点是否符合规则。
- **数据流分析**：遍历 CFG，迭代计算到达定值、活跃变量等（Dragon Book 第 9 章）。
- **优化 pass**：遍历 IR 寻找可优化的模式（常量折叠、DCE、CSE）。
- **lowering / conversion**：遍历 IR 把高层 op 换成低层 op。
- **代码生成**：遍历 IR 生成机器码。

可以说，编译器 = 遍历 + 变换。本章讲"遍历"这一半，变换（重写）留给第六部分（Ch9–Ch11）。

MLIR 的 IR 有两种结构（第 2–4 章建立）：

1. **结构树**：Operation ⊃ Region ⊃ Block ⊃ Operation 的嵌套（Ch2–3）。
2. **数据流图**：Value → OpOperand → Value 的 use-def 链（Ch4）。

对应地，MLIR 有三类遍历：

| 遍历类型 | 走什么结构 | 典型用途 |
|---|---|---|
| **结构遍历（walk）** | 结构树（嵌套） | 遍历所有 op 找某种模式；lowering |
| **数据流遍历（use/user）** | use-list（Ch4） | 找一个值的定义/使用者；数据流分析 |
| **控制流遍历（pred/succ）** | Block 间的跳转边 | CFG 分析；循环检测；支配关系 |

本章逐一展开。其中**控制流遍历有个精巧之处**：predecessor 不是直接存储的，而是从 Block 的 use-list 推导出来的（第 3 章埋的伏笔）。

## 5.2 结构遍历：walk 的递归深度优先

MLIR 最常用的遍历是 `walk`——从某个 Operation 出发，深度优先遍历它包含的所有嵌套结构。源码在 [Visitors.h:127-187](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h#L127)：

```cpp
template <typename Iterator>
void walk(Operation *op, function_ref<void(Operation *)> callback,
          WalkOrder order) {
  if (order == WalkOrder::PreOrder)
    callback(op);                                    // ← 前序：先回调自己

  for (auto &region : Iterator::makeIterable(*op)) {
    for (auto &block : Iterator::makeIterable(region)) {
      for (auto &nestedOp :
           llvm::make_early_inc_range(Iterator::makeIterable(block)))
        walk<Iterator>(&nestedOp, callback, order);  // ← 递归遍历嵌套 op
    }
  }

  if (order == WalkOrder::PostOrder)
    callback(op);                                    // ← 后序：后回调自己
}
```

这段代码做的事：对当前 op，遍历它的每个 Region；对每个 Region，遍历它的每个 Block；对每个 Block，遍历里面的每个 Operation，并**递归**地对那个 Operation 调 walk（因为它可能还嵌套着更深的 Region）。这正是结构树的深度优先遍历。

### 5.2.1 两种顺序：PreOrder 与 PostOrder

`walk` 支持两种访问顺序：

- **PreOrder（先序）**：进入一个 op 时先回调它，再递归子 op。适合"自顶向下"的处理（如从外层函数往内层指令找）。
- **PostOrder（后序，默认）**：先递归子 op，回来再回调自己。适合"自底向上"的处理（如先把内层算完再处理外层——许多 lowering 是这种方向）。

默认是 PostOrder（[Operation.h:788](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L788)）。这个默认有深意：后序保证"处理一个 op 时，它内部的所有 op 都已经处理过了"，这对很多变换（如 lowering）是必需的。

### 5.2.2 三种回调变体

`walk` 有三种回调签名，对应不同的控制能力：

1. **`void` 回调**（[Visitors.h:132-133](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h#L132)）：最简单，遍历完所有 op，不能中断。只能在 post-order 时 erase 当前 op（见 5.5 节）。

```cpp
op->walk([](Operation *op) { /* 处理 op */ });
```

2. **`WalkResult` 回调**（[Visitors.h:195-198](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h#L195)）：可以返回 `advance`（继续）、`interrupt`（停止遍历）、`skip`（跳过子 op 的递归）。**pre-order erase + skip 也合法**（见 5.5 节）。

```cpp
op->walk([](Operation *op) -> WalkResult {
  if (someCondition) return WalkResult::interrupt();
  return WalkResult::advance();
});
```

3. **`WalkStage` 回调**（[Visitors.h:389-399](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h#L389)）：一个 op 会被回调 N+1 次（N = 它的 region 数），用 `WalkStage` 跟踪"现在走到第几个 region 之前/之后"。用于需要在每个 region 边界做事的高级场景。

日常 90% 的场景用前两种。

### 5.2.3 一个实际例子

最典型的 walk 用法——遍历所有 `arith::AddIOp`：

```cpp
moduleOp->walk([](arith::AddIOp addOp) {
  // 对每个 addi op 做处理
  if (addOp.getLhs().getDefiningOp<arith::ConstantOp>())
    // 左操作数是常量，可以做某种优化...
});
```

这个 lambda 会被 walk 反复调用，每次传一个匹配类型的嵌套 op。这是 MLIR 优化 pass 的标准写法——Ch9 讲的 matchAndRewrite 驱动本质就是这种遍历 + 匹配 + 变换的循环。

## 5.3 数据流遍历：use_iterator 与 user_iterator

第 4 章讲了 use-list 的物理实现。现在看怎么遍历它。

### 5.3.1 从 Value 出发：遍历使用者和使用边

从一个 Value 出发，可以双向遍历数据依赖：

```cpp
Value v = ...;

// 下游：谁在使用 v？（遍历 DU chain）
for (OpOperand &use : v.getUses()) {
  Operation *user = use.getOwner();
  // use.getOperandNumber() 告诉你这是 user 的第几个操作数
}

// 同样效果，但只关心 user 不关心 use 的细节：
for (Operation *user : v.getUsers()) {
  // user 是使用 v 的 op
}
```

`getUses()` 返回的 `use_iterator` 直接遍历 Ch4 讲的 `firstUse → nextUse → ...` 链表。`getUsers()` 返回的 `user_iterator` 是它的投射——每个 use 节点映射到它的 owner（[UseDefLists.h:341-355](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L341)）：

```cpp
// user 遍历 = use 遍历 . 映射到 owner
Operation *mapElement(OpOperand &value) const {
  return value.getOwner();
}
```

**一个容易踩的坑**：如果同一个 op 多处引用同一个 Value，`getUsers()` 会**重复返回**同一个 `Operation*`。需要去重时要自己处理。

### 5.3.2 从 Operation 出发：遍历它结果的使用者

从 Operation 侧，可以获取"它的结果被谁用了"：

```cpp
Operation *op = ...;

// 遍历 op 的所有结果的所有 use
for (auto it = op->use_begin(); it != op->use_end(); ++it) { ... }

// 等价的快捷方式
for (Operation *user : op->getUsers()) { ... }

// 查询
bool hasOneUse = op->hasOneUse();
bool noUses = op->use_empty();
```

这两个查询（`hasOneUse`、`use_empty`）在重写（Ch9–Ch10）里极其常用：

- `use_empty()` 判断"这个 op 的结果没人用了"——erase 前的安全检查（Ch10）。
- `hasOneUse()` 判断"这个 op 的结果恰好被一个 op 用"——重写时验证能否安全搬迁（Ch13 的 BufferCastOpFold 就用这个）。

> **编译原理浸润点：数据流分析的物理基础**　Dragon Book 第 9 章的数据流分析都需要"从一个 def 找它的所有 use"——这正是 `getUsers()` 的用途。例如活跃变量分析（live variable analysis）的核心就是"沿着 use-list 判断一个 def 的值在后续是否还被使用"。MLIR 的 use-list 把这个分析所需的能力做成了 O(users) 的遍历，比 Dragon Book 假设的"扫描整个基本块"高效得多。Ch9 讲 matchAndRewrite 时会看到：匹配一个 pattern 经常就是沿着 use-def 链往上看（"这个操作数是不是某个特定 op 的结果"）。

## 5.4 控制流遍历：predecessor 与 successor

控制流遍历处理 Block 之间的跳转。这里有个精巧的设计：**predecessor 和 successor 都不是直接存储的边，而是从 use-list 推导出来的**。

### 5.4.1 求后继（successor）

后继信息存在 Block 的 terminator 上——terminator op 的 `BlockOperand[]` 就是它的后继列表。源码（[Block.cpp:258-261](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L258)）：

```cpp
Block *Block::getSuccessor(unsigned i) {
  assert(i < getNumSuccessors());
  return getTerminator()->getSuccessor(i);
}
```

所以"求后继"= 取 terminator → 读它的 successor operand。terminator（如 `cond_br`、`br`）通过 `BlockOperand` 持有指向后继 Block 的边。

### 5.4.2 求前驱（predecessor）：从 use-list 推导

**前驱没有直接存储的边**。它靠的是第 3 章讲的"Block 身兼二职"——Block 继承 `IRObjectWithUseList<BlockOperand>`，持有 BlockOperand 的 use-list 头。所有指向这个 Block 的 BlockOperand（即所有 terminator 的 successor operand）都挂在它的 use-list 上。

源码（[Block.h:231-233](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h#L231)）：

```cpp
pred_iterator pred_begin() {
  return pred_iterator((BlockOperand *)getFirstUse());
}
```

`pred_begin()` 就是把 `getFirstUse()`（Ch4 讲的 use-list 头）转成 `BlockOperand*` 当迭代器。映射到前驱 Block 的逻辑在 [Block.cpp:324-326](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L324)：

```cpp
Block *PredecessorIterator::unwrap(BlockOperand &value) {
  return value.getOwner()->getBlock();   // BlockOperand 的 owner 是 terminator op，
                                         // 它所在的 Block 就是前驱
}
```

翻译一下：BlockOperand 的 owner 是某个 terminator op（它持有这个 successor 边），那个 op 所在的 Block 就是当前 Block 的前驱。

**这个设计极其巧妙**：MLIR 没有单独存储"前驱边"，而是**复用了 use-list 机制**——terminator 的 successor operand 本身就是 BlockOperand，它自然挂在目标 Block 的 use-list 上。于是"求前驱"就是"遍历 use-list"。这把 Ch4 讲的 use-list 机制推广到了控制流，零额外数据结构。

> **编译原理浸润点：CFG 的边表示**　Dragon Book 第 8.4 节用"邻接表"或"邻接矩阵"表示 CFG 的边。MLIR 选择了更省内存的方式——复用 use-list，让控制流边成为数据流边的特例（BlockOperand 是 OpOperand 的"控制流版本"）。这是"用一种机制统一多种边"的设计哲学，和 Ch2 讲的"一切皆 Operation"同源。

### 5.4.3 单一/唯一前驱的细微差别

[Block.h:248-255](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h#L248) 提供了三个相关查询，初学者容易混：

| API | 语义 | 何时返回 null |
|---|---|---|
| `getSinglePredecessor()` | 有且仅有 **1 个 BlockOperand 边** | 重复边（同一个前驱跳两次）→ null |
| `getUniquePredecessor()` | 所有边来自**同一个 Block** 即可 | 多个不同前驱 → null |
| `hasNoPredecessors()` | 无任何边 | 入口块返回 true |

`getSinglePredecessor` 和 `getUniquePredecessor` 的差别在于"重复边"：如果同一个 Block 通过两条 `BlockOperand` 跳到当前 Block（如 `cond_br` 的两个分支都指向同一个块），前者返回 null（因为有 2 条边），后者返回那个 Block（因为所有边来自同一个块）。这个细节在重写时（Ch10）会影响能否安全删除一个 Block。

## 5.5 变更契约：遍历时能不能改 IR

这是本章的重点，也是编译器写作者最容易踩坑的地方。

### 5.5.1 问题的本质

考虑这段代码：

```cpp
// 危险：边遍历边 erase
for (Operation &op : *block) {
  if (shouldErase(op))
    op.erase();    // ← 遍历器还指着这个 op！它被销毁了，下一步 ++ 会野指针
}
```

`op.erase()` 销毁了当前 op（Ch10 详解 erase），但 for 循环的迭代器还指着它，下一次 `++` 就会访问已释放内存——典型的 use-after-free。这是遍历时改 IR 的根本难题。

### 5.5.2 walk 的变更契约

[Operation.h:767-770](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L767) 给出 walk 的精确规则：

> A callback on a block or operation is allowed to erase that block or operation if either:
> - the walk is in post-order, or
> - the walk is in pre-order and the walk is skipped after the erasure.

翻译成表格：

| 场景 | PostOrder（默认） | PreOrder |
|---|---|---|
| **原地 erase 当前 op/block** | ✅ 允许 | ✅ 允许，但须 `return WalkResult::skip()` |
| **原地替换当前 op** | ❌ 文档未保证 | ❌ 文档未保证 |
| **修改其他未访问的 op** | ⚠️ 无文档保证（应 collect-and-mutate-after） | ⚠️ 无文档保证 |
| **中断遍历** | `return WalkResult::interrupt()` | `return WalkResult::interrupt()` |

为什么 post-order 能 erase 而无需 skip？因为 post-order 是"先递归子 op，回来再回调自己"——回调自己时，子 op 已经遍历完了，erase 自己不会影响后续遍历（它已经没有"后续"了，或者下一个是兄弟节点，不受影响）。

pre-order 则不同——回调自己时，子 op 还没遍历。如果 erase 自己后再继续遍历它的子 op，那些子 op 已经跟着销毁了。所以必须 `skip()` 跳过子 op 的递归。

### 5.5.3 early_increment：遍历器的预取机制

walk 之所以能支持"边遍历边 erase"，靠的是源码里的这一行（[Visitors.h:178-180](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h#L178)）：

```cpp
for (auto &nestedOp :
     llvm::make_early_inc_range(Iterator::makeIterable(block)))
```

`llvm::make_early_inc_range` 是 LLVM 的"提前自增"适配器——在**进入循环体前就预取下一个迭代器**。这样即使循环体里把当前节点 erase 了，遍历器手里还握着"下一个节点的位置"，不会野指针。

对比普通 for 循环：

```cpp
// 普通 for：先进入循环体，体结束才 ++it
for (auto it = ...; it != end; ++it) {
  it->erase();   // 危险：it 还指着被销毁的节点，++it 访问已释放内存
}

// early_inc_range：进入循环体前先 ++it（预取）
for (auto &x : make_early_inc_range(...)) {
  x.erase();     // 安全：迭代器已经在下一个位置
}
```

这是 LLVM 处理"遍历时改容器"的标准手法。MLIR 的 walk 复用了它，所以 walk 回调里可以安全 erase 当前 op（在 post-order 或 pre-order+skip 的前提下）。

### 5.5.4 推荐模式：collect-and-mutate-after

尽管 walk 支持"边遍历边 erase"，MLIR 社区还有一个更安全的模式——**collect-and-mutate-after**（先收集，后变换）：

```cpp
// 安全模式：先收集所有要 erase 的 op，遍历结束后统一 erase
SmallVector<Operation *> toErase;
moduleOp->walk([&](Operation *op) {
  if (shouldErase(op))
    toErase.push_back(op);
});
for (Operation *op : toErase)
  op->erase();
```

这个模式把"遍历"与"变换"分离，彻底避免迭代器失效问题。代价是要额外的 vector 存中间结果，但安全性高得多。Ch9 的 matchAndRewrite 驱动、Ch13 的 BufferCastOpFold 都隐含了这个思路。

> **编译原理浸润点：迭代器失效**　Dragon Book 假设"分析阶段不改 IR"（分析 pass 和变换 pass 分开），所以不讨论迭代器失效。但任何把分析与变换揉在一起的现实编译器（MLIR、LLVM）都必须面对。MLIR 的解法是分层的：底层用 `make_early_inc_range` 让单次遍历内能 erase，上层用 collect-and-mutate-after 让复杂变换更安全。这两层解法对应编译器工程的两个层次——数据结构层（迭代器）与算法层（变换模式）。

## 5.6 遍历速查表

把本章的遍历 API 汇总成一个速查表：

| 想做什么 | API | 复杂度 | 遍历什么 |
|---|---|---|---|
| 递归遍历 IR（DFS） | `op->walk(callback)` | O(N) | 所有嵌套的 op/block/region |
| 遍历 op 的所有 use | `op->use_begin()/use_end()` | O(uses) | 所有使用此 op 结果的 OpOperand |
| 遍历 op 的所有 user | `op->getUsers()` | O(uses) | 每个 result 的所有使用者（可能重复） |
| 遍历 block 前驱 | `block->getPredecessors()` | O(edges) | 所有指向 block 的 BlockOperand |
| 遍历 block 后继 | `block->getSuccessors()` | O(edges) | terminator 的 BlockOperand[] |
| 遍历 block 内 op | `for (auto &op : *block)` | O(ops) | Block 的 iplist |
| 遍历 region 内 block | `for (auto &b : region)` | O(blocks) | Region 的 iplist |

读 IR 的代码几乎都是查这张表。后续章节（Ch9 matchAndRewrite、Ch13 综合）会反复用到这些遍历。

---

## 编译原理浸润点回顾

1. **遍历是编译器基础设施**：本章主题。分析、优化、lowering 都是遍历 + 变换。
2. **数据流分析的迭代算法**：Dragon Book 第 9 章。`getUsers()` 把"找 def 的所有 use"做成了 O(users) 遍历。
3. **CFG 的边表示**：predecessor 从 use-list 推导，复用数据流边机制表示控制流边。这是"统一机制"的体现。
4. **迭代器失效问题**：Dragon Book 不讨论，但现实编译器必须面对。MLIR 用 `make_early_inc_range`（数据结构层）+ collect-and-mutate-after（算法层）两层解法。
5. **侵入式数据结构让遍历高效**：iplist（Ch3）和 use-list（Ch4）都是侵入式的，遍历它们零额外开销。

---

## 本章关键结论

1. **MLIR 有三类遍历**：结构遍历（walk，深度优先走结构树）、数据流遍历（use/user，沿 use-list）、控制流遍历（pred/succ，沿 Block 跳转边）。
2. **walk 是深度优先递归**：遍历 Operation 的每个 Region → 每个 Block → 每个 Operation，递归下去。支持 PreOrder/PostOrder，默认 PostOrder。
3. **predecessor 从 use-list 推导**：Block 的 BlockOperand use-list 就是它的前驱边，无需单独存储。这是"统一机制"的体现。
4. **遍历时改 IR 有精确契约**：post-order 可 erase 当前 op；pre-order 须 erase 后 `return WalkResult::skip()`。靠 `make_early_inc_range` 预取迭代器实现。
5. **推荐 collect-and-mutate-after 模式**：复杂变换先收集再统一处理，安全性最高。

---

## 下一章预告

第一到第四部分（Ch0–Ch5）讲完了"IR 长什么样、怎么遍历"——这是**读** IR。从下一章开始进入第五部分（**写** IR 的构建）。第 6 章回答：一段前端解析出的 AST，怎么用 MLIR 的 C++ API 织成一棵 IR 树？这里有一个贯穿构建过程的心智模型——**两类挂接**：结构挂接（把 op 装进 block）+ 数据流挂接（把 value 接成 operand）。理解了这两类挂接，就理解了所有 IR 构建动作的本质。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-IR-Node组织与遍历插入删除教程.md` §3（walk 三种 variant、变更契约、use_iterator、predecessor/successor）——**保留并重组**
- 遍历速查表来自原文附录 A
- 编译原理铺垫（数据流分析迭代算法、迭代器失效问题）为本书新增

## 参考文献

- **[Aho 2006]** Dragon Book，第 9 章（数据流分析的迭代算法，需要遍历 IR）。
- **[Lattner 2020]** Lattner et al. "MLIR"，walk 与变更契约的工程设计。
