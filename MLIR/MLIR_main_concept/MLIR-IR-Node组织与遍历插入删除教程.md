# MLIR IR 中的 Node 组织：结构、遍历、插入与删除

**基于 LLVM 19.1.7 源码的完整教程**

---

## 摘要

MLIR 中的"图"并非平坦的邻接表，而是一个**由四类 Node（Operation / Block / Region）通过 Intrusive 链表和侵入式 use-def 链交织而成的分层结构**。四种连接机制各自独立运转，又互相制约：

- **「数据流边」(use-def 链)**：OpOperand → Value，intrusive doubly-linked，维护 SSA 数据依赖。
- **「结构边」(iplist)**：Operation ∈ Block，Block ∈ Region — LLVM `iplist`，维护拓扑与嵌套。
- **「控制流边」**：BlockOperand → Block（terminator operand），借 use-def 链复用的 predecessor 遍历。
- **「零架构边」(Operand vs Attribute 语义分野)**：只有 operand 参与数据流图 — 这是 MLIR 可分析、可重写的根基。

本教程以每种**遍历**（walk / result-use / predecessor）、**插入**（create + 自动入链 + setInsertionPoint）、**删除**（erase / remove / RAUW）为主线，逐个打开源码，连接数据结构与算法。最后以 NorthStar 教程 CH-9/CH-14 的 `BufferCastOpFold` → walk() → 匹配 → RAUW → double-erase 为贯穿案例，完成从原理到实践的闭环。

**关键词**：MLIR；IR Node；use-def chain；intrusive list；iplist；遍历；walk；RAUW；Operation Lifetime

---

## 目录

- ## 0. 前置知识（图论 · SSA · 四类边分类学）
- ## 1. 四类核心 node
  - 1.1 Operation — 基本计算单元
  - 1.2 Value — SSA 值的两面：在「定义」处的「储值容器」和一个侵入式链表「头」
  - 1.3 Block — 控制流容器
  - 1.4 Region — 块嵌套容器
- ## 2. 物理内存布局
  - 2.1 Operation 的前缀结果 + 尾分配存储
  - 2.2 `IROperandBase` 侵入式双向链表的「反直觉」back 设计
  - 2.3 iplist — Block 内的 Operation、Region 内的 Block 如何链接
- ## 3. 遍历四种路径
  - 3.1 Walk：递归深度优先遍历
  - 3.2 数据流遍历：use_iterator 和 user_iterator
  - 3.3 控制流遍历：predecessor / successor
  - 3.4 遍历中的变更契约 (WalkResult::skip/interrupt + 安全 erase 规则)
- ## 4. 插入 — Operation 诞生与入链
  - 4.1 Operation::create 的完整路径 (4 个重载 → 一次 malloc)
  - 4.2 操作数自动入链：insertInto 含图解
  - 4.3 OpBuilder 的插入点管理
- ## 5. 删除 — erase / remove / RAUW 全路径
  - 5.1 erase vs remove：生命周期的不同时刻
  - 5.2 ~Operation 的双重断言 (block==nullptr, use_empty)
  - 5.3 RAUW + erase 的标准三步法
  - 5.4 常见反模式
- ## 6. 综合案例：BufferCastOpFold
  - 6.1 实战走读：walk → match → RAUW → double-erase
  - 6.2 模式注册与 pass 管道内的调用链
- ## 附录
  - A. 遍历/变更速查表
  - B. 常见错误 checklist
  - C. 关键源码文件索引

---

## 0. 前置知识

### 0.1 编译器 IR 的"节点与边"

任何编译器的中间表示，都可以抽象为一幅**图**：

$$G = (V, E)$$

图模型的表达能力，决定了编译优化能做多少事。常见的三种抽象：

| 抽象层 | 典型实现 | 特点 |
|---|---|---|
| **AST（语法树）** | Clang AST | 形如源码，嵌套深，适合类型检查；不适合数据流分析 |
| **三地址码** | LLVM IR, MLIR SSA | 扁平化，每条指令 1-2 个输入、1 个输出；适合数据流优化 |
| **图 IR** | Sea-of-Nodes, MLIR 的 use-def 链 | 显式建模数据依赖边；自动暴露并行度 |

MLIR 选择了**类三地址码 + 任意多边 (N-ary) 的 SSA 图**：每条 Operation 可以有多个输入 (operands) 和多个输出 (results)，连通性完全由 use-def 链定义。

### 0.2 SSA 形式速览

**Static Single Assignment** 是 LLVM 和 MLIR 的基础：

> 每个值在其作用域内**只被定义一次**（"static"），这个定义必须**支配**所有使用点（"dominance"）。

好处：
- 每个 use 的唯一定义立即可得 — **O(1) 获取定义者**
- 每个 def 的所有用户也可遍历 — **O(users) 枚举使用者**
- 死代码消除（DCE）：遍历每条 def-use 链，标记活跃性，删除无可达用户的值

MLIR 直接以**双向二部图**实现这个模型：`Value` 持有 `firstUse` 链表头，指向所有使用它的 `OpOperand`；每个 `OpOperand` 持有指针回到它引用的 `Value`。这就是 **use-def 链**。

### 0.3 四类"边"的分类学**

MLIR IR 中存在**四种连接关系**，各自使用截然不同的数据结构，却共同构成 IR 的完整图景：

```text
Operation A                       Operation B
+-----------+                     +-----------+
| OpResult  | --------> OpOperand 0 |            ← 数据流边 (use-def chain)
|           |                OpOperand 1 |            ← 也是数据流边
+-----------+                     +-----------+
       |                                |
       v  (属于 Block iplist)            v  (属于 Block iplist) ← 结构边 (iplist)
  +-------- Block --------------------+
  |                                    |
  +------------------------------------+
       |
       v  (属于 Region 的 Block iplist)   ← 结构边 (iplist)
  +----- Region ---------------------+
  |                                    |
  +------------------------------------+

TerminatorOp in Block A  --BlockOperand-->  Block B   ← 控制流边 (BlockOperand + use-list)
```

**这四种边是理解 MLIR node 组织结构的核心框架**。下面逐一展开。

---

## 1. 四类核心 Node

### 1.1 Operation — 图的基本节点

[`Operation.h:31-88`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h) 是 MLIR 中最核心的类。它的声明：

```cpp
// Operation.h:84-88
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, detail::OperandStorage,
                                    detail::OpProperties, BlockOperand,
                                    Region, OpOperand>
```

三个关键点：

1. **继承 `ilist_node_with_parent<Operation, Block>`** — Operation 本身就是一个双向链表节点（通过 LLVM 的 intrusive list），嵌在 Block 内。所以插入/移除 Operation 到 Block 是 O(1) 的。

2. **继承 `TrailingObjects<...>`** — 这是 LLVM 的尾分配 (trailing objects) 框架，允许在堆上一次性分配 Operation 主体 + OperandStorage + OpProperties + BlockOperand + Region + OpOperand，零额外 malloc。

3. **`alignas(8)`** — 尾分配的指针算术要求对齐。

Operation 的私有字段（[Operation.h:1035-1066](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h)）：

```cpp
private:
  Block *block = nullptr;                             // :1035
  Location location;                                  // :1039
  mutable unsigned orderIndex = 0;                    // :1043
  const unsigned numResults;                          // :1045
  const unsigned numSuccs;                            // :1046
  const unsigned numRegions : 23;                     // :1047  （23 位域）
  bool hasOperandStorage : 1;                         // :1052
  unsigned char propertiesStorageSize : 8;            // :1057  （存储 size/8）
  OperationName name;                                 // :1063
  DictionaryAttr attrs;                               // :1066
```

注意这些**都不是直接存储 operands/results/regions/successors** — 它们全都在前缀区或尾分配区中。这就是 MLIR Operation 最精巧的内存设计（详见 §2.1）。

**API 快速一览**：

| 概念 | 获取方式 | 返回类型 | 存储位置 |
|---|---|---|---|
| operands | `getOperands()` | `OperandRange` | 尾分配的 `OpOperand[]` |
| results | `getResults()` | `ResultRange` | **前缀区**（反向存储） |
| regions | `getRegions()` | `MutableArrayRef<Region>` | 尾分配的 `Region[]` |
| successors | `getSuccessors()` | `SuccessorRange` | 尾分配的 `BlockOperand[]` |
| attributes | `getAttrDictionary()` | `DictionaryAttr` | 成员 `attrs` |
| properties | `getPropertiesStorage()` | `OpaqueProperties` | 尾分配的 `OpProperties` blob |
| uses | `use_begin()/use_end()` | `use_iterator` | 委托给 `getResults().use_begin()` |
| users | `user_begin()/user_end()` | `user_iterator` | `ValueUserIterator` 投射自 uses |

### 1.2 Value — SSA 值的两面性

[`Value.h:96-254`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h) 是一个**轻量级句柄类**（类似智能指针）：

```cpp
class Value {
  detail::ValueImpl *impl;   // :253 — 仅有一个指针成员
public:
  Type getType() const;
  Operation *getDefiningOp() const;
  void replaceAllUsesWith(Value newValue);
  use_iterator use_begin() const { return impl->use_begin(); }
  use_iterator use_end() const { return use_iterator(); }
  bool use_empty() const;
  bool hasOneUse() const;
};
```

`Value` 有两种**子类型**，通过 `ValueImpl` 的多态区分：

1. **`OpResult`** — Operation 的结果。实际存储在前缀区的 `InlineOpResult` / `OutOfLineOpResult` 对象中。
2. **`BlockArgument`** — Block 的参数。存储在 Block 的 `std::vector<BlockArgument>` 中。

**关键是这个双面角色的理解**：

```text
Value:
  (1) 在「产生」端 → 它是一个结果（OpResult）或参数（BlockArgument）
      [定义在此，类型固定，不能赋值只能被使用]

  (2) 在「消费」端 → 它是一个「use-list 的头部」
      [通过 impl->firstUse 链，追踪所有指向它的 OpOperand]
```

`ValueImpl` 作为 `IRObjectWithUseList<OpOperand>` 的子类（[UseDefLists.h:189-294](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h)），内嵌 `firstUse` 头指针。所有从这个 Value 为操作数的 `OpOperand` 都挂在这个链表上。

### 1.3 Block — 控制流容器 + 参数列表

[`Block.h:30-31`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h):

```cpp
class Block : public IRObjectWithUseList<BlockOperand>,
              public llvm::ilist_node_with_parent<Block, Region>
```

Block 同时身兼二职：
- **`IRObjectWithUseList<BlockOperand>`** — 作为 predecessor 遍历的"头部"（terminator 的 BlockOperand 挂在这个链上）
- **`ilist_node_with_parent<Block, Region>`** — 自己又是一个链表节点，挂在父 Region 的 `iplist<Block>` 上

Block 的三个私有成员（[Block.h:392-398](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h)）：

```cpp
llvm::PointerIntPair<Region*, 1, bool> parentValidOpOrderPair;  // 父 Region + order 有效位
OpListType operations;      // = llvm::iplist<Operation>
std::vector<BlockArgument> arguments;
```

Block 的 **使用链用于前驱遍历**，而非数据依赖：

```cpp
// Block.h:231-233
pred_iterator pred_begin() {
  return pred_iterator((BlockOperand *)getFirstUse());
}
```

Pred 遍历是对 Block 的 BlockOperand use-list 的投射（[Block.cpp:324-326](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp)）：

```cpp
Block *PredecessorIterator::unwrap(BlockOperand &value) {
  return value.getOwner()->getBlock();
}
```

### 1.4 Region — Block 嵌套的容器

[`Region.h:24-26`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Region.h):

```cpp
class Region {
private:
  BlockListType blocks;             // = llvm::iplist<Block>   (:331)
  Operation *container = nullptr;   // 指向包含此 region 的 Op  (:334)
};
```

> Region **自身不是**一个 Value 或 IRObject — 它没有 use-list、没有 Location、没有名字。它是一个纯的"块集合容器"。

关键事实：
- `getContext()` / `getLoc()` 委托给 `container`（父 Op），未 attach 则 crash（[Region.cpp:24-27](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Region.cpp)）
- `getRegionNumber()` 通过 `this - &getParentOp()->getRegions()[0]` 的指针减法 O(1) 求得（[Region.cpp:65](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Region.cpp)）——因为 Region 对象在 Operation 尾分配区中是**连续**存储的
- `getArguments()` / `addArgument()` / `insertArgument()` / `eraseArgument()` **全部直接转发给 front()**（入口块）

---

## 2. 物理内存布局

### 2.1 Operation 的尾分配 (trailing objects)

MLIR 中也许最精巧的代码片段 — [Operation.h:42-67](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h) 的注释：

```text
低地址 →                                                       → 高地址

[OutOfLineOpResults...][InlineRes4..0][ Operation body ][OperandStorage][OpProperties][BlockOperand*s][Region*r][OpOperand*m]

<-- 前缀区 (反向) -->            <------------------------ 尾分配区 (正向) ------------------------->
                        ^
                  Operation* 指向此处（create 的返回值）
```

设计决定：

1. **结果反向存储在 Operation\* 之前**，为了支持前 5 个 "inline" 结果和后 6+ 个 "out-of-line" 结果的变长模式。
2. **Operation\* 指针指向主体开头，而非内存块开头**。`destroy()` 必须后退 `prefixAllocSize` 才能拿到原始 `malloc` 指针（[Operation.cpp:208-215](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp)）：

```cpp
void Operation::destroy() {
  char *rawMem = reinterpret_cast<char *>(this)
    - llvm::alignTo(prefixAllocSize(), alignof(Operation));
  this->~Operation();
  free(rawMem);
}
```

3. **一次 malloc，零后续堆分配**。`byteSize + prefixByteSize` 一次计算完成（[Operation.cpp:113-114](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp)）。

### 2.2 use-list 的「反向指针」设计 — 一句入魂

[`UseDefLists.h:96-113`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h) 是所有 IR 节点数据流组织的基础：

```cpp
class IROperandBase {
  IROperandBase *nextUse = nullptr;    // 指向 use-list 中的下一个节点
  IROperandBase **back = nullptr;      // 指向"指向本节点的指针的地址"
  Operation *const owner;              // 回到拥有此 operand 的 Op
};
```

**这是整个报告最难懂、也最重要的 3 个字段。**

#### 直觉化讲解

首先画出链表：

```text
Value V 的 use-list:
V.firstUse ──→ [Op A 的 operand] ──→ [Op B 的 operand] ──→ [Op C 的 operand] ──→ nullptr

问：「B 的前驱是谁？」
答：不是"A 的地址"，是——"A 的 nextUse 字段" 这个指针（的那个位置是谁指过来的）
```

`back` 的类型是 `IROperandBase **` —— **指针的指针**。它的值是：**“谁指向了本节点”的那个指针的地址**。

| 节点 | `back` 的内容 | 解读 |
|---|---|---|
| head (A) | `&(V.firstUse)` | “指向我的是 V 的 firstUse 字段” |
| middle (B) | `&(A.nextUse)` | “指向我的是 A 的 nextUse 字段” |
| tail (C) | `&(B.nextUse)` | “指向我的是 B 的 nextUse 字段” |

**为什么不是指向前驱节点的指针？**

因为 "头部是由 `firstUse` 指向的" 和 "非头部是由前驱的 `nextUse` 指向的" 是**两种不同类型的字段** — 但仍然都是 `IROperandBase*` 类型的存储位置。`back` 统一用 `IROperandBase**` 指向这两个情况，使**头节点删除无需条件判断**：

```cpp
void removeFromCurrent() {
  if (!back) return;
  *back = nextUse;                 // 无论是头还是中间，这一行就够了
  if (nextUse)
    nextUse->back = back;
}
```

追溯：
- 对头 A：`*back = nextUse` → `*(&V.firstUse) = nextUse` → `V.firstUse = B`
- 对中间 B：`*back = nextUse` → `*(&A.nextUse) = nextUse` → `A.nextUse = C`

**O(1)，无 if。** 这就是指针的指针 (double pointer) 的威力。

#### insertInto 的动态演示

[`UseDefLists.h:96-103`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h):

```cpp
template <typename UseListT>
void insertInto(UseListT *useList) {
  back = &useList->firstUse;           // ① 我让 back 指向链表的 firstUse 字段
  nextUse = useList->firstUse;         // ② 我接手当前的头结点
  if (nextUse)
    nextUse->back = &nextUse;          // ③ 老头的 back 现在指到我的 nextUse
  useList->firstUse = this;            // ④ 链表的头现在指向我
}
```

逐步推演（链表之前是空的，然后插入节点 S）：

```
插入前:  V.firstUse = nullptr

Step ①: S.back = &V.firstUse
Step ②: S.nextUse = nullptr       (因为 firstUse 是 nullptr)
Step ③: (跳过，nextUse 为空)
Step ④: V.firstUse = S

结果:  V.firstUse → S
       S.back = &V.firstUse        ← 记住谁指向了它
```

再插入节点 T（链表已有 S）：

```
插入前:  V.firstUse → S → ∅

Step ①: T.back = &V.firstUse
Step ②: T.nextUse = S
Step ③: S.back = &T.nextUse        ← 现在 S 记住：是 T 在指向我
Step ④: V.firstUse = T

结果:  V.firstUse → T → S → ∅
       T.back = &V.firstUse
       S.back = &T.nextUse
```

现在删除 T（头部删除）：

```
操作: T.removeFromCurrent()
  *back = nextUse   → *(&V.firstUse) = S   → V.firstUse = S
  S.back = back     → S.back = &V.firstUse

结果:  V.firstUse → S → ∅
       S.back = &V.firstUse  ← 重建为"我是头"
```

删除 S（现在是惟一的节点，也是头部）：

```
操作: S.removeFromCurrent()
  *back = nextUse   → *(&V.firstUse) = nullptr   → V.firstUse = nullptr
  (nextUse 为空，无后来者)

结果:  V.firstUse = nullptr  ← 链表为空
```

### 2.3 iplist — O(1) 的结构边

[`Block.h:134`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h) 和 [Region.h:44](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Region.h):

```cpp
// Block
using OpListType = llvm::iplist<Operation>;
OpListType operations;

// Region
using BlockListType = llvm::iplist<Block>;
BlockListType blocks;
```

对比标准库与 iplist：

| 容器 | 插入/删除 | 节点存储 | 节点内嵌钩子 |
|---|---|---|---|
| `std::list<T>` | O(1) | 外部分配的 list_node | 否，节点被包装 |
| `llvm::iplist<T>` | O(1) | 节点**自己就是** list_node | 是，通过继承 `ilist_node_with_parent` |

`iplist` 通过 **Intrusive 设计**实现 O(1) 插入/删除 + 零外部开销。Block 移除一个 Operation 时，`iplist` 直接通过内部钩子（[BlockSupport.h:225-238](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/BlockSupport.h)）更新双向链接并调用 `deleteNode`（其中触发 `destroy()`）。

**iplist 与 use-list 的对比**：

| 属性 | iplist (Operation∈Block) | use-list (OpOperand→Value) |
|---|---|---|
| 方向 | 双向（prev + next） | 大致双向（nextUse + back） |
| 存储 | 节点内嵌在 Operation 的 `ilist_node` 基类 | 节点内嵌在 OpOperand 的 `nextUse`/`back` 字段 |
| 头节点 | Block 内嵌的哨兵 (sentinel) | `IRObjectWithUseList::firstUse` |
| 删除 | erase(iplist::iterator) | removeFromCurrent (无需 iterator) |
| 遍历 | iplist::iterator (线性) | 从头开始遍历 nextUse |

---

## 3. 遍历四种路径

### 3.1 Walk：递归深度优先遍历

[`Visitors.h:127-187`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h) 的实现：

```cpp
template <typename Iterator>
void walk(Operation *op, function_ref<void(Operation *)> callback,
          WalkOrder order) {
  if (order == WalkOrder::PreOrder)
    callback(op);

  for (auto &region : Iterator::makeIterable(*op)) {
    for (auto &block : Iterator::makeIterable(region)) {
      for (auto &nestedOp :
           llvm::make_early_inc_range(Iterator::makeIterable(block)))
        walk<Iterator>(&nestedOp, callback, order);     // ← 递归
    }
  }

  if (order == WalkOrder::PostOrder)
    callback(op);                                        // ← 后序回调
}
```

三种 variant：
- **`void` 回调**（[Visitors.h:132-133](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h)）：只能在 post-order 时 erase 当前 op。
- **`WalkResult` 回调**（[Visitors.h:195-198](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h)）：pre-order erase + skip 也合法。
- **`WalkStage` 回调**（[Visitors.h:389-399](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h)）：op 被回调 N+1 次（N=region 数），配合 WalkStage 跟踪当前进度。

调用模板默认 `WalkOrder::PostOrder`、`ForwardIterator`（[Operation.h:788](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h)）。

**变更契约是经过验证的**（见完整文档 [Operation.h:767-770](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h)）：

> 允许 erase 当前 op/block，当且仅当：(1) post-order，或 (2) pre-order 且 erase 后 return skip()。

### 3.2 数据流遍历 — use_iterator ↔ user_iterator

从一个 Value 出发，可以双向遍历数据依赖：

```cpp
Value v = ...;

// 下游：谁在使用 v？
for (OpOperand &use : v.getUses()) {
  Operation *user = use.getOwner();
  // use.getOperandNumber() 是第几个操作数
}

// 同样效果：
for (Operation *user : v.getUsers()) {
  // user 是使用 v 的 Op
}
```

`user_iterator` 本质上是 `use_iterator` 的投射（[UseDefLists.h:341-355](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h)）：

```cpp
// User 遍历 = use 遍历 . 映射到 owner
Operation *mapElement(OpOperand &value) const {
  return value.getOwner();
}
```

**注意**：如果同一个 Op 多处引用同一个 Value，`getUsers()` 会**重复返回**同一个 `Operation*`。

从 Operation 侧获取被使用信息（[Operation.h:838-848](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h)）：

```cpp
// Op 的所有结果的所有 use
for (auto it = op->use_begin(); it != op->use_end(); ++it) { ... }
// 等价快捷方式
for (Operation *user : op->getUsers()) { ... }
bool hasOneUse = op->hasOneUse();
bool noUses = op->use_empty();
```

### 3.3 控制流遍历 — predecessor / successor

**没有直接存储的边**；两种遍历都是**从 use-list 即时推导**的：

**求后继 (successors)** — 取 block 的 `terminator`，读取其 `BlockOperand[]`（[Block.cpp:339-345](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp)）：

```cpp
// Block.cpp:258-261
Block *Block::getSuccessor(unsigned i) {
  assert(i < getNumSuccessors());
  return getTerminator()->getSuccessor(i);
}
```

**求前驱 (predecessors)** — 遍历 Block 自己的 BlockOperand use-list（[Block.h:231-233](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h)）：

```cpp
// Block.h:231-233 — 前驱是从 Block 作为 IRObjectWithUseList<BlockOperand>
// 的 firstUse 链表迭代得到的
pred_iterator pred_begin() {
  return pred_iterator((BlockOperand *)getFirstUse());
}

// Block.cpp:324-326 — 映射：BlockOperand * ↦ Operation * → Block
Block *PredecessorIterator::unwrap(BlockOperand &value) {
  return value.getOwner()->getBlock();
}
```

**单/唯一 前驱的细微差别**（[Block.h:248-255](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Block.h)）：

```text
getSinglePredecessor()  — 有且仅有 1 个 BlockOperand 边；重复边 → null
getUniquePredecessor()  — 所有边来自同一个 Block 即可（碰撞重复）；多个块 → null
hasNoPredecessors()     — 无任何边
```

### 3.4 遍历期间的变更契约 — [已验证]

[Operation.h:767-770](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h) 给出精确规则：

> A callback on a block or operation is allowed to erase that block or operation if either:
>   * the walk is in post-order, or
>   * the walk is in pre-order and the walk is skipped after the erasure.

**已由独立复核确认** (Visitors.h:195-198 / 322-323 / 351-354 逐行确认)。

实现保证（[Visitors.h:178-180](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h)）：

```cpp
// Early increment here in the case where the operation is erased.
for (auto &nestedOp :
     llvm::make_early_inc_range(Iterator::makeIterable(block)))
```

`llvm::make_early_inc_range` 在进入循环体前**预取**下一个迭代器，即使当前节点被 erase 销毁，遍历仍不会野指针。

| 场景 | PostOrder (默认) | PreOrder |
|---|---|---|
| **原地 erase 当前 op/block** | ✅ 允许 | ✅ 允许，但须 `return WalkResult::skip()` |
| **原地替换当前 op** | ❌ 文档未提及 replace | ❌ 文档未提及 replace |
| **修改其他 op** | ⚠️ 无文档保证（collect-and-mutate-after） | ⚠️ 无文档保证 |
| **中断遍历** | `return WalkResult::interrupt()` | `return WalkResult::interrupt()` |

---

## 4. 插入 — Operation 的诞生与挂链

### 4.1 Operation::create 的完整路径

[`Operation.cpp:82-153`](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp) 是 **THE allocator**——MLIR 所有 Operation 实例化的单一路径。调用图：

```text
create(OperationState)                    [.cpp:34]
  └→ create(..., NamedAttrList + RegionRange) [.cpp:51]  (takes bodies)
       └→ create(..., NamedAttrList + numRegions) [.cpp:67]  (populates defaults)
            └→ create(..., DictionaryAttr + numRegions) [.cpp:82]
                 │
                 ├─ 计算 byteSize = totalSizeToAlloc<OperandStorage, OpProperties,
                 │                             BlockOperand, Region, OpOperand>(...)
                 ├─ 计算 prefixByteSize = alignTo(prefixAllocSize(...), alignof(Operation))
                 ├─ malloc(byteSize + prefixByteSize)            [cpp:114]
                 ├─ placement new Operation(rawMem, ...)          [cpp:118]
                 ├─ placement new InlineOpResult[0..4]            [cpp:127-130]
                 ├─ placement new OutOfLineOpResult[5..]          [cpp:131-132]
                 ├─ placement new Region[0..r]                    [cpp:135-136]
                 ├─ placement new OperandStorage(...)              [cpp:139-142]
                 │   └→ for each operand: placement new OpOperand(owner, value[i])
                 │        └→ IROperand(owner, value) ctor → insertIntoCurrent()
                 ├─ placement new BlockOperand[0..s]              [cpp:145-147]
                 └─ setAttrs(attributes)                          [cpp:150]
```

**关键洞察**：操作数的 use-def 链在 OperandStorage 构造时自动建立 — 不需要额外步骤。

### 4.2 自动入链的机理

[UseDefLists.h:130-133](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h):

```cpp
// 构造即入链
IROperand(Operation *owner, IRValueT value)
    : detail::IROperandBase(owner), value(value) {
  insertIntoCurrent();
}
```

流向：[OpOperand ctor] → `insertIntoCurrent()` → `insertInto(DerivedT::getUseList(value))` → `insertInto`（[UseDefLists.h:96-103](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h)）→ 结果：op 的 operand 始终**挂在该 value 的 firstUse 链表上**。

OpOperand::getUseList 的实现（[Value.h:270-272](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h)）：

```cpp
static IRObjectWithUseList<OpOperand> *getUseList(Value value) {
  return value.getImpl();
}
```

### 4.3 OpBuilder 的插入点

[Builders.h:210-449](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h) — 插入点由一个 `(Block *, Block::iterator)` 对定义：

```cpp
class InsertPoint {
  Block *block = nullptr;
  Block::iterator point;
};

// 三种常用位置：
void setInsertionPoint(Operation *op);           // op 之前
void setInsertionPointAfter(Operation *op);       // op 之后
void setInsertionPointAfterValue(Value val);      // 在定义 val 的 op 之后（或 block 首部若 val 是 BlockArgument）
void setInsertionPointToStart(Block *block);      // block 首
void setInsertionPointToEnd(Block *block);        // block 末

// RAII 还原
InsertPoint saveInsertionPoint() const;
void restoreInsertionPoint(InsertPoint ip);
```

实际使用模式：

```cpp
OpBuilder builder(ctx);
builder.setInsertionPointToStart(&block);
auto op = builder.create<MyOp>(loc, operand1, attr);
// op 已挂链到 block + operand1 的 use-list
```

`OpBuilder::create` 模板（[Builders.h:508-517](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h)）：

```cpp
template <typename OpTy, typename... Args>
OpTy create(Location location, Args &&...args) {
  OperationState state(location,
    getCheckRegisteredInfo<OpTy>(location.getContext()));
  OpTy::build(*this, state, std::forward<Args>(args)...);  // 填充 OperationState
  auto *op = create(state);                                  // Operation::create
  // ... dyn_cast 返回
}
```

---

## 5. 删除 — erase / remove / 析构安全

### 5.1 erase vs remove

[Operation.cpp:539-550](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp):

```cpp
// 从 parent block 移除 + 销毁
void Operation::erase() {
  if (auto *parent = getBlock())
    parent->getOperations().erase(this);   // ← 触发 iplist::erase → deleteNode → destroy()
  else
    destroy();                              // ← 无 parent 就直接 destroy
}

// 仅解除与 parent block 的链接，不销毁
void Operation::remove() {
  if (Block *parent = getBlock())
    parent->getOperations().remove(this);
}
```

**规则**：
- `erase()` = 断开与 block 的链接 + 销毁（free memory）
- `remove()` = 仅断开链接，op 仍存活（用于中途转移，如 clone 后移动）

### 5.2 ~Operation 的双重断言

[Operation.cpp:179-205](MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp):

```cpp
Operation::~Operation() {
  assert(block == nullptr && "operation destroyed but still in a block"); // ← 始终有效
#ifndef NDEBUG
  if (!use_empty()) {                                                    // ← 仅 debug 构建
    emitOpError("operation destroyed but still has uses");
    for (Operation *user : getUsers())
      diag.attachNote(user->getLoc()) << "- use: " << *user << "\n";
    llvm::report_fatal_error("operation destroyed but still has uses");
  }
#endif
  // 析构 OperandStorage、BlockOperand、Region、properties...
}
```

> ⚠️ **关键警告**：use_empty 检查在 **release build 下完全不存在**，有 use 的 op 被 erase 后是静默 use-after-free。在 debug 构建下会触发 `report_fatal_error`。安全路径：erase 前必须 RAUW 所有 results 或将所有 uses 置零。

### 5.3 安全的删除模式：RAUW + erase

[`UseDefLists.h:211-216`](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h):

```cpp
template <typename ValueT>
void replaceAllUsesWith(ValueT &&newValue) {
  assert((!newValue ||
          this != OperandType::getUseList(newValue)) &&
         "cannot RAUW a value with itself");
  while (!use_empty())
    use_begin()->set(newValue);
  // 现在所有 uses 都指向 newValue 了，this->firstUse 是 nullptr
}
```

每个 use 的 `set()`（[UseDefLists.h:163-169](MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h)）：

```cpp
void set(IRValueT newValue) {
  removeFromCurrent();       // 从旧 value 的链表中移除本 operand
  value = newValue;
  insertIntoCurrent();       // 插入新 value 的链表
}
```

**这就是 RAUW 的精妙**：不是拷贝列表然后重写；而是用 `while (!use_empty())` 重复**弹出链表头并重新插入新 value 的链表**。O(uses) 时间，零额外空间。

### 5.4 安全删除清单

```text
✅ 安全路径：
   1. 确保 op 不在 block 中 (或调用 erase 时自动解除)
   2. 对所有 result 调用 result.replaceAllUsesWith(someOtherValue)
      或 dropAllUses()
   3. op->erase() (或依赖 PatternRewriter 的 eraseOp)

❌ 常见反模式：
   1. 对仍有 uses 的 op 直接 erase() → release build 静默 use-after-free
   2. erase 后保留 op 指针继续使用 → 已 free
   3. walk 中 erase 其他未访问的 op → 行为未定义
   4. 调用 delete op → 错误（结构中有前缀，free 不动原始 malloc 指针）
   5. RAUW 一个 value 到自己 → assert
```

PatternRewriter 的 `eraseOp` 自动处理 uses 的前提：**matched op**。对于其他 op，须先调用 `replaceAllUsesWith` 再 erase。

---

## 6. 综合案例：BufferCastOpFold

[`DeviceRegionFusion.cpp:203-234`](MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp)（CH-9）和 [NorthStarCanonicalize.cpp:53-82](MLIR-Tutorial/14-fold_and_canonicalization/src/Dialect/NorthStar/IR/NorthStarCanonicalize.cpp)（CH-14）中有一个贯通匹配-重写-删除的经典案例。

### 6.1 场景

```mlir
// 输入：冗余的 cast→softmax→cast 链条
%t0 = "north_star.buffer_cast"(%a) : (!ns.tensor<...>) -> !ns.tensor<...>
%out = "north_star.buffer_cast"(%t0) : (!ns.tensor<...>) -> !ns.tensor<...>
                                          ↑ 这个 cast 是冗余的（输入就是上面 cast 的输出）
```

### 6.2 逐步走读

**第一步：match() — 遍历 use-def 链检查前置条件**

```cpp
LogicalResult BufferCastOpFold::match(BufferCastOp op) const {
  // ① 遍历每个 operand，验证它们来自同一个 defining op（而不是 BlockArgument）
  Operation *above_cast = nullptr;
  for (OpOperand &operand : op->getOpOperands()) {
    // ← 通过 use-def 的反向：operand.get() → value → getDefiningOp()
    Operation *defOp = operand.get().getDefiningOp();
    if (!defOp || (above_cast && defOp != above_cast))
      return failure();               // operand 来自不同 op 或直接来自 BlockArgument
    above_cast = defOp;
  }

  // ② 类型匹配
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    OpOperand &aboveOperand = above_cast->getOpOperand(index);
    if (result.getType() != aboveOperand.get().getType())
      return failure();
  }

  // ③ 每个中间结果只有一个 use（否则跨越会破坏另外的边）
  for (OpResult result : above_cast->getResults()) {
    if (!result.hasOneUse())          // ← 遍历 uses 检查
      return failure();
  }

  return success();
}
```

`match()` 中的图操作全是**数据流遍历**：
- `getDefiningOp()` (operand → value → 定义 op)：O(1)
- `hasOneUse()` (value → 检查 firstUse 链表恰好一个元素)：O(1)
- `getOpOperand(index)` (op → 通过 OperandStorage 的尾分配数组)：O(1)

**第二步：rewrite() — RAUW + erase**

```cpp
void BufferCastOpFold::rewrite(BufferCastOp op,
                                PatternRewriter &rewriter) const {
  Operation *above_cast = op->getOperand(0).getDefiningOp();

  // ① 将所有结果全部重新引到上面 cast 的输入 (RAUW)
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    Value replacement = above_cast->getOperand(index).get();
    rewriter.replaceAllUsesWith(result, replacement);
    //   在每个 use 上调用 set(newValue)：
    //     removeFromCurrent()   — 从旧 results 链表移除
    //     value = newValue
    //     insertIntoCurrent()   — 插入新 replacement value 的链表
    //   now all uses → replacement
  }

  // ② 删除两个 op（顺序重要 — 先删外层，再删里层）
  rewriter.eraseOp(op);         // op 此时已无 uses（我们刚 RAUW 掉）
  rewriter.eraseOp(above_cast); // above_cast 也已无 uses（match 时 hasOneUse 验证过）
}
```

**第三步：注册与 pass 调度**

[CH-14] 通过 ODS 的 `hasCanonicalizer = 1` 自动注册进 `getCanonicalizationPatterns`，`--canonicalize` pass 可自动拾取。
[CH-9] 显式加入 `FrozenRewritePatternSet` 后馈给 `applyPatternsAndFoldGreedily`。

### 6.3 过程中的图状态

```text
应用前：
  %a ──→ buffer_cast ──→ %t0 ──→ buffer_cast ──→ %out ──→ (下游的 softmax)

match() 后确认可折叠：
  %a ──→ [buffer_cast_above] ──→ %t0 ──→ [buffer_cast_op] ──→ %out ──→ (下游)

rewrite 后（RAUW + erase 完成）：
  %a ──→ (下游的 softmax)  ← 两个 buffer_cast 消失了
  use-def 链上的所有 use 从 %out 重新指向了 %a
```

---

## 参考资料

- [MLIR 官方文档 — Operations](https://mlir.llvm.org/docs/LangRef/#operations)
- [MLIR 官方文档 — Understanding the IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)
- [LLVM 19.1.7 源码](MLIR-Tutorial/third_party/llvm-project/mlir/)
- NorthStar Tutorial (violetDelia/MLIR-Tutorial, Apache 2.0)
- Lattner C, Amini M et al. "MLIR: A Compiler Infrastructure for the End of Moore's Law." arXiv:2002.11054, 2020.

---

## 附录 A：遍历与变更速查表

| 操作 | API | 复杂度 | 遍历什么 |
|---|---|---|---|
| 递归遍历 IR (DFS) | `op->walk(callback)` | O(N) | 所有嵌套的 op/block/region |
| 遍历 op 的所有 uses | `op->use_begin()/use_end()` | O(uses) | 所有使用此 op 结果的 OpOperand |
| 遍历 op 的所有 users | `op->getUsers()` | O(uses) | 每个结果的所有使用者的 Operation* (可能重复) |
| 遍历 block 前驱 | `block->getPredecessors()` | O(edges) | 所有以 block 为目标的 BlockOperand |
| 遍历 block 后继 | `block->getSuccessors()` | O(edges) | terminator 的 BlockOperand[] |
| 遍历 block 内 op | `for (auto &op : *block)` | O(ops) | Block 的 iplist |
| 遍历 region 内 block | `for (auto &b : region)` | O(blocks) | Region 的 iplist |

---

## 附录 B：常见错误 checklist

| # | 错误 | 后果 | 正确做法 |
|---|---|---|---|
| 1 | 对仍有 uses 的 op 直接 `erase()` | Release 下静默 use-after-free | 先 `replaceAllUsesWith`（或 `dropAllUses`），再 erase |
| 2 | `delete op` | 前缀区未释放，原始 malloc 指针丢失 | 使用 `op->erase()` 或 `op->destroy()` |
| 3 | walk 中 erase 其他未访问的 op | 未定义行为（文档未保证） | collect-and-mutate-after |
| 4 | RAUW a value 到自己 | assert 失败 | 检查 `this != getUseList(newValue)` |
| 5 | 在 attach 前调用 `region.getContext()` | assert 失败（container==nullptr） | Region 须先关联到父 Op |
| 6 | 依赖 `getUsers()` 不重复 | 同一 Op 多处用同一 value 会重复 | 需要时跳过去重 |
| 7 | 混淆 `getSinglePredecessor`/`getUniquePredecessor` | 条件分支同一 block 时前者返回 null | 按语义选用 |
| 8 | 对 ZeroOperands trait op 动态添加 operands | assert（`setOperands` checks `operands.empty()`） | 创建时确保 hasOperandStorage=true |

---

## 附录 C：关键源码文件索引

| 文件 | 内容 | 重要行号 |
|---|---|---|
| `mlir/include/mlir/IR/Operation.h` | Operation 类定义 + 尾分配注释 | :42-67 (布局图), :84-88 (继承), :767-770 (walk 变更契约), :1035-1066 (字段) |
| `mlir/lib/IR/Operation.cpp` | Operation::create + 析构 + erase/remove | :82-153 (create), :155-175 (ctor), :179-205 (~Op), :539-550 (erase/remove) |
| `mlir/include/mlir/IR/UseDefLists.h` | IROperandBase + use-list 整模块 | :35-114 (IROperandBase), :87-103 (removeFromCurrent + insertInto), :130-133 (ctor 自动入链), :189-294 (IRObjectWithUseList), :211-216 (RAUW) |
| `mlir/include/mlir/IR/Value.h` | Value + OpOperand | :96-254 (Value), :267-286 (OpOperand), :270-272 (getUseList) |
| `mlir/include/mlir/IR/Block.h` | Block 定义 | :30-31 (继承), :134 (OpListType), :220-265 (terminator/successors/predecessors) |
| `mlir/lib/IR/Block.cpp` | Block 实现 | :243-251 (terminator), :269-291 (getSingle/Unique), :307-318 (splitBlock) |
| `mlir/include/mlir/IR/Region.h` | Region 定义 | :24-26 (class), :44 (BlockListType), :331-334 (私有成员) |
| `mlir/include/mlir/IR/Visitors.h` | Walk + WalkResult + ForwardIterator | :27-58 (WalkResult), :127-187 (walk impl), :305-313 (walk disp) |
| `mlir/include/mlir/IR/Builders.h` | OpBuilder + 插入点 | :210-449 (builder), :329-377 (InsertPoint/InsertionGuard), :508-517 (create 模板) |
| `mlir/include/mlir/IR/BlockSupport.h` | BlockOperand + ilist traits | :30-39 (BlockOperand), :225-256 (ilist traits) |
| `9-rewrite_pattern/.../DeviceRegionFusion.cpp` | BufferCastOpFold (CH-9) | :203-234 (match+rewrite), :237-249 (register), :259-286 (pass run) |————————————————————————————————————————————————————————————————————————————————

