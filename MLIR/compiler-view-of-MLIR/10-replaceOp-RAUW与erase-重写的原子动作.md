# 第 10 章 replaceOp、RAUW 与 erase：重写的原子动作

> **本章位置**　第 9 章把 matchAndRewrite 当黑盒用了——`rewriter.replaceOp` 和 `rewriter.eraseOp` 都是"黑盒 API"。本章打开它们：**replaceOp = RAUW + erase**，而 RAUW 是"把一个 Value 的所有使用者改指向另一个 Value"。本章回到第 4 章的 use-list 数据结构，看那两个原语（removeFromCurrent + insertInto）如何组合成 RAUW 的批量搬迁。同时讲 erase 的安全性——为什么"先 RAUW 再 erase"是必须的。**本章并入原计划的 erase 安全内容，消除"用 erase（Ch9）与懂 erase 风险"的前向依赖**——读者在本章用 erase 时立刻懂风险。
>
> **前置依赖**　第 4 章（use-list 数据结构、back 指针、insertInto/removeFromCurrent）、第 9 章（matchAndRewrite 用 rewriter 改图）。
>
> **编译原理切入**　本章从 **SSA 不变量在重写时的维护**立论。重写一个 op 等于搬迁它所有结果的使用边——这是 SSA 要求的"定义-使用关系"的动态调整。传统编译器重写时重建 use-list 是笨重的（要先收集所有 user 再逐个改），MLIR 用第 4 章的指针的指针（`back`）让搬迁成为零分支的 O(uses) 操作——这是把不变量维护做成零成本抽象的典范。本章还会引出**死代码消除（DCE）**——批量安全删除的编译器经典算法。

---

## 10.1 replaceOp 的两个原子动作

第 9 章把 `rewriter.replaceOp(op, newValues)` 当黑盒。它的内部其实是两步：

```text
replaceOp(op, newValues)  =
  ① RAUW：把 op 每个结果的所有使用者，改指向 newValues 对应的值
  ② erase：删掉 op（此时它的结果已无人用，erase 安全）
```

第 9 章的 `SimplifyRedundantTranspose` 里 `rewriter.replaceOp(外层transpose, {%arg0})` 做的就是：先把 `%t1`（外层 transpose 的结果）的所有使用者（return）改指 `%arg0`，再删掉外层 transpose。

这两步对应 SSA 不变量的动态维护：
- **RAUW** 维护"使用边"——改的是 use-list（DU chain）。
- **erase** 维护"定义点"——删的是 op 本身（及其 result 容器）。

顺序必须如此：**先 RAUW 再 erase**。如果先 erase，op 的 result 还被下游引用，erase 会让下游指向已释放内存（use-after-free）。本章逐步打开这两个动作。

## 10.2 RAUW：把一个 Value 的所有使用者改指向另一个 Value

### 10.2.1 RAUW 是干什么的

**一句话**：把一个 Value `V` 的**所有使用者**（所有引用 `V` 的 op 操作数），全部改成引用另一个 Value `W`。改完之后 `V` 没有任何使用者（`use_empty()`），`W` 多出原来 `V` 的那批使用者。

**直觉类比**：想象 `V` 是一根水管的总阀门，A、B、C 三个水龙头都从 `V` 接水（三条 use 边）。RAUW(`V`, `W`) 就是**把这三个龙头的进水管全部从 `V` 拆下来，改装到 `W` 上**。龙头还是那三个龙头（op 不变），只是水源换了。

**关键性质**（第 4 章的源码会让你看清这几条从何而来）：
- **O(uses) 时间**——只遍历 `V` 的 use 链一次。
- **零内存分配**——不 new 任何新对象，**复用**原有的那些 `OpOperand`（只是把它们从 `V` 的链上摘下来、挂到 `W` 的链上）。
- **不删除任何 op**——只改 use-list 的归属，op 自己一个不少（要删 op 得另外调 `eraseOp`）。

源码入口（[UseDefLists.h:210-216](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L210)）短得惊人：

```cpp
template <typename ValueT>
void replaceAllUsesWith(ValueT &&newValue) {
  assert((!newValue || this != OperandType::getUseList(newValue)) &&
         "cannot RAUW a value with itself");   // 不允许把 V 换成自己
  while (!use_empty())
    use_begin()->set(newValue);                 // ★ 全部戏份在这一行
}
```

### 10.2.2 术语澄清：use-def vs def-use（读这节，否则全篇会绕）

RAUW 到底用的是哪个方向？这是读 RAUW 时最容易卡住的地方。编译原理里有两个**方向相反**的标准术语：

| 术语 | 全称 | 方向 | 含义 | MLIR 里的体现 |
|---|---|---|---|---|
| **UD chain (use-def)** | use-definition chain | **use → def**（向上） | 从一个**使用**出发，找它的**定义** | `operand.getDefiningOp()`；OpOperand 的 `value` 字段 |
| **DU chain (def-use)** | definition-use chain | **def → use**（向下） | 从一个**定义**出发，找它所有的**使用** | Value 的 `firstUse` use-list；`value.getUsers()` |

**关键事实：use-list（firstUse 链）就是 def-use (DU) chain 的物理实现。**

那么 RAUW 到底用的是哪个方向？**答：def-use（DU）。** 因为：

- `RewriterBase::replaceAllOpUsesWith(Operation *from, ValueRange to)` 里的 `from` 是一个 **Operation**，它的**结果**就是**定义（def）**。
- "替换 `from` 的所有使用" = 找出 `from` 结果的所有使用者 = 遍历这些结果的 **DU chain（use-list）**。
- 源码印证（[PatternMatch.cpp:114-120](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp#L114)）：

```cpp
void RewriterBase::replaceAllOpUsesWith(Operation *from, ValueRange to) {
  if (auto *rewriteListener = dyn_cast_if_present<Listener>(listener))
    rewriteListener->notifyOperationReplaced(from, to);
  replaceAllUsesWith(from->getResults(), to);   // ★ 拿 from 的【结果】(def)，对每个走 use-list
}
```

**所以你的推理完全正确：RAUW 改的就是 `from` 的 def-use 边（use-list）；它绝不向上找 `from` 的定义，也绝不改任何定义。** `from` 自己就是定义者（它产出结果），我们只动它结果的"下游使用者"。

> **那为什么正文里有时看到 "use-def 边" 这种说法？** 那是沿用了 MLIR 自己的口语化命名——**连头文件都叫 `UseDefLists.h`**，把"存 use-list 的这套机制"笼统叫 use-def。严格讲这不精确：use-list 是 DU chain。读这篇时，凡看到"改 use-list / 改 use 链"，都指 DU 方向；凡看到"找定义（getDefiningOp）"，才指 UD 方向。

**进阶精确版：同一条边有两个读法（RAUW 两个方向都碰）**。一个 `OpOperand` 代表**一条 SSA 边**，连着使用方 op `U` 和被定义值 `V`。从两头读，名字不同：

```text
                 一条 SSA 边 = 一个 OpOperand
   ┌────────────────────┴────────────────────┐
   从 V（def）侧读                              从 U（use）侧读
   它是 V 的 use-list（DU chain）里的一项         它的 value 字段指向 V（UD link）
   "V 被 U 使用"                                "U 的操作数是 V"
   → V.getUsers() 能枚举到 U                     → U 的这个 operand.getDefiningOp() = V
```

RAUW 实际干两件事，正好横跨这两个方向：

1. **遍历**：沿 `V` 的 **DU chain（use-list）** 逐个找到每条边。← 用 def-use 方向
2. **改写**：把每条边的 `value` 字段（**UD link**）从 `V` 改成 `W`，并把节点从 `V` 的 use-list 搬到 `W` 的 use-list。← 改了 use-def 指针 + 移动 def-use 链归属

所以严格讲 RAUW **既遍历 def-use、又改了 use-def 指针**——但**主导框架是 def-use（use-list）**。

### 10.2.3 RAUW 的三个原子动作

`use_begin()->set(newValue)` 里的 `set()`（[UseDefLists.h:163-169](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L163)）由三个原子动作组成（第 4 章已详解每个，这里回顾）：

```cpp
void set(IRValueT newValue) {
  removeFromCurrent();      // ① 把我从【当前 value】的 use 链上摘下来
  value = newValue;         // ② 改我的 value 字段，改指向【新 value】
  insertIntoCurrent();      // ③ 把我插进【新 value】的 use 链
}
```

这三个动作第 4 章 §4.4-4.6 已详解（removeFromCurrent 无分支三步、insertInto 头插四步、set 的组合）。**RAUW 本质就是 while 循环里反复调 set**——每次 set 搬一条 use 边，循环到 use_empty 为止。

### 10.2.4 完整逐步追踪（不跳任何一步）

设 Value `%V` 有**两个使用者**：op `B` 和 op `A`（B 先建、A 后建，因头插法 B 在链头）。Value `%W` **暂时没有使用者**。目标：`V.replaceAllUsesWith(W)`——让 B、A 都改用 `%W`。

初始的 use 链（注意 B 是链头，A 是链尾）：

```text
  Value %V
  └─ firstUse ──► [OpOperand_B] ──nextUse──► [OpOperand_A] ──nextUse──► nullptr
                       │                          │
                    value=%V                   value=%V
                    owner=B                    owner=A
                    back=&(V.firstUse)         back=&(OpOperand_B.nextUse)
                                                  ↑ A 的 back 指向"前驱 B 的 nextUse 字段地址"

  Value %W
  └─ firstUse ──► nullptr           （空）
```

#### 第一轮循环：处理 OpOperand_B（链头节点）

```cpp
while (!use_empty())        // V.firstUse = B ≠ null → 进入循环
    use_begin()->set(W);    // use_begin() = V.firstUse = OpOperand_B → 调 B.set(%W)
```

`B.set(%W)` 执行三个原子动作：

**① removeFromCurrent()**（摘 B 出 V 的链）：
```cpp
// B.back = &(V.firstUse)，非空
*back = nextUse;     // → *(V.firstUse 的地址) = B.nextUse = OpOperand_A
                     //   即  V.firstUse = OpOperand_A     ★ V 的链头现在是 A 了
if (nextUse)         // B.nextUse = A，非空
    nextUse->back = back;   // → A.back = &(V.firstUse)     ★ A 现在是链头，back 指向 V.firstUse
```
摘完后 V 的链：
```text
  Value %V
  └─ firstUse ──► [OpOperand_A] ──nextUse──► nullptr
                       │
                    back=&(V.firstUse)        ← A 升格为链头
```

**② value = newValue**：`B.value = %W` ← B 现在指向 %W 了（UD link 改向）

**③ insertIntoCurrent()**（头插 B 进 W 的链）：
```cpp
back = &useList->firstUse;        // → B.back = &(W.firstUse)
nextUse = useList->firstUse;      // → B.nextUse = W.firstUse = nullptr（W 原来空）
useList->firstUse = this;         // → W.firstUse = OpOperand_B
```
插完后 W 的链：
```text
  Value %W
  └─ firstUse ──► [OpOperand_B] ──nextUse──► nullptr
                       │
                    value=%W         ← 已改
                    back=&(W.firstUse)
```

**第一轮结束时的全局状态**：
```text
  %V : firstUse ──► [OpOperand_A] ──► null        （只剩 A 一个使用者）
  %W : firstUse ──► [OpOperand_B] ──► null        （B 已搬来）
```

#### 第二轮循环：处理 OpOperand_A（现在是链头）

```cpp
while (!use_empty())        // V.firstUse = A ≠ null → 仍进入循环
    use_begin()->set(W);    // use_begin() = V.firstUse = OpOperand_A → 调 A.set(%W)
```

`A.set(%W)`：

**① removeFromCurrent()**（摘 A 出 V 的链）：
```cpp
// A.back = &(V.firstUse)，非空
*back = nextUse;     // → V.firstUse = A.nextUse = nullptr     ★ V 的链头变 null = V 空了
```
摘完后：`%V : firstUse ──► nullptr` ★ V 现在没有任何使用者。

**② value = newValue**：`A.value = %W`

**③ insertIntoCurrent()**（头插 A 进 W 的链）：
```cpp
back = &useList->firstUse;        // → A.back = &(W.firstUse)
nextUse = useList->firstUse;      // → A.nextUse = W.firstUse = OpOperand_B（W 现在有 B 了）
if (nextUse)
    nextUse->back = &nextUse;     // → B.back = &(A.nextUse)     ★ B 降为第二个
useList->firstUse = this;         // → W.firstUse = OpOperand_A
```
插完后 W 的链：
```text
  Value %W
  └─ firstUse ──► [OpOperand_A] ──nextUse──► [OpOperand_B] ──nextUse──► nullptr
                       │                          │
                    value=%W                   value=%W
                    back=&(W.firstUse)         back=&(OpOperand_A.nextUse)
```

**第二轮结束时的全局状态**：
```text
  %V : firstUse ──► nullptr          （空 ✓）
  %W : firstUse ──► [A] ──► [B] ──► null   （A、B 都搬来了，注意顺序：A 在前，因为是头插）
```

#### 终态对照

| | 调用前 | 调用后 |
|---|---|---|
| `%V` 的使用者 | {B, A} | **∅**（空） |
| `%W` 的使用者 | ∅ | {A, B} |
| op `A` 的操作数指向 | `%V` | `%W` |
| op `B` 的操作数指向 | `%V` | `%W` |
| 新建的 `OpOperand` | — | **0 个**（A、B 是原有对象搬家） |
| 删除的 op | — | **0 个**（只动 use-list，不动 op） |

**目标达成**：B、A 两个 op 的操作数都从 `%V` 改指 `%W`，`%V` 变成无人引用。

> 注意一个细节：RAUW 后 `%W` 链里是 `A → B`（A 在前），而调用前 `%V` 链里是 `B → A`（B 在前）。**顺序反了**——因为每次都是"头插"。这不影响正确性（use 集合是无序的语义），但说明 RAUW 不保证 use 顺序。这也是为什么 MLIR 另有 `shuffleUseList`——某些场景（如编译器确定性/测试稳定性）需要显式控制 use 顺序。

### 10.2.5 几个"为什么"（设计动机）

**为什么 while 里用 `use_begin()->set()` 而不是"遍历链表逐个改"？** 因为 `set()` 会**摘掉当前节点**（`removeFromCurrent` 改了 `firstUse`），如果你用普通迭代器"记住 next 再前进"，会和"链表正在被改"打架。这种写法**每次重新取最新链头**，天然避免了迭代器失效——每次循环 `firstUse` 都指向"还没处理的下一个"。而且每次循环链表严格缩短一个节点，**必然终止**。

**为什么零分配？** `set()` 全程在**搬运现有 `OpOperand` 对象**（改它的 `value`/`back`/`nextUse` 三个字段），从不 `new`。所以 RAUW 100 个使用者也不会分配一个字节——这对编译器（动辄百万条 use 边）极其重要。

**为什么 `back` 用指针的指针（再强调）？** 回到 §10.2.4 的追踪：摘头节点 B 时 `*back=nextUse` 展开成 `V.firstUse = A`；摘尾节点 A 时 `*back=nextUse` 展开成 `V.firstUse = null`。**同一句代码，因 `back` 指向不同地址而正确处理了"头"和"非头"两种情况**。若用普通 `prev` 指针，摘头节点要特判 `if (this == firstUse)`——分支预测、代码膨胀。`back` 把这个分支前置到了"插入时设置 back 的值"那一步，换取摘除时的无条件 O(1)。这是经典的**用空间（存地址的地址）换分支 + 用不变量换简洁**。

**为什么有"不能 RAUW 自己到自己"的断言？** 如果 `V` 和 `W` 是同一个 Value，`set()` 会先从 V 链摘、再插回 V 链——在多使用者时会产生混乱（摘一个、插回，链结构被破坏）。断言从源头挡住这种无意义调用。

## 10.3 erase：删除 op 的安全约束

RAUW 把使用者搬走后，op 的 result 就 use_empty 了，这时才能安全 erase。`Operation::erase`（[Operation.cpp:539-550](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L539)）：

```cpp
void Operation::erase() {
  if (auto *parent = getBlock())
    parent->getOperations().erase(this);   // ← 触发 iplist::erase → deleteNode → destroy()
  else
    destroy();                              // ← 无 parent 就直接 destroy
}
```

### 10.3.1 erase vs remove：生命周期的不同时刻

[Operation.cpp:539-550](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L539) 同时定义了 erase 和 remove：

```cpp
// 从 parent block 移除 + 销毁
void Operation::erase() {
  if (auto *parent = getBlock())
    parent->getOperations().erase(this);
  else
    destroy();
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

### 10.3.2 ~Operation 的双重断言（SSA 不变量守卫）

[Operation.cpp:179-205](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L179)：

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
  // 析构 OperandStorage（自动从上游 use-list 摘掉它的 operand）、BlockOperand、Region、properties...
}
```

> ⚠️ **关键警告**：use_empty 检查在 **release build 下完全不存在**（它在 `#ifndef NDEBUG` 里）。有 use 的 op 被 erase 后，在 release 下是**静默 use-after-free**——下游 op 的 operand 指向已释放内存，可能任意时刻崩溃。在 debug 构建下才会触发 `report_fatal_error`。安全路径：erase 前必须 RAUW 所有 results 或将所有 uses 置零。

### 10.3.3 安全删除三步法

```text
✅ 安全路径：
   1. 确保 op 不在 block 中（或调用 erase 时自动解除）
   2. 对所有 result 调用 result.replaceAllUsesWith(someOtherValue) 或 dropAllUses()
   3. op->erase() (或依赖 PatternRewriter 的 eraseOp)

❌ 常见反模式：
   1. 对仍有 uses 的 op 直接 erase() → release build 静默 use-after-free
   2. erase 后保留 op 指针继续使用 → 已 free
   3. walk 中 erase 其他未访问的 op → 行为未定义
   4. 调用 delete op → 错误（结构中有前缀，free 不动原始 malloc 指针）
   5. RAUW 一个 value 到自己 → assert
```

**反模式 4 值得展开**：不能 `delete op`，因为 Operation 的结果反向存在前缀区（第 7 章 §7.5），`delete` 只释放 `Operation*` 指向的位置，但原始 malloc 指针在前缀区之前。必须用 `op->erase()` 或 `op->destroy()`，它们内部会算出 `prefixAllocSize` 并后退到真正的 malloc 指针。

PatternRewriter 的 `eraseOp` 自动处理 uses 的前提：**matched op**。对于其他 op，须先调用 `replaceAllUsesWith` 再 erase。**这就是"先 RAUW 再 erase"顺序的物理根源**——RAUW 把所有下游使用者搬走，result 变 use_empty，erase 才不会触发断言。

> **编译原理浸润点：死代码消除（DCE）**　"先 RAUW 再 erase" 的批量版本就是**死代码消除（Dead Code Elimination, DCE）**——Dragon Book 第 9.4 节的经典算法。DCE 的核心：一个 op 如果没有副作用、且其结果无人使用，它就是死的，可以删。删除后，它的操作数定义 op 可能也变死，递归删除——这正是第 9 章讲的 `addOperandsToWorklist` 的"向上回溯"。MLIR 的 `isOpTriviallyDead` + `eraseOp` 就是 DCE 的工程实现。Ch13 的 BufferCastOpFold 会综合用到它。

## 10.4 replaceOp = RAUW + erase 的完整图景

把 RAUW 和 erase 合起来看 replaceOp。`RewriterBase::replaceOp`（[PatternMatch.cpp:133-150](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp#L133)，精简）：

```cpp
void RewriterBase::replaceOp(Operation *op, ValueRange newValues) {
  // ① 通知监听者（驱动据此更新工作表）
  if (auto *rewriteListener = ...)
    rewriteListener->notifyOperationReplaced(op, newValues);
  // ② RAUW：把 op 每个结果的所有使用者改指向 newValues
  replaceAllUsesWith(op->getResults(), newValues);
  // ③ erase：现在 op 的结果都 use_empty 了，安全删除
  eraseOp(op);
}
```

对照第 9 章 `SimplifyRedundantTranspose` 的 `rewriter.replaceOp(外层transpose, {%arg0})`：
- ② 把 `%t1`（外层 transpose 结果）的所有使用者（return）改指 `%arg0`——RAUW，沿 `%t1` 的 use-list 搬迁 return 的 OpOperand。
- ③ 删除外层 transpose——现在 `%t1` use_empty，erase 安全。

整个过程的物理动作全是第 4 章的两个原语（removeFromCurrent + insertInto）的组合：
- RAUW 的每个 set = removeFromCurrent + 改 value + insertInto。
- erase 触发 ~Operation，析构 OperandStorage 时自动从上游 use-list 摘掉它的 operand（对每个 operand 调 removeFromCurrent）。

**所以重写看起来神秘，底层却异常朴素**——它复用的全是第 4 章已经讲透的东西。匹配（match）沿着 use-def 链去"看"；替换（replace）用 rewriter 改图，而改图的原子动作（replaceOp/eraseOp）本质全是 use-def 边的拆除与重接。

---

## 编译原理浸润点回顾

1. **SSA 不变量在重写时的维护**：本章主题。RAUW 维护 use 边，erase 维护定义点。两者组合成 replaceOp。
2. **零成本抽象**：`back` 指针的指针让 RAUW 的批量搬迁成为零分支 O(uses)。这是把不变量维护做成零成本抽象的典范。
3. **死代码消除（DCE）**：erase 安全的批量版本。Dragon Book 第 9.4 节。MLIR 的 isOpTriviallyDead + eraseOp + addOperandsToWorklist 向上回溯 = DCE 的工程实现。
4. **use-def vs def-use 术语**：本章开篇的澄清。RAUW 改的是 DU chain（use-list），遍历用 DU 方向、改 value 字段改的是 UD link。
5. **指针的指针技巧**（回扣 Ch4）：RAUW 的无分支搬迁依赖它。

---

## 本章关键结论

1. **replaceOp = RAUW + erase**。RAUW 把 op 结果的所有使用者改指向新值，erase 删掉 op。顺序必须先 RAUW 再 erase。
2. **RAUW 是 O(uses)、零分配**。while 循环里反复调 set（removeFromCurrent + 改 value + insertInto），每次搬一条 use 边。不 new 任何对象。
3. **RAUW 改的是 DU chain（use-list）**，遍历用 DU 方向、改 value 字段改的是 UD link。口语化的"use-def 边"说法严格讲是 DU。
4. **`back` 指针的指针让 RAUW 零分支**：摘头节点和中节点的 `*back=nextUse` 是同一句代码。这是把不变量维护做成零成本抽象。
5. **erase 有双重断言**：block==nullptr（始终）+ use_empty（仅 debug）。release 下静默 use-after-free。安全路径是先 RAUW 再 erase。
6. **不能 delete op**：前缀区使原始 malloc 指针不在 Operation* 位置。用 erase/destroy。
7. **DCE 是 erase 安全的批量版本**：isOpTriviallyDead + eraseOp + 向上回溯 = Dragon Book 第 9.4 节 DCE 的工程实现。

---

## 下一章预告

本章讲完了重写的原子动作。第 11 章讲重写的"调度层"——**驱动、工作表与监听者**。这里有一个 Dragon Book 不深究、但 MLIR 必须面对的理论——**不动点与单调框架**。greedy driver 的收敛本质是"反复应用规则到 IR 不再改变"，这正是单调框架下数据流分析迭代到不动点的对偶。我们会立论"为什么 greedy driver 能终止"，并补 Tarski 不动点定理与项重写终止性的理论背景。同时讲 PatternBenefit、fold 先于 pattern、DRR/PDL 声明式写法。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-RAUW-replaceAllUsesWith-机理详解.md` 全文（RAUW 三原子动作、逐步追踪、use-def vs def-use 术语澄清、设计动机）——**全文保留，是 Ch10 主体**
- `docs/MLIR-matchAndRewrite-重写过程教程_精品.md` §4（replaceOp = RAUW + erase）——**保留**
- `docs/MLIR-IR-Node组织与遍历插入删除教程.md` §5（erase vs remove、双重断言、反模式）——**全文并入本章，消除 v1 的前向依赖**
- 编译原理铺垫（SSA 不变量维护、零成本抽象、DCE）为本书新增

## 参考文献

- **[Aho 2006]** Dragon Book，第 9.4 节（死代码消除算法）、第 9 章（SSA 变更）。
- **[Cytron 1991]** Cytron et al. SSA 不变量在重写时的维护。
- **[Lattner 2020]** Lattner et al. "MLIR"，RAUW 与 erase 的设计。
- **LLVM** `Value::replaceAllUsesWith`，RAUW 机制的对比实现。
