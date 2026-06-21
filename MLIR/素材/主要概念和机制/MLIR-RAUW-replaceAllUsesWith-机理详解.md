# RAUW（replaceAllUsesWith）机理详解：一步步追踪一次 use-list 的搬迁

**基于 LLVM 19.1.7 源码**

> 本文是 [从零构建一棵 MLIR IR 树](MLIR-IR-树的构建过程教程_精品.md) 与 [matchAndRewrite 重写过程教程](MLIR-matchAndRewrite-重写过程教程_精品.md) 的专题补充，专门把 **RAUW（replaceAllUsesWith）** 这一个原语讲透。
> 它回答：「**把一个 Value 的所有使用者改指向另一个 Value，底层到底动了什么？**」

---

## 目录

- ## 0. RAUW 是干什么的（功能）
- ## 0.5 术语澄清：use-def vs def-use（先读这段，否则全篇会绕）
- ## 1. 前置：use-list 的数据结构
- ## 2. RAUW 的三个原子动作
- ## 3. 完整例子：逐步追踪（不跳任何一步）
- ## 4. 几个"为什么"
- ## 5. 和上篇、本篇的呼应

---

## 0. RAUW 是干什么的

**一句话**：把一个 Value `V` 的**所有使用者**（所有引用 `V` 的 op 操作数），全部改成引用另一个 Value `W`。改完之后 `V` 没有任何使用者（`use_empty()`），`W` 多出原来 `V` 的那批使用者。

**直觉类比**：想象 `V` 是一根水管的总阀门，A、B、C 三个水龙头都从 `V` 接水（三条 use 边）。RAUW(`V`, `W`) 就是**把这三个龙头的进水管全部从 `V` 拆下来，改装到 `W` 上**。龙头还是那三个龙头（op 不变），只是水源换了。

```cpp
// 调用前：addi、return、sub 三个 op 都用 %V
// 调用：V.replaceAllUsesWith(W);
// 调用后：addi、return、sub 三个 op 都改用 %W；%V 无人引用
```

**关键性质**（后面的源码会让你看清这几条从何而来）：
- **O(uses) 时间**——只遍历 `V` 的 use 链一次。
- **零内存分配**——不 new 任何新对象，**复用**原有的那些 `OpOperand`（只是把它们从 `V` 的链上摘下来、挂到 `W` 的链上）。
- **不删除任何 op**——只改 use-list 的归属，op 自己一个不少（要删 op 得另外调 `eraseOp`）。

源码入口在 [UseDefLists.h:210-216](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L210)，短得惊人：

```cpp
// UseDefLists.h:210-216
template <typename ValueT>
void replaceAllUsesWith(ValueT &&newValue) {
  assert((!newValue || this != OperandType::getUseList(newValue)) &&
         "cannot RAUW a value with itself");   // 不允许把 V 换成自己
  while (!use_empty())
    use_begin()->set(newValue);                 // ★ 全部戏份在这一行
}
```

---

## 0.5 术语澄清：use-def vs def-use（先读这段，否则全篇会绕）

这一节回答一个高频困惑——也是读 RAUW 时最容易卡住的地方。

编译原理里有两个**方向相反**的标准术语：

| 术语 | 全称 | 方向 | 含义 | MLIR 里的体现 |
|---|---|---|---|---|
| **UD chain (use-def)** | use-definition chain | **use → def**（向上） | 从一个**使用**出发，找它的**定义** | `operand.getDefiningOp()`；OpOperand 的 `value` 字段 |
| **DU chain (def-use)** | definition-use chain | **def → use**（向下） | 从一个**定义**出发，找它所有的**使用** | Value 的 `firstUse` use-list；`value.getUsers()` |

**关键事实：use-list（firstUse 链）就是 def-use (DU) chain 的物理实现。**

- "def-use 是从定义看使用" —— ✅ 你说对了。
- "def-use 是不是就是 use-list" —— ✅ 是的，完全等价。

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

> ⚠️ **那为什么正文里有时会看到 "use-def 边" 这种说法？** 那是沿用了 MLIR 自己的口语化命名——**连头文件都叫 `UseDefLists.h`**（[UseDefLists.h](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h)），把"存 use-list 的这套机制"笼统叫 use-def。严格讲这不精确：use-list 是 DU chain。读这篇时，凡看到"改 use-list / 改 use 链"，都指 DU 方向；凡看到"找定义（getDefiningOp）"，才指 UD 方向。

### 进阶精确版：同一条边有两个读法（RAUW 两个方向都碰）

一个 `OpOperand` 代表**一条 SSA 边**，连着使用方 op `U` 和被定义值 `V`。从两头读，名字不同：

```
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

所以严格讲 RAUW **既遍历 def-use、又改了 use-def 指针**——但**主导框架是 def-use（use-list）**。这就是为什么本文强调"它不向上找定义、只改下游使用者"：因为遍历用的是 DU 方向，而"向上找定义"是 UD 方向，RAUW 根本不需要它。

---

## 1. 前置：use-list 的数据结构（只讲 RAUW 用到的部分）

### 1.1 一条单向链表挂在 Value 上

每个 Value 内部（经 `ValueImpl`）有一个**链表头字段 `firstUse`**，指向"第一个使用它的 `OpOperand`"。所有使用这个 Value 的 `OpOperand`，靠各自的 `nextUse` 字段串成一条**单向链**：

```
  Value V
  └─ firstUse ──► [OpOperand₁] ──nextUse──► [OpOperand₂] ──nextUse──► nullptr
                       │                         │
                    value=V                    value=V        ← 每个节点的 value 字段都指向 V
```

这条链就是"`V` 的使用者集合"（= V 的 DU chain）。`V.use_empty()` = (`firstUse == nullptr`)；`V.getUsers()` = 沿 `nextUse` 走一遍。

### 1.2 一个 `OpOperand` 有四个字段

`OpOperand`（继承自 `IROperand`，再继承 `IROperandBase`）的关键字段（[UseDefLists.h:106-113](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L106) + [:183](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L183)）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `owner` | `Operation* const` | 使用方 op（"哪个 op 用了我"），构造时定死，永不变 |
| `value` | `IRValueT`（Value） | 被引用的值（"我引用了哪个 Value"）——**RAUW 改的就是它（UD link）** |
| `nextUse` | `IROperandBase*` | 链表后继（同一 Value 的下一个使用者） |
| `back` | `IROperandBase**` | **★ 指针的指针**——指向"指向我自己的那个指针变量"的地址 |

前三个好懂，`back` 是全文最绕的点，下一节专门讲。

### 1.3 `back` 为什么是"指针的指针"（核心难点，先啃下它）

普通双向链表里，每个节点存一个"前驱指针 `prev`"。但这里**没有 `prev`**，取而代之是一个**更巧妙**的东西：`back` 存的不是"前驱节点"，而是"**指向我自己的那个指针变量，它的内存地址**"。

这听起来很抽象，看两种情况就懂了：

**情况 A：我是链表头节点**
```
  Value V
  └─ firstUse ──► [我]
```
"指向我自己的指针变量"是 `V.firstUse`。所以：
```
  我的 back = &(V.firstUse)        // 指向 V.firstUse 这个字段的地址
```

**情况 B：我是链表中间/末尾的节点**
```
  Value V
  └─ firstUse ──► [前驱P] ──nextUse──► [我]
```
"指向我自己的指针变量"是"前驱 P 的 `nextUse` 字段"。所以：
```
  我的 back = &(P.nextUse)         // 指向前驱 P 的 nextUse 字段的地址
```

**为什么这么设计？** 因为不管我是头节点还是中间节点，"**要把我摘掉，就是把『指向我的那个指针』改成指向我的后继**"。而 `back` 恰好就是"指向我的那个指针"的地址，所以摘除时一句 `*back = nextUse` 就搞定——**不需要 if 判断我是不是头节点**。这就是后面 `removeFromCurrent()` 只有 3 行、没有分支的根源。

> 类比：普通双向链表摘节点要写 `if (我是头) head=next; else prev->next=next;`。MLIR 用 `back` 把"我是不是头"这个分支消掉了——`back` 永远指向"该改的那个指针变量"，一句 `*back=nextUse` 通杀。

现在你具备了读 RAUW 源码的全部前置知识。

---

## 2. RAUW 的三个原子动作

`use_begin()->set(newValue)` 里的 `set()`（[UseDefLists.h:163-169](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L163)）由三个原子动作组成：

```cpp
// UseDefLists.h:163-169
void set(IRValueT newValue) {
  removeFromCurrent();      // ① 把我从【当前 value】的 use 链上摘下来
  value = newValue;         // ② 改我的 value 字段，改指向【新 value】
  insertIntoCurrent();      // ③ 把我插进【新 value】的 use 链
}
```

逐个拆。

### 原子动作 ①：`removeFromCurrent()` —— 从当前链上摘掉自己

[UseDefLists.h:87-93](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L87)：

```cpp
void removeFromCurrent() {
  if (!back)
    return;                 // 我根本没在任何链上（已 drop），直接返回
  *back = nextUse;          // ★ "指向我的指针" 改成 指向我的后继 → 我被绕过
  if (nextUse)
    nextUse->back = back;   // 我后继的 back 改成 原本指向我的那个地址
}
```

两句话：
- `*back = nextUse`：让"指向我的那个指针"（无论是 `V.firstUse` 还是"前驱的 nextUse"）改成指向我的后继 `nextUse`。这一行就把我从链里摘掉了。
- `if (nextUse) nextUse->back = back`：如果我有后继，让后继的 `back` 接管"原本指向我的地址"——这样后继以后也能 O(1) 摘自己。

> 这就是 §1.3 说的"一句 `*back=nextUse` 通杀"的兑现。**两行、无 if（除了 nextUse 空检查）、O(1)。**

### 原子动作 ②：`value = newValue` —— 改指向

一行赋值。把 `OpOperand` 的 `value` 字段（即 UD link）从旧 Value 改成新 Value。此刻这个 `OpOperand` 既不在旧链上（刚摘下来）、也还没进新链（下一步才插），是个"游离"状态，但 `owner` op 还指着它——**所以 op 的这个操作数此刻暂时"没有家"，但很快就会被插进新链**。

### 原子动作 ③：`insertIntoCurrent()` → `insertInto(新value的链)` —— 头插进新链

[UseDefLists.h:186](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L186) 的 `insertIntoCurrent()` 转手调 `insertInto(新 value 的 firstUse 容器)`（[UseDefLists.h:96-103](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L96)）：

```cpp
template <typename UseListT>
void insertInto(UseListT *useList) {
  back = &useList->firstUse;        // ① 我的 back 指向【新 value 的 firstUse 字段地址】
  nextUse = useList->firstUse;      // ② 我接手新 value 原来的头节点
  if (nextUse)
    nextUse->back = &nextUse;       // ③ 老头的 back 改成指向我的 nextUse
  useList->firstUse = this;         // ④ 新 value 的头现在是我
}
```

这是**头插法**：我把自己插到新 value 链的最前面。四步对应 §1.3 的两种情况里"成为新链头"的设置——插完后我的 `back = &新value.firstUse`（情况 A）。

**三个原子动作合起来 = `set(newValue)`**：从旧链摘 → 改 value → 插新链。**一个 OpOperand 对象搬家了，但它还是原来那个对象（`owner` 不变、内存地址不变），只是从"旧 value 的使用者"变成了"新 value 的使用者"。**

---

## 3. 完整例子：逐步追踪（不跳任何一步）

### 3.0 初始设置

- Value `%V` 有**两个使用者**：op `B` 和 op `A`（B 先建、A 后建，因头插法 B 在链头）。
- Value `%W` **暂时没有使用者**。
- 目标：`V.replaceAllUsesWith(W)`——让 B、A 都改用 `%W`。

初始的 use 链（注意 B 是链头，A 是链尾）：

```
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

> 注意两个 `back` 的差别：B 是链头 → `back = &(V.firstUse)`；A 是第二个 → `back = &(B.nextUse)`。这两种情况正好覆盖了"摘头节点"和"摘中间节点"，下面的追踪会让你看到 `*back = nextUse` 如何对两者都成立。

### 3.1 RAUW 进入 while 循环

```cpp
while (!use_empty())        // V.firstUse = B ≠ null → 进入循环
    use_begin()->set(W);    // use_begin() = V.firstUse = OpOperand_B → 调 B.set(%W)
```

#### 第一轮循环：处理 `OpOperand_B`（链头节点）

`B.set(%W)` 执行三个原子动作：

**① `removeFromCurrent()`**（摘 B 出 V 的链）：
```cpp
// B.back = &(V.firstUse)，非空
*back = nextUse;     // → *(V.firstUse 的地址) = B.nextUse = OpOperand_A
                     //   即  V.firstUse = OpOperand_A     ★ V 的链头现在是 A 了
if (nextUse)         // B.nextUse = A，非空
    nextUse->back = back;   // → A.back = &(V.firstUse)     ★ A 现在是链头，back 指向 V.firstUse
```
摘完后 V 的链：
```
  Value %V
  └─ firstUse ──► [OpOperand_A] ──nextUse──► nullptr
                       │
                    value=%V
                    owner=A
                    back=&(V.firstUse)        ← A 升格为链头，back 改指向 V.firstUse
```
（B 此刻被绕过，游离，但 `B.value` 还是 `%V`，`B.owner` 还是 B。）

**② `value = newValue`**：
```
  B.value = %W          ← B 现在指向 %W 了（UD link 改向）
```

**③ `insertIntoCurrent()` → `insertInto(%W 的链)`**（头插 B 进 W 的链）：
```cpp
back = &useList->firstUse;        // → B.back = &(W.firstUse)
nextUse = useList->firstUse;      // → B.nextUse = W.firstUse = nullptr（W 原来空）
if (nextUse) ...                  // nextUse 为 null，跳过
useList->firstUse = this;         // → W.firstUse = OpOperand_B
```
插完后 W 的链：
```
  Value %W
  └─ firstUse ──► [OpOperand_B] ──nextUse──► nullptr
                       │
                    value=%W         ← 已改
                    owner=B
                    back=&(W.firstUse)
```

**第一轮结束时的全局状态**：
```
  %V : firstUse ──► [OpOperand_A] ──► null        （只剩 A 一个使用者）
  %W : firstUse ──► [OpOperand_B] ──► null        （B 已搬来）
```

#### 第二轮循环：处理 `OpOperand_A`（现在是链头）

回到 while 顶部，重新判断：
```cpp
while (!use_empty())        // V.firstUse = A ≠ null → 仍进入循环
    use_begin()->set(W);    // use_begin() = V.firstUse = OpOperand_A → 调 A.set(%W)
```

`A.set(%W)` 三个原子动作：

**① `removeFromCurrent()`**（摘 A 出 V 的链）：
```cpp
// A.back = &(V.firstUse)，非空
*back = nextUse;     // → V.firstUse = A.nextUse = nullptr     ★ V 的链头变 null = V 空了
if (nextUse) ...     // A.nextUse = null，跳过
```
摘完后 V 的链：
```
  Value %V
  └─ firstUse ──► nullptr           ★ V 现在没有任何使用者
```

**② `value = newValue`**：
```
  A.value = %W
```

**③ `insertIntoCurrent()` → `insertInto(%W 的链)`**（头插 A 进 W 的链）：
```cpp
back = &useList->firstUse;        // → A.back = &(W.firstUse)
nextUse = useList->firstUse;      // → A.nextUse = W.firstUse = OpOperand_B（W 现在有 B 了）
if (nextUse)
    nextUse->back = &nextUse;     // → B.back = &(A.nextUse)     ★ B 降为第二个，back 改指向 A.nextUse
useList->firstUse = this;         // → W.firstUse = OpOperand_A
```
插完后 W 的链：
```
  Value %W
  └─ firstUse ──► [OpOperand_A] ──nextUse──► [OpOperand_B] ──nextUse──► nullptr
                       │                          │
                    value=%W                   value=%W
                    owner=A                    owner=B
                    back=&(W.firstUse)         back=&(OpOperand_A.nextUse)
```

**第二轮结束时的全局状态**：
```
  %V : firstUse ──► nullptr          （空 ✓）
  %W : firstUse ──► [A] ──► [B] ──► null   （A、B 都搬来了，注意顺序：A 在前，因为是头插）
```

#### 回到 while 顶部，循环终止

```cpp
while (!use_empty())   // V.firstUse == nullptr → use_empty() 为 true → 退出循环
```

### 3.2 终态对照

| | 调用前 | 调用后 |
|---|---|---|
| `%V` 的使用者 | {B, A} | **∅**（空） |
| `%W` 的使用者 | ∅ | {A, B} |
| op `A` 的操作数指向 | `%V` | `%W` |
| op `B` 的操作数指向 | `%V` | `%W` |
| 新建的 `OpOperand` | — | **0 个**（A、B 是原有对象搬家） |
| 删除的 op | — | **0 个**（只动 use-list，不动 op） |

**目标达成**：B、A 两个 op 的操作数都从 `%V` 改指 `%W`，`%V` 变成无人引用。

> 注意一个细节：RAUW 后 `%W` 链里是 `A → B`（A 在前），而调用前 `%V` 链里是 `B → A`（B 在前）。**顺序反了**——因为每次都是"头插"。这不影响正确性（use 集合是无序的语义），但说明 RAUW 不保证 use 顺序。这也是为什么 MLIR 另有 `shuffleUseList`（[UseDefLists.h:222](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L222)）——某些场景（如编译器确定性/测试稳定性）需要显式控制 use 顺序。

---

## 4. 几个"为什么"（设计动机）

### 4.1 为什么 while 里用 `use_begin()->set()` 而不是"遍历链表逐个改"

```cpp
while (!use_empty())
    use_begin()->set(newValue);    // 每次都处理【当前链头】
```

因为 `set()` 会**摘掉当前节点**（`removeFromCurrent` 改了 `firstUse`），如果你用普通迭代器"记住 next 再前进"，会和"链表正在被改"打架。这种写法**每次重新取最新链头**，天然避免了迭代器失效——每次循环 `firstUse` 都指向"还没处理的下一个"。而且每次循环链表严格缩短一个节点，**必然终止**。

### 4.2 为什么零分配

`set()` 全程在**搬运现有 `OpOperand` 对象**（改它的 `value`/`back`/`nextUse` 三个字段），从不 `new`。所以 RAUW 100 个使用者也不会分配一个字节——这对编译器（动辄百万条 use 边）极其重要。

### 4.3 为什么 `back` 用指针的指针（再强调）

回到 §1.3 和 §3 的追踪：摘头节点 B 时 `*back=nextUse` 展开成 `V.firstUse = A`；摘尾节点 A 时 `*back=nextUse` 展开成 `V.firstUse = null`。**同一句代码，因 `back` 指向不同地址而正确处理了"头"和"非头"两种情况**。若用普通 `prev` 指针，摘头节点要特判 `if (this == firstUse)`——分支预测、代码膨胀。`back` 把这个分支前置到了"插入时设置 back 的值"那一步，换取摘除时的无条件 O(1)。这是经典的**用空间（存地址的地址）换分支 + 用不变量换简洁**。

### 4.4 为什么有"不能 RAUW 自己到自己"的断言

[UseDefLists.h:212-213](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L212)：
```cpp
assert((!newValue || this != OperandType::getUseList(newValue)) &&
       "cannot RAUW a value with itself");
```
如果 `V` 和 `W` 是同一个 Value，`set()` 会先从 V 链摘、再插回 V 链——在多使用者时会产生混乱（摘一个、插回，链结构被破坏）。断言从源头挡住这种无意义调用。

---

## 5. 和上篇、本篇的呼应

- **上篇视角**：上篇 §6 Part C 讲的是"**建**一条 use 边"（`OpOperand` 构造时 `insertInto`）。RAUW 讲的是"**搬**已有的 use 边"——`removeFromCurrent` + `insertInto`，正是上篇那个 `insertInto` 的逆操作 + 再操作。**建、搬、拆（`drop`/`eraseOp`）共用同一套 `back`/`nextUse`/`insertInto`/`removeFromCurrent` 原语。**
- **matchAndRewrite 教程视角**：那篇 §4 讲 `replaceOp = RAUW + erase`。现在你彻底看清了：**`rewriter.replaceOp(op, newValues)` 里的"RAUW"那一步，就是把 op 每个结果的 use-list（DU chain）上的所有 `OpOperand`，逐个 `set()` 搬到新 value 的 use-list**——搬完旧结果就 `use_empty()`，紧接着的 `eraseOp` 才不会触发 `assert(use_empty())` 崩溃。这就是"先 RAUW 再 erase"顺序的物理根源。

一句话收尾：**RAUW 不是魔法，它是"沿着一条 use-list（DU chain），把每个节点摘下来、改个 value 字段（UD link）、头插到另一条 use-list"的循环——精妙全在那个指针的指针 `back` 上，它让"摘节点"这件事变成了无条件的一行 `*back = nextUse`。**

---

## 参考资料

- [LLVM 19.1.7 源码 · UseDefLists.h](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h)（`replaceAllUsesWith` :210、`set` :163、`removeFromCurrent` :87、`insertInto` :96）
- [LLVM 19.1.7 源码 · PatternMatch.cpp](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp)（`replaceAllOpUsesWith` :114、`replaceOp` :133）
- 姊妹篇 [从零构建一棵 MLIR IR 树](MLIR-IR-树的构建过程教程_精品.md) §6（use 机制的诞生）
- 续集 [matchAndRewrite 重写过程教程](MLIR-matchAndRewrite-重写过程教程_精品.md) §4（replaceOp = RAUW + erase）
