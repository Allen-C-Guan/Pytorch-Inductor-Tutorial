# 第 4 章 def-use chain：从数据流分析到 use-list 实现

> **本章位置**　第 2、3 章认识了 IR 的逻辑部件（Operation、Value、Block、Region），但都只点到"数据流边由 use-list 维护"而没展开。本章是全书最底层、也最精巧的一章——打开 def-use chain 的物理实现：那个被很多人称为"全文最难懂"的 `back` 指针（指针的指针）到底解决什么问题，那个只有三行的 `removeFromCurrent` 凭什么做到 O(1) 无分支。
>
> **前置依赖**　第 2 章（Value 的两面性：定义端容器 + 消费端链头）。
>
> **编译原理切入**　本章主题是 **def-use chain 的物理实现**——这是数据流分析与 SSA 的工程基石。Dragon Book 第 9 章的数据流分析（到达定值、活跃变量、可用表达式）都需要两个基本操作：**从一个 use 立即找到它的 def**（O(1)），**从一个 def 枚举它的所有 use**（O(users)）。要同时支持这两个操作，必须用某种"双向链"的数据结构把 def 和 use 连起来。不同的编译器有不同的实现选择——LLVM 用 `Value::use_list`、Sea of Nodes 用显式的边对象、MLIR 用一个精巧的侵入式链表加"指针的指针"。本章讲清 MLIR 的选择，并立论那个看似古怪的 `back` 字段为何让链表的头删、中删、尾删统一成无条件 `*back = nextUse`。

---

## 4.1 数据流分析为什么需要 def-use chain

在打开源码前，先从编译器学科立论：为什么需要 def-use chain？

Dragon Book 第 9 章的数据流分析，本质上都是"在程序的数据依赖关系上做迭代计算"。例如：

- **到达定值（reaching definition）**：每个 use 点能到达的 def 有哪些？
- **活跃变量（live variable）**：每个 def 之后，它的值还会被用到吗？
- **可用表达式（available expression）**：某个表达式在某点是否已经被算过、值还可用？
- **死代码消除（DCE）**：一个 def 如果没有 use，它就是死的，可以删。

这些分析都需要两个方向的查询：

1. **从 use 找 def**（向上）：给定一个操作数引用，它指向哪个值？——这是 SSA 的天然能力，每个操作数自带"指向定义"的指针。
2. **从 def 找 use**（向下）：给定一个值，它被哪些操作数引用？——这需要显式维护"使用链"，即 def-use chain。

第 0 章讲过 SSA 的三大好处，其中之一就是"每个 def 的所有用户可遍历"。这个遍历能力的物理基础，就是本章要讲的 **use-list**。

```text
Value V（定义端）
  └─ firstUse ──► [OpOperand A] ──► [OpOperand B] ──► nullptr
                     │                  │
                  owner=opX          owner=opY
                  value=V            value=V
```

这幅图就是 def-use chain 的物理实现：Value 持有链头 `firstUse`，每个 OpOperand 既知道自己引用的 Value（`value` 字段），又串在链表里（`nextUse` 指向下一个使用者）。注意这是**单向链表**——从 Value 出发能走到所有 use，但从 use 回到 Value 靠的是 `value` 字段（不是链表的反向指针）。这种"单向 nextUse + value 回指"的设计已经足够，因为 SSA 不需要"从 use 删除自己时找前驱"——它用了更巧妙的 `back` 机制（下一节）。

> **术语澄清：def-use vs use-def**　这是编译原理里两个方向相反的标准术语，读本章（和后续章节）时不要搞混：
> - **UD chain（use-def chain）**：use → def（向上）。从一个**使用**出发找它的**定义**。MLIR 里：`operand.getDefiningOp()`、OpOperand 的 `value` 字段。
> - **DU chain（def-use chain）**：def → use（向下）。从一个**定义**出发找它所有的**使用**。MLIR 里：Value 的 `firstUse` use-list、`value.getUsers()`。
>
> **关键事实：MLIR 的 use-list（firstUse 链）就是 DU chain（def-use）的物理实现**。虽然 MLIR 的头文件叫 `UseDefLists.h`（笼统命名），但严格讲 use-list 是 def-use 方向。读到"use-list / use 链"都指 DU 方向，读到"找定义（getDefiningOp）"才指 UD 方向。

## 4.2 use-list 的数据结构：OpOperand 的四个字段

现在打开源码。维护 use-list 的核心类是 [`IROperandBase`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L35)，每个 OpOperand 继承它。OpOperand 的关键字段有四个（[UseDefLists.h:106-113](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L106) + [:183](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L183)）：

| 字段 | 类型 | 含义 |
|---|---|---|
| `owner` | `Operation* const` | 使用方 op（"哪个 op 用了我"），构造时定死，永不变 |
| `value` | `IRValueT`（即 Value） | 被引用的值（"我引用了哪个 Value"）——这是 UD link |
| `nextUse` | `IROperandBase*` | 链表后继（同一 Value 的下一个使用者） |
| `back` | `IROperandBase**` | **★ 指针的指针**——指向"指向我自己的那个指针变量"的地址 |

前三个字段直觉上好懂。`owner` 让我们从 use 找到使用方 op；`value` 让我们从 use 找到被引用的 Value（UD link）；`nextUse` 把同一 Value 的所有 use 串成链表（DU chain）。但第四个字段 `back` 是全文最绕、也最精巧的点，下一节专门讲。

Value 侧的 use-list 容器是 [`IRObjectWithUseList`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L189)，它只持有一个头指针：

```cpp
// UseDefLists.h:189-294（精简）
template <typename OperandType>
class IRObjectWithUseList {
  detail::IROperandBase *firstUse = nullptr;  // use-list 头
};
```

Value（经 ValueImpl）继承 `IRObjectWithUseList<OpOperand>`，所以持有 `firstUse`。Block 继承 `IRObjectWithUseList<BlockOperand>`，也持有 `firstUse`（用于前驱遍历，见 Ch5）。这个模板设计让"持有 use-list"成为可复用的能力——凡是"被引用的实体"都能用同一套机制管理使用者。

## 4.3 `back` 为什么是"指针的指针"（核心难点）

这是本章的核心，也是全书最精巧的设计。务必啃下这一节。

普通双向链表里，每个节点存一个"前驱指针 `prev`"。但 use-list **没有 `prev`**，取而代之是一个更巧妙的东西：`back` 存的不是"前驱节点"，而是"**指向我自己的那个指针变量，它的内存地址**"。

听起来抽象，看两种情况就懂了。

### 4.3.1 情况 A：我是链表头节点

```text
  Value V
  └─ firstUse ──► [我]
```

"指向我自己的指针变量"是 `V.firstUse`。所以：

```text
  我的 back = &(V.firstUse)        // 指向 V.firstUse 这个字段的地址
```

### 4.3.2 情况 B：我是链表中间/末尾的节点

```text
  Value V
  └─ firstUse ──► [前驱P] ──nextUse──► [我]
```

"指向我自己的指针变量"是"前驱 P 的 `nextUse` 字段"。所以：

```text
  我的 back = &(P.nextUse)         // 指向前驱 P 的 nextUse 字段的地址
```

### 4.3.3 为什么这么设计

**因为不管我是头节点还是中间节点，"要把我摘掉，就是把『指向我的那个指针』改成指向我的后继"。** 而 `back` 恰好就是"指向我的那个指针"的地址，所以摘除时一句 `*back = nextUse` 就搞定——**不需要 if 判断我是不是头节点**。

对比普通双向链表，摘一个节点要写：

```cpp
// 普通双向链表
if (我是头) head = next;
else prev->next = next;
```

这里有个分支（`if`）。MLIR 用 `back` 把"我是不是头"这个分支消掉了——`back` 永远指向"该改的那个指针变量"，一句 `*back = nextUse` 通杀。

> **编译原理浸润点：指针的指针技巧**　这个设计不是 MLIR 原创，它是双向链表删除优化的经典技巧，源自 Linux 内核的 `list_head`（`struct list_head` 的 `pprev` 字段就是同样的"指针的指针"）。这种技巧用空间（存地址的地址）换分支（消除 if），在编译器这种"链表操作发生在热路径"的场景下极其有价值——IR 有百万级 use 边，每条边的增删都走这条路径。第 10 章讲 RAUW（批量搬迁 use 边）时，你会看到这个 `back` 设计如何让"搬迁 100 条 use 边"也是零分支的 O(uses)。

## 4.4 insertInto：构造时如何入链（四步动态演示）

现在看 `insertInto`——把一个 OpOperand 插入 Value 的 use-list 头部。源码（[UseDefLists.h:96-103](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L96)）：

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

这是**头插法**：新节点总是插到链表最前面。逐步推演（链表之前是空的，然后插入节点 S）：

```text
插入前:  V.firstUse = nullptr

Step ①: S.back = &V.firstUse
Step ②: S.nextUse = nullptr       (因为 firstUse 是 nullptr)
Step ③: (跳过，nextUse 为空)
Step ④: V.firstUse = S

结果:  V.firstUse → S
       S.back = &V.firstUse        ← 记住谁指向了它
```

再插入节点 T（链表已有 S）：

```text
插入前:  V.firstUse → S → ∅

Step ①: T.back = &V.firstUse
Step ②: T.nextUse = S
Step ③: S.back = &T.nextUse        ← 现在 S 记住：是 T 在指向我
Step ④: V.firstUse = T

结果:  V.firstUse → T → S → ∅
       T.back = &V.firstUse
       S.back = &T.nextUse
```

注意 Step ③：插 T 后，原来的头 S 降为第二个，它的 `back` 从 `&V.firstUse` 改成了 `&T.nextUse`——因为它现在是被 T 的 nextUse 指向的，而不是被 firstUse 指向的。这正是 `back` 设计的精髓：**back 永远指向"当前指向我的那个指针变量"**，无论我是头还是中间节点。

## 4.5 removeFromCurrent：摘除时如何出链（三步无分支）

现在看 `removeFromCurrent`——把一个 OpOperand 从 use-list 摘下来。源码（[UseDefLists.h:87-93](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L87)）：

```cpp
void removeFromCurrent() {
  if (!back)
    return;                 // 我根本没在任何链上（已 drop），直接返回
  *back = nextUse;          // ★ "指向我的指针" 改成 指向我的后继 → 我被绕过
  if (nextUse)
    nextUse->back = back;   // 我后继的 back 改成 原本指向我的那个地址
}
```

两句话、无 `if`（除了 nextUse 空检查）、O(1)。逐步看：

- `*back = nextUse`：让"指向我的那个指针"（无论是 `V.firstUse` 还是"前驱的 nextUse"）改成指向我的后继 `nextUse`。**这一行就把我从链里摘掉了**。
- `if (nextUse) nextUse->back = back`：如果我有后继，让后继的 `back` 接管"原本指向我的地址"——这样后继以后也能 O(1) 摘自己。

回到 4.3 的两种情况，验证 `*back = nextUse` 如何对两者都成立：

**摘头节点 S**（S 是 firstUse 直接指向的）：
```text
操作: S.removeFromCurrent()
  *back = nextUse   → *(&V.firstUse) = S.nextUse = T   → V.firstUse = T
  T.back = back     → T.back = &V.firstUse              ← T 升格为头，back 指向 V.firstUse
```

**摘中间节点 T**（T 是被 S.nextUse 指向的）：
```text
操作: T.removeFromCurrent()
  *back = nextUse   → *(&S.nextUse) = T.nextUse         → S.nextUse = T.nextUse
  ...
```

**同一句代码 `*back = nextUse`，因 `back` 指向不同地址而正确处理了"头"和"非头"两种情况。** 这就是 4.3 节说的"消除了分支"的兑现。

### 4.5.1 完整的删除追踪示例

设 Value V 有两个使用者（B 先建、A 后建，因头插法 B 在链头），现在依次删除 B 和 A：

```text
初始:
  V.firstUse ──► [B] ──nextUse──► [A] ──► nullptr
                  │                │
               value=V         value=V
               back=&(V.firstUse)  back=&(B.nextUse)
```

**删除 B（头节点）**：
```text
B.removeFromCurrent():
  *back = nextUse   → V.firstUse = A          ← V 的链头现在是 A
  A.back = back     → A.back = &(V.firstUse)  ← A 升格为链头

结果:
  V.firstUse ──► [A] ──► nullptr
                  │
               back=&(V.firstUse)
```

**删除 A（现在是唯一的节点，也是头部）**：
```text
A.removeFromCurrent():
  *back = nextUse   → V.firstUse = nullptr    ← V 的链头变 null = V 空了
  (nextUse 为空，无后续)

结果:
  V.firstUse = nullptr   ← 链表为空
```

这就是 use-list 的全部操作。理解了 insertInto（头插四步）和 removeFromCurrent（无分支三步），就理解了 def-use chain 的物理实现。**第 10 章讲 RAUW 时，会看到这两个原语如何组合成"批量搬迁所有 use 边"——那只是 while 循环里反复调 removeFromCurrent + insertInto。**

## 4.6 `set`：改写一条 use 边（摘+改+插）

在进入下一章前，看一个组合原语——`set`。它把一个 OpOperand 从"引用旧 Value"改成"引用新 Value"。源码（[UseDefLists.h:163-169](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L163)）：

```cpp
void set(IRValueT newValue) {
  removeFromCurrent();      // ① 把我从【当前 value】的 use 链上摘下来
  value = newValue;         // ② 改我的 value 字段，改指向【新 value】
  insertIntoCurrent();      // ③ 把我插进【新 value】的 use 链
}
```

这就是 4.4、4.5 两个原语的组合：摘（removeFromCurrent）→ 改指（value =）→ 插（insertInto）。一个 OpOperand 对象搬家了，但它还是原来那个对象（owner 不变、内存地址不变），只是从"旧 Value 的使用者"变成了"新 Value 的使用者"。

`set` 是第 10 章 RAUW 的真正主角——RAUW 本质就是"对所有 use 反复调 set"。本章先建立这个原语，第 10 章再用它组装出 RAUW。

## 4.7 与其他编译器实现的对比

把 MLIR 的 use-list 放在编译器谱系里看，它的设计取舍就清楚了：

| 编译器 | def-use chain 实现 | 取舍 |
|---|---|---|
| **LLVM** | `Value::use_list`，类似的双向链表，`Use` 对象有 `Prev`/`Next`/`Parent` | 也用侵入式链表，但 LLVM 的 `Use` 用传统的 prev 指针（有分支删除） |
| **Sea of Nodes (Graal)** | 显式的边对象，节点和边都是一等公民 | 表达力强（边可以带属性），但对象数量翻倍 |
| **MLIR** | use-list + `back` 指针的指针 | 零分支 O(1) 删除，代价是 `back` 字段稍难理解 |

MLIR 的选择继承了 LLVM 的侵入式链表传统，但用 `back` 指针的指针把"删除时的分支"消除了。这是"用理解成本换运行时性能"的典型取舍——初学者觉得 `back` 难懂，但它让百万级 use 边的操作快了一个量级。

## 4.8 本章建立的心智模型

把这一章收束成一个可记忆的心智模型。MLIR 的 def-use chain 由三个对象协作：

```text
┌─────────────────────────────────────────────────────────────┐
│  Value V（定义端，IRObjectWithUseList<OpOperand>）          │
│    └─ firstUse ──► （use-list 头）                          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  OpOperand（一条 use 边，IROperandBase 子类）                │
│    ├─ owner  = 使用方 op（永不变）                            │
│    ├─ value  = 被引用的 Value（UD link，set 时改）            │
│    ├─ nextUse = 同一 Value 的下一个 use                       │
│    └─ back   = 指向"指向我的指针"的地址（指针的指针）          │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Operation（使用方，通过 OpOperand 引用 Value）              │
│    └─ 它的 operand 存储里有一组 OpOperand                    │
└─────────────────────────────────────────────────────────────┘
```

**两个方向**：
- UD（use → def）：从 OpOperand 的 `value` 字段立即找到被引用的 Value。O(1)。
- DU（def → use）：从 Value 的 `firstUse` 沿 `nextUse` 遍历所有使用者。O(users)。

**两个原语**：
- `insertInto`（4.4）：头插四步，构造新 use 边时调用。Ch8 讲"构造即入链"时用到。
- `removeFromCurrent`（4.5）：无分支三步，删除 use 边时调用。Ch10 讲 RAUW 时用到。

这两个原语组合成 `set`（4.6），`set` 循环组合成 RAUW（Ch10）。**理解了本章，你就理解了 MLIR 一切数据流操作的物理基础。**

---

## 编译原理浸润点回顾

1. **def-use chain 是数据流分析的物理基础**：本章主题。Dragon Book 第 9 章的数据流分析都建立在 def-use 查询能力之上。
2. **侵入式数据结构**：use-list 是侵入式的（OpOperand 自带 nextUse/back 钩子），与 Ch3 的 iplist 同源。这是编译器 IR 零开销数据管理的体现。
3. **指针的指针技巧**：`back` 字段的设计，源自 Linux `list_head` 的 `pprev`。用空间换分支，让链表删除无 if。
4. **SSA 形式化**：本章把 SSA 的"每个 def 的所有 use 可遍历"形式化为 use-list 的物理实现。Ch0 的直觉在此落地。
5. **UD chain vs DU chain**：本章开篇的术语澄清。读后续章节时，"改 use-list"指 DU 方向（搬迁使用者），"找定义"指 UD 方向。

---

## 本章关键结论

1. **def-use chain 的物理实现是 use-list**：Value 持有 `firstUse` 头，每个 OpOperand 串成单向链表（`nextUse`），同时持 `value` 回指被引用的 Value（UD link）。
2. **OpOperand 有四个字段**：owner（使用方 op）、value（被引用值，UD link）、nextUse（DU chain 后继）、back（指针的指针）。
3. **`back` 是全文最精巧的设计**：它指向"指向我的那个指针变量的地址"，让链表的头删、中删、尾删统一成无条件的 `*back = nextUse`。源自 Linux `list_head` 传统。
4. **insertInto 是头插四步**：构造 use 边时入链。Ch8 详解"构造即入链"。
5. **removeFromCurrent 是无分支三步**：删除 use 边时出链。Ch10 的 RAUW 本质是反复调它。
6. **MLIR 的 use-list 继承 LLVM 侵入式传统，用 `back` 消除了删除分支**：理解成本换运行时性能，在百万级 use 边的场景下极其划算。

---

## 下一章预告

本章讲了"数据流边的物理实现"。有了结构（Ch2-3）和数据流边（Ch4），下一章自然要问：**怎么遍历这棵 IR？** 第 5 章讲 walk（深度优先遍历结构树）、use_iterator（遍历数据流）、predecessor/successor（遍历控制流，从 use-list 推导）。其中有一个编译器写作者必知的契约——**遍历期间能不能改 IR？什么时候能 erase 当前 op？** 这个"变更契约"是 Ch5 的重点，它决定了优化 pass 能不能正确地"边遍历边改"。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-IR-Node组织与遍历插入删除教程.md` §1.2（Value 两面性）、§2.2（use-list + back 指针 + insertInto/removeFromCurrent 动态演示）——**保留并重组为编译器视角**
- `docs/MLIR-RAUW-replaceAllUsesWith-机理详解.md` §1（back 指针的直觉化两种情况讲解）——**保留**
- `docs/operator-use-def的形成.md` §8.1（OpOperand 四字段）——**保留**
- 编译原理铺垫（数据流分析需要 def-use chain、UD/DU 术语、Linux list_head 对比）为本书新增

## 参考文献

- **[Aho 2006]** Dragon Book，第 9 章（数据流分析，def-use / use-def chain 的概念）。
- **[Cytron 1991]** Cytron et al. SSA 形式，def-use chain 是其工程实现。
- **[Click 1995]** Sea of Nodes，显式边对象的对比实现。
- **Linux kernel** `include/linux/list.h`，`list_head` 的 `pprev` 指针的指针设计。
