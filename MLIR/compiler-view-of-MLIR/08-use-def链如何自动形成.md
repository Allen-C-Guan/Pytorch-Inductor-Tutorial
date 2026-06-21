# 第 8 章 use-def 链如何自动形成

> **本章位置**　第 4 章讲了 use-list 的静态数据结构（back 指针、insertInto、removeFromCurrent）。第 7 章讲到 `Operation::create` 第 3 步初始化操作数时"触发 use-def 链建立"——但只是一笔带过。本章把这个机制讲透：OpOperand 构造如何自动入链，OperandStorage 如何批量构造，以及 SSA 不变量为何在构造时被自动维护。
>
> **前置依赖**　第 4 章（use-list 数据结构）、第 6 章（七类挂接，尤其挂接⑦）、第 7 章（Operation::create 初始化顺序）。
>
> **编译原理切入**　本章主题对应传统 SSA 构造与 MLIR 的一个关键差异。Cytron 1991 的传统 SSA 构造需要显式插入 φ 节点、显式维护支配边界。MLIR 用 block argument 取代 φ（Ch3），更关键的是——**use-def 边在 OpOperand 构造时自动建立**，无需任何显式调用。这是"正确性内建于数据结构构造，而非事后检查"的设计哲学。本章立论这一设计为何能隐式维护 SSA 的支配不变量。

---

## 8.1 SSA 不变量在构造时的自动维护

第 0 章讲过 SSA 的核心不变量：**每个值只定义一次，定义支配所有使用**。这个不变量在传统 SSA 构造里需要显式维护——Cytron 算法要算支配边界、要插 φ、要重命名变量（[Cytron 1991]）。这是 SSA 构造的经典难点。

MLIR 给出了不同的答案：**use-def 边的建立是构造的副作用，不需要显式调用**。当你创建一个 Operation 并传入 operands 时，那些 use-def 边就自动织出来了——因为 OpOperand 的构造函数会自动把自己插进 Value 的 use-list。这意味着 SSA 的 def-use 关系（Ch4）在构造时就自动正确，无需额外的"建立边"步骤。

这个设计的深层动机：**正确性内建于数据结构构造，而非事后检查**。传统编译器经常"先建 IR，再跑验证 pass 检查对不对"。MLIR 让数据结构本身保证正确——只要构造完成了，use-def 关系就是对的。

## 8.2 OpOperand 的构造即入链

回到第 7 章第 3 步。`Operation::create` 初始化操作数时（[Operation.cpp:139-142](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L139)）：

```cpp
if (needsOperandStorage) {
  new (&op->getOperandStorage()) detail::OperandStorage(
      op, op->getTrailingObjects<OpOperand>(), operands);
}
```

OperandStorage 的构造（[OperationSupport.cpp:239-246](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp#L239)）：

```cpp
detail::OperandStorage::OperandStorage(Operation *owner,
                                       OpOperand *trailingOperands,
                                       ValueRange values)
    : isStorageDynamic(false), operandStorage(trailingOperands) {
  numOperands = capacity = values.size();
  for (unsigned i = 0; i < numOperands; ++i)
    new (&operandStorage[i]) OpOperand(owner, values[i]);  // placement new
}
```

OperandStorage 在 Operation 主体后的尾分配区中，逐个 placement-new 构造 `OpOperand(owner, values[i])`。**关键在于 OpOperand 构造函数的副作用**（[UseDefLists.h:130-133](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L130)）：

```cpp
IROperand(Operation *owner, IRValueT value)
    : detail::IROperandBase(owner), value(value) {
  insertIntoCurrent();   // 构造即入链
}
```

`insertIntoCurrent()` 调用 `insertInto(DerivedT::getUseList(value))`，即把当前 operand 插入 value 的 firstUse 链表（第 4 章 §4.4 详解的"头插四步"）。

**核心洞察**：OpOperand 一旦被构造，就**自动**把自己挂到所引用 Value 的 use-list 上。use-def 链的建立不需要额外显式调用——它是构造函数的副作用。

流向（以 SoftmaxOp 以 `get_tensor_op_1` 的结果为操作数为例）：

```text
SoftmaxOp 的 OperandStorage 构造
  └─ new OpOperand(softmaxOp, %0)
       └─ insertIntoCurrent()
            └─ insertInto(%0.useList)
                 → %0.firstUse = &softmaxOp的operand
```

`get_tensor_op_1` 的结果 value 的 use-list 上从此多了一个节点，指向 SoftmaxOp 的 operand。**这就是数据流"边"的物理实现**，而且它是自动的。

## 8.3 use-list 链表操作图解

第 4 章 §4.4 讲过 insertInto 的四步头插。这里用"添加新使用者"的场景再追踪一次，强化"构造即入链"的动态感。

设 value V 原有使用者链 `A → B`（A 的 operand 引用 V，B 的 operand 引用 V）。现 SoftmaxOp S 新增对 V 的使用：

```text
插入前:  V.firstUse → A_use → B_use → ∅

插入后:  V.firstUse → S_use → A_use → B_use → ∅
          (S_use.nextUse=A_use, A_use.back=&S_use.nextUse)
```

每个 use 节点通过 `owner` 字段可回溯到使用它的 Operation，故遍历 use-list 即可得到所有 user（`getUsers()`，见 [UseDefLists.h:267-274](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L267)）。

注意一个细节：**头插法让新使用者排在链头**。这意味着 use-list 的顺序是"后构造的在前"，而非"先构造的在前"。这不影响正确性（use 集合在 SSA 语义里是无序的），但在需要确定性的场景（如测试稳定性）下，MLIR 另有 `shuffleUseList`（[UseDefLists.h:222](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L222)）来显式控制顺序。

## 8.4 销毁约束：use_empty 断言

构造时 use-def 链自动建立，销毁时呢？这里有一个 SSA 不变量的运行时守卫。

`~Operation`（[Operation.cpp:179-205](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L179)）在 `NDEBUG` 下断言：

```cpp
Operation::~Operation() {
  assert(block == nullptr && "operation destroyed but still in a block");
#ifndef NDEBUG
  if (!use_empty()) {
    emitOpError("operation destroyed but still has uses");
    for (Operation *user : getUsers())
      diag.attachNote(user->getLoc()) << "- use: " << *user << "\n";
    llvm::report_fatal_error("operation destroyed but still has uses");
  }
#endif
  // 析构 OperandStorage、BlockOperand、Region、properties...
}
```

`IRObjectWithUseList` 析构（[UseDefLists.h:197-199](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L197)）同样断言 `use_empty()`。这保证了 SSA 的核心不变量：**一个仍被引用的 value 不能被销毁**，从而避免悬挂引用，维持图的结构一致性。

> ⚠️ **关键警告**（第 10 章会再次强调）：use_empty 检查在 **release build 下完全不存在**——它在 `#ifndef NDEBUG` 里。有 use 的 op 被 erase 后，在 release 下是**静默 use-after-free**。在 debug 构建下才会触发 `report_fatal_error`。安全路径：erase 前必须 RAUW 所有 results 或将所有 uses 置零。这个"先 RAUW 再 erase"的物理根源是第 10 章的主题。

## 8.5 支配性如何被隐式维护

回到本章开头的立论——MLIR 的构造如何隐式维护 SSA 的支配不变量？

关键在于**构建顺序**。第 6 章讲 AST→IR 前端时强调过：递归构建"先算操作数后算使用者"——

```cpp
mlir::Value lhs = mlirGen(*binop.getLHS());   // 先算左操作数
mlir::Value rhs = mlirGen(*binop.getRHS());   // 再算右操作数
return builder.create<AddOp>(location, lhs, rhs);  // 后算加法
```

这个顺序天然保证：**操作数一定先于使用者被构造**。而 OpOperand 构造即入链（8.2），所以 use-def 边建立时，被引用的 Value 一定已经存在。再加上 OpBuilder 的插入点管理（6.2）保证 op 被插入 Block 的位置在操作数定义之后——三者合力，**SSA 的支配不变量在构造时被隐式维护**，无需显式的支配树计算。

> **编译原理浸润点：传统 SSA 构造 vs MLIR 的隐式维护**　Cytron 1991 的传统 SSA 构造是一个复杂的算法：从 CFG 算支配树、算支配边界、在支配边界插 φ、用栈做变量重命名。这是对"已有非 SSA IR，转成 SSA"的场景。MLIR 的场景不同——它是**从零构造 IR**，前端 AST 天然有顺序（先算子表达式，后算父表达式），所以支配性可以靠构建顺序保证，不需要 Cytron 算法。这是一个重要的视角差异：传统 SSA 构造是"转换"（已有 IR → SSA IR），MLIR 是"构造"（从无到有 → 天然 SSA）。后者的支配性维护成本低得多。

## 8.6 本章建立的心智模型

把这一章收束：use-def 链的形成是一个**零显式调用**的过程。

```text
你写：builder.create<AddIOp>(loc, lhs, rhs)
  │
  ├─ OpBuilder::create
  │   ├─ OperationState 收集 operands=[lhs, rhs]
  │   └─ Operation::create(state)
  │       └─ OperandStorage 构造
  │           └─ for each operand:
  │               └─ placement new OpOperand(owner, value)
  │                   └─ 构造函数副作用：insertIntoCurrent()
  │                       └─ insertInto(value 的 use-list)   ← 边自动织出来
  │
  └─ 你拿到 addi 操作（边已自动建好）
```

整个过程中，**你没有写任何一行"建立 use-def 边"的代码**——它内建于 OpOperand 的构造。这是 MLIR 数据结构设计的一个核心哲学：**让正确性成为构造的必然结果**。只要 OpOperand 被构造，它就一定挂在使用链上；只要 Operation 被正确 erase（Ch10），它的 operand 就一定从使用链上摘下来（析构自动调 removeFromCurrent）。

---

## 编译原理浸润点回顾

1. **SSA 不变量的自动维护**：本章主题。use-def 边在构造时自动建立，靠 OpOperand 构造的副作用，而非显式调用。
2. **构造 vs 转换的视角差异**：传统 SSA 构造（Cytron 1991）是"转换已有 IR"，需要算支配边界/插 φ；MLIR 是"从零构造"，靠构建顺序保证支配性，成本低得多。
3. **正确性内建于数据结构构造**：MLIR 的设计哲学。只要构造完成，use-def 关系就自动正确。
4. **头插法与 use 顺序**：新使用者排链头，后构造在前。不影响正确性（use 集合无序），但需要确定性时用 shuffleUseList。

---

## 本章关键结论

1. **use-def 链由 OpOperand 构造自动建立**：构造函数副作用调 `insertIntoCurrent()`，把自己头插进 Value 的 use-list。无需显式调用。
2. **OperandStorage 批量构造**：`Operation::create` 第 3 步，逐个 placement-new OpOperand，每次触发自动入链。
3. **支配性由构建顺序隐式维护**：前端递归"先算操作数后算使用者" + OpBuilder 插入点管理 + 构造即入链，三者合力保证 SSA 支配不变量。
4. **销毁有 use_empty 守卫**：~Operation 在 debug 下断言 use_empty，release 下静默。这就是"先 RAUW 再 erase"的必要性（Ch10 详解）。
5. **头插法让 use 顺序与构造顺序相反**：新使用者排链头。需要确定性时用 shuffleUseList。

---

## 下一章预告

第五部分（IR 的构建）到此结束。我们讲完了 IR 如何从无到有被织出来——七类挂接（Ch6）、create 的内存模型（Ch7）、use-def 链的自动形成（Ch8）。从第 9 章开始进入第六部分——**IR 的重写**。IR 造好后，怎么改它？这里有一个编译器优化的数学基础——**项重写系统（term rewriting）与不动点语义**。第 9 章讲 matchAndRewrite 的全生命周期：一条重写规则如何被定义为"带通配符的查找替换模板"，如何被驱动系统化地套用到整棵 IR 树上。

---

## 原文对照

本章素材主要来自：
- `docs/operator-use-def的形成.md` §8（OpOperand 构造即入链、OperandStorage 批量构造、use-list 操作图解、销毁约束）——**全文保留**
- `docs/MLIR-IR-树的构建过程教程_精品.md` §6 Part C（挂接⑦ 的动态过程，已在 Ch6 引用，本章回扣）
- 编译原理铺垫（传统 SSA 构造 vs MLIR 隐式维护、正确性内建于构造）为本书新增

## 参考文献

- **[Cytron 1991]** Cytron et al. 传统 SSA 构造算法，与 MLIR 隐式维护的对比。
- **[Aho 2006]** Dragon Book，第 9 章（SSA 形式与支配关系）。
- **[Lattner 2020]** Lattner et al. "MLIR"，block argument 与构造即入链的设计。
