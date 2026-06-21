# 第 6 章 从 AST 到 IR：OpBuilder 与两类挂接

> **本章位置**　第 0–5 章讲的是"IR 长什么样、怎么读"。从本章开始进入第五部分——**写** IR 的构建。本章回答：一段前端解析出的 AST，怎么用 MLIR 的 C++ API 织成一棵 IR 树？这里有一个贯穿构建过程的心智模型——**两类挂接**：结构挂接（把节点装进容器）+ 数据流挂接（把 Value 接成 operand）。
>
> **前置依赖**　第 1 章（MLIRContext 拥有 Type/Attribute）、第 2 章（Operation 部件语义）、第 3 章（Block/Region 结构边、iplist）、第 4 章（use-list，数据流边）。
>
> **编译原理切入**　本章主题对应 Dragon Book 第 2–4 章的**前端流水线**——词法/语法/语义分析之后，前端要把 AST 翻译成 IR。传统编译器在语义分析阶段做这个翻译（语法制导翻译，syntax-directed translation，Dragon Book 第 5 章）。MLIR 的构建本质是同时织两张网：**结构树**（谁包含谁）与**数据流图**（谁流向谁）。本章用一个最小例子（计算 `return arg0 + 1` 的函数）逐行展示这两张网如何被一点点织出来。

---

## 6.1 构建的本质：同时织两张网

第 2–3 章讲过，MLIR 的 IR 有两种结构：

- **结构树**：Operation ⊃ Region ⊃ Block ⊃ Operation 的嵌套（谁包含谁）。
- **数据流图**：Value → OpOperand → Value 的 use-def 链（谁的数据流向谁）。

构建 IR 的本质，就是按一定顺序执行**两类「挂接」动作**，把这两张网一点点织起来：

| 挂接类型 | 干什么 | 底层动作 | 你调的接口 |
|---|---|---|---|
| **结构挂接** | 把一个节点装进容器 | iplist 插入 / placement-new Region(op) | `OpBuilder::create<>`、`createBlock`、`setInsertionPoint*` |
| **数据流挂接** | 把一个 Value 接成另一个 op 的操作数 | OpOperand 构造 → insertInto 进 Value 的 use-list（Ch4） | create 时传 operands |

**核心洞察（贯穿本章）**：每一次 `OpBuilder::create<>`，都会**同时**触发两类挂接——结构上把 op 插进当前 Block，数据流上把每个 operand 接进对应 Value 的 use-list。这两件事在 `Operation::create` 内部一次性完成。

本章选一个最小但完整的例子——计算 `return arg0 + 1` 的函数——逐行走读它从无到有 build 成 IR 的全过程，每一步标注：调用了哪个接口、挂接了什么、挂到了谁身上。

## 6.2 三个基础设施：OpBuilder 与 OperationState

构建 IR 需要三个基础设施。MLIRContext（第 1 章已讲）是拥有者，另外两个是干活的工具。

### 6.2.1 OpBuilder：带插入点的工人

[`OpBuilder`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h) 继承自 `Builder`，身兼两职：

```cpp
// Builders.h:50-206 —— Builder 基类：创建 context 级对象（Type/Attr/Loc）
class Builder {
  template <typename Ty, typename... Args> Ty getType(Args&&...);   // get I32Type 等
  IntegerAttr getIntegerAttr(Type type, int64_t value);
  ...
};

// Builders.h:210 —— OpBuilder：在 Builder 之上加"创建 Operation" + "插入点"
class OpBuilder : public Builder {
  class InsertPoint;                       // 一个 (Block*, Block::iterator) 对
  template <typename OpTy, typename... Args>
  OpTy create(Location location, Args&&... args);     // ★ 核心：建 op 并插入
  Block *createBlock(Region *parent, ...);            // ★ 建 block 并插入 region
  void setInsertionPointToStart(Block *);             // 移动插入点
  ...
private:
  Block *block = nullptr;                  // ← 当前插入到哪个 block   (Builders.h:614)
  Block::iterator insertPoint;             // ← block 内的哪个位置     (Builders.h:617)
};
```

**它最有状态的两个字段就是 `block` 和 `insertPoint`**——这俩决定了下一个 `create` 出来的 op 落在哪。这就是"一步一步挂接"的基石：你挪动插入点，再 create，op 就长在那个位置。

插入点的常用设置方式（[Builders.h:329-377](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L329)）：

```cpp
void setInsertionPoint(Operation *op);           // op 之前
void setInsertionPointAfter(Operation *op);       // op 之后
void setInsertionPointAfterValue(Value val);      // 在定义 val 的 op 之后
void setInsertionPointToStart(Block *block);      // block 首
void setInsertionPointToEnd(Block *block);        // block 末

// RAII 还原
InsertPoint saveInsertionPoint() const;
void restoreInsertionPoint(InsertPoint ip);
```

`saveInsertionPoint`/`restoreInsertionPoint` 提供 RAII 式的"临时改插入点、用完还原"——这在递归构建（如 6.5 节的 MLIRGen）里极其有用。

### 6.2.2 OperationState：装箱单

[`OperationState`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/OperationSupport.h#L950) 是一个**栈上的临时结构**，把"造一个 op 需要的全部信息"打包：

```cpp
// OperationSupport.h:950-960
struct OperationState {
  Location location;                                 // :951  这条 op 的源位置
  OperationName name;                                // :952  "arith.addi" 等
  SmallVector<Value, 4> operands;                    // :953  操作数（指向别的 Value）
  SmallVector<Type, 4> types;                        // :955  结果类型
  NamedAttrList attributes;                          // :956  属性（常量、符号名…）
  SmallVector<Block*, 1> successors;                 // :958  后继块（terminator 才有）
  SmallVector<std::unique_ptr<Region>, 1> regions;   // :960  它持有的 region
  // 一堆 add* 方法往里塞：addOperands / addTypes / addAttribute / addRegion / addSuccessors
};
```

它的角色是 **"填表"**：

```text
你（或 OpTy::build）填表 OperationState  ──递交给──→  Operation::create（消费装箱单，分配内存）
```

`addRegion` / `addAttribute` 这些方法就是**往装箱单里塞东西**。**此时还没真正建 op**，东西只是暂存在单子上。`Operation::create` 如何消费这张单（包括尾分配内存模型）是第 7 章的主题——本章只关注"填单"的语义。

> **为什么需要 OperationState 这个中间层？**　这是解耦的工程考量。如果 `Operation::create` 直接吃 8 个参数（operands/types/attrs/regions/successors/...），每个 op 的 build 签名都要重复一遍且容易写错。装箱单把"声明意图"（build 填单）与"实际分配"（create 消费）解耦——build 可以按任何顺序塞东西，create 统一消费。这个模式在编译器里很常见（如 LLVM 的 IRBuilder 也用类似的中间结构）。

## 6.3 目标例子与 MLIR 文本语法

我们要从零造出来的 IR，文本长这样：

```mlir
module {
  func.func @add_one(%arg0 : i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %arg0, %c1 : i32
    return %sum : i32
  }
}
```

语法速读：

| 语法片段 | 含义 | 对应的 IR 实体 |
|---|---|---|
| `module { ... }` | 顶层容器，自带 1 个 Region、1 个 Block | `ModuleOp` |
| `func.func @add_one(...) -> i32 { ... }` | 一个函数；`@add_one` 是符号名属性 | `func::FuncOp` |
| `%arg0 : i32` | 函数的入口块参数 | **`BlockArgument`**（一种 Value） |
| `%c1 = arith.constant 1 : i32` | 常量 op，`1` 是它的属性 | `arith::ConstantOp`；`1` 是 `IntegerAttr` |
| `%sum = arith.addi %arg0, %c1 : i32` | 加法，两个操作数 | `arith::AddIOp`；操作数 = use-def 边 |
| `return %sum : i32` | 终结指令 | `func::ReturnOp`（terminator） |

注意四类东西在本例中全部出现：Attribute（`@add_one`、`1`、函数类型）、Value（`%arg0`、`%c1`、`%sum`）、Block（函数体、module body）、Region（module 和 func 各有一个）。下面逐一把它们"造"出来。

## 6.4 一步步构建（7 步，每步带挂接标注）

### Step 0：备料——context + 方言 + builder + 类型

```cpp
mlir::MLIRContext ctx;
ctx.getOrLoadDialect<mlir::func::FuncDialect>();
ctx.getOrLoadDialect<mlir::arith::ArithDialect>();

mlir::OpBuilder builder(&ctx);                  // 建工人，插入点暂为空
mlir::Location loc = builder.getUnknownLoc();   // 位置（追溯用）
mlir::Type i32 = builder.getI32Type();          // 类型：intern 到 ctx（O(1) 查/插）
```

**挂接标注**：这一步不挂接任何 IR 节点，只是**备料**——`i32` 是个被 context 唯一化的 Type 句柄（Ch1），后面所有地方复用同一个指针。

> 注意 `OpBuilder` 构造后 `block == nullptr`。此时直接 `create` 会得到一个**游离 op**（不在任何 block）。所以下一步必须先建立"根"，再把插入点移进去。

### Step 1：建根 ModuleOp

```cpp
mlir::ModuleOp module = builder.create<mlir::ModuleOp>(loc);
```

**调用的接口**：`OpBuilder::create<ModuleOp>(loc)`（[Builders.h:508-517](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L508)）。展开它的内部，是 MLIR 构建 op 的**标准三段式**：

```cpp
template <typename OpTy, typename... Args>
OpTy create(Location location, Args&&... args) {
  OperationState state(location, getCheckRegisteredInfo<OpTy>(ctx)); // ① 开装箱单
  OpTy::build(*this, state, std::forward<Args>(args)...);            // ② ModuleOp::build 填单
  auto *op = create(state);    // = Operation::create(state) 然后 insert   ③ 分配 + 挂接
  return cast<OpTy>(op);
}
```

**ModuleOp::build 填了什么**（[BuiltinDialect.cpp:133-140](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/BuiltinDialect.cpp#L133)）：

```cpp
void ModuleOp::build(OpBuilder &builder, OperationState &state,
                     std::optional<StringRef> name) {
  state.addRegion()->emplaceBlock();   // ★ 给 module 配 1 个 Region，并往里塞 1 个 Block
}
```

**Operation::create 怎么消费这张单**（[Operation.cpp:82-153](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L82)，完整路径留 Ch7）：

```cpp
for (unsigned i = 0; i != numRegions; ++i)
  new (&op->getRegion(i)) Region(op);     // ★★ 挂接①：placement-new Region，container 指向 op
```

**Step 1 的挂接清单**：

| 挂接 | 谁 → 谁身上 | 持有形式 |
|---|---|---|
| **Region → Op ①** | module 的 1 个 Region，`.container = module` | op 尾部 `Region[]` 数组 + 每个 Region 一个 `container` 回指指针 |
| **Block → Region ④** | Region 里 emplace 了一个 Block | Region 的 iplist（侵入式双向链表） |

> 🏗️ **挂接①（Region→Op）怎么挂？**　Operation 用尾分配在自己堆内存的尾部预留一个 `Region[]` 数组。`Operation::create` 对每个 region 做 placement-new，构造时传入 `op` 指针 → Region 的成员 `container` 回指属主 op。**持有形式 = op 尾部的 `Region[]` + 每个 Region 一个 container 回指。属主 = op。**

> 🏗️ **挂接④（Block→Region）怎么挂？**　Region 持有 `llvm::iplist<Block>`（Ch3）；Block 继承 `ilist_node_with_parent<Block, Region>`，**自己就是链表节点**。`emplaceBlock()` 把一个 `new Block()` splice 进这条链表。**持有形式 = 侵入式双向链表，零额外节点分配，O(1) 插入/删除。属主 = region。**

### Step 2：把插入点移进 module 的 body

```cpp
builder.setInsertionPointToStart(module.getBody());   // module.getBody() = 那个 Block
```

**调用的接口**：`setInsertionPointToStart` → 把 builder 的 `block` 设成 module 的 body 块、`insertPoint` 设成块首。

> 这是"一步一步挂接"的关键动作——**先定位，再 create**。之后所有 `create` 的 op 都会长在这个 block 里。

### Step 3：建 FuncOp（自带 region + 属性）

```cpp
auto fnType = builder.getFunctionType({i32}, {i32});   // (i32) -> i32，也是个 intern 的 Type
mlir::func::FuncOp func =
    builder.create<mlir::func::FuncOp>(loc, "add_one", fnType);
```

**FuncOp::build 填了什么**（[FuncOps.cpp:180-195](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Dialect/Func/IR/FuncOps.cpp#L180)）：

```cpp
void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ...) {
  state.addAttribute("sym_name", builder.getStringAttr(name));   // 函数名属性
  state.addAttribute("function_type", TypeAttr::get(type));      // 函数类型属性
  state.addRegion();   // ★ 配 1 个空 Region（注意：此时还没 block！）
}
```

> ⚠️ **重要澄清**：`func::FuncOp::build` 只 `addRegion()`——**建一个空 region，并不创建入口块，也不创建 `%arg0`！** 入口块和块参数需要单独一步显式创建。下一步处理。

**Operation::create 完成分配后，OpBuilder::insert 把 func 挂进 module 的 body**（[Builders.cpp:432-439](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L432)）：

```cpp
Operation *OpBuilder::insert(Operation *op) {
  if (block)
    block->getOperations().insert(insertPoint, op);   // ★★ 挂接③：把 op 插进当前 block 的 iplist
  return op;
}
```

**Step 3 的挂接清单**：

| 挂接 | 谁 → 谁身上 | 持有形式 |
|---|---|---|
| **Attribute → Op ②** | `sym_name`、`function_type` | op 的 `attrs` 字段（一个指针，指向 context 池里的字典） |
| Region → Op ① | func 的函数体 Region | op 尾部 `Region[]` + container 回指 |
| **Op → Block ③** | func 插进 module 的 body 块 iplist | 侵入式双向链表 |

> 🏗️ **挂接②（Attribute→Op）怎么挂？**　Operation 有一个成员字段 `DictionaryAttr attrs`（[Operation.h:1066](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L1066)）——**就是一个指针**。`DictionaryAttr` 是 context 里被唯一化的字典对象的指针包装（Ch1）。所以 **op 不存属性的副本，只存一个指向 context 拥有的字典的指针**——所有属性先被打包成一个 `DictionaryAttr`，再让 op 拿到它的指针。**属主 = context（StorageUniquer），不是 op。**

> 🏗️ **挂接③（Op→Block）怎么挂？**　Block 持有 `llvm::iplist<Operation>`（Ch3）；Operation 继承 `ilist_node_with_parent<Operation, Block>`，自己就是链表节点。`OpBuilder::insert` 调 `block->getOperations().insert(insertPoint, op)` 把 op 节点 splice 进链表。**持有形式 = 侵入式双向链表，O(1) 插入/删除。属主 = block。**（Step 2 挪插入点的意义正在此：`insertPoint` 决定 splice 到链表的哪个位置。）

### Step 4：建入口块 + 块参数 %arg0

Step 3 结束时 func 的 region 还是空的。现在显式创建入口块，并给它配一个 `i32` 类型的块参数——这一步**同时**完成"Block→Region ④"和"BlockArgument→Block ⑤"两个挂接：

```cpp
mlir::Block *entry = builder.createBlock(&func.getBody(), {i32}, {loc});
mlir::Value arg0 = entry->getArgument(0);   // ★ %arg0 = 入口块的第 0 个 BlockArgument
```

`createBlock` 一次做四件事（[Builders.cpp:441-456](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L441)）：

```cpp
Block *b = new Block();
b->addArguments(argTypes, locs);          // ★ 建块参数 → 挂接⑤
parent->getBlocks().insert(insertPt, b);  // ★ 块入 region → 挂接④
setInsertionPointToEnd(b);                // 把插入点挪进新块
return b;
```

每个块参数的诞生是 `Block::addArgument`（[Block.cpp:152](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L152)）：

```cpp
BlockArgument Block::addArgument(Type type, Location loc) {
  BlockArgument arg = BlockArgument::create(type, this, arguments.size(), loc);
  arguments.push_back(arg);     // 挂接⑤：BlockArgument 进 block 的 arguments 向量
  return arg;
}
```

> 🏗️ **挂接⑤（BlockArgument→Block）怎么挂？**　Block 持有 `std::vector<BlockArgument>`。每个 BlockArgument 内部 impl **显式存了一个 `Block *owner` 字段**回指属主 block。`addArgument` new 一个 `BlockArgumentImpl(type, this, index, loc)` push_back 进向量。**持有形式 = 动态数组 + 每个参数一个 owner 回指指针。属主 = block。**
>
> （**为什么 BlockArgument 要存 owner 指针，而下一步的 OpResult 不存？** 因为块参数住在一个独立的 vector 里，没法靠地址算术定位属主；而 OpResult 按逆序紧贴在 op 主体之前，能靠算术倒推——见 Step 5 的挂接⑥。）

**关键认知**：`%arg0` 是一个 Value（BlockArgument 子类），但它**不是任何 op 的产物**——它由 Block 拥有。这正是"Value 的两种来源"之一（另一种是 OpResult）。

### Step 5：建 arith.constant（属性挂接 + OpResult 挂接）

```cpp
mlir::Attribute oneAttr = builder.getIntegerAttr(i32, 1);   // 1 : i32，一个 IntegerAttr
mlir::Value c1 = builder.create<mlir::arith::ConstantOp>(loc, oneAttr);
```

`Operation::create` 消费装箱单时：
- **挂接②（属性→Op）**：`op->setAttrs(attributes)`（[Operation.cpp:150](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L150)），把 `{value = 1 : i32}` 字典挂到 op 上。
- **挂接⑥（OpResult→Op）**：placement-new 一个 `InlineOpResult`（[Operation.cpp:127-132](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L127)）——这就是 `%c1` 这个 Value。
- **挂接③（Op→Block）**：constant 被 insert 进 entry 块。

> 🏗️ **挂接⑥（OpResult→Op）怎么挂？** —— 全篇最反直觉的设计。这里 OpResult **不存 owner 指针**！结果按**逆序**存在 op 的**前缀区**（`InlineOpResult[0..4]` + 尾部 `OutOfLineOpResult[]`），内存布局是 `| OutOfLine results | Inline results | Operation |`。`OpResult::getOwner()` 靠**反向指针算术**，用 result number 倒推出 op 起始地址。**持有形式 = op 前缀区的定长结果数组（无 owner 字段，靠算术定位属主）。属主 = op。**
>
> （对比挂接⑤：BlockArgument 住独立 vector、没法靠算术定位，所以**必须**存 owner 字段——这就是 ⑤ 和 ⑥ 差别的根因。省下每个结果一个指针 × 大量结果 = 可观的内存。）

注意 constant **无 operand**，所以**不织任何 use-def 边**——常量值走属性，不走 SSA 边（Ch2 的 operand vs attribute 分野在此体现）。

### Step 6：建 arith.addi（数据流挂接 / use-def 边）

```cpp
mlir::Value sum = builder.create<mlir::arith::AddIOp>(loc, arg0, c1);
//    操作数是 arg0(%arg0) 和 c1(%c1) —— 两个已存在的 Value
```

前几步的挂接（①~⑥）织的都是**结构树**。这一步不同：它织出**数据流图**——`%sum` 依赖 `%arg0` 和 `%c1`。这是全篇最重要的挂接⑦。

use-def 是个**逻辑概念**（"谁定义了这个值、谁使用了它"），它背后靠一套**数据结构**支撑（Ch4 已详解）。这里只回顾关键：**一条 use-def 边 = 一个 OpOperand 对象，一身二任**。

```text
              一个 OpOperand{owner=U, value=V}  ＝  "U 用了 V" 这条逻辑边
        ┌───────────────────────────────┴───────────────────────────────┐
   从 U（消费者）侧看                                   从 V（被使用的值）侧看
   它物理长在 U 的尾部 OpOperand[]                       它逻辑链在 V 的 firstUse 链
   → U 能说"我的 operand#i 是 V"（U.getOperand(i)）       → V 能说"我有 U 这个 user"（V.getUsers()）
```

挂接⑦ 的动态过程（Ch4 已详解 insertInto，这里回顾要点）：

**(1) 建 OpOperand**（[OperationSupport.cpp:239-246](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp#L239)）：

```cpp
new (&operandStorage[i]) OpOperand(owner, values[i]);   // addi 的槽#0→%arg0，槽#1→%c1
```

**(2) 构造即入链**：OpOperand 构造初始化完 `owner`/`value` 后，**当场把自己插进那个 value 的 use-list**（[UseDefLists.h:130-133](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L130)）：

```cpp
IROperand(Operation *owner, IRValueT value)
    : detail::IROperandBase(owner), value(value) {
  insertIntoCurrent();      // ← use-def 边在这一行诞生！
}
```

**(3) insertInto**：头插法 4 步（Ch4 §4.4 已详解）。织完后，`%arg0`/`%c1` 的 firstUse 各挂上一个 `OpOperand{owner=addi}`。

**容器视角——织完后的内存真实样子**：

```text
═══ addi 这个 Operation 的内存（一次 malloc：前缀结果 + 主体 + 尾部操作数）═══
addi (Operation)
  ├─ name = "arith.addi"     block = &entry     location = loc
  ├─ 前缀区  OpResult[]   = [ %sum ]
  └─ 尾部区  OperandStorage → OpOperand[] =
        [0] { owner=&addi, value=%arg0, nextUse=nullptr, back=&(%arg0.firstUse) }
        [1] { owner=&addi, value=%c1,  nextUse=nullptr, back=&(%c1 .firstUse) }

═══ 每个 Value 的 use-list（firstUse 单向链）的真实样子 ═══
%arg0  firstUse ──► addi.OpOperand[0] ──► nullptr
%c1   firstUse ──► addi.OpOperand[1] ──► nullptr
%sum  firstUse ──► nullptr                ← 还没有 user
```

> **关键看点——同一个对象被两个容器引用**：`addi.OpOperand[0]` 这**一个**对象，既是 addi 尾部数组的一项（让 addi 能说"我的 operand#0 是 %arg0"），又是 %arg0 的 `firstUse` 链头（让 %arg0 能说"我有 addi 这个 user"）。这就是"一身二任"在内存里的具象。于是两条查询走的是**同一个 OpOperand**：`addi->getOperand(0)` → `%arg0`；`%arg0.getUsers()` → `addi`。这正是 use-def 边能**双向 O(1)** 的物理根基——不是存了两份，而是一根箭头两头都能摸。

### Step 7：建 func.return（终结指令）

```cpp
builder.create<mlir::func::ReturnOp>(loc, sum);   // sum 作为 return 的操作数
```

和 Step 6 同理：`%sum` 被 OpOperand 挂进它自己的 use-list（return 成了 `%sum` 的 user）。同时 return 持有 `IsTerminator` trait，必须是块的最后一个 op。

此刻整棵树织完（织出第 3 条 use-def 边；entry 块以 terminator 结尾 = well-formed）：

```text
ModuleOp                                    ← 根
  └─ region#0
      └─ Block (module body)
          └─ FuncOp @add_one
              ├─ attrs: { sym_name, function_type }
              └─ region#0
                  └─ Block (entry)
                      ├─ arg#0: %arg0 (i32)        ← BlockArgument
                      ├─ ConstantOp → %c1          (value 属性带值)
                      ├─ AddIOp    → %sum          (operands: %arg0, %c1)
                      └─ ReturnOp                  (terminator)
                          └─ operands: %sum

完整的 data-flow 图（共 3 条 use-def 边）：
  %arg0 ──(lhs)──┐
                 ├─→ AddIOp ──→ %sum ──→ ReturnOp
  %c1  ──(rhs)──┘
   ↑ constant 用属性带值，不贡献 use-def 边
```

打印出来就是 6.3 节的那段 `.mlir`。

## 6.5 七类挂接总表

把 6.4 的七类挂接汇总。**构建一棵 IR = 反复执行这张表里的动作**：

| # | 挂接动作 | 被「挂」者 | 挂到「谁」身上 | **持有形式** | 触发接口 | 验证源码 |
|---|---|---|---|---|---|---|
| ① | Region → Op | `Region` | `Operation` | **数组**（op 尾部 `Region[]`）+ container 回指指针 | `Operation::create` 内 `new Region(op)` | [Operation.cpp:135](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L135) |
| ② | Attribute → Op | `DictionaryAttr` | `Operation` | **一个指针**（op 的 `attrs` 字段，指向 context 池） | `op->setAttrs(...)` | [Operation.cpp:150](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L150) |
| ③ | **Op → Block** | `Operation` | `Block` | **侵入式双向链表** `iplist<Operation>` | `OpBuilder::insert` | [Builders.cpp:434](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L434) |
| ④ | Block → Region | `Block` | `Region` | **侵入式双向链表** `iplist<Block>` | `createBlock`/`emplaceBlock` | [Builders.cpp:450](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L450) |
| ⑤ | BlockArgument → Block | `BlockArgument` | `Block` | **动态数组** + owner 回指指针 | `Block::addArgument` | [Block.cpp:152](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L152) |
| ⑥ | OpResult → Op | `OpResult` | `Operation` | **数组**（op 前缀区，**无** owner 字段，靠算术定位） | `Operation::create` placement-new | [Operation.cpp:127](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L127) |
| ⑦ | **Operand → Value（数据流边）** | `OpOperand` | `Value` 的 use-list | op 尾部 `OperandStorage` + Value 的 `firstUse` 单向链（nextUse）配 back 双指针 | `OperandStorage` ctor → `insertInto` | [OperationSupport.cpp:239](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp#L239) |

**两条主轴**：
- ①②③④⑤⑥ 都是**结构挂接**——织"结构树"。除 ⑤⑥ 外，全靠 iplist 或 placement-new 在 `Operation::create` / `OpBuilder::insert` 里自动完成。
- ⑦ 是**数据流挂接**——织"数据流图"，靠 `OpOperand` 构造时的 `insertInto` 自动完成。

> **一个深刻的结论**：`OpBuilder::create<>` 这一个调用，内部就完成了 ①~⑦ 里的好几项（取决于这个 op 有没有 region、operands、attrs）。这就是为什么 MLIR 的构建代码看起来很"干净"——大量挂接细节被 `Operation::create` 吞掉了，你只需声明"我要一个什么样的 op"，剩下交给装箱单 → 分配 → 自动挂接的流水线。`Operation::create` 的内部细节（尤其是尾分配内存模型）是第 7 章的主题。

## 6.6 从 AST 到 MLIR：前端的递归下降

到此你知道"一个 op 怎么建、怎么挂"。但真实前端不是手动一个个 create——它是**递归遍历 AST，每个 AST 节点生成 0~1 个 op**。MLIR 官方 Toy 教程的 [`MLIRGen.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp) 就是这个模式的教科书。

### 6.6.1 整体骨架：一个 OpBuilder + 一个符号表

```cpp
// MLIRGen.cpp:59-97（精简）
class MLIRGenImpl {
  mlir::OpBuilder builder;                                  // 建工人（带插入点）
  llvm::ScopedHashTable<StringRef, mlir::Value> symbolTable; // ★ AST 变量名 → SSA Value
public:
  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());  // 建根（本文 Step 1）
    for (FunctionAST &f : moduleAST) mlirGen(f);                  // 递归处理每个函数
    if (failed(mlir::verify(theModule))) return nullptr;          // 建完 verify
    return theModule;
  }
};
```

> **符号表是 AST 与 SSA 之间的桥梁**：AST 里变量叫 `a`、`b`，SSA 里是 `%0`、`%1`。符号表把"AST 名字"映射到"它对应的那个 Value"。声明变量时登记，引用变量时查表拿到 Value 当操作数。这正是 Dragon Book 第 2.7 节符号表的角色——从名字解析到声明。MLIR 的前端把符号表的 value 端换成了 SSA Value，但"名字 → 声明"的核心机制不变。

### 6.6.2 一个 AST 节点 → 一段 build

**AST 函数 → FuncOp + 进入函数体**（[MLIRGen.cpp:129-154](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L129)）：

```cpp
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  builder.setInsertionPointToEnd(theModule.getBody());     // ← Step 2：挪插入点
  mlir::toy::FuncOp function = mlirGen(*funcAST.getProto()); // ← Step 3：建 FuncOp
  mlir::Block &entryBlock = function.front();              // ← Step 4：取入口块
  // 把函数参数登记进符号表（AST 参数名 → BlockArgument Value）
  for (auto [arg, value] : llvm::zip(protoArgs, entryBlock.getArguments()))
    declare(arg->getName(), value);                        //    %arg0 ↔ "arg0"
  builder.setInsertionPointToStart(&entryBlock);           // ← Step 4：挪插入点进函数体
  mlirGen(*funcAST.getBody());                             // ← Step 5~7：递归生成函数体
}
```

**AST 二元表达式 → AddIOp（use-def 挂接）**（[MLIRGen.cpp:181-212](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L181)）：

```cpp
mlir::Value mlirGen(BinaryExprAST &binop) {
  mlir::Value lhs = mlirGen(*binop.getLHS());   // ← 递归：先算左操作数，返回一个 Value
  mlir::Value rhs = mlirGen(*binop.getRHS());   // ← 递归：再算右操作数
  ...
  return builder.create<AddOp>(location, lhs, rhs);  // ← Step 6：lhs/rhs 当操作数 → 自动织 use-def
}
```

> 注意这里的**递归顺序**：必须先 create 出 `lhs`/`rhs`（它们各自返回一个 Value），再 create 加法 op 把它们当操作数。这天然保证了 SSA 的**支配性**——操作数一定先于使用者被定义。这是 SSA 不变量在构建时的隐式维护，第 8 章会专门讲。

**AST 字面量 → ConstantOp（属性挂接）**（[MLIRGen.cpp:261-284](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L261)）：

```cpp
mlir::Value mlirGen(LiteralExprAST &lit) {
  ...
  auto dataAttribute = mlir::DenseElementsAttr::get(dataType, data);  // 把数据打包成属性
  return builder.create<ConstantOp>(loc(lit.loc()), type, dataAttribute); // ← Step 5：属性挂到 op
}
```

**AST 变量引用 → 符号表查 Value（不建任何 op）**（[MLIRGen.cpp:217-224](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L217)）：

```cpp
mlir::Value mlirGen(VariableExprAST &expr) {
  if (auto variable = symbolTable.lookup(expr.getName()))  // 查表，拿到已登记的 Value
    return variable;                                       // 直接复用，不建新 op
  ...
}
```

### 6.6.3 把 6.6.2 串成一条流水线

```text
ModuleAST
  └─ mlirGen(ModuleAST):  ModuleOp::create()                      → Step 1（建根）
     └─ FunctionAST
        └─ mlirGen(FunctionAST):  create<FuncOp> + 挪插入点 + 登记参数 → Step 2,3,4
           └─ ReturnExprAST
              └─ BinaryExprAST(+)
                 ├─ VariableExprAST(arg0)  → symbolTable.lookup  → %arg0（复用 Value）
                 └─ NumberExprAST(1)       → create<ConstantOp>  → %c1（Step 5，带属性）
                 → create<AddOp>(%arg0, %c1) → %sum              （Step 6，织 use-def 边）
              → create<ReturnOp>(%sum)                            （Step 7，terminator）
```

这条流水线完美对应 6.4 的 Step 1-7。**前端不是手动一个个 create，而是递归遍历 AST，每个节点对应一段 build**。这就是 MLIR 前端的标准模式——Dragon Book 第 5 章的语法制导翻译，在 MLIR 里的具体实现。

> **编译原理浸润点：前端语义分析与 IR 构造**　Dragon Book 第 2-4 章描述的前端流水线（词法→语法→语义→IR 构造）在 MLIR 里被简化了。MLIR 假设你已经有了 AST（由前端工具如 Toy 的 parser 产生），`MLIRGen` 只负责"AST → IR"这一段——即语义分析 + IR 构造。它用 OpBuilder 一砖一瓦地把 AST 翻成 IR 树，每翻一个节点就执行 6.4 的一两步挂接。这个模式是所有 MLIR 前端的通用骨架。

---

## 编译原理浸润点回顾

1. **前端流水线（AST→IR）**：本章主题。对应 Dragon Book 第 2-4 章的前端，尤其是第 5 章的语法制导翻译。MLIR 的 MLIRGen 是这个模式的具体实现。
2. **符号表**：AST 变量名 → SSA Value 的桥梁。Dragon Book 第 2.7 节的符号表角色，在 MLIR 前端里 value 端变成了 SSA Value。
3. **SSA 支配性在构建时的保证**：递归构建"先算操作数后算使用者"天然保证操作数先于使用者定义——这是 SSA 支配性的隐式维护。Ch8 会专门讲。
4. **结构树 vs 数据流图**：本章建立"两类挂接"心智模型——结构挂接（①-⑥）织结构树，数据流挂接（⑦）织数据流图。这是理解所有 IR 构建动作的钥匙。

---

## 本章关键结论

1. **构建 IR = 反复执行两类挂接**：结构挂接（把节点装进容器，6 类）+ 数据流挂接（把 Value 接成 operand，1 类）。七类挂接总表是本章的核心产出。
2. **OpBuilder 是带插入点的工人**：`block` + `insertPoint` 决定下一个 op 落在哪。`create<>` 内部完成"填装箱单 → create 分配 → insert 挂接"三段式。
3. **OperationState 是装箱单**：栈上临时结构，解耦"声明意图"（build 填单）与"实际分配"（create 消费）。
4. **七类挂接各有不同的持有形式**：①Region[] 数组、②attrs 指针、③④iplist、⑤vector+owner、⑥前缀数组（无 owner，算术定位）、⑦OpOperand 一身二任（双向 O(1) 查询的根基）。
5. **前端 = 递归遍历 AST + 符号表 + OpBuilder**：每个 AST 节点对应一段 build。递归顺序天然保证 SSA 支配性。
6. **Operation::create 的内部细节（尾分配）留给 Ch7**：本章只关注"填单 + 挂接"的语义，create 如何一次 malloc 分配所有东西是下一章的主题。

---

## 下一章预告

本章讲了构建的"外壳"——OpBuilder、OperationState、七类挂接。但 `Operation::create` 内部到底怎么把装箱单变成一个真实的、挂在内存里的 Operation 对象？那个让很多人困惑的**尾分配（trailing objects）内存模型**——"前缀结果 + 主体 + 尾分配"——到底是什么样子？为什么 MLIR 要用一次 malloc 分配所有东西？第 7 章打开 `Operation::create` 的源码，把这张"内存布局图"逐字节讲清。同时讲 MLIR 的声明式 op 定义——`.td → .inc → C++` 三段式管线，它让定义新 op 变成填表而非写代码。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-IR-树的构建过程教程_精品.md` §0,2-5（三类基础设施、目标例子、7 步构建、七类挂接总表、AST→IR 前端 MLIRGen）——**全文保留，重新组织为编译器视角叙事**
- §6（use-def 边织出过程）回扣 Ch4，本章只引用不重复详解
- 编译原理铺垫（前端流水线、语法制导翻译、符号表）为本书新增，对应 Dragon Book Ch.2-5

## 参考文献

- **[Aho 2006]** Dragon Book，第 2.7 节（符号表）、第 5 章（语法制导翻译）、第 8.3 节（IR 构造）。
- **[Lattner 2020]** Lattner et al. "MLIR"，OpBuilder 与 OperationState 的设计。
- **MLIR Toy Tutorial** Ch.2，`MLIRGen.cpp`——前端递归下降的标准模式。
