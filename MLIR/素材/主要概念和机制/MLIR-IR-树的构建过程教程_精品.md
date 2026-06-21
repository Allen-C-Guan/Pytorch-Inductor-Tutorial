# 从零构建一棵 MLIR IR 树：AST → Module → 挂接 → 遍历 lowering

**基于 LLVM 19.1.7 源码的完整教程**

> 本文是 [MLIR IR 中的 Node 组织](MLIR-IR-Node组织与遍历插入删除教程.md) 的姊妹篇。
> 那一篇回答「**IR 长什么样、由什么数据结构拼成**」（静态结构）；
> 本篇回答「**这棵树是怎么从无到有、一步一步被构建出来的**」（动态构建）。
> 两者合起来，才是 MLIR IR 的完整图景。

---

## 摘要

MLIR 里没有一棵孤立的"树"。当你构建 IR 时，你其实在**同时编织两张网**：

- **结构树**：`Operation ⊃ Region ⊃ Block ⊃ Operation ⊃ …` 的嵌套——谁包含谁。
- **数据流图**：`Value → OpOperand → Value` 的 use-def 链——谁的数据流向谁。

构建 IR 的本质，就是按一定顺序执行**两类「挂接（attach）」动作**，把这两张网一点点织起来：

| 挂接类型 | 干什么 | 底层动作 | 你调的接口 |
|---|---|---|---|
| **结构挂接** | 把一个节点装进容器 | iplist 插入 / placement-new `Region(op)` | `OpBuilder::create<>`、`createBlock`、`setInsertionPoint*` |
| **数据流挂接** | 把一个 Value 接成另一个 op 的操作数 | `OpOperand` 构造 → `insertInto` 进 Value 的 use-list | create 时传 operands |

本教程选一个最小但完整的例子——计算 `return arg0 + 1` 的函数——逐行走读它**从 AST 一路 build 成 IR**的全过程，每一步都标注：

1. **调用了哪个 MLIR/LLVM 关键接口**（`OpBuilder::create` → `OpTy::build` → `Operation::create` → `insert`）；
2. **这一步「挂接」了什么、挂到了谁身上**（Region→Op、Block→Region、Op→Block、Attr→Op、Operand→Value…）；
3. 最后如何用 `walk()` **遍历这棵树**，作为 lowering（渐进下沉）的起点。

**关键词**：IR 构建；OpBuilder；OperationState；插入点；挂接；use-def；AST → MLIR；walk；progressive lowering

---

## 目录

- ## 0. 前置：MLIR IR = 结构树 ∪ 数据流图
- ## 1. 三个基础设施：MLIRContext / OpBuilder / OperationState
- ## 2. 目标例子与 MLIR 文本语法
- ## 3. 一步步构建（7 步，每步带「挂接」标注）
- ## 4. 「挂接」本质：一张总表
- ## 5. 从 AST 到 MLIR：前端的递归下降（Toy `MLIRGen`）
- ## 6. 遍历这棵树 → lowering 的起点
- ## 7. 全局俯瞰：结构树 × 数据流图的数据结构全景
- ## 附录：构建 API 速查 + 关键源码索引

---

## 0. 前置：MLIR IR = 结构树 ∪ 数据流图

### 0.1 把 IR 当"盖房子"来理解

构建一个 IR，就像盖一栋房子，要同时干两件事：

- **搭骨架（结构树）**：先打地基（`Module`），再砌墙（`FuncOp`），墙里放房间（`Block`），房间里摆家具（`Operation`）。这是**谁包住谁**的嵌套关系。
- **走管线（数据流图）**：在骨架里布置水管/电线——这间房的水（数据）从哪间房接过来、流向哪间房。这是**数据怎么流**的依赖关系。

MLIR 的所有构建动作，无非是这两类：

```
                     【结构树】谁包含谁
        ModuleOp ─region→ [Block: { FuncOp ─region→ [Block: { addi, constant, return }] }]
            │
            └ 这是一棵「嵌套树」，靠 iplist + Region.container 指针维系

                     【数据流图】谁的数据流向谁
        %arg0 ──┐
                ├──→ addi ──→ %sum ──→ return
        %c1  ──┘
            │
            └ 这是一张「SSA 图」，靠 OpOperand → Value 的 use-def 链维系
```

> **核心洞察（贯穿全文）**：每一次 `OpBuilder::create<>`，都会**同时**触发两类挂接——结构上把 op 插进当前 Block，数据流上把每个 operand 接进对应 Value 的 use-list。这两件事在 `Operation::create` 内部一次性完成。

### 0.2 编译栈定位

```
源语言前端 ──AST──→ 【本文边界：AST → MLIR】 ──→ MLIR 多层 dialect ──lowering──→ LLVM IR ──→ 机器码
                        │                         (func/arith/.../north_star)
                     你在这里：
                     用 OpBuilder 一砖一瓦
                     把 AST 翻成 IR 树
```

本文聚焦编译栈最前端的这一段：**怎么把前端解析出的 AST，用 MLIR 的 C++ API 织成一棵 IR 树**。下游的 lowering（dialect 之间逐层下沉）只是"在这棵树上继续走读 + 重写"，留到 §6 开个头，详尽内容见 [NorthStar 教程 CH-12/15](../MLIR-Tutorial)。

### 0.3 设计哲学：为什么是 `OpBuilder` + `OperationState` + 插入点？

回头看，这三个东西解决了三个工程问题：

| 要解决的问题 | MLIR 的答案 | 为什么不换个更简单的做法 |
|---|---|---|
| **怎么保证 IR 节点被正确地"放进容器"？** | `OpBuilder` 维护一个**插入点** `(Block*, iterator)`，`create` 出来的 op 自动插进去 | 若让调用方手动 `block->push_back(op)`，极易忘记挂接 → 产生游离 op（不在任何 block 里，析构断言爆炸）。集中托管 = 不可能忘记 |
| **op 的字段那么多（operands/types/attrs/regions/successors），怎么统一传递？** | 用 `OperationState` 当**"装箱单"**：`OpTy::build` 往里塞，`Operation::create` 取出来 | 若 create 直接吃 8 个参数，每个 op 的 build 签名都要重复一遍且容易写错。装箱单 = 解耦"声明意图"与"实际分配" |
| **Type/Attr 创建太频繁，怎么去重？** | `MLIRContext` 里的 **StorageUniquer** 做 intern（唯一化） | 两个 `i32` 在内存里是同一个指针。对比 = 指针比较 O(1)，省内存 |

---

## 1. 三个基础设施

### 1.1 `MLIRContext` —— 整个宇宙的拥有者

`MLIRContext` 是一切 IR 对象的**根拥有者**。但"拥有"在这里有精确含义：**用一组具体的数据结构（哈希表、分配器、值成员）把对象攥在手心，对象的生命周期与 context 绑定，context 析构时它们一起消失。** 本节就回答你最关心的那句——"**用什么形式持有？容器、字典、还是什么？**"

```cpp
mlir::MLIRContext ctx;
ctx.getOrLoadDialect<mlir::func::FuncDialect>();   // 用到哪个方言就加载哪个
ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
```

#### 1.1.1 第一层：`MLIRContext` 自己是个空壳（pImpl）

**第一个反直觉点**：`MLIRContext` 类本身几乎不持有任何东西——它用 C++ 的 **pImpl（pointer to implementation）手法**，整个类只有一个私有成员，一个指向"隐藏实现对象"的智能指针：

```cpp
// MLIRContext.h:297
class MLIRContext {
private:
  const std::unique_ptr<MLIRContextImpl> impl;   // ← 唯一成员
};
```

真正的持有结构全在那个对头文件不可见的 [`MLIRContextImpl`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp)（struct 定义起于 :123）里。pImpl 的好处：把内部容器挪到 .cpp，对外只暴露 `getImpl()` 引用，头文件干净、ABI 稳定。

#### 1.1.2 第二层：`MLIRContextImpl` 用这些容器持有东西

下表是它**实际用哪些容器持有哪些东西**（全部经源码逐行核对 + 对抗式复核）：

| 持有什么 | 容器/形式 | key → value | 源码 |
|---|---|---|---|
| **已加载的 dialect 实例** | `DenseMap<StringRef, std::unique_ptr<Dialect>> loadedDialects` | namespace 名 → owned Dialect | [MLIRContext.cpp:196](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L196) |
| **可加载 dialect 的构造器** | `DialectRegistry dialectsRegistry`（值对象；内部 `std::map<string, pair<TypeID, allocator>>`） | namespace → "怎么 new" | [MLIRContext.cpp:197](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L197) |
| **type 唯一化池** | `StorageUniquer typeUniquer`（按值内嵌） | TypeID → 一池 Storage | [MLIRContext.cpp:214](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L214) |
| **attribute 唯一化池** | `StorageUniquer attributeUniquer` | TypeID → 一池 Storage | [MLIRContext.cpp:246](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L246) |
| **affine 唯一化池** | `StorageUniquer affineUniquer` | TypeID → 一池 Storage | [MLIRContext.cpp:207](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L207) |
| **出现过的 op 名** | `llvm::StringMap<std::unique_ptr<OperationName::Impl>> operations` | op 全名 → 描述符 | [MLIRContext.cpp:183](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L183) |
| **已注册 op（多索引）** | `DenseMap<TypeID,RON>` + `StringMap<RON>` + `SmallVector<RON>`（排序快照） | TypeID / 名字 → RegisteredOperationName | [MLIRContext.cpp:186-191](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L186) |
| **已注册 type/attr 元信息** | `DenseMap<TypeID, AbstractType*>` + `DenseMap<StringRef, AbstractType*>`（attr 同构） | TypeID / 名字 → 元信息 | [MLIRContext.cpp:213/221](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L213) |
| **上述小对象的内存** | `llvm::BumpPtrAllocator abstractDialectSymbolAllocator` | —— | [MLIRContext.cpp:180](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L180) |
| **热点 type/attr 缓存** | 值成员 `IntegerType int32Ty; Float32Type f32Ty; BoolAttr trueAttr; ...` | —— | [MLIRContext.cpp:224-261](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L224) |
| 诊断引擎 / 线程池 / action handler | `DiagnosticEngine diagEngine` / `ThreadPoolInterface*` / `std::function` | —— | :137 / :173 / :131 |

**读法**：dialect 实例用 `DenseMap<名字, unique_ptr>` 持有（context 独占所有权）；type/attr 用三个 `StorageUniquer` 值成员持有（下一节的戏份）；op 名用 `StringMap` 持有。**几乎没有"裸指针悬空"——能用 `unique_ptr`/值成员的地方都用了；少数裸指针（如 `AbstractType*`）的内存由 `BumpPtrAllocator` 兜底，context 析构时整体回收。**

所有权全景图：

```
MLIRContext
 └─ unique_ptr ─→ MLIRContextImpl (藏在 .cpp 里)
                  ├─ DenseMap<StringRef, unique_ptr<Dialect>> loadedDialects   ← dialect 全在这里
                  ├─ StorageUniquer typeUniquer       ┐
                  ├─ StorageUniquer attributeUniquer  ├─ 三个 intern 池（type/attr/affine）
                  ├─ StorageUniquer affineUniquer     ┘
                  ├─ StringMap<unique_ptr<OpName::Impl>> operations  ← op 名表
                  ├─ DenseMap<TypeID,AbstractType*> registeredTypes  (+ nameToType / attr 同构)
                  ├─ BumpPtrAllocator   ← 给上面那些小对象分配内存
                  ├─ int32Ty / f32Ty / trueAttr / ...（缓存的热点 type/attr 实例）
                  └─ diagEngine / threadPool / actionHandler
```

#### 1.1.3 第三层：`StorageUniquer` —— type/attribute 的"对象池"

这是回答"`IntegerType::get(ctx, 32)` 到底把对象存在哪"的关键。`StorageUniquer` 内部（[StorageUniquer.cpp](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Support/StorageUniquer.cpp)）用**两张按 TypeID 分桶的哈希表**组织：

```cpp
// StorageUniquer.cpp:347-353
DenseMap<TypeID, std::unique_ptr<ParametricStorageUniquer>> parametricUniquers;  // 带参数的类型（如 i32、NSTensor<...>）
DenseMap<TypeID, BaseStorage *> singletonInstances;                              // 单例类型（如 index、none）
```

- 每注册一种带参数的类型（比如 `IntegerType`），就用它的 `TypeID` 在 `parametricUniquers` 里建一个专属桶 `ParametricStorageUniquer`。
- 为了多线程减少锁竞争，**每个桶又按 hash 值的低几位切成多个 `Shard`（默认 8 个，须为 2 的幂）**，每片各持一把读写锁。
- **真正装"唯一化后的 Storage 对象"的容器**，是每个 Shard 里的：

```cpp
// StorageUniquer.cpp:42-47, 76
struct HashedStorage { unsigned hashValue; BaseStorage *storage; };   // 存的是指针，不是对象副本
using StorageTypeSet = DenseSet<HashedStorage, StorageKeyInfo>;        // = Shard::instances
```

即：**一个 `DenseSet`（哈希集合），元素是"预计算 hash + 指向 storage 的裸指针"。** storage 本身的内存来自一个 `BumpPtrAllocator`——**分配后不单独释放，直到 context 析构才整体回收，因此叫"immortal（永生）"**。这就是"同一个 i32 永远是同一个指针"的物理来源。

```
typeUniquer
 └─ parametricUniquers: DenseMap<TypeID, unique_ptr<ParametricStorageUniquer>>
      └─ [TypeID = IntegerType] → ParametricStorageUniquer
            └─ shards[8]（按 hash 切片）
                  └─ shard.instances: DenseSet<HashedStorage>
                        └─ { hash, &IntegerTypeStorage(width=32, signless) }   ← 唯一的 i32 实例
                                （storage 内存来自 BumpPtrAllocator，永生）
```

#### 1.1.4 "持有"的实际动作：一次 `IntegerType::get(ctx, 32)` 的旅程

把上面三层串起来。你写 `builder.getI32Type()`（最终落到 `IntegerType::get(ctx, 32)`），完整路径：

```
IntegerType::get(ctx, 32)                                  // 你调的
  MLIRContext.cpp:1060-1091
   ├─ 先查缓存：context 里的 int32Ty 成员（命中就直接返回，O(1)）         ← 大多数 i32 走这条
   └─ 未命中 → Base::get(ctx, 32, signless)                  // StorageUserBase::get (StorageUniquerSupport.h:173)
        └─ TypeUniquer::get<IntegerType>(ctx, 32, ...)       // TypeSupport.h:212
             └─ ctx->getTypeUniquer().get<IntegerTypeStorage>(initFn, typeID, 32, signless)
                  // ↑ 拿的是 context 持有的那个 typeUniquer 成员（MLIRContext.cpp:1010）
                  └─ StorageUniquer::get (StorageUniquer.h:194-219)
                       ├─ getKey  → KeyTy = (32, Signless)
                       ├─ getHash → hashValue
                       ├─ isEqual = lambda（调 IntegerTypeStorage::operator==）
                       └─ ctorFn  = lambda（调 IntegerTypeStorage::construct）
                            └─ getParametricStorageTypeImpl → getOrCreate (StorageUniquer.cpp:263)
                                 ├─ parametricUniquers[TypeID] 取出 IntegerType 的桶
                                 ├─ getShard(hashValue) 选分片
                                 ├─ 读锁 find_as(LookupKey)：命中 → 返回已有 storage 指针 ✅
                                 └─ 未命中 → 写锁 getOrCreateUnsafe → insert_as + ctorFn() 真正 new 一个 storage
```

**结论**：`IntegerType` 对象（`IntegerTypeStorage`）的内存由 context 的 `typeUniquer` 池持有，**不属于任何 op**。`IntegerType` 这个 C++ 类本身只是 `ImplType *impl` 一个指针——[Types.h:20-31](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Types.h) 注释明说 *"wraps a pointer to the storage object owned by MLIRContext"*。`Attribute` 完全同构（[Attributes.h:19-25](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Attributes.h)，*"references to immortal ... storage owned by MLIRContext"*）。

> 这就解释了本篇反复出现的论断——**"type/attribute 属于 context，不属于 op"**：op 只是在自己的 `attrs` 字段里存了**一个指向 context 池中字典对象的指针**（见 §3 Step 3 的 Attribute 挂接）。op 销毁不会动 context 池里的一根毫毛。

### 1.2 `OpBuilder` —— 干活的工人

[`OpBuilder`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h) 继承自 [`Builder`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L50)，身兼两职：

```cpp
// Builders.h:50-206 —— Builder 基类：创建 context 级对象（Type/Attr/Loc）
class Builder {
  template <typename Ty, typename... Args> Ty getType(Args&&...);   // get I32Type 等
  template <typename Attr, typename... Args> Attr getAttr(Args&&...);
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

### 1.3 `OperationState` —— 装箱单

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
  ...
  // 一堆 add* 方法往里塞：addOperands / addTypes / addAttribute / addRegion / addSuccessors
};
```

它的角色是 **"填表"**：

```
你（或 OpTy::build）填表 OperationState  ──递交给──→  Operation::create（消费装箱单，分配内存）
```

`addRegion` / `addAttribute` 这些方法就是**往装箱单里塞东西**，比如 `state.addRegion()` 塞一个空 region，`state.addAttribute(name, attr)` 塞一个属性。**此时还没真正建 op**，东西只是暂存在单子上。

---

## 2. 目标例子与 MLIR 文本语法

我们要从零造出来的 IR，文本长这样（你应能独立读懂）：

```mlir
module {
  func.func @add_one(%arg0 : i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %sum = arith.addi %arg0, %c1 : i32
    return %sum : i32
  }
}
```

### 2.1 语法速读（读懂 `.mlir` 的最小集合）

| 语法片段 | 含义 | 对应的 IR 实体 |
|---|---|---|
| `module { ... }` | 顶层容器，自带 1 个 Region、1 个 Block | `ModuleOp` |
| `func.func @add_one(...) -> i32 { ... }` | 一个函数；`@add_one` 是符号名属性 | `func::FuncOp`，结果类型是 `FunctionType` 属性 |
| `%arg0 : i32` | 函数的入口块参数，类型 i32 | **`BlockArgument`**（一种 `Value`） |
| `%c1 = arith.constant 1 : i32` | 常量 op，`1` 是它的属性 | `arith::ConstantOp`；`1` 是一个 `IntegerAttr` |
| `%sum = arith.addi %arg0, %c1 : i32` | 加法，两个操作数 `%arg0`/`%c1`，产出一个 `%sum` | `arith::AddIOp`；操作数 = use-def 边 |
| `return %sum : i32` | 终结指令，把 `%sum` 返回 | `func::ReturnOp`（terminator） |

注意四类东西在本例中**全部出现**：

- **Attribute**：`@add_one`（StringAttr，函数名）、`1`（IntegerAttr，常量值）、函数类型（TypeAttr）。
- **Value**：`%arg0`（BlockArgument）、`%c1`/`%sum`（OpResult）。
- **Block**：函数体里那个 `{ ... }` 对应的块；module 也有自己的 body 块。
- **Region**：`module { }` 和 `func.func { }` 各有一个 Region。

下面逐一把它们"造"出来。

---

## 3. 一步步构建（7 步，每步带「挂接」标注）

> 下方 C++ 是**示意写法**（用标准方言 `func`/`arith`，便于看清四类挂接）。你仓库里**可编译的同类代码**是 [NorthStar 教程 CH-5 的 `main.cpp`](../MLIR-Tutorial/5-define_operation/main.cpp#L240)，它用 `north_star::ConstOp/BufferOp/SoftmaxOp` 做完全一样的事——文末 §5.4 会对照。

### Step 0：准备 —— context + 方言 + builder + 位置 + 类型

```cpp
mlir::MLIRContext ctx;
ctx.getOrLoadDialect<mlir::func::FuncDialect>();
ctx.getOrLoadDialect<mlir::arith::ArithDialect>();

mlir::OpBuilder builder(&ctx);                  // 建工人，插入点暂为空
mlir::Location loc = builder.getUnknownLoc();   // 位置（追溯用，这里用未知）
mlir::Type i32 = builder.getI32Type();          // 类型：intern 到 ctx（O(1) 查/插）
```

**挂接标注**：这一步不挂接任何 IR 节点，只是**备料**——`i32` 是个被 context 唯一化的 Type 句柄，后面所有地方复用同一个指针。

> 注意 `OpBuilder` 构造后 `block == nullptr`（[Builders.h:614](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L614)）。此时直接 `create` 会得到一个**游离 op**（不在任何 block）。所以下一步必须先建立"根"，再把插入点移进去。

**此刻全景**（本步新增 = 无 IR 节点，只是备料）：

```
IR 树：（空 —— 还没有任何 Operation / Region / Block）

context 内部（不在 IR 树里，但已就位，见 §1.1）：
  loadedDialects = { "builtin", "func", "arith" }
  typeUniquer 池里有 IntegerTypeStorage(width=32)   ← i32 住这里
builder.block == nullptr                            ← 插入点空
```

### Step 1：建根 `ModuleOp`

```cpp
mlir::ModuleOp module = builder.create<mlir::ModuleOp>(loc);
//    ↑ 此时插入点仍是空，但 ModuleOp 是根，不需要插进别的 block
```

**调用了哪些接口**：`OpBuilder::create<ModuleOp>(loc)` ([Builders.h:508-517](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L508))。展开它的内部，是 MLIR 构建 op 的**标准三段式**：

```cpp
// Builders.h:508-517（简化）
template <typename OpTy, typename... Args>
OpTy create(Location location, Args&&... args) {
  OperationState state(location, getCheckRegisteredInfo<OpTy>(ctx)); // ① 开装箱单
  OpTy::build(*this, state, std::forward<Args>(args)...);            // ② ModuleOp::build 填单
  auto *op = create(state);    // = Operation::create(state) 然后 insert   ③ 分配 + 挂接
  return cast<OpTy>(op);
}
```

**`ModuleOp::build` 填了什么单**（[BuiltinDialect.cpp:133-140](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/BuiltinDialect.cpp#L133)）：

```cpp
void ModuleOp::build(OpBuilder &builder, OperationState &state,
                     std::optional<StringRef> name) {
  state.addRegion()->emplaceBlock();   // ★ 给 module 配 1 个 Region，并往里塞 1 个 Block
  if (name)
    state.addAttribute("sym_name", builder.getStringAttr(*name));  // 可选属性
}
```

**`Operation::create` 怎么消费这张单**（[Operation.cpp:82-153](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L82)）：

```cpp
// 摘自 Operation.cpp:134-150（简化）
for (unsigned i = 0; i != numRegions; ++i)
  new (&op->getRegion(i)) Region(op);     // ★★ 挂接①：placement-new Region，并把 Region.container 指向 op
...
op->setAttrs(attributes);                  // ★★ 挂接②：把属性字典挂到 op 上
```

**Step 1 的挂接清单**（本例 `create<ModuleOp>(loc)` 没传 name，故无属性）：

| 挂接 | 谁 → 谁身上 | 哪行代码 |
|---|---|---|
| **Region → Op ①** | `module` 的 1 个 Region，`.container = module` | [Operation.cpp:135-136](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L135)（`new Region(op)`）|
| **Block → Region ④** | Region 里 emplace 了一个 Block | [BuiltinDialect.cpp:135](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/BuiltinDialect.cpp#L135)（`addRegion()->emplaceBlock()`）|

> 🏗️ **数据结构 · 挂接①（Region→Op）怎么挂？**
> `Operation` 用 LLVM 的**尾分配（trailing objects）**：在自己这块堆内存的**尾部预留一个 `Region[]` 定长数组**（[Operation.h:86](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L86) 的 `TrailingObjects<..., Region, ...>`）。`Operation::create` 对每个 region 做 placement-new，构造时传入 `op` 指针 → Region 的成员 `Operation *container` 回指属主 op（[Region.h:334](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Region.h#L331)）。
> **持有形式 = op 尾部的 `Region[]` 数组 + 每个 Region 一个 `container` 回指指针。属主 = op。**

> 🏗️ **数据结构 · 挂接④（Block→Region）怎么挂？**
> Region 持有一个 **`llvm::iplist<Block>`**（侵入式双向链表，[Region.h:44](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Region.h#L44)）；Block 继承 `ilist_node_with_parent<Block, Region>`，**自己就是链表节点**（不再外部分配节点包装）。`emplaceBlock()` 把一个 `new Block()` splice 进这条链表。
> **持有形式 = 侵入式双向链表，零额外节点分配，O(1) 插入/删除。属主 = region。**

**此刻全景**（★ = 相比上一步新增；本步新增 1 op + 1 region + 1 block，root 不在任何 block 里）：

```
★ ModuleOp                          ← 根；插入点为空 → insert 跳过，它是游离 root
  └─ ★ region#0
      └─ ★ Block (空)              ← emplaceBlock 建的 module body 块
          ↑ Region.container 指回 ModuleOp

builder.block == nullptr（仍未设）
```

### Step 2：把插入点移进 module 的 body

```cpp
builder.setInsertionPointToStart(module.getBody());   // module.getBody() = 那个 Block
```

**调用的接口**：[`setInsertionPointToStart`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L434) → 把 builder 的 `block` 设成 module 的 body 块、`insertPoint` 设成块首。

> 这是"一步一步挂接"的关键动作——**先定位，再 create**。之后所有 `create` 的 op 都会长在这个 block 里。

**此刻全景**（树结构不变，只挪了 builder 的指针）：

```
ModuleOp
  └─ region#0
      └─ Block (空)
          ↑ ↑ ↑
          ★ builder.block 现在指向这里，insertPoint = begin()
          （下一次 create 的 op 会落到这个 block 里）

builder.block → module body block 的 begin()
```

### Step 3：建 `FuncOp`（自带 region + 入口块 + 块参数）

```cpp
auto fnType = builder.getFunctionType({i32}, {i32});   // (i32) -> i32，也是个 intern 的 Type
mlir::func::FuncOp func =
    builder.create<mlir::func::FuncOp>(loc, "add_one", fnType);
```

**`FuncOp::build` 填了什么**（[FuncOps.cpp:180-195](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Dialect/Func/IR/FuncOps.cpp#L180)）：

```cpp
void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                   FunctionType type, ...) {
  state.addAttribute("sym_name", builder.getStringAttr(name));   // 函数名属性
  state.addAttribute("function_type", TypeAttr::get(type));      // 函数类型属性
  state.addRegion();   // ★ 配 1 个空 Region（注意：此时还没 block！）
  ...
}
```

> ⚠️ **重要澄清（本教程的关键修正点）**：`func::FuncOp::build` 只 `addRegion()`——**建一个空 region，并不创建入口块，也不创建 `%arg0`！** 入口块和块参数需要**单独一步**显式创建。两种做法：
> - 调 `FunctionOpInterface` 的 `func.addEntryBlock()`（便利方法，内部就是 `new Block` + `addArguments`）；
> - 直接用通用的 `OpBuilder::createBlock(&func.getBody(), {i32}, {loc})`——**本教程选这条**，因为它把"建 Block + 建 BlockArgument"两个挂接动作完全摊开给你看。
>
> （对照：Toy 方言的 [`toy::FuncOp::build`](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/Dialect.cpp#L206) 直接调 `buildWithEntryBlock(..., type.getInputs())` 把入口块和参数在 build 里一次建好——那是 Toy **自愿**加的便利，标准 `func::FuncOp` 没这么做。这也是为什么 Toy 的 `mlirGen` 能直接 `function.front()` 拿到带参数的入口块。）

**`Operation::create` 完成分配后，`OpBuilder::insert` 把 func 挂进 module 的 body**（[Builders.cpp:432-439](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L432)）：

```cpp
Operation *OpBuilder::insert(Operation *op) {
  if (block)
    block->getOperations().insert(insertPoint, op);   // ★★ 挂接③：把 op 插进当前 block 的 iplist
  return op;
}
```

而 `OpBuilder::create(OperationState&)` 就是 `insert(Operation::create(state))`（[Builders.cpp:468-470](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L468)）——**先分配，再挂接**。

**Step 3 的挂接清单**：

| 挂接 | 谁 → 谁身上 | 代码 |
|---|---|---|
| **Attribute → Op ②** | `sym_name`、`function_type` | [FuncOps.cpp:183-185](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Dialect/Func/IR/FuncOps.cpp#L183) |
| Region → Op ① | func 的函数体 Region，`.container = func` | [FuncOps.cpp:187](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Dialect/Func/IR/FuncOps.cpp#L187) + [Operation.cpp:135-136](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L135) |
| **Op → Block ③** | func 被插进 module 的 body 块 iplist | [Builders.cpp:434](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L434) |

> 🏗️ **数据结构 · 挂接②（Attribute→Op）怎么挂？** —— 回答"属性用什么形式挂到 op"
> `Operation` 有一个成员字段 **`DictionaryAttr attrs`**（[Operation.h:1066](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L1066)）——**就是一个指针**。`DictionaryAttr` 本身是 context 里被唯一化的"字典对象"的指针包装，指向 §1.1.3 那个 `attributeUniquer` 池里的 immortal storage。所以 **op 不存属性的副本，只存一个指向 context 拥有的字典的指针**——所有属性（sym_name、function_type、value…）先被打包成一个 `DictionaryAttr`，再让 op 拿到它的指针。
> **持有形式 = op 上的一个 `DictionaryAttr` 指针（指向 context 池里的字典）。属主 = context（StorageUniquer），不是 op。**

> 🏗️ **数据结构 · 挂接③（Op→Block）怎么挂？**
> Block 持有 **`llvm::iplist<Operation>`**（侵入式双向链表，Block.h 的 `OpListType operations`）；Operation 继承 `ilist_node_with_parent<Operation, Block>`，**自己就是链表节点**。`OpBuilder::insert` 调 `block->getOperations().insert(insertPoint, op)` 把 op 节点 splice 进链表（[Builders.cpp:434](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L434)）。
> **持有形式 = 侵入式双向链表，O(1) 插入/删除。属主 = block。**（Step 2 挪插入点的意义正在此：`insertPoint` 决定 splice 到链表的哪个位置。）

**此刻全景**（★ 新增 FuncOp；注意 func 的 region 是**空的**——还没有入口块，也没有 `%arg0`）：

```
ModuleOp
  └─ region#0
      └─ Block (module body)               ← builder.block 仍在这里
          └─ ★ FuncOp @add_one             ← 新增；insert 把它 splice 进 module body 的 iplist
              ├─ attrs: { sym_name="add_one", function_type=(i32)->i32 }   ← 挂接②（指针指向 context）
              └─ region#0                  ← 空！（FuncOp::build 只 addRegion()，没建 block）
```

### Step 4：建函数体的入口块 + 块参数 `%arg0`

Step 3 结束时 func 的 region 还是空的。现在显式创建入口块，并给它配一个 `i32` 类型的块参数——这一步**同时**完成"Block→Region ④"和"BlockArgument→Block ⑤"两个挂接：

```cpp
// 在 func 的 region 里建一个入口块，带一个 i32 类型、loc 位置的参数
mlir::Block *entry = builder.createBlock(&func.getBody(), {i32}, {loc});
mlir::Value arg0 = entry->getArgument(0);   // ★ %arg0 = 入口块的第 0 个 BlockArgument
```

`createBlock` 一次做四件事（[Builders.cpp:441-456](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L441)）：

```cpp
Block *b = new Block();
b->addArguments(argTypes, locs);          // ★ 建块参数 → 挂接⑤（BlockArgument→Block）
parent->getBlocks().insert(insertPt, b);  // ★ 块入 region → 挂接④（Block→Region，iplist）
setInsertionPointToEnd(b);                // 把插入点挪进新块
return b;
```

每个块参数的诞生是 [`Block::addArgument`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L152)：

```cpp
BlockArgument Block::addArgument(Type type, Location loc) {
  BlockArgument arg = BlockArgument::create(type, this, arguments.size(), loc);  // this = entry 块
  arguments.push_back(arg);     // 挂接⑤：BlockArgument 进 block 的 arguments 向量
  return arg;
}
```

> 🏗️ **数据结构 · 挂接⑤（BlockArgument→Block）怎么挂？**
> Block 持有 **`std::vector<BlockArgument>`**（Block.h 的 `arguments`）。每个 BlockArgument 内部 impl **显式存了一个 `Block *owner` 字段**（[Value.h:305](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h#L305) 的 `BlockArgumentImpl::owner`）回指属主 block。`addArgument` new 一个 `BlockArgumentImpl(type, this, index, loc)` push_back 进向量。
> **持有形式 = 动态数组（`std::vector`）+ 每个参数一个 owner 回指指针。属主 = block。**
>
> （**为什么 BlockArgument 要存 owner 指针，而下一步的 OpResult 不存？** 因为块参数住在一个独立的 vector 里，没法靠地址算术定位属主；而 OpResult 按逆序紧贴在 op 主体之前，能靠算术倒推——见 Step 5 的挂接⑥。）

**Step 4 的挂接清单**：

| 挂接 | 谁 → 谁身上 | 代码 |
|---|---|---|
| **Block → Region ④** | entry 块 splice 进 func.region#0 的 iplist | [Builders.cpp:450](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L450) |
| **BlockArgument → Block ⑤** | `%arg0` 这个 Value，owner = entry 块 | [Block.cpp:152-155](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L152) |

> **关键认知**：`%arg0` 是一个 **Value**（具体是 `BlockArgument` 子类），但它**不是任何 op 的产物**——它由 Block 拥有。这正是"Value 的两种来源"之一（另一种是 OpResult，下一步见）。它现在可以被任何 op 当作操作数使用。

**此刻全景**（★ 新增入口块 + `%arg0`；func 的 region 不再空）：

```
ModuleOp
  └─ region#0
      └─ Block (module body)
          └─ FuncOp @add_one
              ├─ attrs: { sym_name, function_type }
              └─ region#0
                  └─ ★ Block (entry)            ← 新增（createBlock 建的）
                      └─ ★ arg#0: %arg0 (i32)   ← 新增 BlockArgument
builder.block → entry 块（createBlock 自动把插入点设到 entry 末尾）
```

### Step 5：建 `arith.constant`（**属性挂接② + OpResult 挂接⑥**）

```cpp
mlir::Attribute oneAttr = builder.getIntegerAttr(i32, 1);   // 1 : i32，一个 IntegerAttr
mlir::Value c1 = builder.create<mlir::arith::ConstantOp>(loc, oneAttr);
```

`ConstantOp::build`（由 ODS 生成）把 `oneAttr` 作为名为 `value` 的属性塞进装箱单，并声明结果类型 `i32`。`Operation::create` 消费时：

- **挂接②（属性→Op）**：`op->setAttrs(attributes)`（[Operation.cpp:150](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L150)），把 `{value = 1 : i32}` 这个字典挂到 op 上。
- **挂接⑥（OpResult→Op）**：placement-new 一个 `InlineOpResult`（[Operation.cpp:127-132](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L127)）——这就是 `%c1` 这个 Value，它的 defining op = 这个 constant。
- **挂接③（Op→Block）**：constant 被 insert 进 entry 块。

> 🏗️ **数据结构 · 挂接⑥（OpResult→Op）怎么挂？** —— 全篇最反直觉的设计
> 这里 OpResult **不存 owner 指针**！结果按**逆序**存在 op 的**前缀区**（`InlineOpResult[0..4]` + 尾部 `OutOfLineOpResult[]`），内存布局是 `| OutOfLine results | Inline results | Operation |`（[Value.h:368-432](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h#L368)）。`OpResult::getOwner()` 靠**反向指针算术**，用 result number 倒推出 op 起始地址（Value.cpp `OpResultImpl::getOwner`）。
> **持有形式 = op 前缀区的定长结果数组（无 owner 字段，靠算术定位属主）。属主 = op。**
> （对比挂接⑤：BlockArgument 住独立 vector、没法靠算术定位，所以**必须**存 owner 字段——这就是 ⑤ 和 ⑥ 差别的根因。省下每个结果一个指针 × 大量结果 = 可观的内存。）

**此刻全景**（★ 新增 constant；注意 constant **无 operand**，所以**不织任何 use-def 边**——常量值走属性，不走 SSA 边）：

```
ModuleOp
  └─ region#0 └─ Block (module body)
      └─ FuncOp @add_one
          └─ region#0 └─ Block (entry)
              ├─ arg#0: %arg0 (i32)
              └─ ★ ConstantOp                      ← 新增，insert 进 entry
                  ├─ attrs: { value = 1 : i32 }    ← 挂接②（指针指向 context）
                  └─ result#0: %c1 (OpResult)      ← 挂接⑥（前缀区，算术定位 owner）
data-flow 边：无（constant 无 operand）
```

### Step 6：建 `arith.addi`（数据流挂接 / use-def 边）

```cpp
mlir::Value sum = builder.create<mlir::arith::AddIOp>(loc, arg0, c1);
//    操作数是 arg0(%arg0) 和 c1(%c1) —— 两个已存在的 Value
```

前几步的挂接（①~⑥）织的都是**结构树**——谁包含谁（注意：⑥ OpResult→Op 也算结构挂接——结果长在 op 自己的前缀区、属 op 自身结构，不是跨 op 的数据流边；**跨 op 的数据流边只有 ⑦**）。这一步不同：它织出**数据流图**——`%sum` 依赖 `%arg0` 和 `%c1`。这是全篇最重要的挂接，也是 use-def 机制的诞生处。

但别急着看代码怎么挂。use-def 是个**逻辑概念**（"谁定义了这个值、谁使用了它"），它背后靠一套**数据结构**支撑。这一节的目标，就是让**逻辑概念和数据结构一一咬合**。分三层讲：先认识机制里的几个对象（Part A），再把 use-def 概念逐条翻译成数据结构（Part B），最后才看挂接的动态过程（Part C）。

---

#### Part A — use-def 机制里的"演员"（对象 · 抽象 · 数据结构）

use-def 涉及五个核心对象（下面用 **【甲】~【戊】** 标注——**这是对象编号，和全文 §4 的挂接编号 ①~⑦ 不是一回事，别混**）。按 use-def 的"两半"分组看更清楚：**def 侧** = Value（甲）及其两种来源 OpResult（丁）/BlockArgument（戊）；**use 侧** = Operation 作消费者（乙）+ OpOperand 作唯一的边对象（丙）。

**【甲】`Value` —— 被定义、被使用的那个"值"（数据流的主角）**

- **抽象**：一个 SSA 值——"只定义一次、可被多处使用"。本例 `%arg0`、`%c1`、`%sum` 都是 Value。源码（[Value.h:87-92](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h#L87)）："a computable value that has a type and a set of users"。
- **两种来源（也只有两种）**：**OpResult**（op 算出的结果：`%c1`←constant、`%sum`←addi）与 **BlockArgument**（块的入口参数：`%arg0`←entry 块）。
- **数据结构**：`Value` 类极轻——**只是一个指针句柄**，全类只有一个成员 `detail::ValueImpl *impl`（[Value.h:96/253](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h#L96)）；所有方法转手问 `impl`。真正持数据的是 `ValueImpl`（`OpResultImpl`/`BlockArgumentImpl` 两个子类）：它自己的字段是 `typeAndKind`（把 type 和 kind 打包），并**经基类 `IRObjectWithUseList` 继承一个 use-list 头 `firstUse`**。

**【乙】`Operation`（op）—— 既是"生产者"又是"消费者"**

- **抽象**：一个计算节点——做加法、取常量…。它**产**结果（OpResult）、**用**操作数（operand Value）。addi 产出 `%sum`、消费 `%arg0`/`%c1`。
- **数据结构**（use-def 相关的两块存储）：
  - **前缀区 `OpResult[]`** —— 它**产**的若干结果（addi 的 `%sum` 在这）；
  - **尾部区一个 `OperandStorage`**（trailing object）——它内部持 `OpOperand[]` 数组，装它**用**的若干操作数对应的边记录（addi 有 2 个）。

**【丙】`OpOperand` —— 一条"使用"的记录（数据流边的载体，边对象）**

- **抽象**：代表"**op U 的第 i 个操作数槽 引用了 Value V**"这一条关系。**它不是节点、不是 op，是一条边（edge）**。先分清两个词：`operand`（小写、值）= 被使用的 Value（`%arg0`）；`OpOperand`（类、对象）= 记录这条使用关系的边（"addi 的槽#0 用了 `%arg0`"）。一句话：operand 是"箭头指向的目标值"，OpOperand 是"箭头本身"。
- **数据结构**：`OpOperand : public IROperand<OpOperand, Value>`（[Value.h:267](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h#L267)，**public 继承，不是别名**），四个字段：`owner`（`Operation*`，使用方）、`value`（`Value`，被引用值）、`nextUse`/`back`。后两个把它**串进一条链表——这条链表挂在被引用的那个 Value 身上，称为它的 use-list**（Part B 会讲它怎么和 def 对应）。它还隐含一个**operand number**——即自己在 owner 尾部数组里的下标（`getOperandNumber()`），靠地址偏移算出，不是独立字段。
- **和 op 的关系（易混点）**："OpOperand 被 op 拥有"指**它的内存/生命周期归 owner op 管**（长在 op 尾部、op 死它死），**不是**说它是 op 的一种——它是 op 的一根"入线接线柱"。

**【丁】`OpResult` —— 一条"定义"的载体（op 产出的那半边）**：Value 的一种，表示"由某 op 产出"；作为 `OpResultImpl` **物理长在产出它的 op 的前缀区**（`OpResult[]`），靠地址算术反查属主 op（`getDefiningOp()`，无独立 owner 字段）。

**【戊】`BlockArgument` —— 一条"定义"的载体（块参数那半边）**：Value 的另一种，表示"块的入口参数"；作为 `BlockArgumentImpl` **存在 block 的 `std::vector<BlockArgument>` 里**，带**显式 `Block *owner` 字段**反查属主块。

**两个"基类"是粘合剂**：`IRObjectWithUseList<OpOperand>`（ValueImpl 的基类，**内嵌 `firstUse` 头**，[UseDefLists.h:290](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L290)——每个 Value 经 ValueImpl 白嫖一个）与 `IROperandBase`（OpOperand 的基类，提供 `owner`/`nextUse`/`back` + `insertInto`/`removeFromCurrent` 动作）。

> 🔗 **顺带：控制流边 `BlockOperand` 与【丙】同构**。MLIR 把同一套 `IROperand`/`IRObjectWithUseList` 模板复用到控制流——terminator op 引用后继 Block 时用 `BlockOperand`（`IROperand<BlockOperand, Block*>`），它链进 Block 自己的 use-list，这就是 Block 能 O(1) 反查"谁跳到我"（前驱）的来源。**数据流边 = OpOperand，控制流边 = BlockOperand，同一套机制**。本例 addi/return 无后继边，故不展开。

---

#### Part B — use-def 这个逻辑概念，如何用数据结构表示（核心）

> 本节是整章的北极星：让**逻辑概念**和**数据结构**一一咬合。

use-def 这个逻辑概念，对任何一个 Value `V` 只有两种关系：

- **def**：`V` 被**恰好一个**生产者 `D` 定义（D 是某 op 或某 block）；
- **uses**：`V` 可能被**零个或多个**消费者 `U1, U2, …` 使用。

把这两句话逐条翻译成数据结构：

| 逻辑概念 | 数据结构怎么表示 | 查询 API |
|---|---|---|
| **`V` 由 `D` 定义**（1 个 def） | OpResult：V 作为 `OpResultImpl` **长在 D 的前缀区 `OpResult[]`**，地址算术反查 D；BlockArgument：V 在 block 的 vector，带 `owner` 字段 | `V.getDefiningOp()` |
| **`V` 被 `U` 使用**（1 条 use） | 一个 `OpOperand{owner=U, value=V}`：**物理**在 U 的尾部 `OpOperand[]`，**逻辑**链在 V 的 `firstUse` 链 | — |
| **`V` 的全部 uses**（use 集合） | V 的 `firstUse → nextUse → …` 单向链，每节点一个 OpOperand | `getUses()`→OpOperand(边，含槽位) / `getUsers()`→Operation(消费 op) |
| **`D` 产出的全部结果** | D 的前缀区 `OpResult[]`（result#0…n） | `D.getResults()` |
| **`U` 消费的全部操作数** | U 的尾部 `OpOperand[]`（operand#0…m） | `U.getOperands()` |

**最关键的是第二条：一条"use"逻辑关系 = 一个 `OpOperand` 对象，而且这个对象一身二任——**

```
              一个 OpOperand{owner=U, value=V}  ＝  "U 用了 V" 这条逻辑边
        ┌───────────────────────────────┴───────────────────────────────┐
   从 U（消费者）侧看                                   从 V（被使用的值）侧看
   它物理长在 U 的尾部 OpOperand[]                       它逻辑链在 V 的 firstUse 链
   → U 能说"我的 operand#i 是 V"（U.getOperand(i)）       → V 能说"我有 U 这个 user"（V.getUsers()）
```

这就是 use-def 能**双向**查询的根源：**同一根箭头（OpOperand），从消费者那头摸是 operand，从被使用值那头摸是 user。** 结构挂接（谁包含谁）做不到这种双向——所以 use-def 必须单独一套机制。

**为什么 def 和 uses 表示得不对称？** 更深一层的原因是**结构差异**：use 一侧专门有 `OpOperand` 这个**边对象**来承载关系，所以多个 use 能各占一个节点串成链；def 一侧**没有独立的边对象**——定义关系直接由 Value 自身兼任（`OpResultImpl`/`BlockArgumentImpl` 就是 Value 本体），故一个 def 无需链、靠反查属主即可。表象上的基数不对称（一个值有**多个** use 要链表 `firstUse → nextUse`、只有**一个** def）只是这个结构差异的结果，不是原因。

落回本例（建完 addi 后）：

```
%arg0 :  def = entry 块(block arg)   uses = { addi }     firstUse → [OpOperand{owner=addi, value=%arg0}] → ∅
%c1  :  def = constant(op result)    uses = { addi }     firstUse → [OpOperand{owner=addi, value=%c1 }] → ∅
%sum :  def = addi(op result)         uses = { }(暂无)     firstUse → ∅   ← 长在 addi 前缀区，getDefiningOp()=addi
```

> 图中每个 `[OpOperand{…}]` 节点就是一个 OpOperand；它的 `value` 字段恒等于左列那个 Value（故这里显式写出以印证"两端"），`owner` 字段才是 use 的去向（addi）。

> 一句话总括：**use-def 不是一张额外的图，而是"长在 Value 和 op 身上的指针"**——Value 长出一条 use 链（`firstUse`），op 长出 operand 数组（尾部）和 result 数组（前缀），中间用 `OpOperand` 这根"边"把它们缝起来。

#### Part C — 挂接过程：一条 use-def 边是怎么织出来的（机制通用，本例 addi 触发 ×2）

机制讲清了（Part A/B），现在看动态——`builder.create<AddIOp>(loc, arg0, c1)` 这一下，怎么把两条"use 边"织出来。

**(1) 建 OpOperand**：`Operation::create` 消费 `state.operands`，对每个 operand 执行（[OperationSupport.cpp:239-246](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp#L239)）：

```cpp
new (&operandStorage[i]) OpOperand(owner, values[i]);   // addi 的槽#0→%arg0，槽#1→%c1
```

**(2) 构造即入链**：`OpOperand` 构造（继承自 `IROperand`）初始化完 `owner`/`value` 后，**当场把自己插进那个 value 的 use-list**（[UseDefLists.h:130-133](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L130)）：

```cpp
IROperand(Operation *owner, IRValueT value)
    : detail::IROperandBase(owner), value(value) {
  insertIntoCurrent();      // ← use-def 边在这一行诞生！
}
// insertIntoCurrent() → insertInto( OpOperand::getUseList(value) )
//   getUseList(value) = value.getImpl()   （取 value 的链头容器，Value.h:270）
```

**(3) insertInto：头插法 4 步**（[UseDefLists.h:96-103](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L96)）——"把自己缝进 use 链"的实际动作：

```cpp
void insertInto(UseListT *useList) {
  back = &useList->firstUse;       // ① 我的 back 指向 value 的 firstUse 字段
  nextUse = useList->firstUse;     // ② 我接手原来的头节点
  if (nextUse)
    nextUse->back = &nextUse;      // ③ 老头的 back 改指我的 nextUse 字段
  useList->firstUse = this;        // ④ value 的头现在是我
}
```

织完之后的状态，就是 Part B 那张"落回本例"图：`%arg0`/`%c1` 的 firstUse 各挂上一个 `OpOperand{owner=addi}`，`%sum` 长在 addi 前缀区。

> 💡 **`back` 为什么是指针的指针？** `back` 存的不是"前驱节点"，而是"**指向我的那个指针变量的地址**"——链首时是 `&value->firstUse`，链中时是 `&前驱->nextUse`。于是摘除自己时一句 `*back = nextUse`（见下）对链首和链中**都成立**，不用 if。

**(4) 读回来（查询侧）**：use 串成 `firstUse → nextUse → …` 单向链，查询开销极低——这也是 Part B 那张映射表里各 API 的实际走法：

```cpp
%arg0.use_empty();   // = (firstUse == nullptr)               → false（addi 在用）
%arg0.hasOneUse();   // = firstUse && firstUse->nextUse==null  → true
for (Operation *user : %arg0.getUsers())   // 沿 nextUse 遍历 → { addi }
%sum .getDefiningOp();                       // addi（OpResult 地址算术 / BlockArgument owner 字段）
```

**(5) 拆 / 换（生命周期）**：
- **摘边 `removeFromCurrent()`**（[UseDefLists.h:86-93](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L86)）：`if(!back) return; *back = nextUse; if(nextUse) nextUse->back = back;`——`back` 双指针的兑现，一行摘掉自己。
- **换值 RAUW**：`replaceAllUsesWith(newValue)`（[UseDefLists.h:210-216](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L210)）= `while(!use_empty()) use_begin()->set(newValue)`；每个 `set` 内部 `removeFromCurrent() → value=newValue → insertIntoCurrent()`。**O(uses)、零额外空间。**
- **为什么删一个有结果的 op 必须先 RAUW 它的结果**：op 的 **operand 边**（它引用上游 value）由析构 `~OperandStorage` 自动从上游 use-list 摘掉；但 op 的 **result 值**可能正被下游 op 当 operand 引用（下游的 OpOperand 链在它的 firstUse 上）。直接 free 会让下游指向已释放内存——所以必须先 `result.replaceAllUsesWith(别的值)`（或保证 `use_empty()`）再 erase。debug 构建里 `~Operation` 的 `assert(use_empty())` 兜底。

> **为什么这一步最关键**：结构挂接（谁包含谁）只决定"拓扑形状"，数据流挂接才决定"语义依赖"。所有优化（DCE、GVN、CSE…）都建立在 use-def 图上——而这张图就是 Part B 那些长在 Value / op 身上的指针织出来的。

> 🏗️ **挂接⑦（Operand→Value，数据流边）的持有形式**：op 尾部 `OperandStorage`（内部持 `OpOperand[]`，物理属主 = op）+ 挂进 Value 的 `firstUse` **单向链表（nextUse）配 back 双指针**（逻辑边，O(1) 摘除）；一条边 = 一个 OpOperand **一身二任**（既是 op 的 operand，又是 value 的 user）。详见 Part B 的概念↔结构映射表、Part C 的拆/换。

**Step 6 的挂接清单**：

| 挂接 | 谁 → 谁身上 | 代码 |
|---|---|---|
| **Operand → Value（数据流边）⑦ ×2** | 两个 `OpOperand` 分别挂进 `%arg0`、`%c1` 的 use-list | [OperationSupport.cpp:239-247](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp#L239) |
| OpResult → Op ⑥ | `%sum` 是 addi 的结果 | [Operation.cpp:127-132](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L127) |
| Op → Block ③ | addi 插进 entry 块 | [Builders.cpp:434](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L434) |

**此刻全景**（★ 新增 addi + `%sum`；织出 **2 条** use-def 边）：

```
ModuleOp
  └─ region#0 └─ Block (module body)
      └─ FuncOp @add_one
          └─ region#0 └─ Block (entry)
              ├─ arg#0: %arg0 (i32)
              ├─ ConstantOp → %c1
              └─ ★ AddIOp                         ← 新增
                  ├─ operands: lhs=%arg0, rhs=%c1 ← 挂接⑦ ×2（织 use-def 边）
                  └─ result#0: %sum (OpResult)    ← 挂接⑥

data-flow 图（★ 新增 2 条边）：
  %arg0 ──(lhs)──┐
                 ├─→ AddIOp → %sum
  %c1  ──(rhs)──┘
```

**🔍 容器视角——把数据结构的"最终内容"摊开看**（上面抽象映射在内存里的真实样子）：

```
═══ addi 这个 Operation 的内存（一次 malloc：前缀结果 + 主体 + 尾部操作数）═══
addi (Operation)
  ├─ name = "arith.addi"     block = &entry     location = loc
  ├─ 前缀区  OpResult[]   = [ %sum ]                          ← result#0（InlineOpResult, type=i32）
  └─ 尾部区  OperandStorage → OpOperand[] =
        [0] { owner=&addi, value=%arg0, nextUse=nullptr, back=&(%arg0.firstUse) }
        [1] { owner=&addi, value=%c1,  nextUse=nullptr, back=&(%c1 .firstUse) }

═══ 每个 Value 的 use-list（firstUse 单向链）的真实样子 ═══
%arg0  (BlockArgument：type=i32, owner=entry 块, 住在 entry.arguments 向量里)
  firstUse ──► addi.OpOperand[0] ──► nullptr
                 ↑ 与上面 addi 数组里的 [0] 是【同一个对象】；它的 back 指回 %arg0.firstUse 这个字段

%c1   (OpResult of constant：type=i32, 住在 constant 前缀区)
  firstUse ──► addi.OpOperand[1] ──► nullptr
                 ↑ 与 addi 数组里的 [1] 是【同一个对象】

%sum  (OpResult of addi：type=i32, 住在 addi 前缀区)
  firstUse ──► nullptr                ← 还没有 user（Step 7 的 return 才会把它挂进来）
```

> **关键看点——同一个对象被两个容器引用**：`addi.OpOperand[0]` 这**一个**对象，既是 addi 尾部数组的一项（让 addi 能说"我的 operand#0 是 %arg0"），又是 %arg0 的 `firstUse` 链头（让 %arg0 能说"我有 addi 这个 user"）。这就是 Part B"一身二任"在内存里的具象。于是两条查询走的是**同一个 OpOperand**：
> - `addi->getOperand(0)` → 顺 `addi.OpOperand[0].value` → `%arg0` ✅
> - `%arg0.getUsers()` → 顺 `%arg0.firstUse` → `addi.OpOperand[0].owner` → `addi` ✅
>
> 这正是 use-def 边能**双向 O(1)** 的物理根基——不是存了两份，而是一根箭头两头都能摸。

### Step 7：建 `func.return`（终结指令）

```cpp
builder.create<mlir::func::ReturnOp>(loc, sum);   // sum 作为 return 的操作数
```

和 Step 6 同理：`%sum` 被 `OpOperand` 挂进它自己的 use-list（return 成了 `%sum` 的 user）。同时 return 持有 `IsTerminator` trait，必须是块的最后一个 op。

此刻整棵树织完（★ 新增 return；织出第 **3** 条 use-def 边；entry 块现在以 terminator 结尾 = well-formed）：

```
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
                      └─ ★ ReturnOp                ← 新增（terminator）
                          └─ operands: %sum        ← 挂接⑦（第 3 条 use-def 边）

完整的 data-flow 图（共 3 条 use-def 边）：
  %arg0 ──(lhs)──┐
                 ├─→ AddIOp ──→ %sum ──→ ReturnOp
  %c1  ──(rhs)──┘     │ addi 织 2 条 │  return 织 1 条
   ↑ constant 用属性带值，不贡献 use-def 边
```

打印出来就是 §2 的那段 `.mlir`。至此 entry 块以 `return` 这个 terminator 收尾，整棵树 well-formed，可以被 `verify` 检查、被后续 pass 走读（见 §6）。

---

## 4. 「挂接」本质：一张总表

把 §3 的七类挂接汇总。**构建一棵 IR = 反复执行这张表里的动作**：

| # | 挂接动作 | 被「挂」者 | 挂到「谁」身上 | **持有形式（数组？指针？链表？）** | 触发它的接口 | 验证源码 |
|---|---|---|---|---|---|---|
| ① | Region → Op | `Region` | `Operation` | **数组**（op 尾部 `Region[]`）+ 每个 Region 一个 `container` 回指指针 | `Operation::create` 内 `new Region(op)` | [Operation.cpp:135-136](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L135) |
| ② | Attribute → Op | `DictionaryAttr` | `Operation` | **一个指针**（op 的 `attrs` 字段，指向 context 池里的字典对象） | `op->setAttrs(...)` | [Operation.cpp:150](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L150) |
| ③ | **Op → Block** | `Operation` | `Block` | **侵入式双向链表** `iplist<Operation>` | `OpBuilder::insert` | [Builders.cpp:434](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L434) |
| ④ | Block → Region | `Block` | `Region` | **侵入式双向链表** `iplist<Block>` | `createBlock`/`emplaceBlock` | [Builders.cpp:450](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L450) |
| ⑤ | BlockArgument → Block | `BlockArgument`(Value) | `Block` | **动态数组** `std::vector<BlockArgument>` + 每个参数一个 owner 回指指针 | `Block::addArgument` | [Block.cpp:152-155](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L152) |
| ⑥ | OpResult → Op | `OpResult`(Value) | `Operation` | **数组**（op 前缀区 `InlineOpResult[0..4]`+尾部 `OutOfLineOpResult[]`），**无** owner 字段（靠算术定位） | `Operation::create` placement-new | [Operation.cpp:127-132](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L127) |
| ⑦ | **Operand → Value（数据流边）** | `OpOperand` | `Value` 的 use-list | **逻辑双向可查**：op 尾部 `OperandStorage`（内部持 `OpOperand[]`，物理属主=op）+ 挂进 Value 的 `firstUse` **单向链表（nextUse）配 back 双指针**（逻辑边，支持 O(1) 摘除） | `OperandStorage` ctor → `insertInto` | [OperationSupport.cpp:239](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp#L239) |

> 读法：①⑥ 是**数组**（①贴 op 尾部，⑥贴 op 前部）；② 是**单个指针**（指向 context 池）；③④ 是**侵入式双向链表（iplist）**；⑦ 的逻辑侧是**单向链表（nextUse）配 back 双指针**——不是 iplist 那种真双向，但因 `back` 存的是"指向我的指针变量的地址"，摘除时一句 `*back=nextUse` 即可，效果等同双向；⑤ 是 **vector + 回指指针**。注意 ⑥ 与众不同——不存 owner，靠算术倒推（见 Step 5 callout）。

**两条主轴**：
- ①②③④⑤⑥ 都是**结构挂接**——织"结构树"。除 ⑤⑥ 外，全靠 iplist 或 placement-new 在 `Operation::create` / `OpBuilder::insert` 里自动完成。
- ⑦ 是**数据流挂接**——织"数据流图"，靠 `OpOperand` 构造时的 `insertInto` 自动完成。

> 一个深刻的结论：**`OpBuilder::create<>` 这一个调用，内部就完成了 ①~⑦ 里的好几项**（取决于这个 op 有没有 region、operands、attrs）。这就是为什么 MLIR 的构建代码看起来很"干净"——大量挂接细节被 `Operation::create` 吞掉了，你只需声明"我要一个什么样的 op"，剩下交给装箱单 → 分配 → 自动挂接的流水线。

---

## 5. 从 AST 到 MLIR：前端的递归下降（Toy `MLIRGen`）

到这里你已经知道"一个 op 怎么建、怎么挂"。但真实前端不是手动一个个 create——它是**递归遍历 AST，每个 AST 节点生成 0~1 个 op**。MLIR 官方 [Toy 教程](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp) 的 `MLIRGen.cpp` 就是这个模式的教科书。

### 5.1 整体骨架：一个 `OpBuilder` + 一个符号表

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

> **符号表是 AST 与 SSA 之间的桥梁**：AST 里变量叫 `a`、`b`，SSA 里是 `%0`、`%1`。符号表把"AST 名字"映射到"它对应的那个 Value"。声明变量时登记，引用变量时查表拿到 Value 当操作数。

### 5.2 一个 AST 节点 → 一段 build（对应本文 §3 的每一步）

**AST 函数 → `FuncOp` + 进入函数体**（[MLIRGen.cpp:129-154](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L129)）：

```cpp
mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
  builder.setInsertionPointToEnd(theModule.getBody());     // ← Step 2：挪插入点到 module body
  mlir::toy::FuncOp function = mlirGen(*funcAST.getProto()); // ← Step 3：建 FuncOp
  mlir::Block &entryBlock = function.front();              // ← Step 4：取入口块
  // 把函数参数登记进符号表（AST 参数名 → BlockArgument Value）
  for (auto [arg, value] : llvm::zip(protoArgs, entryBlock.getArguments()))
    declare(arg->getName(), value);                        //    %arg0 ↔ "arg0"
  builder.setInsertionPointToStart(&entryBlock);           // ← Step 4：挪插入点进函数体
  mlirGen(*funcAST.getBody());                             // ← Step 5~7：递归生成函数体
}
```

**AST 二元表达式 → `AddIOp`（use-def 挂接）**（[MLIRGen.cpp:181-212](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L181)）：

```cpp
mlir::Value mlirGen(BinaryExprAST &binop) {
  mlir::Value lhs = mlirGen(*binop.getLHS());   // ← 递归：先算左操作数，返回一个 Value
  mlir::Value rhs = mlirGen(*binop.getRHS());   // ← 递归：再算右操作数
  ...
  return builder.create<AddOp>(location, lhs, rhs);  // ← Step 6：lhs/rhs 当操作数 → 自动织 use-def
}
```

> 注意这里的**递归顺序**：必须先 create 出 `lhs`/`rhs`（它们各自返回一个 Value），再 create 加法 op 把它们当操作数。这天然保证了 SSA 的**支配性**——操作数一定先于使用者被定义。

**AST 字面量 → `ConstantOp`（属性挂接）**（[MLIRGen.cpp:261-284](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp#L261)）：

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

### 5.3 把 5.2 串成一条流水线

```
ModuleAST
  └─ mlirGen(ModuleAST):  ModuleOp::create()                      → Step 1（建根）
     └─ FunctionAST
        └─ mlirGen(FunctionAST):  create<FuncOp> + 挪插入点 + 登记参数 → Step 2,3,4
           └─ ReturnExprAST
              └─ BinaryExprAST(+)
                 ├─ VariableExprAST(arg0)  → symbolTable.lookup  → %arg0（复用 Value）
                 └─ NumberExprAST(1)       → create<ConstantOp>  → %c1（Step 5，带属性）
                 → create<AddIOp>(%arg0,%c1)                       → %sum（Step 6，use-def）
              → create<ReturnOp>(%sum)                             → Step 7
```

**对应关系**：每个 AST 叶子/节点，要么查表复用一个已有 Value，要么 `create` 一个 op（并自动完成 §4 表里的若干挂接）。AST 的嵌套结构 → IR 的结构树；AST 表达式的子表达式关系 → IR 的 use-def 数据流图。

### 5.4 对照你仓库里的 NorthStar CH-5

[NorthStar `main.cpp` 的 `CH5()`](../MLIR-Tutorial/5-define_operation/main.cpp#L240) 是**同样的模式、省去 AST 的硬编码版本**：

```cpp
mlir::OpBuilder builder(&context);                                    // Step 0
auto module = builder.create<mlir::ModuleOp>(loc, "NorthStar");       // Step 1
builder.setInsertionPointToStart(module.getBody());                   // Step 2

auto const_1 = builder.create<mlir::north_star::ConstOp>(             // Step 5：ConstOp 带 DenseElementsAttr
    loc, tensor_type_1, mlir::DenseElementsAttr::get(...));           //       （属性挂接）
auto buffer_op = builder.create<mlir::north_star::BufferOp>(          // Step 6：BufferOp 吃 const_1 等作操作数
    loc, mlir::ValueRange({const_1, const_3}));                       //       （use-def 挂接）
auto softmax_op = builder.create<mlir::north_star::SoftmaxOp>(        // Step 6 续：softmax 吃 get_tensor 结果
    loc, get_tensor_op_1, 1);
```

类型那几行 `NSTensorType::get(&context, ...)`（[main.cpp:262-264](../MLIR-Tutorial/5-define_operation/main.cpp#L262)）就是 Step 0 的"备料"——intern 到 context。整个 CH-5 没有用到 AST，但它**逐行执行的挂接动作**和 Toy 的 `MLIRGen` 一模一样——只是 AST 的角色被你（程序员）代替了。

---

## 6. 遍历这棵树 → lowering 的起点

树织好之后，怎么"走读"它来做优化和 lowering？

### 6.1 `walk()`：对结构树做深度优先遍历

最常用的遍历入口是 `Operation::walk`（实现见 [Visitors.h:127](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h#L127)，走向见下一段）：

```cpp
module->walk([](mlir::Operation *op) {
  llvm::outs() << op->getName() << " @ block " << op->getBlock() << "\n";
});
// 输出顺序（PostOrder 默认）：
//   arith.constant
//   arith.addi
//   func.return
//   func.func
//   module
```

`walk` 沿着**结构树**（region→block→op）递归下降，对每个 op 调一次回调。它**不沿数据流走**——要查"谁用了 `%sum`"得用 `Value::getUsers()`（沿上文 use-def 链的 `firstUse` 遍历）。

### 6.2 lowering 的本质：在遍历中重写

[MLIR 的设计哲学之一是 progressive lowering（渐进下沉）](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)——高层 dialect（如 `arith`/`north_star`）的 op，被一组**重写模式（Rewrite Pattern）**逐个换成更低层的 op（最终落到 `llvm.` dialect）。其骨架：

1. **遍历**：`Pass` 框架对每个 op 触发模式匹配（内部走 walk 之类）。
2. **匹配**：模式检查 op 的结构/操作数/属性是否符合（`match`）。
3. **重写**：用 `PatternRewriter` 建新 op（又走一遍本文 §3 的挂接流程！）、把旧 op 的结果 `replaceAllUsesWith` 到新结果、`eraseOp` 删旧 op（RAUW 与 erase 的机理见上文 Step 6 的「数据结构·挂接⑦」）。

例如，本例的 `arith.addi %arg0, %c1 : i32` 在 lowering 到 LLVM 时，会被换成 `llvm.add`：

```
arith.addi  ──(ConversionPass: walk + match + rewrite)──→  llvm.add
```

**这里的精妙闭环**：lowering 阶段"建新 op"，调的还是 `OpBuilder::create<>`（或 `PatternRewriter::create`，它继承自 `OpBuilder`），走的还是本文 §3 的"装箱单 → Operation::create → 自动挂接"流水线。**你一旦理解了 IR 是怎么 build 出来的，就同时理解了它是怎么被 lowering 重写的**——因为构建和重写共用同一套挂接原语。

> NorthStar 教程的主线正是这条：CH-12（Conversion）开始把 `north_star` 的 op 逐步 lowering，CH-15 落到 LLVM IR。届时你会看到 [BufferCastOpFold](MLIR-IR-Node组织与遍历插入删除教程.md) 那样的 match→RAUW→erase，本质上就是"在已建好的树上，按规则拆/重组"。

---

## 7. 全局俯瞰：结构树 × 数据流图的数据结构全景

> 前六章是一砖一瓦「往上盖」（§3 的 7 步挂接）。本章退后一步，站在**建好的 IR 顶层俯瞰**：这棵树到底由哪几套数据结构缝成？它们各自维系什么关系？每个结构用一句话点透本质。
>
> 用法：当「总览 + 速查」。要全局图就来本章，细节回查 §3/§4，RAUW 改图的细节见姊妹篇 [RAUW 机理详解](MLIR-RAUW-replaceAllUsesWith-机理详解.md)。

### 7.0 总纲：两套独立结构，缝合成一棵 IR

MLIR 的 IR 不是单一数据结构，而是**两套几乎不相关的结构缝在一起**：

| 结构 | 回答「什么关系」 | 维系它的数据结构 | 对应挂接（§4） |
|---|---|---|---|
| **结构树** | 谁包含谁 | 侵入式链表（iplist）+ 回指指针 + 尾/前缀分配数组 | ①②③④⑤⑥ |
| **数据流图** | 谁的数据流向谁 | OpOperand 边对象 + Value 的 use-list | ⑦ |

**唯一交汇点**：`Operation` 这个对象**同时**是结构树的一个节点（住在某个 Block 的 iplist 里）和数据流图的一个计算节点（消费 operand、产出 result）。两套结构除此之外各管各的、互不干扰。

```
   ┌──────────── 结构树（谁包含谁）────────────┐
   │  Module ▸ Block ▸ Func ▸ Block ▸ {Op,Op,Op} │   靠 iplist 链表 + 回指指针层层咬合
   └─────────────────────────────────────────────┘
                       │
              交汇点 = Operation
            （既是树的节点，又是图的计算节点）
                       │
   ┌──────────── 数据流图（谁流向谁）────────────┐
   │   Value ◄── OpOperand ──► Operation          │   靠 OpOperand 边对象 + use-list 维系
   └─────────────────────────────────────────────┘
```

> **一句话本质（总纲）**：**MLIR IR = 结构树 ∪ 数据流图，两套独立数据结构缝合成一棵树；唯一的缝合点是 `Operation` 这个对象——它既是结构树的节点，又是数据流图的计算节点。其余结构各管各的。**

### 7.1 结构树全景：嵌套靠链表，反查靠回指

把 §2 那段 `add_one` 的 IR，**结构树**连同每层用的数据结构标出来：

```
ModuleOp
 └─ Region[] (op 尾部数组) ①                         ← Region→Op：数组 + container 回指
     └─ region#0  ─container─► 回指 ModuleOp
         └─ iplist<Block> ④                          ← Block→Region：侵入式双向链表
             └─ Block (module body)
                 └─ iplist<Operation> ③              ← Op→Block：侵入式双向链表
                     └─ FuncOp @add_one
                         ├─ OpResult[] (op 前缀区) ⑥        ← OpResult→Op：前缀数组，算术反查，无 owner 字段
                         ├─ attrs: DictionaryAttr 指针 ②    ← Attr→Op：单个指针 → context 池
                         └─ Region[] ①
                             └─ region#0 (函数体)
                                 └─ iplist<Block> ④
                                     └─ Block (entry)
                                         ├─ arguments: vector<BlockArgument> ⑤   ← BlockArgument→Block：动态数组 + owner 回指
                                         └─ iplist<Operation> ③
                                             ├─ ConstantOp  (前缀 OpResult[] ⑥ / attrs ②)
                                             ├─ AddIOp      (同上)
                                             └─ ReturnOp    (同上)
```

> **一句话本质（结构树）**：**结构树的嵌套，靠 `iplist<Operation>`（Block 里）和 `iplist<Block>`（Region 里）两条侵入式双向链表层层咬合——节点就是 Operation/Block 自己（零额外分配），O(1) 插删；每个 Region/Block/Op 各存一个回指父节点的指针（container / getParent / getBlock），于是既能向下遍历、又能 O(1) 向上找属主。op 自己的部件（结果、属性、子 Region）则不进链表，而是长在 op 的前缀/尾部数组里、或存一个指针。**

各挂接的持有形式速记：

| 挂接 | 持有形式 | 本质一句 |
|---|---|---|
| ① Region→Op | op **尾部数组** + 每个 Region 一个 `container` 回指 | 定长子数组 + 反查指针 |
| ② Attr→Op | op 上**一个指针**（→ context 池的字典） | 属性不属于 op，op 只存指针 |
| ③ Op→Block | **iplist** 侵入式双向链表 | 节点即 Operation，O(1) 插删 |
| ④ Block→Region | **iplist** 侵入式双向链表 | 同上，节点即 Block |
| ⑤ BlockArgument→Block | **vector** + 每个参数一个 `owner` 回指 | 动态数组 + 反查指针 |
| ⑥ OpResult→Op | op **前缀数组**，**无** owner 字段（算术反查） | 省指针，靠地址倒推属主 |

### 7.2 数据流图全景：一条边，两个方向

数据流图只有**一种**核心对象——`OpOperand`（一条 SSA 边）。但同一条边，从两头读，名字不同：

```
                       一条 SSA 边 = 一个 OpOperand 对象
          ┌──────────────────────────┴──────────────────────────┐
   从【def 一侧】读（Value 的视角）                  从【use 一侧】读（op 的视角）
   它是 Value 的 use-list 里的一个节点              它的 value 字段指向 Value
   "Value 被这个 op 使用"                            "这个 op 的操作数是 Value"
        ↓ 这一头叫【def-use (DU)】                       ↓ 这一头叫【use-def (UD)】
```

#### def-use（1 个定义 → N 个使用）：链表，Value 端维护

```
  Value %sum  (一个【定义/def】)
  ┌───────────────┐
  │ firstUse ─────┼─► [OpOperand] ─nextUse─► [OpOperand] ─nextUse─► null
  └───────────────┘     owner=return           owner=print
                        value=%sum             value=%sum
   ↑ Value 用一条 firstUse 单向链表（配 back 双指针，O(1) 摘除）把所有引用它的 OpOperand 串起来
```

> **一句话本质（def-use）**：**Value 用一条 `firstUse` 单向链表（配 `back` 双指针实现 O(1) 摘除）把所有引用它的 `OpOperand` 串联起来，每个 OpOperand 的 `owner` 字段指向使用者 op——于是「1 个 Value ↔ N 个 Operation」的 def-use 关系，由 Value（def 这一端）用一条链表维护。**

#### use-def（N 个使用 → 1 个定义）：指针，OpOperand 端维护

```
  OpOperand(owner=return)        OpOperand(owner=print)
    │ value=%sum ─────┐             │ value=%sum ─────┐
    └─────────────────┼─────────────┴─────────────────┘
                       ▼                           ▼
                     Value %sum ◄──────────────────┘   ← N 个 use 的 value 指针都指向这 1 个 def
                         │  (OpResult？地址算术反查 op ／ BlockArgument？存 owner=Block)
                         ▼
                     AddIOp（定义方 op）
```

> **一句话本质（use-def）**：**每个 OpOperand 存一个 `value` 指针指向它引用的 Value，再由 `Value.getDefiningOp()` 反查定义 op（OpResult 靠地址算术、BlockArgument 靠 owner 字段）——于是「N 个使用 → 1 个定义」的 use-def 关系，由每个 OpOperand（use 这一端）用**一个指针**维护；SSA 保证每个使用恰有 1 个定义，所以一指针足矣，无需链表。**

#### 为什么一个用链表、一个用指针（不对称的根因）

| | def-use (DU) | use-def (UD) |
|---|---|---|
| 基数 | **1 → N**（一个定义被多处用） | **N → 1**（多处用指向一个定义） |
| 数据结构 | **链表**（要枚举 N 个 use） | **单指针**（只存 1 个 def） |
| 维护端 | Value（def） | OpOperand（use） |

> **一句话本质（不对称）**：**def-use 用链表、use-def 用指针，纯粹是 SSA 基数的必然——「一个定义可被 N 处用」要枚举故用链表，「一个使用只有一个定义」存一个故用指针；不是为对称而设计。**

### 7.3 四类「持有形式」速记

把全篇出现过的数据结构归成四类，背下这四类就背下了整棵 IR 的骨架：

| 持有形式 | 用在哪 | 一句话 |
|---|---|---|
| **侵入式双向链表（iplist）** | ③ Op→Block、④ Block→Region | 节点即对象本身，零额外分配，O(1) 插删——结构树的主力 |
| **尾/前缀分配数组（trailing objects）** | ① Region[]（尾部）、⑥ OpResult[]（前缀） | 一块连续内存装不定长部件，省一次 malloc |
| **单个指针** | ② attrs→context 池；⑦ use-def 的 value→def | 最轻，指向别处拥有的对象 |
| **动态数组 + 回指指针** | ⑤ BlockArgument→Block | vector 存对象 + 每对象一个 owner 反查 |

> **特例（⑦ OpOperand 一身二任）**：边对象 OpOperand **物理**上是 op 尾部 `OperandStorage` 数组的一项（让 op 说"我的 operand#i 是 V"），**逻辑**上是 Value use-list 的链表节点（让 Value 说"我有这个 op 当 user"）——同一对象被两处引用，正是 def-use / use-def 能双向 O(1) 的物理根基。

### 7.4 七句话记住整棵 IR

1. **MLIR IR = 结构树 ∪ 数据流图**，两套独立数据结构缝合成一棵树，唯一缝合点是 `Operation`（既是树节点、又是计算节点）。
2. **结构树**靠 `iplist<Operation>` / `iplist<Block>` 两条侵入式双向链表层层嵌套，配回指指针（container / getParent / getBlock）实现向下遍历 + 向上 O(1) 反查。
3. **op 的部件**（结果、属性、子 Region）长在 op 的前缀/尾部数组里或存一个指针，不进链表。
4. **数据流图**只有一种边对象 `OpOperand`，它一身二任：图里的边（`value`→def、`owner`→use）+ Value use-list 的链表节点（`nextUse`/`back`）。
5. **def-use（1→N）**：Value 用一条 `firstUse` 链表串联所有引用它的 OpOperand，由 def 端维护。
6. **use-def（N→1）**：每个 OpOperand 存一个 `value` 指针指向它的 Value，由 use 端维护；SSA 让它只需一指针。
7. **def-use 用链表、use-def 用指针**的不对称，纯粹是「一个定义多处用、一个使用一个定义」的 SSA 基数所致。

---

## 附录 A：构建 API 速查

| 你想做的事 | 调用 | 出处 |
|---|---|---|
| 建一个 op 并自动插入当前位置 | `builder.create<OpTy>(loc, args...)` | [Builders.h:508](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L508) |
| 建一个 block 并插入 region | `builder.createBlock(region, it, argTypes, locs)` | [Builders.cpp:441](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp#L441) |
| 给 block 加参数（产生 BlockArgument Value） | `block.addArgument(type, loc)` | [Block.cpp:152](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp#L152) |
| 挪插入点到块首/块尾 | `builder.setInsertionPointToStart/End(block)` | [Builders.h:434/439](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L434) |
| 保存/恢复插入点（RAII） | `OpBuilder::InsertionGuard guard(builder);` | [Builders.h:351](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L351) |
| 取一个 context 级类型 | `builder.getI32Type()` / `Ty::get(ctx, ...)` | [Builders.h:84](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L84) |
| 取一个 context 级属性 | `builder.getIntegerAttr(type, val)` | [Builders.h:111](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L111) |
| 建只建不插（游离 op） | `Operation::create(state)` | [Operation.cpp:34](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L34) |
| 遍历结构树 | `op->walk(callback)` | [Visitors.h:127](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Visitors.h#L127) |

---

## 附录 B：关键源码索引

| 文件 | 内容 | 重要行号 |
|---|---|---|
| [`Builders.h`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h) | `Builder` / `OpBuilder` / 插入点 | :50-206（Builder），:210-618（OpBuilder），:508-517（create 模板），:614-617（插入点字段）|
| [`Builders.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Builders.cpp) | `insert` / `createBlock` / `create` 实现 | :432-439（insert），:441-456（createBlock），:468-470（create→insert）|
| [`OperationSupport.h`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/OperationSupport.h) | `OperationState`（装箱单） | :950-1075（结构 + add* 方法）|
| [`Operation.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp) | `Operation::create`（THE allocator） | :34-48（create(state)），:82-153（分配 + 全部挂接），:127-150（results/regions/operands/attrs）|
| [`OperationSupport.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp) | `OperandStorage` ctor（织 use-def） | :239-247 |
| [`BuiltinDialect.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/BuiltinDialect.cpp) | `ModuleOp::build` / `create` | :133-146 |
| [`Block.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Block.cpp) | `Block::addArgument`（BlockArgument 诞生） | :152-156 |
| [`FuncOps.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Dialect/Func/IR/FuncOps.cpp) | `func::FuncOp::build` | :180-195 |
| Toy [`MLIRGen.cpp` (Ch2)](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/MLIRGen.cpp) | AST → MLIR 递归下降范本 | :65-82（Module），:129-178（Function），:181-212（BinaryExpr），:261-284（Literal/属性），:217-224（Var/符号表）|
| Toy [`Dialect.cpp` (Ch2)](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch2/mlir/Dialect.cpp) | `toy::FuncOp::build`（建入口块+参数） | :206-212 |
| NorthStar [`main.cpp` (CH-5)](../MLIR-Tutorial/5-define_operation/main.cpp) | 你仓库里的真实 build 示例 | :240-314（`CH5()`）|

---

## 参考资料

- [MLIR 官方文档 — Understanding the IR Structure](https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/)
- [MLIR Toy 教程 — Chapter 2: Emitting Basic MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)
- 姊妹篇 [MLIR IR 中的 Node 组织：结构、遍历、插入与删除](MLIR-IR-Node组织与遍历插入删除教程.md)
- [LLVM 19.1.7 源码（本仓库 submodule）](../MLIR-Tutorial/third_party/llvm-project/mlir/)
- NorthStar Tutorial (violetDelia/MLIR-Tutorial, Apache 2.0)
