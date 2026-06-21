# MLIR Operation 的构建机制与 Use-Def 链形成
## 基于 NorthStar 方言 SoftmaxOp 的源码级分析

**报告类型**：技术研究报告  
**分析对象**：MLIR Operation 的构建、实例化与 SSA use-def 链机制  
**案例方言**：NorthStar Dialect（`mlir::north_star`）  
**基础版本**：LLVM 19.1.7  
**日期**：2026 年 6 月 14 日

---

## 摘要

本报告以 NorthStar 方言的 `SoftmaxOp` 为贯穿案例，通过源码级走读，系统阐明 MLIR 中一个 Operation 从 TableGen 声明、经 ODS 代码生成、到运行时 IR 节点实例化、再到 SSA use-def 链形成的完整生命周期。报告覆盖三个层次：声明层（`.td`）、生成层（`.inc`）、运行时 IR 层（`Operation` / `OperationState` / `OpOperand` / `OpResult`），重点剖析 `OperationState` 作为"建造说明书"的角色、`$_state` / `$_builder` 替换变量的语义、`Operation::create` 的尾分配（trailing objects）内存模型，以及 Operand 与 Attribute 的语义分野。核心结论是：MLIR 通过将"数据流边"（Operand / Value）与"编译期配置"（Attribute / Properties）严格分离，并以侵入式双向链表维护 use-def 关系，使 IR 既具备图结构、又支持高效的依赖追踪与等价判断，为渐进式 lowering 奠定了基础。

**关键词**：MLIR；Operation；SSA；Use-Def 链；TableGen / ODS；OperationState；Trailing Objects；数据流图

---

## 1. 引言

### 1.1 研究背景

MLIR（Multi-Level Intermediate Representation）是 LLVM 社区提出的可扩展编译基础设施，其核心创新在于允许通过 Dialect（方言）机制定义任意多个层次化、可组合的中间表示，并通过 Conversion（lowering）在层级之间渐进转换，最终收束到 LLVM IR 与机器码。整体编译栈定位为：

```
源语言前端 → 高层 IR → MLIR 多层 dialect → LLVM IR → Machine Code
```

在该栈中，Operation 是 MLIR IR 的基本执行单元与图节点。一切变换（Pass）、重写（Rewrite Pattern）、转换（Conversion）都以 Operation 为操作对象。因此，理解 Operation 的内部结构、构建过程与连接机制（use-def 链），是理解整个 MLIR 运行机制的基石。

### 1.2 问题与研究范围

尽管 MLIR 官方文档与教程对 Operation 的"使用方式"有较好描述，但对以下三个内部机制的源码级阐释仍较分散：

1. **声明-生成-实例化的三段式构建管线**：从 `.td` 声明如何变为可实例化的 `Operation` 对象？
2. **`Operation::create` 的内存模型**：结果、操作数、属性、区域如何在一个堆对象中紧凑布局？
3. **SSA use-def 链的形成机制**：当一个 Op 以另一个 Op 的结果为操作数时，"边"是如何在数据结构层面建立的？

本报告以 NorthStar 方言的 `SoftmaxOp`（一元 softmax 操作，含 `axis` 属性）为案例，对上述机制进行源码级剖析。

### 1.3 方法论

报告所有论断均基于对 LLVM 19.1.7 源码与 NorthStar 教程生成产物（`*.inc`）的实际读取，标注真实文件名与行号，杜绝臆测。采用"原理先行 → 数据结构 → 控制流程 → 设计哲学"的递进结构。

---

## 2. 前置概念

### 2.1 SSA 形式与数据流图

MLIR 采用 SSA（Static Single Assignment，静态单赋值）形式：每个值在其作用域内仅被定义一次，值的传递通过"边"连接。在 IR 文本中，值以 `%name` 表示（SSA value）。一个 Operation 的操作数（operand）是来自其他 Op 结果或 Block 参数的 SSA value，这些 operand 构成数据流图的边。

### 2.2 Operation 作为 IR 节点

Operation 的官方定义见 `mlir/include/mlir/IR/Operation.h:31-83`。其注释明确指出：

> "Operation is the basic unit of execution within MLIR."

一个 Operation 由以下部件组成：

| 部件 | 类型 | 语义 |
|---|---|---|
| Name | `OperationName` | 操作名，形如 `dialect.opname` |
| Results | `OpResult[]` | 产生的 SSA value，即图的"输出边" |
| Operands | `OpOperand[]` | 消费的 SSA value，即图的"输入边" |
| Attributes / Properties | `DictionaryAttr` / `Properties` | 编译期固定参数 |
| Regions | `Region[]` | 嵌套子 IR |
| Successors | `BlockOperand[]` | 控制流后继 |
| Location | `Location` | 源码位置 |

### 2.3 Operand 与 Attribute 的语义分析

这是本报告的核心区分。二者虽都"挂在 Operation 上"，但语义截然不同：

- **Operand**：运行时数据流的一边，其值由别的 Op 计算得出，参与数据流图。
- **Attribute**：编译期固定的配置参数，不随执行变化，不参与数据流图。

对 `SoftmaxOp` 而言，`input`（输入张量）是 operand，`axis = 1`（沿第 1 维归约）是 attribute。

### 2.4 ODS / TableGen 的声明-生成范式

MLIR 使用 TableGen 语言描述方言实体（Type / Attribute / Operation），由 `mlir-tblgen` 生成 C++ 代码（`.inc` 文件）。手写的 `.h` / `.cpp` 依赖这些生成产物。这一范式使方言定义声明化、自动化、可校验。

---

## 3. 构建管线的三层层级

Operation 的构建跨越三层：

```
┌─────────────────────────────────────────────────────────────┐
│ 第 1 层：声明层（.td）                                       │
│   NorthStarOps.td — SoftmaxOp 的 TableGen 声明              │
│   描述：操作数类型、结果类型、属性、builder 模板、verify 标记 │
└────────────────────────┬────────────────────────────────────┘
                         │ mlir-tblgen
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 第 2 层：生成层（.inc）                                      │
│   NorthStarOps.h.inc — 类声明 + Properties struct           │
│   NorthStarOps.cpp.inc — build() / verify() / accessor 实现 │
└────────────────────────┬────────────────────────────────────┘
                         │ C++ 编译 + 运行时调用
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ 第 3 层：运行时 IR 层                                       │
│   OpBuilder::create → build() → Operation::create → Operation│
│   内存中实例化的不可变 IR 节点 + use-def 链                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.1 声明层：NorthStarOps.td

`SoftmaxOp` 的声明见 `NorthStarOps.td:59-70`：

```tablegen
def NorthStar_SoftmaxOp : NorthStar_UnaryOp<"softmax", AnyNSTensor, AnyNSTensor,
                                            [], (ins I64Attr:$axis)>{
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis),
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis =
                    $_builder.getIntegerAttr(odsBuilder.getIntegerType(64,true), axis);
                $_state.addTypes(input.getType());
            }]>
    ];
}
```

它继承自 `NorthStar_UnaryOp`（`NorthStarOps.td:28-37`），该基类将 `arguments` 展开为 `(ins OperandType:$input, <attributes>)`，`results` 为 `(outs resultType:$result)`。因此 `SoftmaxOp` 声明：一个 `AnyNSTensor` 操作数 `$input`，一个 `I64Attr` 属性 `$axis`，一个 `AnyNSTensor` 结果。

### 3.2 生成层：NorthStarOps.cpp.inc / .h.inc

经 `mlir-tblgen` 生成，关键产物：

- `SoftmaxOp::build` 重载集，其中手写 builder 的展开见 `NorthStarOps.cpp.inc:1107-1112`
- `Properties` 嵌套结构体，见 `NorthStarOps.h.inc:1432-1451`
- `getAxis` / `setAxis` / `getAxisAttr` 等 accessor，见 `NorthStarOps.cpp.inc:1098-1105`

### 3.3 运行时 IR 层

由 `main.cpp:300` 的 `builder.create<mlir::north_star::SoftmaxOp>(loc, get_tensor_op_1, 1)` 触发，调用链贯穿 `OpBuilder::create` → `SoftmaxOp::build` → `Operation::create`，最终在堆上实例化一个 `Operation` 对象并建立 use-def 链。

---

## 4. OperationState：建造说明书

### 4.1 数据结构定义

`OperationState` 定义于 `OperationSupport.h:950-1075`。其数据成员：

```cpp
struct OperationState {
  Location location;
  OperationName name;
  SmallVector<Value, 4> operands;                  // 操作数收集器
  SmallVector<Type, 4> types;                      // 结果类型
  NamedAttrList attributes;                        // 可丢弃属性
  SmallVector<Block*, 1> successors;               // 控制流后继
  SmallVector<std::unique_ptr<Region>, 1> regions; // 嵌套区域
  Attribute propertiesAttr;                        // 未注册 Op 的属性回退
private:
  OpaqueProperties properties;   // ← inherent 属性（Properties struct）
  TypeID propertiesId;
  llvm::function_ref<void(OpaqueProperties)> propertiesDeleter;
  llvm::function_ref<void(OpaqueProperties, const OpaqueProperties)> propertiesSetter;
  ...
};
```

### 4.2 字段语义分类

`OperationState` 的字段可归为三类：

1. **身份字段**：`location`、`name` — 决定"是什么 Op、在哪里"
2. **结构字段**：`operands`、`types`、`successors`、`regions` — 决定"图的拓扑"
3. **配置字段**：`attributes`、`properties` / `propertiesAttr` — 决定"编译期参数"

### 4.3 临时性与所有权

`OperationSupport.h:946-949` 注释明确：

> "This object is a large and heavy weight object meant to be used as a temporary object on the stack."

`OperationState` 是**栈上临时对象**，充当"建造说明书"。`build` 函数向其填料；`Operation::create` 消费它并构造不可变的 `Operation`。`OperationState` 析构时（`OperationSupport.cpp:196-199`）若持有 properties 则调用 deleter 释放。

---

## 5. TableGen 替换变量的语法与语义

### 5.1 `$_state` 与 `$_builder` 的定义

`.td` 中 `OpBuilder` dag 的 `[{ }]` 代码块是**代码模板**（code template），而非普通 C++。`mlir-tblgen` 在展开时进行文本替换。`$_xxx` 是预定义的替换变量：

| 替换变量 | 展开为 | 实际类型 |
|---|---|---|
| `$_state` | `odsState` | `OperationState &` |
| `$_builder` | `odsBuilder` | `OpBuilder &` |

### 5.2 替换的源码证据

将 `NorthStarOps.td:63-67` 与生成产物 `NorthStarOps.cpp.inc:1107-1112` 逐字对照：

```cpp
void SoftmaxOp::build(::mlir::OpBuilder &odsBuilder,
                      ::mlir::OperationState &odsState,
                      ::mlir::Value input, int64_t axis) {
    odsState.addOperands(input);                   // $_state  → odsState
    odsState.getOrAddProperties<Properties>().axis =
        odsBuilder.getIntegerAttr(
            odsBuilder.getIntegerType(64,true), axis);  // $_builder → odsBuilder
    odsState.addTypes(input.getType());            // $_state  → odsState
}
```

替换关系清晰：`$_state` → `odsState`，`$_builder` → `odsBuilder`。

### 5.3 设计动机

将变量名交由生成器决定而非硬编码在 `.td`，实现关注点分离：`.td` 是方言中立描述，不依赖生成后的 C++ 签名细节；框架参数（builder、state）由 `$_xxx` 隐式注入，用户参数（input、axis）在 `(ins ...)` 显式声明。这一抽象屏障使 MLIR 升级内部实现时不影响方言 `.td` 文件。

---

## 6. 构建阶段：build 函数填充 OperationState

本节逐一打开 build 函数体中的三行，揭示其内部实现。

### 6.1 `addOperands(input)`：操作数收集

`OperationSupport.h:1023` 声明，`OperationSupport.cpp:212-214` 实现：

```cpp
void OperationState::addOperands(ValueRange newOperands) {
  operands.append(newOperands.begin(), newOperands.end());
}
```

语义：将传入的 `Value`（SSA value 引用）追加到 `operands` 向量。`input` 是指向 `get_tensor_op_1` 结果的引用，即数据流图的一条边。此时尚未建立 use-def 链，仅完成"记录意图"。

### 6.2 `getOrAddProperties<Properties>()`：属性装箱

先看 `Properties` 结构体的真实定义，见 `NorthStarOps.h.inc:1432-1451`：

```cpp
struct Properties {
  using axisTy = ::mlir::IntegerAttr;
  axisTy axis;   // 注意：字段类型是 IntegerAttr，不是 int64_t

  auto getAxis() { return ::llvm::cast<::mlir::IntegerAttr>(this->axis); }
  void setAxis(const ::mlir::IntegerAttr &propValue) { this->axis = propValue; }
  bool operator==(const Properties &rhs) const {
    return rhs.axis == this->axis && true;
  }
  bool operator!=(const Properties &rhs) const { return !(*this == rhs); }
};
```

**关键观察**：`Properties::axis` 的类型是 `::mlir::IntegerAttr`（一个 Attribute），而非原生 `int64_t`。即 Properties 内部存储的仍是 Attribute，只是用强类型 struct 组织。

`getOrAddProperties<Properties>()` 实现见 `OperationSupport.h:998-1014`：

```cpp
template <typename T>
T &getOrAddProperties() {
  if (!properties) {
    T *p = new T{};                  // 堆上构造空 Properties
    properties = p;                  // 存入 OpaqueProperties（void* 包装）
    propertiesDeleter = [](OpaqueProperties prop) {
      delete prop.as<const T *>();
    };
    propertiesSetter = [](OpaqueProperties new_prop, const OpaqueProperties prop) {
      *new_prop.as<T *>() = *prop.as<const T *>();
    };
    propertiesId = TypeID::get<T>(); // 类型标识，防串台
  }
  assert(propertiesId == TypeID::get<T>() && "Inconsistent properties");
  return *properties.as<T *>();      // 返回引用，供 .axis = ... 赋值
}
```

语义：惰性创建 Properties 实例，返回引用以供直接成员赋值。

### 6.3 `getIntegerAttr(...)`：int → Attribute 装箱

```cpp
$_builder.getIntegerAttr(odsBuilder.getIntegerType(64,true), axis)
```

分两步：

1. `getIntegerType(64, true)` → 获取 `i64`（signless）Type 对象；
2. `getIntegerAttr(type, axis)` → 将 `int64_t` 的 `1` 装箱为 `IntegerAttr`（内部持有 `APInt`）。

**装箱的必要性**：IR 中一切静态值都是 Attribute。原生 `int64_t` 无类型、不可 intern、不可序列化，不能直接进入 IR。`IntegerAttr` 将"整数值 + 其 Type"打包为一等公民对象（经 StorageUniquer 唯一化）。此过程类比 Java 的 `int` → `Integer` 装箱（boxing）。

### 6.4 `addTypes(input.getType())`：结果类型登记

`OperationSupport.h:1025-1027`：

```cpp
void addTypes(ArrayRef<Type> newTypes) {
  types.append(newTypes.begin(), newTypes.end());
}
```

将结果类型（此处为输入张量类型 `!ns.tensor<2x2xf32, device=0>`）登记到 `types` 向量。注意：此处仅登记**类型**，真正的 `OpResult`（SSA value 容器）在 `Operation::create` 阶段才分配。

---

## 7. 实例化阶段：Operation::create 的内存模型

`build` 返回后，`OpBuilder::create` 调用 `Operation::create(state)` 进行"浇筑"。这是最关键、也最精巧的环节。

### 7.1 入口与转发

`OpBuilder::create<OpTy>` 模板见 `Builders.h:508-517`：

```cpp
template <typename OpTy, typename... Args>
OpTy create(Location location, Args &&...args) {
  OperationState state(location,
                       getCheckRegisteredInfo<OpTy>(location.getContext()));
  OpTy::build(*this, state, std::forward<Args>(args)...);  // 填充 state
  auto *op = create(state);                                 // 实例化
  auto result = dyn_cast<OpTy>(op);
  assert(result && "builder didn't return the right type");
  return result;
}
```

`Operation::create(const OperationState&)` 见 `Operation.cpp:34-48`，转发到字段级 `create` 重载 `Operation.cpp:82-153`。

### 7.2 内存布局：尾分配（Trailing Objects）与前缀结果

`Operation` 类继承自 `llvm::TrailingObjects`（`Operation.h:84-88`），采用 LLVM 的尾分配惯用法。其内存布局见 `Operation.h:42-67` 注释：

```
地址低 ────────────────────────────────────────────► 地址高
┌──────────────────────────┬──────────────────────────────────────────┐
│  OpResult (反向存储)      │              Operation 主体              │
│ [Result_{n-1} ... R1 R0] │  + OperandStorage + OpProperties +       │
│        ▲                 │    BlockOperand[] + Region[] + OpOperand[]│
│        │                 │                                          │
│   前缀区(prefix)         │           尾分配区(trailing)              │
└──────────────────────────┴──────────────────────────────────────────┘
                           ^
                           └── Operation* 指向此处
```

**两个反直觉设计**：

1. **结果反向存储在 Operation 之前**：对 3 个结果的 Op，内存为 `[Result2, Result1, Result0, Operation]`。这是为了支持 `OutOfLineOpResult`（结果序号 > 5 时）的变长存储。

2. **结果分两种形态**（见 `Value.h:366-432`）：
   - `InlineOpResult`（前 5 个结果）：序号编码进 `ValueImpl::typeAndKind` 的 3 个 bit，无需额外字段，省空间。
   - `OutOfLineOpResult`（第 6 个及以后）：需要额外的 `outOfLineIndex` 字段。

### 7.3 大小计算与分配

`Operation.cpp:90-116`：

```cpp
unsigned numTrailingResults = OpResult::getNumTrailing(resultTypes.size());
unsigned numInlineResults  = OpResult::getNumInline(resultTypes.size());
unsigned numOperands = operands.size();
int opPropertiesAllocSize = llvm::alignTo<8>(name.getOpPropertyByteSize());
bool needsOperandStorage = operands.empty()
    ? !name.hasTrait<OpTrait::ZeroOperands>() : true;

size_t byteSize = totalSizeToAlloc<detail::OperandStorage, detail::OpProperties,
                                   BlockOperand, Region, OpOperand>(
    needsOperandStorage ? 1 : 0, opPropertiesAllocSize, numSuccessors,
    numRegions, numOperands);
size_t prefixByteSize = llvm::alignTo(
    Operation::prefixAllocSize(numTrailingResults, numInlineResults),
    alignof(Operation));
char *mallocMem = reinterpret_cast<char *>(malloc(byteSize + prefixByteSize));
void *rawMem = mallocMem + prefixByteSize;   // Operation* 跳过前缀
Operation *op = ::new (rawMem) Operation(...);  // placement new
```

要点：

- 一次 `malloc` 同时分配结果前缀区 + Operation 主体 + 所有尾分配对象，**零额外堆开销**。
- `Operation*` 指向主体起始（前缀区之后），因此结果需要通过指针算术回溯访问。
- 若 Op 已知无操作数（`ZeroOperands` trait）且当前无操作数，则不分配 `OperandStorage`，节省内存。

### 7.4 对象初始化顺序

`Operation.cpp:125-152`：

```cpp
// 1. 初始化结果（placement new 到前缀区）
for (unsigned i = 0; i < numInlineResults; ++i, ++resultTypeIt)
  new (op->getInlineOpResult(i)) detail::InlineOpResult(*resultTypeIt, i);
for (unsigned i = 0; i < numTrailingResults; ++i, ++resultTypeIt)
  new (op->getOutOfLineOpResult(i)) detail::OutOfLineOpResult(*resultTypeIt, i);

// 2. 初始化区域
for (unsigned i = 0; i != numRegions; ++i)
  new (&op->getRegion(i)) Region(op);

// 3. 初始化操作数 ← use-def 链在此建立
if (needsOperandStorage) {
  new (&op->getOperandStorage()) detail::OperandStorage(
      op, op->getTrailingObjects<OpOperand>(), operands);
}

// 4. 初始化后继
for (unsigned i = 0; i != numSuccessors; ++i)
  new (&blockOperands[i]) BlockOperand(op, successors[i]);

// 5. 设置属性（须在 properties 初始化之后）
op->setAttrs(attributes);
```

构造函数本体见 `Operation.cpp:155-175`，初始化 `location` / `name` / 计数字段，并在 `NDEBUG` 下校验方言已注册、调用 `name.initOpProperties` 初始化 properties。

---

## 8. Use-Def 链的形成机制

本节是报告的技术核心。当一个 Op 以另一 Op 的结果为操作数时，"数据流边"如何在数据结构层面建立？

### 8.1 数据结构：侵入式双向链表

MLIR 的 use-def 关系由 `IROperandBase` 维护，定义于 `UseDefLists.h:35-114`。这是一个**侵入式双向链表**（intrusive doubly-linked list），每个 operand 节点持有：

```cpp
class IROperandBase {
  IROperandBase *nextUse = nullptr;   // 同一 value 的下一个使用
  IROperandBase **back = nullptr;     // 指向前驱节点的 nextUse 字段地址
  Operation *const owner;             // 拥有此 operand 的 Op
};
```

Value 侧的 use-list 容器是 `IRObjectWithUseList`（`UseDefLists.h:189-294`），仅持有一个头指针：

```cpp
template <typename OperandType>
class IRObjectWithUseList {
  detail::IROperandBase *firstUse = nullptr;  // use-list 头
};
```

`back` 采用"指向指针的指针"（`IROperandBase**`）而非"指向前驱节点"，使**头插与任意位置删除均为 O(1)**，无需特判头节点。

### 8.2 插入操作 insertInto

`UseDefLists.h:97-103`：

```cpp
template <typename UseListT>
void insertInto(UseListT *useList) {
  back = &useList->firstUse;        // back 指向 value 的 firstUse 字段
  nextUse = useList->firstUse;      // 接到原链表头
  if (nextUse)
    nextUse->back = &nextUse;       // 原头节点的 back 改指向我
  useList->firstUse = this;         // 我成为新头
}
```

这是经典的双向链表头插法，O(1)。

### 8.3 OpOperand 的构造与自动入链

`OpOperand` 继承 `IROperand<OpOperand, Value>`（`Value.h:280-285`），其构造函数 `UseDefLists.h:130-133`：

```cpp
IROperand(Operation *owner, IRValueT value)
    : detail::IROperandBase(owner), value(value) {
  insertIntoCurrent();   // 构造即入链
}
```

`insertIntoCurrent`（`UseDefLists.h:186`）调用 `insertInto(DerivedT::getUseList(value))`，即把当前 operand 插入 value 的 firstUse 链表。

**核心洞察**：OpOperand 一旦被构造，就**自动**把自己挂到所引用 Value 的 use-list 上。use-def 链的建立不需要额外显式调用——它是构造函数的副作用。

### 8.4 OperandStorage 的批量构造

`OperationSupport.cpp:239-246`：

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

`OperandStorage` 在 Operation 主体后的尾分配区中，逐个 placement-new 构造 `OpOperand(owner, values[i])`。每次构造都触发 8.3 节的自动入链。因此，当 `SoftmaxOp` 以 `get_tensor_op_1` 的结果为操作数时：

```
SoftmaxOp 的 OperandStorage 构造
  └─ new OpOperand(softmaxOp, %get_tensor_result)
       └─ insertIntoCurrent()
            └─ insertInto(%get_tensor_result.useList)
                 → %get_tensor_result.firstUse = &softmaxOp的operand
```

`get_tensor_op_1` 的结果 value 的 use-list 上从此多了一个节点，指向 SoftmaxOp 的 operand。**这就是数据流"边"的物理实现**。

### 8.5 use-list 的链表操作图解

设 value V 原有使用者链 `A → B`（A 的 operand 引用 V，B 的 operand 引用 V）。现 SoftmaxOp S 新增对 V 的使用：

```
插入前:  V.firstUse → A_use → B_use → ∅

插入后:  V.firstUse → S_use → A_use → B_use → ∅
          (S_use.nextUse=A_use, A_use.back=&S_use.nextUse)
```

每个 use 节点通过 `owner` 字段可回溯到使用它的 Operation，故遍历 use-list 即可得到所有 user（`getUsers()`，见 `UseDefLists.h:267-274`）。

### 8.6 销毁约束：use_empty 断言

`Operation.cpp:179-191` 的析构函数在 `NDEBUG` 下断言：

```cpp
if (!use_empty()) {
  emitOpError("operation destroyed but still has uses");
  ...
  llvm::report_fatal_error("operation destroyed but still has uses");
}
```

`IRObjectWithUseList` 析构（`UseDefLists.h:197-199`）同样断言 `use_empty()`。这保证了 SSA 的核心不变量：**一个仍被引用的 value 不能被销毁**，从而避免悬挂引用，维持图的结构一致性。

`replaceAllUsesWith`（RAUW，`UseDefLists.h:210-216`）通过逐个 `use_begin()->set(newValue)` 重定向所有使用者，`set`（`UseDefLists.h:163-169`）先 `removeFromCurrent` 再 `insertIntoCurrent`，完成链表节点的迁移。

---

## 9. Operand 与 Attribute 的语义边界再讨论

### 9.1 判断准则

设计一个新 Op 时，对每个参数判断其归属：

> 该值是否由另一条计算链路产生？是否随程序执行而变化？

- 是 → operand；
- 否（编译期固定的元数据/配置）→ attribute。

### 9.2 设计哲学

将 operand 与 attribute 分离的收益：

1. **依赖追踪**：Pass 做死代码消除时，沿 operand 边追踪活跃性，attribute 不参与。
2. **常量折叠**：attribute 是 compile-time 已知，可直接参与折叠计算。
3. **图等价判断**：两个 SoftmaxOp 若 operand 相同且 `axis` 相同（见 `Properties::operator==`，`NorthStarOps.h.inc:1443-1447`）即等价。
4. **文本表示清晰**：operand 在 `(...)`，inherent attr 在 `{ }`。

### 9.3 Properties 机制的演进动机

Properties 是 MLIR 较新引入（约 2023 年）的机制。此前 inherent attr 全部存于一个 `DictionaryAttr`（字符串→Attribute 映射），存在三缺陷：

1. 访问需字符串查找，慢；
2. 序列化 / 哈希 / 比较需逐项重建；
3. 无类型安全（`dict.get("axis")` 返回 `Attribute`，需手动 cast）。

`Properties` 将已知 inherent attr 提升为强类型 struct：访问变为直接成员 `prop.axis`（O(1)、编译期类型 `IntegerAttr`）；同时生成代码保留 Properties ↔ DictionaryAttr 的双向转换（`NorthStarOps.cpp.inc:1016-1053` 的 `setPropertiesFromAttr` / `getPropertiesAsAttr`），兼容文本打印与 bytecode 序列化。`discardable attribute`（存于 `NamedAttrList`）则留给 Pass 临时打标记，与 inherent attr 严格区分。

---

## 10. 完整生命周期案例：SoftmaxOp

### 10.1 构建上下文

`main.cpp:291-300`：

```cpp
auto get_tensor_op_1 = builder.create<mlir::north_star::GetTensorOp>(
    loc, tensor_type_1, buffer_op, 0);          // 产生 %0
...
auto softmax_op =
    builder.create<mlir::north_star::SoftmaxOp>(loc, get_tensor_op_1, 1);
```

### 10.2 完整调用链

```
① builder.create<SoftmaxOp>(loc, get_tensor_op_1, 1)
   │  [Builders.h:508-517]
   ├─ 构造 OperationState state{loc, "north_star.softmax"}
   ├─ SoftmaxOp::build(odsBuilder, state, get_tensor_op_1, 1)
   │     [NorthStarOps.cpp.inc:1107-1112]
   │   ├─ state.addOperands(%0)                     → state.operands=[%0]
   │   ├─ state.getOrAddProperties<Properties>().axis
   │   │     = IntegerAttr(i64, 1)                  → state.properties.axis=i64(1)
   │   └─ state.addTypes(ns.tensor<2x2xf32,dev=0>)  → state.types=[<...>]
   ├─ create(state) → Operation::create(state)  [Operation.cpp:34-48]
   │     [Operation.cpp:82-153]
   │   ├─ malloc(prefix + body + trailing)
   │   ├─ placement new Operation(...)
   │   ├─ 初始化 OpResult（前缀区）→ %1 = softmax 结果
   │   ├─ placement new OperandStorage(softmaxOp, trailingOps, [%0])
   │   │     └─ new OpOperand(softmaxOp, %0)
   │   │           └─ insertIntoCurrent() → %0.useList 头插
   │   │                 【use-def 链建立：softmaxOp 成为 %0 的 user】
   │   └─ setAttrs(...)
   └─ dyn_cast<SoftmaxOp> 返回
```

### 10.3 use-def 链的最终状态

```
%0 (get_tensor_op_1 的结果)
  └─ firstUse → [SoftmaxOp 的 operand]
                   └─ owner = SoftmaxOp*

SoftmaxOp
  ├─ operands[0] → %0   (OpOperand，持有 value=%0)
  ├─ results[0]  → %1   (新产生的 OpResult)
  └─ properties.axis = IntegerAttr(i64, 1)
```

### 10.4 文本表示（.dump() 输出）

```mlir
%1 = "north_star.softmax"(%0) <{axis = 1 : i64}>
    : (!ns.tensor<2x2xf32, device = 0>) -> !ns.tensor<2x2xf32, device = 0>
```

`%0` 是 operand（数据流边），`axis = 1 : i64` 是 inherent attribute（来自 Properties），二者共同构成 SoftmaxOp 这个 IR 节点。

---

## 11. 结论

本报告通过对 MLIR Operation 构建管线、内存模型与 use-def 链机制的源码级分析，得出以下结论：

1. **三段式构建管线**：Operation 的诞生经历声明层（`.td`）→ 生成层（`.inc`）→ 运行时实例化层。`$_state` / `$_builder` 等替换变量是 TableGen 代码模板的占位符，由 `mlir-tblgen` 替换为真实 C++ 标识符，实现了方言声明与生成实现的解耦。

2. **OperationState 是建造说明书**：作为栈上临时对象，它收集 operands / types / properties / regions / successors，由 `build` 填充、`Operation::create` 消费。

3. **尾分配内存模型**：`Operation` 通过 `llvm::TrailingObjects` 在单次 `malloc` 中紧凑布局结果前缀区、Operation 主体与所有尾分配对象（OperandStorage、OpProperties、BlockOperand、Region、OpOperand），零额外堆开销；结果反向存储并区分 inline / out-of-line 双形态以优化空间。

4. **use-def 链由构造自动建立**：侵入式双向链表（`IROperandBase` + `IRObjectWithUseList::firstUse`）维护每个 value 的使用者列表。`OpOperand` 的 placement new 构造触发自动头插，使"边"的建立成为构造的副作用，O(1) 完成。

5. **Operand / Attribute 的语义分野是 MLIR 的根本设计**：数据流边与编译期配置的严格分离，支撑了依赖追踪、常量折叠、图等价判断等编译分析，是渐进式 lowering 得以实现的基础。Properties 机制以强类型 struct 替代 DictionaryAttr 查找，兼顾性能与类型安全。

这些机制共同构成了 MLIR 作为可扩展、多层次、图结构中间表示的工程基石。

---

## 附录 A：关键源码文件索引

### A.1 NorthStar 方言层（教程）

| 文件 | 内容 |
|---|---|
| NorthStarOps.td | Op 的 TableGen 声明 |
| main.cpp | CH5 入口，构造并 dump IR |
| NorthStarOps.cpp | 手写 verify 实现 |
| NorthStarOps.h.inc | 生成声明 |
| NorthStarOps.cpp.inc | 生成实现 |

### A.2 MLIR 核心层（LLVM 19.1.7）

| 文件 | 内容 |
|---|---|
| Operation.h | Operation 类与内存布局注释 |
| Operation.cpp | `Operation::create` 实现 |
| OperationSupport.h | `OperationState`、`OperandStorage`、`OpaqueProperties` |
| OperationSupport.cpp | `addOperands`、`OperandStorage` 构造 |
| Builders.h | `OpBuilder::create<OpTy>` 模板 |
| UseDefLists.h | `IROperandBase`、`IROperand`、`IRObjectWithUseList` |
| Value.h | `ValueImpl`、`OpResultImpl`、`InlineOpResult`、`OutOfLineOpResult` |

## 参考文献

[1] MLIR 官方文档. Operations. https://mlir.llvm.org/docs/LangRef/#operations  
[2] MLIR 官方文档. Understanding the IR Structure. https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/  
[3] MLIR 官方文档. Operation Definition Specification (ODS). https://mlir.llvm.org/docs/DefiningDialects/Operations/  
[4] LLVM Project. MLIR Source Tree (LLVM 19.1.7). `third_party/llvm-project/mlir/`  
[5] violetDelia. NorthStar MLIR Tutorial. Apache-2.0 License.  
[6] Lattner C, Amini M. MLIR: A Compiler Infrastructure for the End of Moore's Law. arXiv:2002.11054, 2020.

---

*报告完*
