# 第 7 章 Operation::create 的完整路径与尾分配内存模型

> **本章位置**　第 6 章讲了构建的"外壳"——OpBuilder、OperationState、七类挂接。但 `Operation::create` 内部到底怎么把装箱单变成一个真实的、挂在内存里的 Operation 对象？本章打开 create 的源码，讲清两件事：**声明式 op 定义**（`.td → .inc → C++` 三段式管线）与**尾分配内存模型**（一次 malloc 分配所有东西）。尾分配与 create 在本章内聚——因为它是 create 时的分配技巧，脱离 create 讲布局是空中楼阁。
>
> **前置依赖**　第 1 章（Type/Attribute 属于 context、唯一化）、第 2 章（Operation 字段、Operand vs Attribute 分野、Properties）、第 4 章（use-list 数据结构）、第 6 章（OpBuilder 三段式、OperationState 装箱单）。
>
> **编译原理切入**　本章从两个编译器学科问题立论。**第一，声明式 IR 定义与代码生成**——编译器框架用 DSL 描述 IR 指令再生成实现，这是 GCC 的 machine descriptions、LLVM 的 TableGen、MLIR 的 ODS 共同的传统。**第二，编译器对内存紧凑性的执念**——IR 节点数量级达百万，一次 malloc 零额外堆开销是工程必需。MLIR 用 LLVM 的尾分配（trailing objects）技巧实现了这一点。本章追问一个反直觉的问题：Operation 为什么不 intern（不像 Type/Attribute）？答案揭示了一个深层设计原则——**无身份值对象 vs 有身份实体对象**。

---

## 7.1 构建管线的三层层级

第 6 章讲过，`OpBuilder::create<OpTy>` 内部是三段式（开装箱单 → build 填单 → create 消费）。但一个 Operation 的诞生其实跨越**三层**，比这三段式更深：

```text
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

本章逐层打开。第 1 层讲 `.td` 怎么声明 op，第 2 层讲 `mlir-tblgen` 生成什么，第 3 层讲 `Operation::create` 如何分配内存并初始化。

## 7.2 声明层：TableGen 的 `.td`

MLIR 用 **TableGen** 语言描述方言实体（Type / Attribute / Operation），由 `mlir-tblgen` 生成 C++ 代码（`.inc` 文件）。手写的 `.h` / `.cpp` 依赖这些生成产物。这一范式使方言定义**声明化、自动化、可校验**——这是声明式 IR 设计的传统。

以 NorthStar 教程的 `SoftmaxOp` 为例（[NorthStarOps.td:59-70](../MLIR-Tutorial/5-define_operation/include/Dialect/NorthStar/IR/NorthStarOps.td#L59)）：

```tablegen
def NorthStar_SoftmaxOp : NorthStar_UnaryOp<"softmax", AnyNSTensor, AnyNSTensor,
                                            [], (ins I64Attr:$axis)>{
    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis),
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis =
                    $_builder.getIntegerAttr($_builder.getIntegerType(64,true), axis);
                $_state.addTypes(input.getType());
            }]>
    ];
}
```

它继承自 `NorthStar_UnaryOp` 基类（[NorthStarOps.td:28-37](../MLIR-Tutorial/5-define_operation/include/Dialect/NorthStar/IR/NorthStarOps.td#L28)），该基类把 `arguments` 展开为 `(ins OperandType:$input, <attributes>)`，`results` 为 `(outs resultType:$result)`。因此 `SoftmaxOp` 声明：一个 `AnyNSTensor` 操作数 `$input`，一个 `I64Attr` 属性 `$axis`，一个 `AnyNSTensor` 结果。

关键点：
- **声明的是 op 的"形状"**——有哪些 operand、哪些 attribute、哪些 result，以及 builder 模板。
- **手写 builder**：`OpBuilder<(ins ...), [{ 代码 }]>` 里的 `[{ }]` 是代码模板，`mlir-tblgen` 会展开成真正的 C++（下一节讲展开规则）。

> **编译原理浸润点：声明式 IR 定义**　用 DSL 描述 IR 指令再生成实现，是编译器框架的成熟传统。GCC 用 `.md`（machine description）文件描述目标指令；LLVM 用 TableGen 描述指令、寄存器、调度；MLIR 把 TableGen 应用到了方言的 op/type/attribute 定义。这种声明式范式的好处是：把"op 长什么样"（声明）与"op 怎么实现"（生成代码）分离，让 op 定义可校验、可工具化。Dragon Book 不讨论代码生成（它是编译器的元层话题），但任何大型编译器都用某种声明式机制管理海量指令定义。Ch1 讲的 dialect 加载机制、本章的 ODS，都是这个传统的体现。

## 7.3 生成层：`$_state` 与 `$_builder` 的替换语义

`mlir-tblgen` 把 `.td` 翻译成 `.inc`。理解生成的关键，是 **TableGen 替换变量**的语义。

### 7.3.1 `$_state` 与 `$_builder` 的定义

`.td` 中 `OpBuilder` dag 的 `[{ }]` 代码块是**代码模板**（code template），而非普通 C++。`mlir-tblgen` 在展开时进行文本替换。`$_xxx` 是预定义的替换变量：

| 替换变量 | 展开为 | 实际类型 |
|---|---|---|
| `$_state` | `odsState` | `OperationState &` |
| `$_builder` | `odsBuilder` | `OpBuilder &` |

### 7.3.2 替换的源码证据

将 `.td` 与生成产物逐字对照。`.td` 里的手写 builder：

```tablegen
OpBuilder<(ins "::mlir::Value":$input, "int64_t":$axis),
    [{
        $_state.addOperands(input);
        $_state.getOrAddProperties<Properties>().axis =
            $_builder.getIntegerAttr($_builder.getIntegerType(64,true), axis);
        $_state.addTypes(input.getType());
    }]>
```

生成产物 `NorthStarOps.cpp.inc:1107-1112`：

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

### 7.3.3 设计动机

将变量名交由生成器决定而非硬编码在 `.td`，实现**关注点分离**：`.td` 是方言中立描述，不依赖生成后的 C++ 签名细节；框架参数（builder、state）由 `$_xxx` 隐式注入，用户参数（input、axis）在 `(ins ...)` 显式声明。这一抽象屏障使 MLIR 升级内部实现时不影响方言 `.td` 文件。

## 7.4 OperationState：建造说明书（细节）

第 6 章介绍了 OperationState 的角色（装箱单）。这里打开它的数据结构（[OperationSupport.h:950-1075](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/OperationSupport.h#L950)）：

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

字段可归为三类：

1. **身份字段**：`location`、`name` — 决定"是什么 Op、在哪里"。
2. **结构字段**：`operands`、`types`、`successors`、`regions` — 决定"图的拓扑"。
3. **配置字段**：`attributes`、`properties` / `propertiesAttr` — 决定"编译期参数"。

`OperationSupport.h:946-949` 的注释明确：*"This object is a large and heavy weight object meant to be used as a temporary object on the stack."*——它是**栈上临时对象**，充当"建造说明书"。`build` 函数向其填料；`Operation::create` 消费它并构造不可变的 `Operation`。

### 7.4.1 build 函数的三行拆解

回到 SoftmaxOp 的 build 函数体，逐行打开。

**`addOperands(input)`**（[OperationSupport.h:1023](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/OperationSupport.h#L1023)，[OperationSupport.cpp:212-214](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/OperationSupport.cpp#L212)）：

```cpp
void OperationState::addOperands(ValueRange newOperands) {
  operands.append(newOperands.begin(), newOperands.end());
}
```

语义：将传入的 `Value`（SSA value 引用）追加到 `operands` 向量。`input` 是指向某个 op 结果的引用，即数据流图的一条边。**此时尚未建立 use-def 链**（那是 `Operation::create` 的事），仅完成"记录意图"。

**`getOrAddProperties<Properties>()`**（[OperationSupport.h:998-1014](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/OperationSupport.h#L998)）：

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

**关键观察**：Properties 内部存储的仍是 Attribute（`IntegerAttr`），而非原生 `int64_t`（[NorthStarOps.h.inc:1432-1451](../MLIR-Tutorial/5-define_operation/build/5-define_operation/include/Dialect/NorthStar/IR/NorthStarOps.h.inc#L1432)）：

```cpp
struct Properties {
  using axisTy = ::mlir::IntegerAttr;
  axisTy axis;   // 字段类型是 IntegerAttr，不是 int64_t
  bool operator==(const Properties &rhs) const {
    return rhs.axis == this->axis && true;
  }
};
```

这是 Ch2 讲的"Properties 用强类型 struct 组织 Attribute"的具体实现。

**`getIntegerAttr(...)`：int → Attribute 装箱**：

```cpp
$_builder.getIntegerAttr($_builder.getIntegerType(64,true), axis)
```

分两步：先 `getIntegerType(64, true)` 得到 `i64` Type（从 context 池，Ch1）；再 `getIntegerAttr(type, axis)` 把 `int64_t` 的 `1` 装箱为 `IntegerAttr`（内部持有 `APInt`，存进 context 的 attributeUniquer 池）。**装箱的必要性**：IR 中一切静态值都是 Attribute。原生 `int64_t` 无类型、不可 intern、不可序列化，不能直接进入 IR。此过程类比 Java 的 `int` → `Integer` 装箱（boxing）。

**`addTypes(input.getType())`**（[OperationSupport.h:1025-1027](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/OperationSupport.h#L1025)）：

```cpp
void addTypes(ArrayRef<Type> newTypes) {
  types.append(newTypes.begin(), newTypes.end());
}
```

将结果类型登记到 `types` 向量。**注意：此处仅登记类型，真正的 OpResult（SSA value 容器）在 `Operation::create` 阶段才分配**。

## 7.5 运行时层：Operation::create 的尾分配内存模型

`build` 返回后，`OpBuilder::create` 调用 `Operation::create(state)` 进行"浇筑"。这是最关键、也最精巧的环节。

### 7.5.1 入口与转发

`Operation::create(const OperationState&)` 见 [Operation.cpp:34-48](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L34)，转发到字段级 `create` 重载 [Operation.cpp:82-153](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L82)。

### 7.5.2 内存布局：尾分配与前缀结果

`Operation` 类继承自 `llvm::TrailingObjects`（[Operation.h:84-88](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L84)），采用 LLVM 的尾分配惯用法。其内存布局见 [Operation.h:42-67](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L42) 的注释：

```text
地址低 ──────────────────────────────────────────────────────► 地址高

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

1. **结果反向存储在 Operation\* 之前**：对 3 个结果的 Op，内存为 `[Result2, Result1, Result0, Operation]`。这是为了支持 `OutOfLineOpResult`（结果序号 > 5 时）的变长存储。
2. **结果分两种形态**（[Value.h:366-432](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h#L366)）：
   - `InlineOpResult`（前 5 个结果）：序号编码进 `ValueImpl::typeAndKind` 的 3 个 bit，无需额外字段，省空间。
   - `OutOfLineOpResult`（第 6 个及以后）：需要额外的 `outOfLineIndex` 字段。

### 7.5.3 大小计算与分配

[Operation.cpp:90-116](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L90)：

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
- **一次 `malloc` 同时分配结果前缀区 + Operation 主体 + 所有尾分配对象，零额外堆开销**。
- `Operation*` 指向主体起始（前缀区之后），因此结果需要通过指针算术回溯访问（Ch6 挂接⑥）。
- 若 Op 已知无操作数（`ZeroOperands` trait）且当前无操作数，则不分配 `OperandStorage`，节省内存。

### 7.5.4 对象初始化顺序

[Operation.cpp:125-152](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L125)：

```cpp
// 1. 初始化结果（placement new 到前缀区）
for (unsigned i = 0; i < numInlineResults; ++i, ++resultTypeIt)
  new (op->getInlineOpResult(i)) detail::InlineOpResult(*resultTypeIt, i);
for (unsigned i = 0; i < numTrailingResults; ++i, ++resultTypeIt)
  new (op->getOutOfLineOpResult(i)) detail::OutOfLineOpResult(*resultTypeIt, i);

// 2. 初始化区域
for (unsigned i = 0; i != numRegions; ++i)
  new (&op->getRegion(i)) Region(op);     // ← Ch6 挂接①

// 3. 初始化操作数 ← use-def 链在此建立（Ch8 详解）
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

构造函数本体见 [Operation.cpp:155-175](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L155)，初始化 `location` / `name` / 计数字段，并在 `NDEBUG` 下校验方言已注册、调用 `name.initOpProperties` 初始化 properties。

注意第 3 步——**操作数初始化触发 use-def 链建立**。OperandStorage 的构造会逐个 placement-new OpOperand，每个 OpOperand 构造时自动入链（Ch4 的 insertInto）。这个"构造即入链"的机制是第 8 章的主题，本章只指出它在 create 内部发生。

### 7.5.5 销毁：destroy 的指针回退

由于结果反向存在 Operation\* 之前，`destroy()` 必须后退 `prefixAllocSize` 才能拿到原始 `malloc` 指针（[Operation.cpp:208-215](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L208)）：

```cpp
void Operation::destroy() {
  char *rawMem = reinterpret_cast<char *>(this)
    - llvm::alignTo(prefixAllocSize(), alignof(Operation));
  this->~Operation();
  free(rawMem);
}
```

这就是为什么 Ch10 会强调"**不能 `delete op`**"——前缀区未释放，原始 malloc 指针丢失。必须用 `op->erase()` 或 `op->destroy()`。

> **编译原理浸润点：编译器对内存紧凑性的执念**　为什么 MLIR 用这么复杂的尾分配？因为 IR 节点数量级达百万。一个典型的神经网络 lowering，中间会生成几十万到几百万个 Operation。每个 Operation 如果用普通 C++ 对象（多个独立堆分配），内存碎片与分配开销会累积成灾难。尾分配把 Operation 主体 + 所有附属对象（OperandStorage、OpProperties、BlockOperand、Region、OpOperand）塞进一次 malloc，零额外堆开销——这在百万级规模下节省的内存与分配时间是数量级的提升。Dragon Book 不讨论这种工程优化（它假设 IR 是逻辑结构），但任何高性能编译器（LLVM、MLIR、Graal）都必须做类似的紧凑布局。LLVM 的 `TrailingObjects` 框架、C 的 flexible array member、Linux 内核的 `kmalloc` + 结构体尾部数组，都是同一思想的工程实现。

## 7.6 为什么 Operation 不被唯一化

对比一个深层问题：Type 与 Attribute 被 context 唯一化（Ch1），Operation 为什么不？

答案揭示了一个深层设计原则——**无身份值对象 vs 有身份实体对象**：

- **Type、Attribute 是无身份的（value-equal）**：`i32` 就是 `i32`，没有"哪一个 i32"之分。结构相等的 Type 在内存里是同一个对象，等价判断退化为指针比较。唯一化对此类对象有意义。
- **Operation 是有身份的（identity-bearing）**：同样是 `addi %a, %b`，这一个和那一个是不同的运算——它们发生在程序的不同位置（不同的 location）、嵌套在不同的 Block 里、可能有不同的属性。两个"长得一样"的 Operation 是不同的对象，不能合并。

这个区分对应编译器的一个经典概念：**值语义 vs 实体语义**。Type/Attribute 是值（只关心它是什么），Operation 是实体（关心它是哪一个）。把这两类对象用不同的所有权策略管理，是 MLIR 内存模型的精髓：

| 对象类型 | 语义 | 所有权 | 唯一化 |
|---|---|---|---|
| Type / Attribute | 值（value-equal） | context 池（immortal） | ✅ 唯一化 |
| Operation | 实体（identity-bearing） | 所在的 Block（Ch3） | ❌ 不唯一化 |

> **编译原理浸润点：值语义 vs 实体语义**　这个区分不是 MLIR 独创，它是编程语言理论的基础概念。函数式语言强调值语义（`1 == 1` 永远成立，不关心"哪一个 1"），命令式语言强调实体语义（两个对象即使内容相同也是不同的对象）。编译器 IR 必须同时处理这两类——Type/常量是值，指令/运算是实体。MLIR 用"唯一化 vs 不唯一化"明确表达了这一区分。LLVM 也有类似的区分（`Type` 是值、`Instruction` 是实体），MLIR 继承了这个设计。

## 7.7 完整生命周期案例：SoftmaxOp

把本章三层串起来。`main.cpp:291-300`：

```cpp
auto get_tensor_op_1 = builder.create<mlir::north_star::GetTensorOp>(
    loc, tensor_type_1, buffer_op, 0);          // 产生 %0
...
auto softmax_op =
    builder.create<mlir::north_star::SoftmaxOp>(loc, get_tensor_op_1, 1);
```

完整调用链：

```text
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
   │   ├─ malloc(prefix + body + trailing)          ← 尾分配，一次分配
   │   ├─ placement new Operation(...)
   │   ├─ 初始化 OpResult（前缀区）→ %1 = softmax 结果
   │   ├─ placement new OperandStorage(softmaxOp, trailingOps, [%0])
   │   │     └─ new OpOperand(softmaxOp, %0)
   │   │           └─ insertIntoCurrent() → %0.useList 头插
   │   │                 【use-def 链建立：softmaxOp 成为 %0 的 user】  ← Ch8 详解
   │   └─ setAttrs(...)
   └─ dyn_cast<SoftmaxOp> 返回
```

最终的 IR 文本（`.dump()` 输出）：

```mlir
%1 = "north_star.softmax"(%0) <{axis = 1 : i64}>
    : (!ns.tensor<2x2xf32, device = 0>) -> !ns.tensor<2x2xf32, device = 0>
```

`%0` 是 operand（数据流边），`axis = 1 : i64` 是 inherent attribute（来自 Properties），二者共同构成 SoftmaxOp 这个 IR 节点。

---

## 编译原理浸润点回顾

1. **声明式 IR 定义**（本章主题）：`.td → .inc → C++` 三段式是 GCC `.md`、LLVM TableGen 传统的延续。让 op 定义声明化、可校验、可工具化。
2. **尾分配内存技巧**（本章主题）：一次 malloc 分配所有，源于编译器对百万级 IR 节点内存紧凑性的执念。LLVM `TrailingObjects`、C flexible array member 同源。
3. **值语义 vs 实体语义**：Type/Attribute 是值（唯一化），Operation 是实体（不唯一化）。MLIR 用不同所有权策略管理两类对象。
4. **装箱（boxing）**：int → IntegerAttr 的装箱类比 Java int → Integer，源于"IR 中一切静态值都是 Attribute"的设计。
5. **可扩展 IR 哲学**（Ch0/Ch1 引入，本章通过 ODS 落实）：声明式 op 定义让定义新 op 变成填表，不需要改框架。

---

## 本章关键结论

1. **Operation 的构建跨越三层**：声明层（.td）→ 生成层（.inc）→ 运行时层（create）。`$_state`/`$_builder` 是 TableGen 的替换变量，解耦声明与生成。
2. **OperationState 是栈上建造说明书**：build 填单（身份/结构/配置三类字段），create 消费。addOperands 只记录意图，不建立 use-def 链。
3. **Operation::create 用尾分配一次 malloc 分配所有**：前缀区（结果反向存储）+ 主体 + 尾分配区（OperandStorage/OpProperties/BlockOperand/Region/OpOperand）。零额外堆开销，源于编译器对百万级节点内存紧凑性的执念。
4. **结果分 Inline/OutOfLine 两种形态**：前 5 个 inline（序号编码进 bit），第 6+ 个 out-of-line（额外字段）。省空间。
5. **Operation 不被唯一化**：它是 identity-bearing 实体（有 location/嵌套），不是 value-equal 值。这与 Type/Attribute 的唯一化形成对比，对应"值语义 vs 实体语义"的设计原则。
6. **操作数初始化触发 use-def 链建立**：OperandStorage 构造时逐个 placement-new OpOperand，每个自动入链。这个机制是 Ch8 的主题。

---

## 下一章预告

本章讲了 create 的内存模型与初始化顺序。其中第 3 步——**操作数初始化触发 use-def 链建立**——只是一笔带过。第 8 章专门讲这个机制：OpOperand 构造如何自动入链，OperandStorage 如何批量构造，use-list 链表操作的图解，以及销毁约束 use_empty 断言。本章建立了数据结构（Ch4）与 create（Ch7）两个前提，Ch8 把它们缝起来——**SSA 不变量在构造时如何被自动维护**。

---

## 原文对照

本章素材主要来自：
- `docs/operator-use-def的形成.md` §3-7（三段式构建管线、OperationState、TableGen 替换变量、Operation::create 内存模型、对象初始化顺序、SoftmaxOp 完整案例）——**全文保留，重新组织为编译器视角叙事**
- `docs/MLIR-IR-Node组织与遍历插入删除教程.md` §2.1（尾分配内存布局图）——**并入本章**
- 编译原理铺垫（声明式 IR 定义传统、尾分配的内存紧凑性动机、值语义 vs 实体语义）为本书新增

## 参考文献

- **[Aho 2006]** Dragon Book，第 8 章（IR 构造，尾分配的内存紧凑性动机）。
- **[Lattner 2020]** Lattner et al. "MLIR"，TableGen/ODS 与 Operation 内存模型的设计。
- **LLVM** `TrailingObjects` 文档与 `ilist` 文档，尾分配与侵入式链表的工程传统。
- **GCC** `.md` machine description，声明式 IR 定义的早期传统。
