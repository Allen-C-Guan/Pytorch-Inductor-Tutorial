# 第 2 章 Operation：一切皆操作及其部件语义

> **本章位置**　第 1 章讲完了"拥有者"。从本章开始，我们看被拥有的东西——IR 的逻辑部件。第 2、3 章只讲**逻辑部件是什么**（Operation 装什么、Block/Region 怎么嵌套），不讲数据流边的物理实现（留给 Ch4）——避免在读者还不认识 Value 的使用语义时就讲 use-list。
>
> **前置依赖**　第 0 章（五大概念、一切皆操作）、第 1 章（Type/Attribute 属于 context）。
>
> **编译原理切入**　本章从 IR 表示理论出发。第 0 章讲过 IR 表示三派（AST/三地址码/图IR）。MLIR 的选择是"一切皆 Operation"——这是对传统编译器"指令/函数/模块分层"的颠覆性统一。Dragon Book 第 8.3 节把指令定义为"运算符 + 操作数"，MLIR 把这个定义推到极致：一个加法是 Operation，一个函数是 Operation，连整个模块也是 Operation。这种统一不是炫技——它让遍历、重写、转换只需一套机制。本章要讲清这个统一的代价与收益，并立论 MLIR 可分析性的根基：**Operand 与 Attribute 的语义分野**。

---

## 2.1 把编译器 IR 抽象为图

Dragon Book 第 8.3 节把三地址码指令形式化为"运算符 + 一组操作数 + 一个结果"。这其实是在说：编译器 IR 可以抽象为一幅图 $G = (V, E)$——节点是指令，边是数据依赖（一个指令的结果是另一个指令的操作数）。

不同的 IR 设计，差别在于"节点有哪些种类"、"边有哪些种类"。MLIR 给出的答案极为简洁：

- **节点只有一种：Operation**。一切计算单元都是 Operation。
- **边有两类**：数据流边（Operand，下一章的 def-use chain）与结构边（Region 包含 Block，Block 包含 Operation）。

这是 MLIR 与传统编译器最显眼的区别。在 LLVM 里，`Instruction`、`Function`、`BasicBlock`、`Module` 是四个不同的类；在 MLIR 里，它们全是 Operation，差异由 trait/interface（2.5 节）表达。我们先看 Operation 这个统一抽象长什么样，再看它为什么能统一这么多东西。

## 2.2 Operation 类的源码骨架

[`Operation`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h) 是 MLIR 中最核心的类。它的声明（[Operation.h:84-88](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L84)）揭示了两件关键的事：

```cpp
// Operation.h:84-88
class alignas(8) Operation final
    : public llvm::ilist_node_with_parent<Operation, Block>,
      private llvm::TrailingObjects<Operation, detail::OperandStorage,
                                    detail::OpProperties, BlockOperand,
                                    Region, OpOperand>
```

**第一个关键点：继承 `ilist_node_with_parent<Operation, Block>`**。这说明 Operation 本身就是一个双向链表节点（通过 LLVM 的 intrusive list，即侵入式链表），嵌在 Block 内。所以"把一个 Operation 插入 Block"或"从 Block 移除 Operation"是 O(1) 的。这个链表的细节属于 Ch3（Block 内部的组织），这里只要记住：**Operation 自己就是链表节点，不需要外层包装**。

**第二个关键点：继承 `TrailingObjects<...>`**。这是 LLVM 的尾分配（trailing objects）框架，允许在堆上一次性分配 Operation 主体 + 一堆附属对象（OperandStorage、OpProperties、BlockOperand、Region、OpOperand），零额外 malloc。这是 MLIR Operation 最精巧的内存设计，它的完整布局留给 Ch7（与 `Operation::create` 同章）详解——因为尾分配本质上是 create 时的分配技巧，脱离 create 讲布局是空中楼阁。本章只关注 Operation 的**逻辑部件**。

> **为什么先讲逻辑、后讲布局？**　这是好教材"具体先于抽象"的原则。尾分配内存布局是高度抽象的内存工程，读者在没有看到"Operation 长什么样、怎么被创建出来"之前，理解不了"为什么要这样布局"。所以本书把尾分配推迟到 Ch7（讲 create 时），本章只讲 Operation 的字段语义。这种排序也保证了顺序友好——Ch7 只依赖 Ch6 之前的章节。

## 2.3 Operation 的字段：一个 Operation"装"了什么

抛开尾分配的内存技巧，一个 Operation 在逻辑上"装"这些东西（对应 [Operation.h:1035-1066](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h#L1035) 的私有字段）：

```cpp
private:
  Block *block = nullptr;                             // :1035  我属于哪个 Block（nullptr = 游离）
  Location location;                                  // :1039  源码位置（用于诊断/调试）
  mutable unsigned orderIndex = 0;                    // :1043  块内序号（遍历优化）
  const unsigned numResults;                          // :1045  结果数量
  const unsigned numSuccs;                            // :1046  后继数量（terminator 才有）
  const unsigned numRegions : 23;                     // :1047  region 数量（23 位域）
  bool hasOperandStorage : 1;                         // :1052  是否有操作数存储
  unsigned char propertiesStorageSize : 8;            // :1057  properties 存储大小
  OperationName name;                                 // :1063  操作名，如 "arith.addi"
  DictionaryAttr attrs;                               // :1066  属性字典
```

注意这些**都不是直接存储 operands/results/regions/successors**——它们全在前缀区或尾分配区中（详见 Ch7）。这里我们先关注语义：每个字段回答一个关于这个 Operation 的问题。

| 字段 | 回答的问题 | 逻辑角色 |
|---|---|---|
| `name` | "我是什么 op？" | 身份 |
| `location` | "我在源码哪一行？" | 追溯（诊断用） |
| `block` | "我挂在哪个块里？" | 结构归属 |
| `numResults` | "我产出几个值？" | 输出 |
| `numRegions` | "我包含几个 region？" | 嵌套 |
| `numSuccs` | "我有几个后继块？" | 控制流 |
| `attrs` | "我的编译期属性有哪些？" | 静态配置 |
| 操作数（尾分配） | "我用谁的结果作为输入？" | 数据流输入 |

把这八行记下来，你就掌握了 Operation 的全部逻辑语义。下面逐一展开其中最关键的几个。

### 2.3.1 name：Operation 的身份

`name` 是 [`OperationName`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h) 类型，形如 `"arith.addi"`、`"func.func"`、`"north_star.softmax"`——一个 dialect 前缀加一个 op 名。它回答"这个 Operation 是哪种运算"。

注意 `name` 不只是个字符串：`OperationName` 内部持有一个指向 `OperationName::Impl` 的指针（context 持有这些 Impl，见 Ch1 §1.3 的 `operations` 表），Impl 上记录了这个 op 是否已注册、有哪些 trait/interface、properties 元信息等。所以 `name` 是"身份 + 元信息"的合一。

### 2.3.2 location：编译期的源码追溯

每个 Operation 都带一个 `Location`（[Location.h](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Location.h)），记录它来自源码的哪个位置。这看起来不起眼，但对编译器的**诊断（diagnostics）**至关重要——报错时能精确指出"这个 op 来自源文件第几行"。MLIR 的 Location 本身也是一种 Attribute（存在 context 池里，见 Ch1），有 `FileLineColLoc`、`UnknownLoc`、`FusedLoc` 等变体。

### 2.3.3 attrs：属性字典

`attrs` 是一个 `DictionaryAttr`——即一个"名字 → Attribute"的映射，挂在 Operation 上。例如一个卷积 op 可能有 `attrs = {strides: [1,1], padding: "SAME"}`。注意 **Attribute 属于 context 池（Ch1），op 只持有一个指针**，所以 op 销毁不动 context 池。

### 2.3.4 操作数与结果：数据流的输入输出

操作数（Operand）与结果（Result）是数据流边，下一章（Ch4）专门讲它们的物理实现。这里只建立逻辑概念：

- **操作数**：这个 Operation 消费的 SSA 值（来自别的 op 的结果或 block 参数）。
- **结果**：这个 Operation 产出的 SSA 值（被别的 op 当操作数用）。

它们是 MLIR IR 的"图边"——operand 是入边（指向定义者），result 是出边（被使用者引用）。这条边的物理实现（use-list）极其精巧，是 Ch4 的主题。

## 2.4 Value：SSA 值的两面性

既然操作数与结果都是 SSA 值，我们必须看清楚"SSA 值"在 MLIR 里到底是什么。[`Value`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Value.h) 是一个**轻量级句柄类**（类似智能指针）：

```cpp
// Value.h:96-254（精简）
class Value {
  detail::ValueImpl *impl;   // :253 — 仅有一个指针成员
public:
  Type getType() const;
  Operation *getDefiningOp() const;
  use_iterator use_begin() const;
  bool use_empty() const;
  bool hasOneUse() const;
};
```

`Value` 只有一个指针成员 `impl`，指向 `ValueImpl`。`ValueImpl` 有两个子类型，对应 SSA 值的两种产生方式：

1. **`OpResult`**——Operation 的结果。一个加法 op 产出的 `%sum` 就是 `OpResult`。
2. **`BlockArgument`**——Block 的参数（详见 Ch3）。函数的 `%arg0` 就是 `BlockArgument`。

**关键洞察：Value 有"两面性"**，这是理解整个 use-def 机制的钥匙：

```text
Value:
  (1) 在「产生」端 → 它是一个结果（OpResult）或参数（BlockArgument）
      [定义在此，类型固定，被使用]

  (2) 在「消费」端 → 它是一个「use-list 的头部」
      [通过 impl->firstUse 链，追踪所有指向它的 OpOperand]
```

这两面性不是偶然——它直接对应 SSA 的两个方向：

- "定义-使用链"（def-use / DU chain）：从一个定义出发，找它的所有使用。Value 的 use-list 就是这条链的物理实现。
- "使用-定义链"（use-def / UD chain）：从一个使用出发，找它的定义。`OpOperand` 持有的 `value` 字段就是这条链。

第 4 章会逐字节打开这个机制。本章只要记住：**Value 既是"被定义的东西"，又是"被引用的中心"**——这两种身份由同一个对象承担，是 MLIR（也是 LLVM）实现高效 use-def 查询的基础。

> **本章不讲 use-list 的物理结构**。读者现在只需要知道 Value 持有"一个使用链的头"。这个链的节点结构（OpOperand 的四字段）、那个精巧的 `back` 指针、insertInto/removeFromCurrent 的逐步演示，全部留到 Ch4——因为只有在读者认识了 Value 的两面性（本章）之后，use-list 的存在动机才能自然显现。

## 2.5 Operand 与 Attribute 的语义分野：MLIR 可分析性的根基

现在到了本章的核心论断。一个 Operation 上挂着很多东西——操作数、结果、属性、region。其中最容易混淆的是 **Operand 与 Attribute**：二者都"挂在 Operation 上"，但语义截然不同。这个区分是 MLIR 能做数据流分析、死代码消除、常量折叠的根基。

### 2.5.1 定义与准则

先给定义：

- **Operand（操作数）**：运行时数据流的一边。它的值由别的 Operation 计算得出（或来自 BlockArgument），**参与数据流图**。
- **Attribute（属性）**：编译期固定的配置参数。它**不随执行变化、不参与数据流图**。

判断准则一句话：**"这个值是否由另一条计算链路产生？是否随程序执行而变化？"** 是 → operand；否（编译期固定的元数据/配置）→ attribute。

举个例子。NorthStar 教程里的 `SoftmaxOp`：

```mlir
%1 = "north_star.softmax"(%input) <{axis = 1 : i64}> : (!ns.tensor<...>) -> !ns.tensor<...>
```

- `%input` 是 operand——它是别的 op 算出来的张量，随执行变化，参与数据流。
- `axis = 1` 是 attribute——它在编译期就定死了，不参与数据流。

### 2.5.2 为什么这个分野是 MLIR 的根本设计

把 operand 与 attribute 分开，是 MLIR 的有意设计，支撑了四件编译器必须做的事：

1. **依赖追踪与死代码消除**：Pass 做死代码消除时，沿 operand 边追踪活跃性——"这个 op 的结果被谁用了"。Attribute 不参与这条追踪（`strides` 不是数据依赖）。如果 operand 和 attribute 混在一起，DCE 就分不清"这个输入算完了才能跑"和"这是个常量配置"。
2. **常量折叠**：attribute 是 compile-time 已知值，可直接参与折叠计算（`axis = 1` 可以直接用）。如果它是 operand，编译器不知道它的值，没法折叠。
3. **图等价判断**：两个 `SoftmaxOp` 若 operand 相同且 `axis` 相同，就是等价的，可以 CSE（公共子表达式消除）。这个判断基于 operand + attribute 的精确语义。
4. **文本表示清晰**：MLIR 文本里 operand 在 `(...)`，inherent attribute 在 `<{...}>` 或 `{...}`，一眼可辨。

> **编译原理浸润点**　这个"运算输入 vs 编译期配置"的区分，对应 Dragon Book 第 8.5 节讨论的"常量折叠所需的静态信息"。MLIR 把它从语义约定提升为 IR 的结构化区分——operand 与 attribute 是 Operation 上**两类不同的部件**，不是同一类东西的不同写法。这是 MLIR 比"纯文本 IR"（如早期的汇编）更适合做自动优化的根本原因。

### 2.5.3 Properties：inherent attribute 的强类型升级

`attrs` 字段是"可丢弃属性"（discardable attributes）的传统容器，但现代 MLIR 还有一个更精细的机制——**Properties**。Properties 是 MLIR 较新引入（约 2023 年）的机制，用于把"inherent attribute"（op 固有属性，如 `SoftmaxOp` 的 `axis`）从字符串字典提升为强类型 struct。

此前 inherent attribute 全部存在一个 `DictionaryAttr`（名字→Attribute 映射），有三个缺陷：访问需字符串查找（慢）、序列化/哈希/比较需逐项重建、无类型安全。`Properties` 把已知 inherent attribute 提升为强类型 struct：

```cpp
// NorthStarOps.h.inc:1432-1451（生成的 Properties）
struct Properties {
  using axisTy = ::mlir::IntegerAttr;
  axisTy axis;   // 字段类型是 IntegerAttr，不是 int64_t
  bool operator==(const Properties &rhs) const {
    return rhs.axis == this->axis && true;
  }
};
```

Properties 内部存储的仍是 Attribute（`IntegerAttr`），只是用强类型 struct 组织。访问变成直接成员 `prop.axis`（O(1)、编译期类型），而不是 `dict.get("axis")`（字符串查找、运行期类型）。

Properties 的尾分配细节属于 Ch7（它存在 Operation 尾分配区的 `OpProperties` blob 里）。本章只要记住：**MLIR 用 Properties 机制，把编译期配置做成强类型、O(1) 访问**——这是对 operand/attribute 分野的进一步工程强化。

## 2.6 API 速查：怎么读 Operation 的各个部件

把上面讲的字段汇总成一个 API 速查表（[Operation.h](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Operation.h) 各处）：

| 想获取 | API | 返回类型 | 存储位置 |
|---|---|---|---|
| operands | `getOperands()` | `OperandRange` | 尾分配的 `OpOperand[]` |
| results | `getResults()` | `ResultRange` | 前缀区（反向存储，详见 Ch7） |
| regions | `getRegions()` | `MutableArrayRef<Region>` | 尾分配的 `Region[]` |
| successors | `getSuccessors()` | `SuccessorRange` | 尾分配的 `BlockOperand[]` |
| attributes | `getAttrDictionary()` | `DictionaryAttr` | 成员 `attrs` |
| properties | `getPropertiesStorage()` | `OpaqueProperties` | 尾分配的 `OpProperties` blob |
| uses（我的结果被谁用） | `use_begin()/use_end()` | `use_iterator` | 委托给 `getResults().use_begin()` |
| users（使用者） | `user_begin()/user_end()` | `user_iterator` | 从 uses 投射（详见 Ch4/Ch5） |

这个表回答"我想读 Operation 的某类部件，调哪个方法"。读 IR 的代码几乎都是查这个表。其中"uses/users"的物理实现涉及 use-list，留给 Ch4。

## 2.7 "一切皆 Operation"的代价与收益

最后回扣本章开头的立论。MLIR 把指令、函数、模块统一为 Operation，这个设计：

**收益**：
- **一套机制处理所有层次**。遍历（walk）、重写（rewrite）、转换（conversion）只需写一遍，对 module、func、arith 都适用。Ch5（遍历）、Ch9（重写）会反复体现这点。
- **trait/interface 表达差异**。Operation 之间的不同（这个是函数吗？这个能终止 block 吗？）用 trait 和 interface 表达，而不是子类化。例如 `func::FuncOp` 有 `FunctionOpInterface`，`arith::AddIOp` 没有——但它们都是 Operation。
- **可扩展性天然**。定义一个新 op 就是定义一个新的 Operation 子类（通过 ODS，见 Ch7），不需要改框架。

**代价**：
- **类型区分靠 trait/interface**，不如强类型子类直观。读代码时要频繁 `dyn_cast<FuncOp>(op)` 判断"这个 Operation 到底是不是函数"。
- **字段统一，但不同 op 用到的字段子集不同**。一个加法 op 没有 region，但 Operation 的字段框架仍为 region 留了位置（尾分配时按需分配，详见 Ch7）。

这个代价是 MLIR 为了"统一 + 可扩展"付出的。Dragon Book 里的 IR（三地址码）是面向单一层次的，不需要这种统一；MLIR 要跨层次，统一的 Operation 抽象就成了必然选择。

---

## 编译原理浸润点回顾

1. **IR 表示理论**（三地址码 + 图 IR 融合）：本章主题。"一切皆 Operation"是 MLIR 对 Dragon Book 第 8.3 节指令定义的极致统一。
2. **SSA 值的两面性**：本章引入 Value 的两面性（定义端容器 / 消费端链头），Ch4 会讲其物理实现。
3. **operand vs attribute 的语义分野**：本章核心论断。对应 Dragon Book 第 8.5 节的"常量折叠所需静态信息"，但 MLIR 把它结构化为 IR 部件。这个分野是后续 DCE（Ch10）、fold（Ch11）的根基。
4. **可扩展 IR 哲学**：trait/interface 表达差异，让统一抽象不牺牲灵活性。

---

## 本章关键结论

1. **MLIR 的 IR 是一幅图，节点只有 Operation，边有两类**（数据流边 = operand，结构边 = region/block 嵌套）。这是 MLIR 对 IR 表示理论的回答。
2. **Operation 是统一抽象**。指令、函数、模块都是 Operation，差异由 trait/interface 表达。这让遍历/重写只需一套机制。
3. **Operation 的字段回答八个问题**（name/location/block/numResults/numRegions/numSuccs/attrs/operands）。其中 operands/results/regions 是数据结构与控制流的核心，attrs 是编译期配置。
4. **Value 有两面性**：既是"被定义的东西"（OpResult/BlockArgument），又是"被引用的中心"（use-list 头部）。这两种身份由同一对象承担，是 use-def 查询的基础。use-list 物理实现留 Ch4。
5. **Operand 与 Attribute 的语义分野是 MLIR 可分析性的根基**。前者是数据流边（运行时变），后者是编译期配置（静态）。这个分野支撑 DCE、常量折叠、CSE。Properties 机制把 inherent attribute 做成强类型 struct。

---

## 下一章预告

本章讲了 Operation 这个"统一节点"。但 Operation 不是孤立存在的——它要么挂在一个 Block 里，要么自己包含若干个 Region。第 3 章打开 Block 与 Region 的源码，回答：Block 为什么就是 Dragon Book 定义的基本块？MLIR 用什么取代了传统 SSA 的 φ 节点？Region 又如何表达结构化作用域？这里有一个对 SSA 的工程简化——**Block argument 取代 φ**——它是 MLIR 区别于 LLVM 的一个重要设计选择。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-IR-Node组织与遍历插入删除教程.md` §0-1（Operation 类的继承与字段、API 速查表、Value 两面性）——**保留并重组为编译器视角叙事**
- `docs/operator-use-def的形成.md` §2.2-2.3（Operand vs Attribute 分野、Properties 机制）
- 编译原理铺垫（IR 图抽象、operand/attribute 分野对应 Dragon Book）为本书新增

## 参考文献

- **[Aho 2006]** Dragon Book，第 8.3 节（三地址码指令定义）、第 8.5 节（常量折叠与静态信息）。
- **[Lattner 2020]** Lattner et al. "MLIR"，"一切皆 Operation" 与 trait/interface 机制的论述。
