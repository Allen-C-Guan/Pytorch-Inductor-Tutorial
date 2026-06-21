# MLIR 方言类型的解析机制：以 NorthStar `NSTensorType` 为例

> **摘要**　MLIR 中，IR 实体（Type／Attribute／Operation）具有内存对象与 assembly 文本两种等价存在形态，二者由互逆的 `print`／`parse` 过程连接。本文以 NorthStar 方言的 `NSTensorType` 解析 `!ns.ns_tensor<2x3xf32, 0>` 为实例，形式化地刻画 `parse` 的输入／输出契约，逐 token 重建其递归下降解析过程，并从编译器分层的角度归纳其语义本质：`parse` 是一个集词法、语法识别、语义校验、IR 构造于一体的反序列化器，为 `print` 的精确逆运算。文末讨论其副作用模型与一处实现边界条件。

---

## 1. 引言

MLIR 的 IR 实体在系统中有两种等价的存在形态：

- **内存对象**（runtime object）：唯一化于 `MLIRContext` 管理的 arena 中，是 IR 的本体；
- **assembly 文本**（printable form）：线性字符串，是对象的序列化形态。

二者由一对互逆过程连接：`print` 将对象序列化为文本，`parse` 将文本反序列化为对象。本文以第 4 章（`4-define_attribute`）的 `NSTensorType` 为标本，解析输入 `!ns.ns_tensor<2x3xf32, 0>`，剖析 `parse` 过程的接口契约、内部机制与语义本质。

涉及代码：

- 手写解析器：[`NorthStarTypes.cpp:51-69`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L51-L69)
- TableGen 生成的分派器：[`NorthStarTypes.cpp.inc`](../MLIR-Tutorial/build/4-define_attribute/include/Dialect/NorthStar/IR/NorthStarTypes.cpp.inc)

---

## 2. 接口契约

[`NSTensorType::parse`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L51) 的签名为：

```cpp
Type NSTensorType::parse(AsmParser &parser);
```

它以一个 `AsmParser` 对象为输入，返回一个 `Type` 值类。二者均非朴素数据类型，需展开其内部结构方能厘清契约。

### 2.1 输入：`AsmParser` 的内部结构

`AsmParser` 并非裸字符串，而是一个封装了源文本、词法分析器与解析游标的对象。其内核结构如下（以输入进入本方言解析器时的状态为准）：

```
AsmParser
├─ 源码 buffer : "ns_tensor<2x3xf32, 0>"     // 实际被加工的原材料
│               （框架已在更外层剥离 "!ns." 方言前缀）
├─ Lexer      : 将字符流切分为 token 序列
│    [kw:ns_tensor] [<] [2] [x] [3] [x] [type:f32] [,] [0] [>]
├─ 游标        : 进入 generatedTypeParser 时指向 ns_tensor；
│               经 KeywordSwitch 消费后、进入 NSTensorType::parse 时指向 '<'
└─ context    : MLIRContext*，构造对象时所需
```

即：入口形式上是一个解析器对象，实质被加工的原材料为其内部持有的字符串。

### 2.2 输出：`Type` 值类与唯一化 storage

返回类型 `Type` 实为 `NSTensorType`（向上隐式转换）。`Type` 是**值类**——其大小等于一个指针，本身不承载数据，仅包装一个指向 `TypeStorage` 的指针。本例产物的展开结构如下：

```
NSTensorType  (= Type，值类，sizeof = 一个指针)
└─ impl : TypeStorage* ──► arena 中的唯一化 storage
                            │
                            ▼
          detail::NSTensorTypeStorage      // 见 NorthStarTypes.cpp.inc:43-70
          ├─ shape       : ArrayRef<int64_t> = [2, 3]
          ├─ elementType : Type             = f32 (Float32Type)
          ├─ device_id   : int64_t          = 0
          │
          └─〔继承自 TypeStorage〕
             ├─ TypeID kind   // 标识具体子类，为 isa / dyn_cast 的依据
             └─ 由 StorageUniquer 全局唯一化：
                相同 (shape, elementType, device_id) 三元组全局唯一
```

即：出口形式上是一个值类，实质是指向 arena 中唯一化 storage 的指针，storage 内承载解析所得的三个字段及类型标识。

综上，`parse` 的接口契约可表述为：**以一个封装了字符串的解析器对象为输入，产出一个指向 arena 中唯一化 storage 的值类指针。**

---

## 3. 解析过程：递归下降的游标推进

完整的类型解析由**三段接力**构成，每段消费文本的一段并向下交棒：

1. **框架层**：MLIR `Parser` 识别 `!` 与方言名 `ns`，剥离前缀 `!ns.`，路由至 [`NorthStarDialect::parseType`](../MLIR-Tutorial/build/4-define_attribute/include/Dialect/NorthStar/IR/NorthStarTypes.cpp.inc#L107)；
2. **分派层**：[`generatedTypeParser`](../MLIR-Tutorial/build/4-define_attribute/include/Dialect/NorthStar/IR/NorthStarTypes.cpp.inc#L19-L29) 以 `KeywordSwitch` 读取 mnemonic `ns_tensor`，匹配 `.Case("ns_tensor")`，调用 `NSTensorType::parse`；
3. **体解析层**：[`NSTensorType::parse`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L51) 消费 `<...>` 体并构造对象。

进入第 3 段时，mnemonic 已被消费，游标位于 `<`。此后 `parse` 以一串 `parseXxx` 调用逐 token 推进游标，每步对应产生式右部一个符号的识别：

| 步 | 游标后剩余输入 | 调用 | 消费 token | 状态 / 中间结果 |
|---|---|---|---|---|
| 0 | `!ns.ns_tensor<2x3xf32, 0>` | MLIR 框架 `Parser` | `!ns.` | 路由至方言 `parseType` |
| 1 | `ns_tensor<2x3xf32, 0>` | 进入 `generatedTypeParser` | — | `KeywordSwitch` 启动 |
| 2 | `ns_tensor<2x3xf32, 0>` | [`KeywordSwitch` 读 keyword](../MLIR-Tutorial/build/4-define_attribute/include/Dialect/NorthStar/IR/NorthStarTypes.cpp.inc#L20-L24) | `ns_tensor` | 命中 `.Case`，调用 `NSTensorType::parse` |
| 3 | `<2x3xf32, 0>` | 进入 `NSTensorType::parse` | — | 体解析开始 |
| 4 | `<2x3xf32, 0>` | [`parseLess()`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L52) | `<` | — |
| 5 | `2x3xf32, 0>` | [`parseDimensionList(allowDynamic=1, withTrailingX=1)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L55-L56) | `2x3x` | dimensions = [2, 3] |
| 6 | `f32, 0>` | [`parseType(elementType)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L61) | `f32` | elementType = Float32Type（递归 builtin parser） |
| 7 | `, 0>` | [`parseComma()`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L63) | `,` | — |
| 8 | `0>` | [`parseInteger(device_id)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L65) | `0` | device_id = 0 |
| 9 | `>` | [`parser.getChecked<NSTensorType>(...)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L67-L68) | （不消费文本） | verify 通过 → StorageUniquer intern → 得到 NSTensorType |
| 10 | `>` | 返回 `generatedTypeParser` | — | value = 对象；返回 `OptionalParseResult(success)` |
| 11 | `>` | 返回 `parseType` | — | `has_value()` 为真 → 返回该对象 |

整个函数体即一条产生式 `ns_tensor := '<' dimlist type ',' int '>'` 的手写递归下降展开；`AsmParser` 为一带游标的递归下降解析器。

---

## 4. 语义本质：`parse` 作为反序列化器

### 4.1 定义

> **`parse` 是 IR 对象的反序列化器**：它以 IR 的可打印序列化形态（assembly 文本）为输入，按递归下降规则逐 token 消费、确认结构合法，同时将识别出的片段重新物化（reify）为 arena 中唯一化的 IR 对象并施加语义校验。它是 `print` 的逆运算，是"文本表示 → 对象本体"的正向桥梁。

### 4.2 语义维度

由上述实例可归纳 `parse` 的六个语义维度，每条对应一项编译器基础概念：

**（1）反序列化 / 逆运算。** IR 具内存对象与文本两种形态；`print` 序列化，`parse` 反序列化，二者互逆。此与 JSON、protobuf、LLVM IR（`.ll`）的解析同构。

**（2）递归下降语法分析。** `parse` 的函数体是产生式的手写展开，每个 `parseXxx` 消费一个文法符号，调用顺序即产生式右部。

**（3）语法识别与语义构造合一（无独立 AST）。** 传统编译器中 parse 仅产出 AST，语义分析为独立的后续阶段。MLIR 的 `parse` 跳过 AST 中间层：边识别结构边经 `getChecked` 构造 IR 对象，并将语义校验（[verify](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L39-L49)）织入构造过程。其依据在于：assembly 文本本即 IR 的忠实序列化，不存在"高级语言 → IR"的语义鸿沟。

**（4）上下文相关、有副作用。** `parse` 携带 `MLIRContext`，构造时访问其 `StorageUniquer` 执行 intern——首次出现的三元组在 arena 中分配新 storage。故 `parse` 非纯函数，它读写 context 的全局唯一化表。

**（5）可失败、空值传播。** 任一 token 失配即返回空 `Type()`，沿调用链上传，与 `generatedTypeParser` 的 `OptionalParseResult` 三态契约对接，构成统一的错误协议。

**（6）"分派认领 + 体解析"两段式。** 完整解析由 `KeywordSwitch` 认领 mnemonic 与手写 `parse` 消费体两部分组成。此为多方言可扩展类型空间所必然——单个 `parse` 仅处理其 mnemonic 之后的部分。

### 4.3 与传统编译器分层的对照

| 编译器阶段 | 传统编译器 | 本文实例中的对应 |
|---|---|---|
| Lexing（词法） | lexer 切分 token | `AsmParser` 内部 Lexer（隐式） |
| Parsing（语法） | token → AST | `parseLess`／`parseDimensionList`／`parseType` 等逐 token 消费 |
| Semantic analysis（语义） | 类型检查、符号决议 | 合并构造：`getChecked` 触发的 `verify` |
| IR 构造 | AST → IR（独立一趟） | `parse` 直接产出 IR 对象（无独立 AST） |

**关键洞察**：MLIR（同 LLVM IR）将词法、语法分析、语义校验、IR 构造四步压缩进单一 `parse` 函数。其可行性根植于输入的性质——被解析者并非源语言，而是 IR 自身的文本序列化，故无需经 AST 中间表示再行 lowering。

---

## 5. 副作用与边界条件

### 5.1 实例触发的副作用

针对输入 `!ns.ns_tensor<2x3xf32, 0>`，本次解析实际产生的副作用如下：

- **游标推进**：自输入首字符推进至 `>` 前，共消费 `ns_tensor<2x3xf32, 0>`；
- **arena 分配**：若三元组 `(shape=[2,3], f32, 0)` 首次出现，`StorageUniquer` 分配并登记新的 `NSTensorTypeStorage`；否则复用既有实例，无分配；
- **诊断**：无——`verify` 全部通过，不发射错误。

### 5.2 边界条件：终止符 `>` 的未消费残留

正常解析路径下，终止符 `>` 未被任何 `parseXxx` 显式消费：[`L65-66`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L65-L66) 的 `parseGreater()` 仅在 `parseInteger` 失败的容错分支执行，而 `0` 解析成功，故该调用被跳过。然而 [`print`（L83）](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L83) 恒输出 `>`。

由此，`parse` 与 `print` 在终止符上存在**不对称**：按字面代码，合法输入 `<2x3xf32, 0>` 解析后将残留 `>`，可能阻塞外层操作的解析。此现象可能源于教程实现的一处疏漏（常规实现应在 `parseInteger` 之后补以 `parseGreater()`），亦可能由未在当前文件体现的机制兜底。

此为第 10 章（`NS-opt`）与第 11 章（lit／FileCheck）阶段应予实证检验的首要疑点：构造含 `!ns.ns_tensor<2x3xf32, 0>` 的 IR，验证 round-trip 的完整性；若解析报错，根因大概率位于此处。

---

## 6. 结论

本文以 `NSTensorType` 解析 `!ns.ns_tensor<2x3xf32, 0>` 为实例，刻画了 MLIR 方言类型解析的完整图景。接口层面，`parse` 以封装字符串的 `AsmParser` 为输入、以指向 arena 唯一化 storage 的值类为输出；过程层面，它经框架、分派、体解析三段接力，以递归下降方式逐 token 推进游标；语义层面，它是集词法、语法、语义、构造于一体的反序列化器，为 `print` 的精确逆运算。这一机制揭示了 MLIR 作为 IR 优先编译器的一项本质特征：**当输入即 IR 的序列化形态时，传统编译流水线中相互分离的词法、语法、语义与构造阶段，被自然压缩为单一的 `parse` 过程。**
