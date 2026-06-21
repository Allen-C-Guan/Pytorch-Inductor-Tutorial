# 第 12 章 专题：type 的 parse/print 往返

> **本章位置**　第七部分（专题与综合）的第一章。前十一章讲的是 IR 在内存里怎么表示、构造、重写。本章换个视角——IR 的**文本形态**：怎么把内存里的 IR 对象序列化成文本（print），又怎么把文本反序列化回内存对象（parse）。这是一个编译器前端的经典话题。
>
> **前置依赖**　第 1 章（Type 属于 context、唯一化）、第 2 章（Type 是 Operation 的部件）。
>
> **编译原理切入**　本章从**词法/语法/语义分析分层**立论——Dragon Book 第 2-4 章的标准流水线。MLIR 的 parse 把这四步压缩进单一函数，因为被解析者是 **IR 的序列化形态**而非源语言——不存在"高级语言 → IR"的语义鸿沟，所以无需独立 AST。本章以 NorthStar 的 `NSTensorType` 解析 `!ns.ns_tensor<2x3xf32, 0>` 为标本，逐 token 重建递归下降解析过程，并归纳 parse 作为"反序列化器"的语义本质。

---

## 12.1 序列化与反序列化：IR 的两种存在形态

MLIR 的 IR 实体在系统中有两种等价的存在形态：

- **内存对象**（runtime object）：唯一化于 `MLIRContext` 管理的 arena 中（Ch1），是 IR 的本体。
- **assembly 文本**（printable form）：线性字符串，是对象的序列化形态。

二者由一对互逆过程连接：`print` 将对象序列化为文本，`parse` 将文本反序列化为对象。这与 JSON、protobuf、LLVM IR（`.ll`）的解析同构——都是序列化/反序列式的往返（round-trip）。

本章以 NorthStar 方言的 `NSTensorType` 解析 `!ns.ns_tensor<2x3xf32, 0>` 为标本，剖析 parse 过程。

## 12.2 接口契约

[`NSTensorType::parse`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L51) 的签名为：

```cpp
Type NSTensorType::parse(AsmParser &parser);
```

它以一个 `AsmParser` 对象为输入，返回一个 `Type` 值类。二者均非朴素数据类型，需展开其内部结构方能厘清契约。

### 12.2.1 输入：AsmParser 的内部结构

`AsmParser` 并非裸字符串，而是一个封装了源文本、词法分析器与解析游标的对象。其内核结构如下（以输入进入本方言解析器时的状态为准）：

```text
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

### 12.2.2 输出：Type 值类与唯一化 storage

返回类型 `Type` 实为 `NSTensorType`（向上隐式转换）。`Type` 是**值类**——其大小等于一个指针，本身不承载数据，仅包装一个指向 `TypeStorage` 的指针（Ch1）。本例产物的展开结构：

```text
NSTensorType  (= Type，值类，sizeof = 一个指针)
└─ impl : TypeStorage* ──► arena 中的唯一化 storage
                            │
                            ▼
          detail::NSTensorTypeStorage
          ├─ shape       : ArrayRef<int64_t> = [2, 3]
          ├─ elementType : Type             = f32 (Float32Type)
          ├─ device_id   : int64_t          = 0
          │
          └─〔继承自 TypeStorage〕
             ├─ TypeID kind   // 标识具体子类，为 isa / dyn_cast 的依据
             └─ 由 StorageUniquer 全局唯一化：
                相同 (shape, elementType, device_id) 三元组全局唯一
```

即：出口形式上是一个值类，实质是指向 arena 中唯一化 storage 的指针（Ch1 的唯一化机制在 parse 输出端的体现）。

综上，parse 的接口契约可表述为：**以一个封装了字符串的解析器对象为输入，产出一个指向 arena 中唯一化 storage 的值类指针。**

## 12.3 解析过程：递归下降的游标推进

完整的类型解析由**三段接力**构成，每段消费文本的一段并向下交棒：

1. **框架层**：MLIR `Parser` 识别 `!` 与方言名 `ns`，剥离前缀 `!ns.`，路由至 `NorthStarDialect::parseType`。
2. **分派层**：`generatedTypeParser` 以 `KeywordSwitch` 读取 mnemonic `ns_tensor`，匹配 `.Case("ns_tensor")`，调用 `NSTensorType::parse`。
3. **体解析层**：[`NSTensorType::parse`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L51) 消费 `<...>` 体并构造对象。

进入第 3 段时，mnemonic 已被消费，游标位于 `<`。此后 parse 以一串 `parseXxx` 调用逐 token 推进游标，每步对应产生式右部一个符号的识别：

| 步 | 游标后剩余输入 | 调用 | 消费 token | 状态 / 中间结果 |
|---|---|---|---|---|
| 0 | `!ns.ns_tensor<2x3xf32, 0>` | MLIR 框架 Parser | `!ns.` | 路由至方言 parseType |
| 1 | `ns_tensor<2x3xf32, 0>` | 进入 generatedTypeParser | — | KeywordSwitch 启动 |
| 2 | `ns_tensor<2x3xf32, 0>` | KeywordSwitch 读 keyword | `ns_tensor` | 命中 .Case，调用 NSTensorType::parse |
| 3 | `<2x3xf32, 0>` | 进入 NSTensorType::parse | — | 体解析开始 |
| 4 | `<2x3xf32, 0>` | [`parseLess()`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L52) | `<` | — |
| 5 | `2x3xf32, 0>` | [`parseDimensionList(allowDynamic=1, withTrailingX=1)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L55) | `2x3x` | dimensions = [2, 3] |
| 6 | `f32, 0>` | [`parseType(elementType)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L61) | `f32` | elementType = Float32Type（递归 builtin parser） |
| 7 | `, 0>` | [`parseComma()`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L63) | `,` | — |
| 8 | `0>` | [`parseInteger(device_id)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L65) | `0` | device_id = 0 |
| 9 | `>` | [`parser.getChecked<NSTensorType>(...)`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L67) | （不消费文本） | verify 通过 → StorageUniquer intern → 得到 NSTensorType |
| 10 | `>` | 返回 generatedTypeParser | — | value = 对象；返回 OptionalParseResult(success) |
| 11 | `>` | 返回 parseType | — | has_value() 为真 → 返回该对象 |

整个函数体即一条产生式 `ns_tensor := '<' dimlist type ',' int '>'` 的手写递归下降展开；`AsmParser` 为一带游标的递归下降解析器。

> **编译原理浸润点：递归下降解析**　Dragon Book 第 4.4 节描述递归下降解析（recursive descent parsing）——为每个非终结符写一个过程，过程体按产生式右部逐个调用其他过程或匹配 token。MLIR 的 `parse` 函数体就是这个模式：`parseLess`/`parseDimensionList`/`parseType`/`parseComma`/`parseInteger` 对应产生式右部的各个符号。这是手写解析器的标准写法，与 Dragon Book 描述的 LL(1) 递归下降同构。

## 12.4 parse 的语义本质：反序列化器

### 12.4.1 定义

> **`parse` 是 IR 对象的反序列化器**：它以 IR 的可打印序列化形态（assembly 文本）为输入，按递归下降规则逐 token 消费、确认结构合法，同时将识别出的片段重新物化（reify）为 arena 中唯一化的 IR 对象并施加语义校验。它是 `print` 的逆运算，是"文本表示 → 对象本体"的正向桥梁。

### 12.4.2 语义维度

由上述实例可归纳 parse 的六个语义维度，每条对应一项编译器基础概念：

**（1）反序列化 / 逆运算。** IR 具内存对象与文本两种形态；`print` 序列化，`parse` 反序列化，二者互逆。此与 JSON、protobuf、LLVM IR（`.ll`）的解析同构。

**（2）递归下降语法分析。** parse 的函数体是产生式的手写展开，每个 parseXxx 消费一个文法符号，调用顺序即产生式右部。

**（3）语法识别与语义构造合一（无独立 AST）。** 传统编译器中 parse 仅产出 AST，语义分析为独立的后续阶段。MLIR 的 parse 跳过 AST 中间层：边识别结构边经 `getChecked` 构造 IR 对象，并将语义校验（verify）织入构造过程。其依据在于：assembly 文本本即 IR 的忠实序列化，不存在"高级语言 → IR"的语义鸿沟。

**（4）上下文相关、有副作用。** parse 携带 `MLIRContext`，构造时访问其 `StorageUniquer` 执行 intern——首次出现的三元组在 arena 中分配新 storage。故 parse 非纯函数，它读写 context 的全局唯一化表。

**（5）可失败、空值传播。** 任一 token 失配即返回空 `Type()`，沿调用链上传，与 `generatedTypeParser` 的 `OptionalParseResult` 三态契约对接，构成统一的错误协议。

**（6）"分派认领 + 体解析"两段式。** 完整解析由 `KeywordSwitch` 认领 mnemonic 与手写 parse 消费体两部分组成。此为多方言可扩展类型空间所必然——单个 parse 仅处理其 mnemonic 之后的部分。

### 12.4.3 与传统编译器分层的对照

| 编译器阶段 | 传统编译器 | 本文实例中的对应 |
|---|---|---|
| Lexing（词法） | lexer 切分 token | AsmParser 内部 Lexer（隐式） |
| Parsing（语法） | token → AST | parseLess／parseDimensionList／parseType 等逐 token 消费 |
| Semantic analysis（语义） | 类型检查、符号决议 | 合并构造：getChecked 触发的 verify |
| IR 构造 | AST → IR（独立一趟） | parse 直接产出 IR 对象（无独立 AST） |

**关键洞察**：MLIR（同 LLVM IR）将词法、语法分析、语义校验、IR 构造四步压缩进单一 parse 函数。其可行性根植于输入的性质——被解析者并非源语言，而是 IR 自身的文本序列化，故无需经 AST 中间表示再行 lowering。

> **编译原理浸润点：为什么 MLIR 能跳过 AST？**　Dragon Book 第 2-4 章描述的前端流水线是为"源语言 → IR"设计的——源语言（如 C++）有复杂的语法语义，需要 AST 中间层做类型检查、重载决议、模板实例化等。但 MLIR 的 parse 输入是 **IR 的序列化形态**——它本身就是 IR，只是用文本表示。把文本 IR 翻成内存 IR 不需要语义鸿沟的跨越（没有"源语言 vs IR"的概念差异），所以可以边解析边构造。这是 MLIR（和 LLVM IR）对传统前端流水线的简化——把四步压缩成一步。理解这一点，就理解了为什么 MLIR 的 parse 看起来既像 parser 又像 constructor。

## 12.5 副作用与边界条件

### 12.5.1 实例触发的副作用

针对输入 `!ns.ns_tensor<2x3xf32, 0>`，本次解析实际产生的副作用：

- **游标推进**：自输入首字符推进至 `>` 前，共消费 `ns_tensor<2x3xf32, 0>`。
- **arena 分配**：若三元组 `(shape=[2,3], f32, 0)` 首次出现，StorageUniquer 分配并登记新的 `NSTensorTypeStorage`；否则复用既有实例，无分配。
- **诊断**：无——verify 全部通过，不发射错误。

### 12.5.2 边界条件：终止符 `>` 的未消费残留

正常解析路径下，终止符 `>` 未被任何 parseXxx 显式消费：[`L65-66`](../MLIR-Tutorial/4-define_attribute/src/Dialect/NorthStar/IR/NorthStarTypes.cpp#L65) 的 `parseGreater()` 仅在 `parseInteger` 失败的容错分支执行，而 `0` 解析成功，故该调用被跳过。然而 print 恒输出 `>`。

由此，parse 与 print 在终止符上存在**不对称**：按字面代码，合法输入 `<2x3xf32, 0>` 解析后将残留 `>`，可能阻塞外层操作的解析。此现象可能源于教程实现的一处疏漏（常规实现应在 parseInteger 之后补以 parseGreater()），亦可能由未在当前文件体现的机制兜底。

> 此为 round-trip 测试应予实证检验的首要疑点：构造含 `!ns.ns_tensor<2x3xf32, 0>` 的 IR，验证 round-trip 的完整性；若解析报错，根因大概率位于此处。这正是 MLIR lit/FileCheck 测试框架（NorthStar CH-11）的价值——它能在 CI 里自动验证 parse/print 的互逆性。

## 12.6 parse 与全书其他章节的呼应

parse 看似是个独立专题，但它与全书的多个机制呼应：

- **Type 唯一化（Ch1）**：parse 输出的 Type 是 arena 中唯一化的 storage 指针。parse 调 `getChecked` 触发 StorageUniquer intern。
- **Type 作为 Operation 部件（Ch2）**：parse 出来的 Type 最终挂在 op 的 result/operand 上。
- **递归下降（本章）**：parse 函数体是产生式的手写展开。
- **词法/语法/语义分层（本章）**：MLIR 把四步压缩进单一 parse，因输入是 IR 序列化。

---

## 编译原理浸润点回顾

1. **词法/语法/语义分析分层**：本章主题（标本）。Dragon Book 第 2-4 章。MLIR 把四步压缩进单一 parse。
2. **递归下降解析**：parse 函数体是产生式手写展开。Dragon Book 第 4.4 节。
3. **AST vs IR**：MLIR 的 parse 跳过 AST，因输入是 IR 序列化。与 Dragon Book "源语言 → AST → IR" 形成对比。
4. **序列化/反序列化**：parse/print 互逆，同构于 JSON/protobuf/.ll 解析。

---

## 本章关键结论

1. **IR 有两种等价形态**：内存对象（arena 唯一化）+ assembly 文本。parse/print 互逆连接。
2. **parse 的接口契约**：输入 AsmParser（封装字符串+游标+context），输出 Type（指向 arena 唯一化 storage 的值类指针）。
3. **parse 是三段接力**：框架层（剥离方言前缀）→ 分派层（KeywordSwitch 认领 mnemonic）→ 体解析层（逐 token 消费体）。
4. **parse 是反序列化器**：六个语义维度——反序列化、递归下降、语法语义合一（无独立 AST）、上下文相关有副作用、可失败空值传播、分派认领+体解析两段式。
5. **MLIR 跳过 AST 的根据**：被解析者是 IR 序列化，无"源语言→IR"语义鸿沟。这简化了 Dragon Book 描述的四步流水线。
6. **终止符不对称疑点**：教程实现的边界条件，应通过 lit round-trip 测试验证。

---

## 下一章预告

本书的最后一章。第 13 章是全书压轴——**BufferCastOpFold 闭环**。它把前十一章的所有机制串成一个真实的优化 pass：walk（遍历，Ch5）→ match（模式匹配，Ch9）→ RAUW（边搬迁，Ch10）→ erase（节点删除，Ch10）。一个冗余 cast 折叠如何把全书的机制串成"发现→变换→清理"的标准优化 pass 结构。

---

## 原文对照

本章素材主要来自：
- `docs/ns-tensor-parse-walkthrough.md` 全文（接口契约、三段接力、递归下降过程、反序列化器语义、六个维度、边界条件）——**完整迁移，补 round-trip 检验建议**
- 编译原理铺垫（词法/语法/语义分层、递归下降、AST vs IR）为本书新增，对应 Dragon Book Ch.2-4

## 参考文献

- **[Aho 2006]** Dragon Book，第 2-4 章（编译流水线分层：词法/语法/语义/IR 构造）、第 4.4 节（递归下降解析）。
- **[Lattner 2020]** Lattner et al. "MLIR"，assembly format 与 parse/print 的设计。
- **LLVM IR** `.ll` 格式，序列化/反序列化的对比传统。
