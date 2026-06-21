# 第 1 章 MLIRContext：对象的所有权与唯一化

> **本章位置**　第 0 章建立了 MLIR 的全景，但全景图里有一个角色被一笔带过——**MLIRContext**。所有 Operation、Type、Attribute 都"活"在一个 context 里。本章打开这个"拥有者"的源码，回答：它用什么数据结构把成千上万的 IR 对象攥在手心？为什么 Type 和 Attribute 要被"唯一化"，而 Operation 不需要？
>
> **前置依赖**　第 0 章（编译流水线、可扩展 IR 哲学）。
>
> **编译原理切入**　本章从编译器一个古老的问题出发——**符号表与类型表**。传统编译器（Dragon Book 第 2.7 节）用符号表记录"这个名字指向哪个声明"，用类型表记录"哪些类型是等价的"。MLIR 把这套思想推到了极致：Type 与 Attribute 不只被记录，还被**唯一化（intern）**——结构相同的 Type 在内存里是同一个对象。这与函数式语言里的 hash-consing、Lisp 里的 symbol intern 同源。理解这一点，才能理解 MLIR 为什么能做到"type 相等比较是 O(1) 的指针比较"。

---

## 1.1 编译器为什么需要"对象的所有权者"

在进入 MLIR 源码之前，先从编译器学科的角度立论：**为什么编译器需要一个"统一拥有所有 IR 对象"的实体？**

考虑一个编译器在运行中要创建多少对象：一个稍大的程序，其 AST 或 IR 动辄有几十万到上百万个节点；每个节点上挂的类型（`i32`、`tensor<...>`）、属性（常量、名字）更是数倍于此。如果这些对象"各自为政"——由谁创建就归谁释放——立刻会面临三个工程难题：

1. **生命周期错综复杂**。一个 `i32` 类型可能被几千个 op 引用，它到底归谁所有？何时释放？稍有差池就是悬挂指针或内存泄漏。
2. **重复创建浪费内存**。如果每用一次 `i32` 就 new 一个新的类型对象，同样的信息会重复存储几千份。
3. **等价判断低效**。要判断两个类型是否相等，如果它们是不同对象，就得逐字段比较；如果它们保证是同一对象，指针比较即可。

编译器学科的解法是引入一个**统一的所有权者**，传统上由"符号表/类型表"承担。Dragon Book 第 2.7 节描述的符号表，本质就是把"名字 → 声明"的映射集中管理。MLIR 把这个思想做了两层强化：

- **统一所有权**：所有 IR 对象（Type、Attribute、加载的 Dialect）的生命周期绑定到 `MLIRContext`，context 析构时整体回收。
- **唯一化（intern）**：结构相同的 Type 与 Attribute 在内存里只存一份，等价判断退化为指针比较。

这两个性质是 MLIR 能高效处理海量 IR 节点的工程基石。本章余下部分打开源码，看 MLIR 如何实现它们。

## 1.2 第一层：MLIRContext 是个"空壳"（pImpl 手法）

第一个反直觉点：[`MLIRContext`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/MLIRContext.h) 这个类本身**几乎不持有任何东西**。它用 C++ 的 **pImpl（pointer to implementation）手法**，整个类只有一个私有成员——一个指向"隐藏实现对象"的智能指针：

```cpp
// MLIRContext.h:297
class MLIRContext {
private:
  const std::unique_ptr<MLIRContextImpl> impl;   // ← 唯一成员
};
```

真正的持有结构全藏在对头文件不可见的 [`MLIRContextImpl`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L123) 里。

> **编译原理浸润点：pImpl 手法**　pImpl 不是编译器专属，它是 C++ 工程的经典封装技巧（Herb Sutter 在 *Exceptional C++* 中系统论述 [Sutter 1999]）：把实现细节挪到 .cpp 文件，对外只暴露一个不透明指针。好处有三：头文件干净、ABI 稳定（改内部容器不影响二进制兼容）、编译防火墙（修改实现不触发重编译）。Dragon Book 不讲 pImpl（它是语言工程问题，不是编译原理），但任何大型编译器（LLVM、GCC、MLIR）都用它来管理复杂的内部状态。读 MLIR 源码时要记住：`MLIRContext` 是门面，`MLIRContextImpl` 才是真身。

## 1.3 第二层：MLIRContextImpl 用哪些容器持有哪些东西

`MLIRContextImpl` 是真正"攥着一切"的结构。下表是它**实际用哪些容器持有哪些东西**（全部经源码逐行核对）：

| 持有什么 | 容器/形式 | key → value | 源码 |
|---|---|---|---|
| **已加载的 dialect 实例** | `DenseMap<StringRef, std::unique_ptr<Dialect>> loadedDialects` | namespace 名 → owned Dialect | [MLIRContext.cpp:196](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L196) |
| **可加载 dialect 的构造器** | `DialectRegistry dialectsRegistry`（内部 `std::map`） | namespace → "怎么 new" | [MLIRContext.cpp:197](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L197) |
| **type 唯一化池** | `StorageUniquer typeUniquer`（按值内嵌） | TypeID → 一池 Storage | [MLIRContext.cpp:214](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L214) |
| **attribute 唯一化池** | `StorageUniquer attributeUniquer` | TypeID → 一池 Storage | [MLIRContext.cpp:246](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L246) |
| **affine 唯一化池** | `StorageUniquer affineUniquer` | TypeID → 一池 Storage | [MLIRContext.cpp:207](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L207) |
| **出现过的 op 名** | `llvm::StringMap<std::unique_ptr<OperationName::Impl>> operations` | op 全名 → 描述符 | [MLIRContext.cpp:183](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L183) |
| **已注册 op（多索引）** | `DenseMap<TypeID,RON>` + `StringMap<RON>` + `SmallVector<RON>` | TypeID / 名字 → RegisteredOperationName | [MLIRContext.cpp:186-191](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L186) |
| **已注册 type/attr 元信息** | `DenseMap<TypeID,AbstractType*>` + `DenseMap<StringRef,AbstractType*>` | TypeID / 名字 → 元信息 | [MLIRContext.cpp:213/221](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L213) |
| **上述小对象的内存** | `llvm::BumpPtrAllocator abstractDialectSymbolAllocator` | —— | [MLIRContext.cpp:180](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L180) |
| **热点 type/attr 缓存** | 值成员 `IntegerType int32Ty; Float32Type f32Ty; BoolAttr trueAttr; ...` | —— | [MLIRContext.cpp:224-261](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/MLIRContext.cpp#L224) |

**怎么读这张表**：

- **dialect 实例**用 `DenseMap<名字, unique_ptr>` 持有——context 独占所有权，context 析构时 unique_ptr 自动释放。
- **type/attr** 用三个 `StorageUniquer` 值成员持有（下一节的戏份）。
- **op 名**用 `StringMap` 持有。
- **几乎没有"裸指针悬空"**——能用 `unique_ptr`/值成员的地方都用了；少数裸指针（如 `AbstractType*`）的内存由 `BumpPtrAllocator` 兜底，context 析构时整体回收。

把这张表画成所有权全景图：

```text
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

这张图回答了本章开头的第一个问题——"用什么形式持有？"答案是**一组 STL 容器 + LLVM 的分配器**，没有魔法，全是扎实的工程数据结构。

## 1.4 第三层：StorageUniquer —— type/attribute 的"对象池"

现在回答本章开头的第二个问题——"为什么 Type/Attribute 要被唯一化？怎么实现的？"

### 1.4.1 唯一化（intern）的编译原理背景

唯一化（intern，又称 hash-consing）是编译器与函数式语言里的经典技术。其核心思想是：**结构相等的对象只存一份，所有引用指向同一份**。这样，等价判断退化为指针比较——O(1) 而非逐字段比较。

Dragon Book 没有专章讲 intern（它把符号表当作"名字 → 声明"的映射），但符号表的字符串池（string pool）本质上就是 intern：同一个标识符字符串在内存里只存一份。Lisp 的 symbol、Erlang 的 atom、Java 的 `String.intern()`、OCaml 的 polymorphic comparison 对原子类型的优化，都是同一思想的不同工程实现。

MLIR 把 intern 应用到了所有"结构相等"的 IR 对象上：Type 与 Attribute。于是，`builder.getI32Type()` 无论调用多少次，返回的永远是**同一个指针**；两个 `i32` 比较相等就是 `a == b` 的指针比较。

### 1.4.2 StorageUniquer 的数据结构

`StorageUniquer`（[StorageUniquer.cpp](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Support/StorageUniquer.cpp)）用**两张按 TypeID 分桶的哈希表**组织：

```cpp
// StorageUniquer.cpp:347-353
DenseMap<TypeID, std::unique_ptr<ParametricStorageUniquer>> parametricUniquers;  // 带参数的类型（如 i32、NSTensor<...>）
DenseMap<TypeID, BaseStorage *> singletonInstances;                              // 单例类型（如 index、none）
```

- 每注册一种带参数的类型（比如 `IntegerType`），就用它的 `TypeID` 在 `parametricUniquers` 里建一个专属桶 `ParametricStorageUniquer`。
- 为了减少多线程锁竞争，**每个桶又按 hash 值的低几位切成多个 `Shard`（默认 8 个，须为 2 的幂）**，每片各持一把读写锁。
- **真正装"唯一化后的 Storage 对象"的容器**，是每个 Shard 里的：

```cpp
// StorageUniquer.cpp:42-47, 76
struct HashedStorage { unsigned hashValue; BaseStorage *storage; };   // 存的是指针，不是对象副本
using StorageTypeSet = DenseSet<HashedStorage, StorageKeyInfo>;        // = Shard::instances
```

即：**一个 `DenseSet`（哈希集合），元素是"预计算 hash + 指向 storage 的裸指针"**。storage 本身的内存来自一个 `BumpPtrAllocator`——**分配后不单独释放，直到 context 析构才整体回收，因此叫"immortal（永生）"**。

这就是"同一个 i32 永远是同一个指针"的物理来源：

```text
typeUniquer
 └─ parametricUniquers: DenseMap<TypeID, unique_ptr<ParametricStorageUniquer>>
      └─ [TypeID = IntegerType] → ParametricStorageUniquer
            └─ shards[8]（按 hash 切片）
                  └─ shard.instances: DenseSet<HashedStorage>
                        └─ { hash, &IntegerTypeStorage(width=32, signless) }   ← 唯一的 i32 实例
                                （storage 内存来自 BumpPtrAllocator，永生）
```

> **为什么用 BumpPtrAllocator 而不是逐个释放？**　这是编译器对 IR 对象生命周期的关键取舍。Type/Attribute 的特点是"创建多、永不单独销毁、随 context 一起消失"。对这种模式，逐个 `delete` 既慢（每个 delete 要更新 free list）又无必要（反正最后一次性回收）。LLVM 的 `BumpPtrAllocator` 就是为这种模式设计的：分配是 bump 一个指针（O(1)，无碎片），销毁是整块归还。Dragon Book 不讨论分配器，但任何写编译器的人都必须懂这个工程权衡——它是 MLIR 能高效处理百万级 IR 对象的隐藏支柱。

### 1.4.3 "持有"的实际动作：一次 IntegerType::get 的旅程

把上面三层串起来。你写 `builder.getI32Type()`（最终落到 `IntegerType::get(ctx, 32)`），完整路径：

```text
IntegerType::get(ctx, 32)                                  // 你调的
  MLIRContext.cpp:1060-1091
   ├─ 先查缓存：context 里的 int32Ty 成员（命中就直接返回，O(1)）         ← 大多数 i32 走这条
   └─ 未命中 → Base::get(ctx, 32, signless)                  // StorageUniquerSupport.h:173
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

注意这条路径上的两个优化：

1. **热点缓存**：`i32`、`f32` 这类极常用的类型，context 直接把它们作为值成员缓存（`int32Ty`、`f32Ty`），连哈希查都省了。`IntegerType::get` 一开始就先查这个缓存。
2. **分片降低锁竞争**：多线程编译时，不同 hash 的类型查找走不同 Shard，锁不冲突。

**结论**：`IntegerType` 对象（`IntegerTypeStorage`）的内存由 context 的 `typeUniquer` 池持有，**不属于任何 op**。`IntegerType` 这个 C++ 类本身只是 `ImplType *impl` 一个指针——[Types.h:20-31](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Types.h) 注释明说 *"wraps a pointer to the storage object owned by MLIRContext"*。`Attribute` 完全同构（[Attributes.h:19-25](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Attributes.h)，*"references to immortal ... storage owned by MLIRContext"*）。

这就解释了一个贯穿全书的论断——**"type/attribute 属于 context，不属于 op"**：op 只是在自己的 `attrs` 字段里存了**一个指向 context 池中对象的指针**。op 销毁不会动 context 池里的一根毫毛（这一点在 Ch10 讲 erase 时会再次体现）。

## 1.5 Dialect 的加载：可扩展性的运行时入口

第 0 章讲过，MLIR 的核心创新是可扩展——任何人可以定义自己的 dialect。这一节看 dialect 在运行时是如何被"加载"进 context 的。

```cpp
mlir::MLIRContext ctx;
ctx.getOrLoadDialect<mlir::func::FuncDialect>();   // 用到哪个方言就加载哪个
ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
```

`getOrLoadDialect` 的语义是"获取或加载"：如果该 dialect 已经在 `loadedDialects` 里，直接返回；否则从 `dialectsRegistry` 找到它的构造器，new 一个实例，塞进 `loadedDialects`。

这个"用到才加载"的设计（lazy loading）对应编译器的实际工作模式：一次编译只用得到少数几个 dialect，没必要把所有 dialect 都加载进来。这也是 MLIR 可扩展性不会拖慢编译的原因——dialect 多了不影响单次编译的开销。

一个 dialect 被加载时，它做的关键事情是**注册自己定义的 Type/Attribute/Operation**。这些注册会把 dialect 的元信息（`AbstractType`、`AbstractAttribute`、`RegisteredOperationName`）写进 context 的相应表里（见 1.3 节的表）。从此，context 就"认识"了这个 dialect 的所有实体。

> **编译原理浸润点：可扩展 IR 哲学**　Dialect 机制是 MLIR 论文 [Lattner 2020] 的核心贡献。传统编译器（LLVM、GCC）的指令集是固定的，要加一条指令得改编译器源码；MLIR 把"定义指令"的权力下放给 dialect，让编译器变成了一个"IR 框架"而非"IR 本身"。这与 Lisp 让用户定义新语法（macro）、Prolog 让用户定义新谓词的精神一脉相承——把元语言下放到用户层。本书第 7 章讲 `Operation::create` 的声明式 .td 定义时，会再次回到这个哲学：MLIR 不仅让 dialect 可扩展，连"怎么声明一个 op"都是声明式的（TableGen）。

## 1.6 本章的核心论断

把这一章收束成几个可记忆的论断：

1. **MLIRContext 是统一的所有权者**。所有 IR 对象（Type、Attribute、加载的 Dialect）的生命周期绑定到 context，context 析构时整体回收。对象所有权没有散落各处，杜绝了悬挂指针。
2. **MLIRContext 本身是空壳，真身是 MLIRContextImpl**。这是 pImpl 手法的体现——门面与实现分离，头文件干净、ABI 稳定。
3. **Type 与 Attribute 被唯一化（intern）**。结构相等的对象只存一份，等价判断退化为 O(1) 的指针比较。物理实现是 `StorageUniquer` 的分片哈希表 + BumpPtrAllocator。
4. **Type/Attribute 属于 context，不属于 op**。op 只持有指向 context 池的指针。这个论断会在 Ch2（Operation 的 attrs 字段）、Ch10（erase 不动 context 池）反复出现。
5. **Operation 不被唯一化**。这个对比很重要：两个"长得一样"的 Operation 是不同的对象（它们有不同的 location、可能嵌套在不同的地方）。Operation 的所有权由它所在的 Block 持有（详见 Ch3、Ch7），不归 context 池。这个差异是 MLIR 的有意设计——Type/Attribute 是无身份的（value-equal），Operation 是有身份的（identity-bearing）。

第 5 点尤其值得强调。它揭示了一个深层设计原则：**编译器要区分"无身份的值对象"与"有身份的实体对象"**。Type、Attribute 是前者——`i32` 就是 `i32`，没有"哪一个 i32"之分；Operation 是后者——同样是 `addi`，这一个和那一个是不同的运算，发生在程序的不同位置。把这两类对象用不同的所有权策略管理，是 MLIR 内存模型的精髓。

---

## 编译原理浸润点回顾

本章涉及的编译原理概念，及其在后续章节的回扣：

1. **符号表/类型表的传统设计**：本章主题。MLIR 把它强化为"统一所有权 + 唯一化"。
2. **唯一化（intern / hash-consing）**：本章主题。源于 Lisp symbol、Erlang atom 等函数式语言实践。后续 Ch7 会对比"Operation 为什么不 intern"。
3. **pImpl（编译指示封装）**：本章的 C++ 工程手法。不是编译原理，但任何大型编译器都用。
4. **可扩展/多层次 IR 哲学**：第 0 章引入，本章通过 dialect 加载落实。Ch7 的声明式 .td 定义会再次回到这个哲学。

---

## 本章关键结论

1. **MLIRContext 用 pImpl 隐藏实现**：`MLIRContext` 只有一个 `unique_ptr<MLIRContextImpl>` 成员，真身藏在 .cpp。
2. **MLIRContextImpl 用 STL 容器 + LLVM 分配器持有所有 IR 对象**：dialect 用 `DenseMap`，type/attr 用三个 `StorageUniquer`，op 名用 `StringMap`。
3. **StorageUniquer 实现唯一化**：分片哈希表 + BumpPtrAllocator，使结构相等的 type/attr 在内存里唯一。
4. **Type/Attribute 属于 context，不属于 op**：op 只持有指向 context 池的指针，op 销毁不动池。
5. **Operation 不被唯一化**：它有身份（location/嵌套），所有权归 Block 而非 context。这是"无身份值对象"与"有身份实体对象"的区分。

---

## 下一章预告

本章讲完了"拥有者"。从下一章开始，我们看被拥有的东西——IR 的逻辑部件。第 2 章打开 `Operation` 的源码，回答：一个 Operation 到底"装"了什么？为什么 MLIR 说"一切皆 Operation"？这里有一个区分是 MLIR 可分析性的根基——**Operand 与 Attribute 的语义分野**：前者是数据流边（运行时变），后者是编译期配置（静态）。理解了这个分野，才理解为什么数据流分析只沿 Operand 走，而不碰 Attribute。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-IR-树的构建过程教程_精品.md` §1.1（MLIRContext 的三层结构、容器表、StorageUniquer、IntegerType::get 旅程）——**全文保留，重新组织为编译器视角叙事**
- §1.5（dialect 加载）综合了原文与 MLIR 源码
- 编译原理铺垫（符号表/类型表传统、intern/hash-consing 背景、pImpl 工程手法）为本书新增

## 参考文献

- **[Aho 2006]** Dragon Book，第 2.7 节（符号表）。
- **[Sutter 1999]** Herb Sutter. *Exceptional C++: 47 Engineering Puzzles, Programming Problems, and Solutions*. Addison-Wesley, 1999.（pImpl / 编译防火墙）
- **[Lattner 2020]** Lattner et al. "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." arXiv:2002.11054.
- **[Ershov 1958]** Andrei Ershov 对 hash-consing 的早期工作（编译器上下文）。
