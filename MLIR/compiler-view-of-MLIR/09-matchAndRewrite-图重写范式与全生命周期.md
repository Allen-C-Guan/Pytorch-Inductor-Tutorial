# 第 9 章 matchAndRewrite：图重写范式与全生命周期

> **本章位置**　第五部分（构建）讲完了 IR 怎么造出来。从本章开始进入第六部分——**IR 的重写**。IR 造好后，怎么改它？本章讲 matchAndRewrite 的全生命周期：一条重写规则如何被定义为"带通配符的查找替换模板"，如何被驱动系统化地套用到整棵 IR 树上。
>
> **前置依赖**　第 5 章（遍历，walk/变更契约）、第 4 章（use-def，match 沿它走）。
>
> **编译原理切入**　本章从**项重写系统（term rewriting）与图重写**立论——这是编译器优化的数学基础。一条重写规则 = 一个带通配符的"查找替换模板"：左半边是模式（要找的子图），右半边是替换（换成什么）。优化本质上是"反复套用重写规则直到 IR 不再改变（到达不动点）"。Dragon Book 第 8 章的 peephole optimization 是这个范式的雏形，MLIR 把它工程化为 `OpRewritePattern`。本章还会补一个 Dragon Book 不深究的理论——**项重写系统的合流性（confluence）**：为什么规则应用顺序不影响最终结果（Church-Rosser 性质）。

---

## 9.1 图重写作为优化的形式化框架

你写代码时一定用过正则替换：`s/foo\(([^)]*)\)/bar(\1)/g`——找到所有"长得像 `foo(任意)`"的地方，换成 `bar(同样的内容)`。编译器优化里这叫**项重写（term rewriting）/ 图重写（graph rewriting）**，只不过被替换的不是文本，而是 **IR 图里的 op 子图**。

一条重写规则（Rewrite Pattern）就是一条这样的"查找替换模板"：

```text
              左半边：match（要找的子图模样）                  右半边：rewrite（换成什么样）
        ┌─────────────────────────────┐            ┌──────────────────────────┐
        │  transpose                  │            │                          │
        │     ↑                       │   ──→      │   （直接用 x，两个 transpose 抵消）│
        │  transpose                  │            │                          │
        │     ↑                       │            └──────────────────────────┘
        │     x                       │
        └─────────────────────────────┘
```

MLIR 里这条规则（Toy 方言的 `SimplifyRedundantTranspose`）写成 C++ 只有两行核心：

```cpp
// examples/toy/Ch4/mlir/ToyCombine.cpp:28-53（精简）
struct SimplifyRedundantTranspose : public OpRewritePattern<TransposeOp> {
  LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const override {
    Value transposeInput = op.getOperand();                       // 拿到外层 transpose 的操作数
    TransposeOp transposeInputOp =
        transposeInput.getDefiningOp<TransposeOp>();              // 顺着 use-def 链往上看：它是不是又一个 transpose？
    if (!transposeInputOp) return failure();                      // 不是 → 不匹配，放弃
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});      // 是 → 把外层 transpose 换成"内层 transpose 的输入 x"
    return success();
  }
};
```

读懂这两行，就懂了 matchAndRewrite 的 80%。后面所有机制都是为了把"这样的一堆规则"**系统化地、自动地、反复地**套用到整棵 IR 树上。

> **编译原理浸润点：项重写系统**　Dragon Book 第 8 章的 peephole optimization 是图重写的雏形——在指令序列上做局部替换。MLIR 把这个思想推广到了任意 op 子图。更形式化地说，这是**项重写系统（term rewriting system）**——一组重写规则 $l \to r$，反复应用到项上直到无法再应用（到达 normal form）。Baader & Nipkow 的 *Term Rewriting and All That* [Baader 1998] 是这个领域的标准教材。项重写系统有两个关键性质：**合流性（confluence）**——规则应用顺序不影响最终结果（Church-Rosser）；**终止性（termination）**——规则应用不会无限循环。MLIR 的 greedy driver 依赖前者（规则顺序不影响正确性，只影响效率），用 benefit 控制后者（第 11 章详谈）。这两个性质是图重写作为优化框架的数学基石。

## 9.2 两个继承家族：Pattern 家 vs Rewriter 家

matchAndRewrite 涉及两组类，分属两个家族，初学者最容易把职责搞混。先给一张全景：

```text
═══════════════ 家族 A：Pattern（"规则"——描述怎么匹配、怎么换）═══════════════
Pattern  [PatternMatch.h:73]               ← 元数据：匹配谁（root）、收益（benefit）、上下文
  └─ RewritePattern  [:246]                ← 加虚函数 match / rewrite / matchAndRewrite（用户实现）
       ├─ detail::OpOrInterfaceRewritePatternBase<SourceOp>  [:318]   ← 把 Operation* cast 成 SourceOp 类型
       │     ├─ OpRewritePattern<SourceOp>         [:357]   ← ★ 最常用：按具体 Op 类型匹配
       │     └─ OpInterfaceRewritePattern<SourceOp> [:371]   ← 按 Interface 匹配
       └─ OpTraitRewritePattern<TraitType>          [:383]   ← 按 Trait 匹配

═══════════════ 家族 B：Rewriter（"改图工具"——建新 op、改 use-def、删 op）═══════════════
OpBuilder  [Builders.h:210]                ← 第 6 章主角：create<>、插入点（建 IR）
  └─ RewriterBase  [PatternMatch.h:400]    ← 在 OpBuilder 上加"改图 API"+ Listener（监听者）
       ├─ IRRewriter  [:766]               ← 非 pattern 场景的改图工具（手动改 IR 用）
       └─ PatternRewriter  [:785]          ← ★ pattern 里用的改图工具（驱动会监听它）
```

**一句话区分**：家族 A（Pattern）是"**规则**"——你写的每个 `struct XxxPattern : OpRewritePattern<FooOp>` 都是一条规则；家族 B（Rewriter）是"**改图的手**"——`matchAndRewrite(op, rewriter)` 第二个参数 `rewriter` 就是这只手，你用它来 create/replace/erase。规则描述"换成什么"，手负责"真的动手"。

### 9.2.1 家族 A：Pattern —— 规则的元数据与虚函数

最底层 `Pattern`（[PatternMatch.h:73](../MLIR-Tutorial/third_project/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L73)）**只存元数据、不含任何匹配逻辑**：

```cpp
class Pattern {
  const void *rootValue;        // 要匹配的 root：可能是 op 名 / InterfaceID / TraitID / "any"
  RootKind rootKind;            // 上面那位的"种类"枚举
  const PatternBenefit benefit; // 收益（越大越优先尝试）
  SmallVector<OperationName, 2> generatedOps;  // 这条规则可能生成哪些 op
  ...
};
```

它有四种"按什么匹配 root"的构造方式，对应四种规则：

| 规则类型 | 按……匹配 | 典型用途 |
|---|---|---|
| `OpRewritePattern<SourceOp>` | **具体 op 名** | 只优化某一种 op（如 `toy.transpose`） |
| `OpInterfaceRewritePattern<Iface>` | **接口 ID** | 优化所有实现了某接口的 op |
| `OpTraitRewritePattern<Trait>` | **trait ID** | 优化所有带某 trait 的 op（如所有 `ConstantLike`） |
| `RewritePattern` + `MatchAnyOpTypeTag` | **任意 op** | 极少用，通配一切 |

中间层 `detail::OpOrInterfaceRewritePatternBase<SourceOp>`（[PatternMatch.h:318](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L318)）干一件关键的事：**把驱动传进来的裸 `Operation*` 安全地 cast 成你想要的强类型 `SourceOp`**：

```cpp
// PatternMatch.h:323-332
void rewrite(Operation *op, PatternRewriter &rewriter) const final {
  rewrite(cast<SourceOp>(op), rewriter);   // 把 Operation* cast 成 toy::TransposeOp 等
}
LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final {
  return matchAndRewrite(cast<SourceOp>(op), rewriter);
}
```

这就是为什么你写 `matchAndRewrite(TransposeOp op, ...)` 时拿到的 `op` 已经是 `TransposeOp` 类型、能直接 `.getOperand()`——cast 由基类包办了。

### 9.2.2 两种写法：单步 vs 两步

`RewritePattern` 提供两种实现方式：

```cpp
// PatternMatch.h:264-271 —— RewritePattern 给的"默认 matchAndRewrite"
virtual LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const {
  if (succeeded(match(op))) {     // 先 match
    rewrite(op, rewriter);        // 成功才 rewrite
    return success();
  }
  return failure();
}
```

- **单步**：只 override `matchAndRewrite(SourceOp, rewriter)`——匹配和改写写在一个函数里（Toy `SimplifyRedundantTranspose` 就是）。
- **两步**：分别 override `match(SourceOp)`（只判断，不动 IR）和 `rewrite(SourceOp, rewriter)`（只改，假设已匹配）。

两种等价，但语义侧重不同：

| 写法 | 特点 | 适用 |
|---|---|---|
| 单步 `matchAndRewrite` | 灵活，可边 match 边 build | 匹配需要"造点东西才知道行不行"的复杂场景 |
| 两步 `match`+`rewrite` | **结构性强制 match-before-mutate**（match 失败则 rewrite 根本不会被调） | 推荐**初学者**；逻辑清晰 |

### 9.2.3 家族 B：Rewriter —— 改图的手 + 监听者机制

`RewriterBase`（[PatternMatch.h:400](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L400)）**public 继承 `OpBuilder`**：

```cpp
class RewriterBase : public OpBuilder {     // PatternMatch.h:400
  // 继承自 OpBuilder 的全部能力：create<>、插入点、getType/getAttr……
  // 在此之上，新增两块能力：
  //   ① 改图 API（OpBuilder 没有）：replaceOp / eraseOp / replaceAllUsesWith / modifyOpInPlace / …
  //   ② 一个 Listener（监听者）：每次改图都"广播"通知
  struct Listener : public OpBuilder::Listener { ... };
};
```

> **关键衔接（承接第 6 章）**：上篇你用 `OpBuilder::create<>` 建 IR；现在 `RewriterBase` 继承了 `OpBuilder`，所以**在 pattern 里 `rewriter.create<>()` 走的是和第 6 章一模一样的"装箱单 → `Operation::create` → 七类挂接"流水线**。重写阶段的"建新 op"和构建阶段的"建 op"是同一套原语——这是为什么第 6 章说"理解了 IR 怎么 build，就同时理解了它怎么被重写"。

`RewriterBase` 最常用的改图方法（[PatternMatch.h:522-707](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L522)）：

```cpp
void replaceOp(Operation *op, ValueRange newValues);      // ★ 把 op 的结果换成 newValues，再 erase op
void replaceOp(Operation *op, Operation *newOp);
template <typename OpTy, ...> OpTy replaceOpWithNewOp(Operation *op, ...);
void eraseOp(Operation *op);                              // ★ 删一个 use_empty 的 op
void replaceAllUsesWith(Value from, Value to);            // RAUW（Ch10 详解）
void replaceAllOpUsesWith(Operation *from, ValueRange to);
// 还有：inlineBlockBefore / mergeBlocks / moveOpBefore / splitBlock ……
```

这些方法底下都连着 `Listener`——**每改一下都会发通知**，这是 9.3 节 driver-as-listener 机制的基础。`replaceOp` 和 `eraseOp` 的内部细节（RAUW + erase）是第 10 章的主题——本章只把它们当黑盒用。

`PatternRewriter`（[PatternMatch.h:785](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L785)）在 `RewriterBase` 上**只多加了一个钩子**：

```cpp
class PatternRewriter : public RewriterBase {
  virtual bool canRecoverFromRewriteFailure() const { return false; }   // ← 唯一新增！
};
```

它**没有** override `replaceOp`/`eraseOp`——用的是 `RewriterBase` 的默认实现（立即生效、不留撤销记录）。这个细节是"greedy driver 没有回滚"的根源（第 11 章详谈）。

> **三者何时用谁**：在 pattern 里**永远用 `PatternRewriter`**（驱动在监听它）；在 Pass 里手动改 IR、又没在 pattern 上下文里时，用 `IRRewriter`；只是建 IR 不改图，直接用 `OpBuilder`。

## 9.3 驱动就是 rewriter 的监听者（本章最关键的设计）

一条规则写好后，谁来调用它、按什么顺序、改完之后会怎样？答案是 **greedy pattern rewrite driver**（贪心重写驱动），入口是 `applyPatternsAndFoldGreedily`（[GreedyPatternRewriteDriver.cpp:896](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L896)）。

先讲一个贯穿全篇的洞见。看驱动的类定义和构造函数：

```cpp
// GreedyPatternRewriteDriver.cpp:322
class GreedyPatternRewriteDriver : public RewriterBase::Listener {   // ★ 驱动"是一个"监听者
  PatternRewriter rewriter;   // :359  驱动自己持有一个 rewriter
  Worklist worklist;          // :366  工作表
  PatternApplicator matcher;  // :403  按 benefit 选规则的引擎
};

// GreedyPatternRewriteDriver.cpp:411-434（构造函数）
GreedyPatternRewriteDriver(...) : rewriter(ctx), matcher(patterns) {
  matcher.applyDefaultCostModel();   // ① 按 benefit 给规则排序
  rewriter.setListener(this);        // ② ★★★ 把"自己"设成 rewriter 的监听者
}
```

**第 ② 行是全文的戏眼**：`rewriter.setListener(this)` 把驱动对象自己注册成了 `rewriter` 的监听者。从此，pattern 里每调一次 `rewriter.xxx()`，rewriter 都会回调驱动自己的 `notifyXxx` 方法。驱动就在这些回调里**自动维护工作表**：

```text
   pattern 代码里你写的：                 rewriter 内部做的事：              驱动（监听者）听见后做的：
   rewriter.create<NewOp>(...)    →   建 op + insert + notifyOperationInserted  →  addToWorklist(newOp)
   rewriter.replaceOp(op, vals)   →   RAUW + erase + notifyOperationReplaced/Erased  →  把相关 op 重塞工作表
   rewriter.eraseOp(op)           →   删 op + notifyOperationErased              →  addOperandsToWorklist + worklist.remove
   rewriter.notifyMatchFailure()  →   （不改图，只通知）                          →  记一条 debug 日志
```

**这就是为什么"你只管改图、工作表自动同步"**：你写的 pattern 代码看起来就是在朴素地 create/replace/erase，但每一次调用都顺带让驱动更新了工作表——因为驱动在监听。**如果你绕过 rewriter 直接 `op->erase()`，通知不会发出，驱动的工作表就会指向已释放的 op → 崩溃**。所以重写的第一条契约就是：**pattern 里改图必须走 rewriter**。

> **设计哲学：为什么要"驱动就是监听者"？**　这是关注点分离的典范。规则只管"改成什么"（局部逻辑），驱动管"改完通知谁、工作表怎么同步"（全局调度）。如果让每条规则自己负责"我换完该通知谁"，每条规则都要懂全局——这违反了规则的独立性。把通知机制做成 rewriter 的副作用，让驱动作为监听者自动接收，是"集中式调度 + 分布式规则"的优雅解法。这个模式（observer/listener）在软件工程里很常见，MLIR 把它用得淋漓尽致。

## 9.4 工作表：一个去重的 LIFO 栈

工作表（[GreedyPatternRewriteDriver.cpp:198-273](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L198)）是贪心驱动的核心数据结构：

```cpp
class Worklist {
  std::vector<Operation *> list;             // :224  实际存 op 的数组（后进先出）
  DenseMap<Operation *, unsigned> map;       // :227  op → 在 list 里的下标（实现 O(1) 查/删 + 去重）
};
```

三个设计要点：
- **LIFO（后进先出）**：`pop()` 从 `list.back()` 取。LIFO 让"刚改出来的 op"优先被重新尝试，符合"局部优化先做透"的贪心直觉。
- **去重（set 语义）**：`push()` 用 map 判断 op 是否已在表里，已在就不重复入表——避免同一个 op 被重复处理。
- **O(1) 删除**：`remove()` 把 `list[idx]` 置 `nullptr`，不搬移数组——删一个已死 op 极快。`pop()` 跳过尾部 nullptr。

**持有形式 = 一个 `vector`（顺序容器）+ 一个 `DenseMap`（下标索引，实现去重与 O(1) 删除）。没有用 `std::set`，因为 LIFO + 随机置空比有序树更轻。**

> **编译原理浸润点：工作表驱动 = 数据流到不动点**　Dragon Book 第 9 章的数据流分析也用"工作表算法"（worklist algorithm）——把待分析的节点丢进工作表，反复 pop 处理，处理时若结果变化就把后继节点丢回去，直到工作表空（收敛到不动点）。MLIR 的 greedy driver 是这个算法在"重写"上的对偶：数据流分析求"信息"的不动点，重写求"IR 形态"的不动点。两者都是单调框架下的迭代到收敛。第 11 章会专门讲这个不动点理论。

## 9.5 主循环：pop → 死代码 → fold → 规则匹配

`processWorklist`（[GreedyPatternRewriteDriver.cpp:436](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L436)）的主循环对一个 op 依次试四件事，命中任一就 `continue` 回到循环顶：

```cpp
while (!worklist.empty()) {
  Operation *op = worklist.pop();                                  // Step A：取出一个 op

  // Step B：如果是"显然没用的死 op"（无结果且无副作用/结果无人用），直接删
  if (isOpTriviallyDead(op)) { rewriter.eraseOp(op); changed = true; continue; }

  // Step C：先尝试 fold（常量折叠/局部化简）。ConstantLike 的 op 不 fold（否则无限折叠）
  if (!op->hasTrait<OpTrait::ConstantLike>()) {
    SmallVector<OpFoldResult> foldResults;
    if (succeeded(op->fold(foldResults))) { /* 用 foldResults 替换结果或原地改 */ continue; }
  }

  // Step D：fold 不行，再交给 PatternApplicator 按 benefit 逐条试规则
  LogicalResult r = matcher.matchAndRewrite(op, rewriter, canApply, onFailure, onSuccess);
  if (succeeded(r)) { changed = true; ++numRewrites; }
}
```

**四步标注**（每步"动了什么 / 谁在听"）：

| 步 | 干什么 | 动了什么 | 驱动听见谁的通知 → 工作表怎么变 |
|---|---|---|---|
| **A · pop** | 从工作表栈顶取一个 op | 不动 IR | 取出即从表里消失 |
| **B · 死代码** | `isOpTriviallyDead` → `eraseOp` | 删 op（摘出 block） | `notifyOperationErased` → `addOperandsToWorklist`（操作数的定义 op 重塞工作表） |
| **C · fold** | `op->fold()` 尝试局部化简 | 可能原地改 op，或换结果 | 原地改 → `notifyOperationModified` → 重塞；换结果走 replaceOp → 同 B |
| **D · 规则** | `matcher.matchAndRewrite(...)` 按序试你写的 pattern | 规则内部用 rewriter 改图 | 取决于规则调了什么（create/replace/erase），对应通知全部被驱动接住 |

### 9.5.1 PatternApplicator：按 benefit 选规则

Step D 里的 `matcher.matchAndRewrite(...)` 是 `PatternApplicator::matchAndRewrite`（[PatternApplicator.cpp:130](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L130)）。它的工作：

1. **按 op 名查表**：`patterns.find(op->getName())` 找到所有"匹配这个 op"的规则（一张 `DenseMap<OperationName, SmallVector<RewritePattern*>>`）。
2. **按 benefit 从高到低试**：规则在 `applyDefaultCostModel` 时已按 benefit 降序排好。逐条调 `pattern->matchAndRewrite(op, rewriter)`，**一旦某条成功就 break**。
3. **失败则换下一条**：每试完一条把游标 `++`，所以同一条规则对同一个 op 不会被重复试。

> **benefit 是什么**（[PatternMatch.h:34-63](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L34)）：一个 `unsigned short`，0..65534（65535 是"永不匹配"哨兵）。**越大越先试**。你构造 pattern 时传，如 `OpRewritePattern(ctx, /*benefit=*/1)`。benefit 的深层意义在第 11 章讲——它影响"规则接力"的效率，但不影响正确性（合流性保证）。

### 9.5.2 改完之后：工作表怎么"滚雪球"

这是重写能"接力"（A 换完触发 B）的关键。驱动在 `notifyXxx` 回调里往工作表塞东西：

```cpp
void notifyOperationInserted(Operation *op, ...) { addToWorklist(op); }   // 新建的 op → 入表
void notifyOperationModified(Operation *op)      { addToWorklist(op); }   // 原地改的 op → 入表
```

最有意思的是**删除时的"向上回溯"**（[GreedyPatternRewriteDriver.cpp:694-725](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L694) `addOperandsToWorklist`）：

```cpp
void addOperandsToWorklist(Operation *op) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) continue;                       // 块参数没有定义 op
    // ★ 启发式：只有当该 operand 的使用者 ≤ 2 个时，才把它的定义 op 入表
    if (该 operand 有超过 2 个使用者) continue;
    addToWorklist(defOp);
  }
}
void notifyOperationErased(Operation *op) {
  addOperandsToWorklist(op);   // 删 op → 它的操作数可能少了一个 user，也许现在能被简化/删除
  worklist.remove(op);         // 同时把被删的 op 自己从工作表摘掉（避免悬空指针）
}
```

这个"向上回溯"让重写能**滚雪球**：删掉一个 op 后，它的操作数定义 op 也被重新检查——也许那个 op 现在也没人用了，可以再删；也许它现在能被某条规则简化。这就是为什么一组规则能"接力"——A 换完，B 接着换，C 再接着换，直到没有任何规则能再应用。

## 9.6 一个完整例子：transpose(transpose(x)) → x

把 9.1 的规则放进 driver 跑一遍。输入 IR（Toy 方言）：

```mlir
func.func @transpose_transpose(%arg0 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %t0 = toy.transpose %arg0 : tensor<2x3xf32>        // 第一次转置
  %t1 = toy.transpose %t0 : tensor<2x3xf32>          // 第二次转置，抵消了第一次
  return %t1 : tensor<2x3xf32>
}
```

驱动启动时，所有 op 被塞进工作表（LIFO，所以 return 在栈顶）。逐步追踪：

**第一轮：pop 出 return**
- 不是死代码（有结果被外部用）。
- fold 失败（return 无 fold 规则）。
- 没有匹配 return 的 pattern。工作表不变。

**第二轮：pop 出 %t1（外层 transpose）**
- 不是死代码。
- fold 失败。
- 试 `SimplifyRedundantTranspose`：
  - `match`：`op.getOperand()` = `%t0`；`%t0.getDefiningOp<TransposeOp>()` = 内层 transpose ✅。匹配成功。
  - `rewrite`：`rewriter.replaceOp(外层transpose, {内层transpose.getOperand()})` = 把 `%t1` 换成 `%arg0`。
    - replaceOp 内部：RAUW（让 return 改用 `%arg0`）+ erase 外层 transpose。
    - 驱动听见 `notifyOperationErased(外层transpose)` → `addOperandsToWorklist` → 把 `%t0` 的定义 op（内层 transpose）重塞工作表。

**第三轮：pop 出内层 transpose（%t0）**
- 现在内层 transpose 的结果 `%t0` 没人用了（外层被删了）→ `isOpTriviallyDead` 返回 true → `eraseOp`。
- 驱动听见 → `addOperandsToWorklist` → `%arg0` 是块参数，无定义 op，不入表。

**工作表空了，收敛**。最终 IR：

```mlir
func.func @transpose_transpose(%arg0 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  return %arg0 : tensor<2x3xf32>                      // 两个 transpose 消失
}
```

这就是 matchAndRewrite 的完整生命周期——从规则定义，到驱动调度，到工作表滚雪球，到收敛。整个过程你只写了一条两行的规则，其余全由 driver 自动完成。

> **这个例子体现了项重写的合流性**：不管驱动先处理 return 还是先处理 transpose，最终结果都是 `%arg0`。规则的应用顺序只影响"走几步到达结果"，不影响结果本身。这就是 Church-Rosser 性质在工程上的体现。

## 9.7 设计契约（matchAndRewrite 的游戏规则）

用 matchAndRewrite 时有几个必须遵守的契约：

1. **改图必须走 rewriter，不能直接 op->erase()**。绕过 rewriter 通知不会发出，工作表会悬空。
2. **match-before-mutate**：match 阶段不应改 IR（改了又失败会留下半改的 IR，无回滚）。两步写法（match/rewrite 分离）结构性保证这一点；单步写法靠自觉。
3. **没有回滚**：greedy driver 不支持事务。一旦 rewrite 开始改图，就不能"撤销"。所以 match 要充分——确认能换再换。
4. **返回 failure 不算错误**：match 失败是正常的（规则不适用），返回 `failure()` 或调 `rewriter.notifyMatchFailure(op, "原因")` 即可。后者会记 debug 日志，便于排查。
5. **erase 的 op 必须 use_empty**：erase 一个仍有 use 的 op 会在 debug 下崩溃（Ch10 详解）。replaceOp 自动保证这一点（它先 RAUW 再 erase）。

这些契约的深层根源在第 10 章（RAUW + erase 的物理机制）和第 11 章（driver 的终止性与合流性）会逐一展开。

---

## 编译原理浸润点回顾

1. **项重写系统 / 图重写**：本章主题。一条规则 = 一个带通配符的查找替换模板。源于 Dragon Book 第 8 章 peephole，MLIR 推广到任意 op 子图。
2. **合流性（confluence）与 Church-Rosser**：规则应用顺序不影响最终结果。这是 greedy driver "规则顺序只影响效率不影响正确性"的数学基础。
3. **工作表驱动 = 数据流到不动点**：与 Dragon Book 第 9 章的工作表算法对偶。Ch11 详谈不动点理论。
4. **observer/listener 模式**：驱动作为 rewriter 的监听者，是"集中式调度 + 分布式规则"的解耦设计。
5. **数据流分析（match 沿 use-def 链）**：match 本质是沿 use-list 反向走（`getDefiningOp`），这是第 4 章 use-list 的应用。

---

## 本章关键结论

1. **一条重写规则 = 一个带通配符的查找替换模板**。左半边 match（要找的子图），右半边 rewrite（换成什么）。写成 C++ 是 override `matchAndRewrite(op, rewriter)`。
2. **两个继承家族**：Pattern 家（规则，家族 A）vs Rewriter 家（改图的手，家族 B）。Pattern 描述"换成什么"，Rewriter 负责"真的动手"。永远用 `PatternRewriter`（驱动在监听它）。
3. **驱动就是 rewriter 的监听者**：`rewriter.setListener(this)`。pattern 里每次 create/replace/erase 都触发通知，驱动据此自动维护工作表。改图必须走 rewriter，绕过则工作表悬空。
4. **工作表是去重 LIFO 栈**：vector + DenseMap，O(1) 查/删/去重。LIFO 让局部优化先做透。
5. **主循环四步**：pop → 死代码检查 → fold → 规则匹配。命中任一就 continue。改完通过 notify 滚雪球（addOperandsToWorklist 向上回溯）。
6. **合流性保证规则顺序不影响正确性**：只影响效率（benefit 控制）。

---

## 下一章预告

本章把 matchAndRewrite 当黑盒用了——`rewriter.replaceOp(op, vals)` 和 `rewriter.eraseOp(op)` 都是"黑盒 API"。但它们的内部到底怎么改图？第 10 章打开它们：**replaceOp = RAUW + erase**，而 RAUW 是"把一个 Value 的所有使用者改指向另一个 Value"。这里会回到第 4 章的 use-list 数据结构，看那两个原语（removeFromCurrent + insertInto）如何组合成 RAUW 的批量搬迁。同时讲 erase 的安全性——为什么"先 RAUW 再 erase"是必须的，以及 erase 的反模式。本章并入原 v1 的 Ch11（erase 安全），消除前向依赖。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-matchAndRewrite-重写过程教程_精品.md` §0-3（图重写范式、两个继承家族、Toy transpose 例子、driver 主循环、工作表、PatternApplicator、滚雪球）——**全文保留，重新组织为编译器视角叙事**
- 编译原理铺垫（项重写系统、合流性、Church-Rosser、工作表算法与数据流不动点）为本书新增，对应 Baader & Nipkow、Dragon Book Ch.9

## 参考文献

- **[Baader 1998]** Franz Baader, Tobias Nipkow. *Term Rewriting and All That*. Cambridge University Press, 1998.（项重写系统的标准教材，合流性与终止性）
- **[Aho 2006]** Dragon Book，第 8 章（peephole optimization，图重写的雏形）、第 9 章（工作表算法）。
- **[Lattner 2020]** Lattner et al. "MLIR"，OpRewritePattern 与 greedy driver 的设计。
- **[Church 1936]** Alonzo Church, J. Barkley Rosser. λ-calculus 的 Church-Rosser 定理——合流性的数学源头。
