# 从匹配到替换：matchAndRewrite 如何改写一棵 MLIR IR 树

**基于 LLVM 19.1.7 源码的完整教程**

> 本文是 [从零构建一棵 MLIR IR 树](MLIR-IR-树的构建过程教程_精品.md) 的续集。
> 那一篇回答「**IR 树是怎么从无到有 build 出来的**」（构建：`OpBuilder::create` → `Operation::create` → 七类挂接）；
> 本篇回答「**这棵已经建好的树，是怎么被一条条重写规则（Rewrite Pattern）改写的**」（重写：`matchAndRewrite` → `replaceOp` → RAUW + erase）。
> 两者合起来，才是 MLIR IR 的完整生命周期——**建起来**，再**改下去**。

---

## 摘要

上一篇里，构建 IR 的本质是反复执行两类「挂接」（结构挂接 + 数据流挂接），把两张网一点点织起来。这一篇要讲的重写（rewrite），**没有引入任何新的底层数据结构**——它复用的全是上一篇已经讲透的东西：

- **匹配（match）** = 沿着上一篇织好的 **use-def 链**（`OpOperand`→`Value`）和**结构树**去"看"一个 op 长什么样、它的操作数是谁定义的；
- **替换（replace）** = 用 rewriter 改图，而**改图的两个原子动作 `replaceOp` / `eraseOp`，本质全是 use-def 边的拆除与重接**——用的就是上一篇 Step 6 讲的 `replaceAllUsesWith` 和 `removeFromCurrent`。

所以重写看起来神秘，底层却异常朴素。本教程要让你建立的**核心心智模型**有三句话：

1. **一个重写规则 = 一个带通配符的"图替换模板"**：左半边（match）描述"什么样的 op 子图"，右半边（rewrite）描述"换成什么样的新子图"。
2. **替换 = RAUW + erase**：把旧 op 结果的所有使用者（`OpOperand`）改指向新值（`replaceAllUsesWith`），再把旧 op 删掉（`eraseOp`）。两步都是 use-def 操作。
3. **驱动（driver）就是 rewriter 的监听者（listener）**：你在 pattern 里调 `rewriter.create<>` / `rewriter.replaceOp`，rewriter 会"广播"通知（`notifyOperationInserted/Erased/...`），驱动听见后自动维护它的工作表（worklist）——**你只管改图，工作表的同步是框架白送的**。

本教程选一个最小但经典的例子——Toy 方言的 `transpose(transpose(x)) → x`——逐行走读它**从 match 到 rewrite 到工作表更新**的全过程，每一步都标注：

1. **调用了哪个 MLIR 关键接口**（`OpRewritePattern::matchAndRewrite` → `PatternRewriter::replaceOp` → `replaceAllUsesWith` → `eraseOp`）；
2. **这一步"动了"哪条 use-def 边 / 哪个 op**（哪个 `OpOperand` 被改了 value、哪个 op 被摘出 block）；
3. 驱动如何**听见通知、把相关 op 重新塞回工作表**，从而让重写"滚雪球"般收敛。

**关键词**：matchAndRewrite；OpRewritePattern；PatternRewriter；RewriterBase；replaceOp / eraseOp / replaceAllUsesWith；PatternBenefit；greedy driver；worklist；listener；notifyMatchFailure；progressive lowering

---

## 目录

- ## 0. 前置：什么是"图重写"（graph rewriting）
- ## 1. 两个继承家族：Pattern 家 与 Rewriter 家
- ## 2. 目标例子与 MLIR 文本语法
- ## 3. matchAndRewrite 的完整生命周期（driver 主循环，逐步标注）
- ## 4. 替换的本质：replaceOp = RAUW + erase（接续上篇 use-def）
- ## 5. 一个完整例子逐步追踪（Toy `transpose(transpose(x)) → x`）
- ## 6. 设计契约与陷阱（match-before-mutate、没有回滚、benefit、终止性）
- ## 7. NorthStar 实战（split vs combined 两种写法 + Pass 集成）
- ## 8. 延伸：fold 先于 pattern、DRR/PDL 声明式写法、与 CH-12 的衔接
- ## 附录 A：Rewriter API 速查
- ## 附录 B：关键源码索引

---

## 0. 前置：什么是"图重写"

### 0.1 把重写当"带通配符的查找替换"来理解

你写代码时一定用过正则替换：`s/foo\(([^)]*)\)/bar(\1)/g`——找到所有"长得像 `foo(任意)`"的地方，换成 `bar(同样的内容)`。编译器优化里这叫**项重写（term rewriting）/ 图重写（graph rewriting）**，只不过被替换的不是文本，而是 **IR 图里的 op 子图**。

一个重写规则（Rewrite Pattern）就是一条这样的"查找替换模板"：

```
              左半边：match（要找的子图模样）                  右半边：rewrite（换成什么样）
        ┌─────────────────────────────┐            ┌──────────────────────────┐
        │  transpose                  │            │                          │
        │     ↑                       │   ──→      │   （直接用 x，两个 transpose 抵消）│
        │  transpose                  │            │                          │
        │     ↑                       │            └──────────────────────────┘
        │     x                       │
        └─────────────────────────────┘
```

MLIR 里这条规则（Toy 方言）写成 C++ 是两行：

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

### 0.2 编译栈定位

```
源语言 ──AST──→ 【上篇边界：AST → MLIR】 ──→ MLIR 多层 dialect ──【本篇边界：在一棵 MLIR 上跑重写】──→ 继续下层 dialect ──→ LLVM IR
                                              (func/arith/north_star/...)
                                                       │
                                                    你在这里：
                                                    用一组 Rewrite Pattern
                                                    把高层 op 逐步换成低层 op
```

上篇停在"树建好了"。本篇讲**树建好之后的第一件大事：重写（rewrite）**。重写是 MLIR 的核心动词——优化（常量折叠、死代码消除、算术化简）、**渐进 lowering（progressive lowering，把 `arith.addi` → `llvm.add`，`north_star.softmax` → 一串 `scf`/`llvm`）**，本质上都是"在树上反复套用重写规则"。所以 NorthStar 教程 CH-9（Rewrite Pattern）正是这一篇的对应章。

### 0.3 设计哲学：为什么要"模式 + 驱动"这套架构

| 要解决的问题 | MLIR 的答案 | 为什么不换个更简单的做法 |
|---|---|---|
| **怎么让"优化规则"可组合、可复用？** | 每条规则写成独立的 `OpRewritePattern`，互不感知；一个 Pass 往 `RewritePatternSet` 里塞一堆 | 若把所有优化硬编码进一个巨大的 visitor，每加一条规则就要改这个庞然大物、且规则间会互相踩踏。独立 pattern = 插件化 |
| **规则之间可能"接力"（A 换完触发 B），怎么自动滚下去直到没法再优化？** | 驱动维护一个**工作表（worklist）**，每次改图就把受影响的 op 重新入表，循环到收敛（fixpoint） | 若让规则自己负责"我换完该通知谁"，每条规则都要懂全局。集中式工作表 = 规则只管局部，调度交给驱动 |
| **改图有风险（删 op、改 use-def），怎么保证驱动的工作表不"看到"半改的 IR？** | **驱动就是 rewriter 的监听者**：所有改图必须走 rewriter，rewriter 每改一步就发通知，驱动据此更新工作表 | 若允许 pattern 直接 `op->erase()`、绕过 rewriter，驱动的工作表会指向已释放的 op → 崩溃。强制走 rewriter = 通知必然被听见 |
| **规则可能不适用（match 失败），怎么优雅放弃？** | `matchAndRewrite` 返回 `failure()`，或更地道地 `return rewriter.notifyMatchFailure(op, "原因")` | 若用异常/C++错误码，写法和读法都重。`LogicalResult` + 通知 = 轻量且可被 `-debug` 观测 |

---

## 1. 两个继承家族

matchAndRewrite 涉及两组类，分属两个家族，初学者最容易把它们的职责搞混。先给一张全景：

```
═══════════════ 家族 A：Pattern（"规则"——描述怎么匹配、怎么换）═══════════════
Pattern  [PatternMatch.h:73]               ← 元数据：匹配谁（root）、收益（benefit）、上下文
  └─ RewritePattern  [:246]                ← 加虚函数 match / rewrite / matchAndRewrite（用户实现）
       ├─ detail::OpOrInterfaceRewritePatternBase<SourceOp>  [:318]   ← 把 Operation* cast 成 SourceOp 类型
       │     ├─ OpRewritePattern<SourceOp>         [:357]   ← ★ 最常用：按具体 Op 类型匹配
       │     └─ OpInterfaceRewritePattern<SourceOp> [:371]   ← 按 Interface 匹配（一族 op 共一个接口）
       └─ OpTraitRewritePattern<TraitType>          [:383]   ← 按 Trait 匹配（注意：直接继承 RewritePattern）

═══════════════ 家族 B：Rewriter（"改图工具"——建新 op、改 use-def、删 op）═══════════════
OpBuilder  [Builders.h:210]                ← 上篇主角：create<>、插入点（建 IR）
  └─ RewriterBase  [PatternMatch.h:400]    ← 在 OpBuilder 上加"改图 API"+ Listener（监听者）
       ├─ IRRewriter  [:766]               ← 非 pattern 场景的改图工具（手动改 IR 用）
       └─ PatternRewriter  [:785]          ← ★ pattern 里用的改图工具（驱动会监听它）
```

**一句话区分**：家族 A（Pattern）是"**规则**"——你写的每个 `struct XxxPattern : OpRewritePattern<FooOp>` 都是一条规则；家族 B（Rewriter）是"**改图的手**"——`matchAndRewrite(op, rewriter)` 第二个参数 `rewriter` 就是这只手，你用它来 create/replace/erase。规则描述"换成什么"，手负责"真的动手"。

### 1.1 家族 A：Pattern —— 规则的元数据与虚函数

最底层 `Pattern`（[PatternMatch.h:73](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L73)）**只存元数据、不含任何匹配逻辑**：

```cpp
// PatternMatch.h:73-231（摘关键成员）
class Pattern {
  const void *rootValue;        // 要匹配的 root：可能是 op 名 / InterfaceID / TraitID / "any"
  RootKind rootKind;            // 上面那位的"种类"枚举
  const PatternBenefit benefit; // 收益（越大越优先尝试，见 §6.3）
  SmallVector<OperationName, 2> generatedOps;  // 这条规则可能生成哪些 op（驱动用来推断工作表）
  ...
};
```

它有四种"按什么匹配 root"的构造方式（[PatternMatch.h:155-198](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L155)），对应四种规则：

| 规则类型 | 按……匹配 | 构造时传入 | 典型用途 |
|---|---|---|---|
| `OpRewritePattern<SourceOp>` | **具体 op 名** | `SourceOp::getOperationName()` | 只优化某一种 op（如 `toy.transpose`） |
| `OpInterfaceRewritePattern<Iface>` | **接口 ID** | `Iface::getInterfaceID()` | 优化所有实现了某接口的 op（如所有 `DistributeParallelOp`） |
| `OpTraitRewritePattern<Trait>` | **trait ID** | `TypeID::get<Trait>()` | 优化所有带某 trait 的 op（如所有 `ConstantLike`） |
| `RewritePattern` + `MatchAnyOpTypeTag` | **任意 op** | 特殊 tag | 极少用，通配一切 |

> **易混点（源码核对）**：`OpRewritePattern` 和 `OpInterfaceRewritePattern` 是亲兄弟（都继承自 `detail::OpOrInterfaceRewritePatternBase<SourceOp>`，[PatternMatch.h:357/371](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L357)）；而 `OpTraitRewritePattern` **直接继承 `RewritePattern`**（[PatternMatch.h:384](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L384)），比前两者高一级——它不做 `cast<SourceOp>`，因为 trait 不绑定具体 op 类型。

中间层 `detail::OpOrInterfaceRewritePatternBase<SourceOp>`（[PatternMatch.h:318-351](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L318)）干一件关键的事：**把驱动传进来的裸 `Operation*` 安全地 cast 成你想要的强类型 `SourceOp`**：

```cpp
// PatternMatch.h:323-332（摘）
void rewrite(Operation *op, PatternRewriter &rewriter) const final {          // ← final！
  rewrite(cast<SourceOp>(op), rewriter);   // 把 Operation* cast 成 toy::TransposeOp 等
}
LogicalResult match(Operation *op) const final { return match(cast<SourceOp>(op)); }
LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final {
  return matchAndRewrite(cast<SourceOp>(op), rewriter);
}
```

这就是为什么你写 `matchAndRewrite(TransposeOp op, ...)` 时拿到的 `op` 已经是 `TransposeOp` 类型、能直接 `.getOperand()`——cast 由基类包办了。

#### 1.1.1 两种写法：单步 `matchAndRewrite` vs 两步 `match`/`rewrite`

`RewritePattern` 提供两种实现方式（[PatternMatch.h:236-244](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L236) 注释明说）：

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

两种等价，但语义侧重不同（§6.1 详谈）：

| 写法 | 特点 | 适用 |
|---|---|---|
| 单步 `matchAndRewrite` | 灵活，可边 match 边 build | 匹配需要"造点东西才知道行不行"的复杂场景 |
| 两步 `match`+`rewrite` | **结构性强制 match-before-mutate**（match 失败则 rewrite 根本不会被调） | 推荐**初学者**；逻辑清晰、不易写出"改了一半又失败"的 bug |

NorthStar 那个文件里两种写法都有，正好可对比（见 §7）。

### 1.2 家族 B：Rewriter —— 改图的手 + 监听者机制

`RewriterBase`（[PatternMatch.h:400](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L400)）**public 继承 `OpBuilder`**——这意味着：

```cpp
class RewriterBase : public OpBuilder {     // PatternMatch.h:400
  // 继承自 OpBuilder 的全部能力：create<>、插入点、getType/getAttr……（上篇 §1.2 讲的全能用得上）
  // 在此之上，新增两块能力：
  //   ① 改图 API（OpBuilder 没有）：replaceOp / eraseOp / replaceAllUsesWith / modifyOpInPlace / …
  //   ② 一个 Listener（监听者）：每次改图都"广播"通知
  struct Listener : public OpBuilder::Listener { ... };   // :402
};
```

> **关键衔接（承接上篇）**：上篇你用 `OpBuilder::create<>` 建 IR；现在 `RewriterBase` 继承了 `OpBuilder`，所以**在 pattern 里 `rewriter.create<>()` 走的是和上篇一模一样的"装箱单 → `Operation::create` → 七类挂接"流水线**。重写阶段的"建新 op"和构建阶段的"建 op"是同一套原语——这是为什么上篇说"理解了 IR 怎么 build，就同时理解了它怎么被重写"。

`RewriterBase` 最常用的改图方法（[PatternMatch.h:522-707](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L522)）：

```cpp
void replaceOp(Operation *op, ValueRange newValues);      // ★ 把 op 的结果换成 newValues，再 erase op
void replaceOp(Operation *op, Operation *newOp);          //   同上，但换成一个新 op 的结果
template <typename OpTy, ...> OpTy replaceOpWithNewOp(Operation *op, ...);  // 建+换 一步到位
void eraseOp(Operation *op);                              // ★ 删一个 use_empty 的 op
void replaceAllUsesWith(Value from, Value to);            // 把 from 的所有使用者改指 to（不删任何 op）
void replaceAllOpUsesWith(Operation *from, ValueRange to);
// 还有：inlineBlockBefore / mergeBlocks / moveOpBefore / splitBlock ……
```

这些方法底下都连着 `Listener`——**每改一下都会发通知**，这是 §3 driver-as-listener 机制的基础。

`PatternRewriter`（[PatternMatch.h:785](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L785)）在 `RewriterBase` 上**只多加了一个钩子**：

```cpp
// PatternMatch.h:785-795
class PatternRewriter : public RewriterBase {
  virtual bool canRecoverFromRewriteFailure() const { return false; }   // ← 唯一新增！§6.2 会讲它为何是 false
};
```

它**没有** override `replaceOp`/`eraseOp`——用的是 `RewriterBase` 的默认实现（立即生效、不留撤销记录）。这个细节是 §6.2 "为什么 greedy 没有回滚"的根源。

> **三者何时用谁**：在 pattern 里**永远用 `PatternRewriter`**（驱动在监听它）；在 Pass 里手动改 IR、又没在 pattern 上下文里时，用 `IRRewriter`；只是建 IR 不改图，直接用 `OpBuilder`。

---

## 2. 目标例子与 MLIR 文本语法

我们要演示"被改写"的例子。Toy 方言里有个 `transpose` op，对张量做转置。两次连续转置等于没转置，这是经典的可优化模式。

**改写前**：

```mlir
func.func @transpose_transpose(%arg0 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %t0 = toy.transpose %arg0 : tensor<2x3xf32>        // 第一次转置
  %t1 = toy.transpose %t0 : tensor<2x3xf32>          // 第二次转置，抵消了第一次
  return %t1 : tensor<2x3xf32>
}
```

**改写后**（`SimplifyRedundantTranspose` 套用后）：

```mlir
func.func @transpose_transpose(%arg0 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  return %arg0 : tensor<2x3xf32>                      // 两个 transpose 消失，return 直接用 %arg0
}
```

### 2.1 语法速读

| 语法片段 | 含义 | use-def 视角（接续上篇） |
|---|---|---|
| `%t0 = toy.transpose %arg0` | 转置 op，操作数 `%arg0`，结果 `%t0` | `%t0` 是 `OpResult`；一个 `OpOperand{owner=transpose0, value=%arg0}` 挂进 `%arg0` 的 use-list |
| `%t1 = toy.transpose %t0` | 第二次转置，操作数是 `%t0` | `OpOperand{owner=transpose1, value=%t0}` 挂进 `%t0` 的 use-list |
| `return %t1` | 返回 `%t1` | `OpOperand{owner=return, value=%t1}` 挂进 `%t1` 的 use-list |

改写要做的，就是：让 `return` 不再用 `%t1` 而改用 `%arg0`（**改 use-def 边**），然后把 `%t1`、`%t0` 这两个没人用的 transpose 删掉（**摘 op 出 block**）。这正好对应上篇 §4 的挂接⑦（use-def 边）和挂接③（op→block）——**重写就是反向拆除这些挂接**。

---

## 3. matchAndRewrite 的完整生命周期（driver 主循环）

一条规则写好后，谁来调用它、按什么顺序、改完之后会怎样？答案是 **greedy pattern rewrite driver**（贪心重写驱动），入口是 `applyPatternsAndFoldGreedily`（[GreedyPatternRewriteDriver.cpp:896](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L896)）。它的主循环（[GreedyPatternRewriteDriver.cpp:436-633](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L436) `processWorklist`）是本篇的"心脏"。

> **承接上篇 §6**：上篇讲了 `walk()` 遍历树做 lowering 的"起点"。本篇就是那个"遍历中重写"的真实实现——它不是手动 walk，而是**工作表驱动**：把 op 丢进工作表，反复 pop 出来尝试规则，改图后又把受影响的 op 丢回去，直到工作表空了（收敛）。

### 3.1 驱动就是 rewriter 的监听者（本篇最关键的设计）

先讲一个贯穿全篇的洞见。看驱动的类定义和构造函数：

```cpp
// GreedyPatternRewriteDriver.cpp:322
class GreedyPatternRewriteDriver : public RewriterBase::Listener {   // ★ 驱动"是一个"监听者
  PatternRewriter rewriter;   // :359  驱动自己持有一个 rewriter
  Worklist worklist;          // :366  工作表
  PatternApplicator matcher;  // :403  按 benefit 选规则的引擎
  ...
};

// GreedyPatternRewriteDriver.cpp:411-434（构造函数，精简）
GreedyPatternRewriteDriver(...) : rewriter(ctx), config(config), matcher(patterns) {
  matcher.applyDefaultCostModel();   // ① 按 benefit 给规则排序
  rewriter.setListener(this);        // ② ★★★ 把"自己"设成 rewriter 的监听者
}
```

**第 ② 行是全文的戏眼**：`rewriter.setListener(this)` 把驱动对象自己注册成了 `rewriter` 的监听者。从此，pattern 里每调一次 `rewriter.xxx()`，rewriter 都会回调驱动自己的 `notifyXxx` 方法。驱动就在这些回调里**自动维护工作表**：

```
   pattern 代码里你写的：                 rewriter 内部做的事：              驱动（监听者）听见后做的：
   rewriter.create<NewOp>(...)    →   建 op + insert 进 block + notifyOperationInserted  →  addToWorklist(newOp)
   rewriter.replaceOp(op, vals)   →   RAUW + erase + notifyOperationReplaced/Erased       →  把相关 op 重塞工作表
   rewriter.eraseOp(op)           →   删 op + notifyOperationErased                        →  addOperandsToWorklist + worklist.remove
   rewriter.notifyMatchFailure()  →   （不改图，只通知）                                    →  记一条 debug 日志
```

**这就是为什么"你只管改图、工作表自动同步"**：你写的 pattern 代码看起来就是在朴素地 create/replace/erase，但每一次调用都顺带让驱动更新了工作表——因为驱动在监听。**如果你绕过 rewriter 直接 `op->erase()`，通知不会发出，驱动的工作表就会指向已释放的 op → 崩溃。所以 §6 的第一条契约就是：pattern 里改图必须走 rewriter。**

### 3.2 工作表：一个去重的 LIFO 栈

工作表（[GreedyPatternRewriteDriver.cpp:198-273](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L198)）是贪心驱动的核心数据结构：

```cpp
// GreedyPatternRewriteDriver.cpp:198-228（摘）
class Worklist {
  std::vector<Operation *> list;             // :224  实际存 op 的数组（后进先出）
  DenseMap<Operation *, unsigned> map;       // :227  op → 在 list 里的下标（实现 O(1) 查/删 + 去重）
};
```

> 🏗️ **数据结构 · 工作表为什么长这样？**
> - **LIFO（后进先出）**：`pop()` 从 `list.back()` 取（[GreedyPatternRewriteDriver.cpp:256](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L256)）。LIFO 让"刚改出来的 op"优先被重新尝试，符合"局部优化先做透"的贪心直觉。
> - **去重（set 语义）**：`push()` 用 `map.insert({op, size})` 的返回值判断 op 是否已在表里，已在就不重复入表（[GreedyPatternRewriteDriver.cpp:246](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L246)）——避免同一个 op 被重复处理。
> - **O(1) 删除**：`remove()` 把 `list[idx]` 置 `nullptr`（[GreedyPatternRewriteDriver.cpp:270](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L270)），不搬移数组——删一个已死 op 极快。`pop()` 跳过尾部 nullptr。
>
> **持有形式 = 一个 `vector`（顺序容器）+ 一个 `DenseMap`（下标索引，实现去重与 O(1) 删除）。没有用 `std::set`，因为 LIFO + 随机置空比有序树更轻。**

### 3.3 主循环：pop → 死代码 → fold → 规则匹配（逐步标注）

`processWorklist`（[GreedyPatternRewriteDriver.cpp:436](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L436)）的主循环对一个 op 依次试四件事，命中任一就 `continue` 回到循环顶：

```cpp
// GreedyPatternRewriteDriver.cpp:457-633（精简为四步）
while (!worklist.empty()) {
  Operation *op = worklist.pop();                                  // Step A：取出一个 op

  // Step B：如果是"显然没用的死 op"（无结果且无副作用/结果无人用），直接删
  if (isOpTriviallyDead(op)) { rewriter.eraseOp(op); changed = true; continue; }

  // Step C：先尝试 fold（常量折叠/局部化简）。注意：ConstantLike 的 op 不 fold（否则无限折叠）
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
| **B · 死代码** | `isOpTriviallyDead` → `eraseOp` | 删 op（挂接③反向：摘出 block） | `notifyOperationErased` → 把它的操作数的定义 op 重塞工作表（`addOperandsToWorklist`，见下） |
| **C · fold** | `op->fold()` 尝试局部化简 | 可能原地改 op，或换结果 | 原地改 → `notifyOperationModified` → 重塞工作表；换结果走 replaceOp → 同 B |
| **D · 规则** | `matcher.matchAndRewrite(op, rewriter, ...)` 按序试你写的 pattern | 规则内部用 rewriter 改图 | 取决于规则调了什么（create/replace/erase），对应通知全部被驱动接住 |

#### 3.3.1 PatternApplicator：按 benefit 选规则

Step D 里那个 `matcher.matchAndRewrite(...)` 是 `PatternApplicator::matchAndRewrite`（[PatternApplicator.cpp:130](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L130)）。它的工作：

1. **按 op 名查表**：`patterns.find(op->getName())`（[PatternApplicator.cpp:145](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L145)）找到所有"匹配这个 op"的规则（一张 `DenseMap<OperationName, SmallVector<RewritePattern*>>`）。
2. **按 benefit 从高到低试**：规则在 `applyDefaultCostModel` 时已按 benefit 降序排好（[PatternApplicator.cpp:106](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L106) 的 `stable_sort`）。逐条调 `pattern->matchAndRewrite(op, rewriter)`（[PatternApplicator.cpp:212](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L212)），**一旦某条成功就 break**。
3. **失败则换下一条**：每试完一条（无论成败）就把它的游标 `++`，所以同一条规则对同一个 op 不会被重复试（[PatternApplicator.cpp:185](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L185)）。

> **benefit 是什么**（[PatternMatch.h:34-63](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L34)）：一个 `unsigned short`，0..65534（65535 是"永不匹配"哨兵）。**越大越先试**。你构造 pattern 时传，如 `OpRewritePattern(ctx, /*benefit=*/1)`。NorthStar 里 `BufferCastOpDeviceRegionFusion` benefit=100、`BufferCastOpFold` benefit=2（§7 会讲为什么一大一小）。详见 §6.3。

#### 3.3.2 改完之后：工作表怎么"滚雪球"

这是重写能"接力"（A 换完触发 B）的关键。驱动在三个 `notifyXxx` 回调里往工作表里塞东西：

```cpp
// GreedyPatternRewriteDriver.cpp:671-692（精简）
void notifyOperationInserted(Operation *op, ...) { addToWorklist(op); }   // 新建的 op → 入表
void notifyOperationModified(Operation *op)      { addToWorklist(op); }   // 原地改的 op → 入表
```

最有意思的是**删除时的"向上回溯"**（[GreedyPatternRewriteDriver.cpp:694-725](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L694) `addOperandsToWorklist`）：

```cpp
// GreedyPatternRewriteDriver.cpp:694-725（精简）—— 删 op 时，把它操作数的定义 op 也重塞工作表
void addOperandsToWorklist(Operation *op) {
  for (Value operand : op->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp) continue;                       // 块参数没有定义 op
    // ★ 启发式：只有当该 operand 的使用者 ≤ 2 个时，才把它的定义 op 入表
    if (该 operand 有超过 2 个使用者) continue;  // 见 :708-721
    addToWorklist(defOp);
  }
}
void notifyOperationErased(Operation *op) {
  addOperandsToWorklist(op);   // 删 op → 它的操作数可能少了一个 user，也许现在能被简化/删除
  worklist.remove(op);         // 同时把被删的 op 自己从工作表摘掉（避免悬空指针）
}
```

**为什么是"≤ 2 个使用者"才回溯**？源码注释（[GreedyPatternRewriteDriver.cpp:696-700](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L696)）说得很直白：删掉这个 op 后，那个 operand 最多只剩 1 个使用者了——可能变成 0 个使用者（→ 可删除），或只剩 1 个使用者（→ 可能有新的化简机会）。**如果它本来就有很多使用者，删一个不改变大局，就不必回溯**（避免工作表爆炸）。这是个工程上的启发式剪枝。

> ⚠️ **精度修正（对抗式验证发现）**：注意 `notifyOperationReplaced`（[GreedyPatternRewriteDriver.cpp:754-762](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L754)）**本身不动工作表**，它只是把通知转发给可选的 `config.listener`。replace 的工作表更新是**间接**发生的——因为默认 `replaceOp` 内部最终调了 `eraseOp`（见 §4），`eraseOp` 触发的 `notifyOperationErased` 才真正改工作表。别误以为"replace 直接刷新工作表"。

### 3.4 外层：反复迭代到收敛

`processWorklist` 跑空一次工作表只是"一轮"。真正收敛在 `RegionPatternRewriteDriver::simplify`（[GreedyPatternRewriteDriver.cpp:826-894](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L826)）的外层 `do-while`：

```cpp
// GreedyPatternRewriteDriver.cpp:826-894（精简）
LogicalResult simplify(...) && {
  bool continueRewrites = false;
  do {
    if (++iteration > config.maxIterations) break;   // 迭代上限（默认 10，防不收敛）
    worklist.clear();                                 // 每轮清空
    region.walk([&](Operation *op){ addToWorklist(op); });  // 把 region 里所有 op 重新入表
    continueRewrites = processWorklist();             // 跑这一轮
    continueRewrites |= succeeded(simplifyRegions(rewriter, region, ...));  // 顺带做区域化简
  } while (continueRewrites);                         // 只要这轮有变化，就再来一轮
  return success(!continueRewrites);                  // 上一轮无变化 = 收敛
}
```

**收敛（fixpoint）= "跑一轮，IR 一点没变"**。这正是上篇 §6.2 提到的 progressive lowering 的运转方式——规则反复套用，直到这组规则再也改不动为止。

---

## 4. 替换的本质：replaceOp = RAUW + erase

现在聚焦改图的两个原子动作。**它们全是 use-def 操作，用的就是上篇 Step 6 讲透的 `OpOperand` / `firstUse` / `removeFromCurrent` 机制**。本节就让你看到"替换"在数据结构层面到底动了什么。

### 4.1 replaceOp = replaceAllOpUsesWith + eraseOp

看默认实现（[PatternMatch.cpp:133-142](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp#L133)）：

```cpp
// PatternMatch.cpp:133-142
void RewriterBase::replaceOp(Operation *op, ValueRange newValues) {
  assert(op->getNumResults() == newValues.size() && "incorrect # of replacement values");
  replaceAllOpUsesWith(op, newValues);   // ① 把 op 所有结果的"使用者"改指到 newValues（RAUW）
  eraseOp(op);                           // ② 删掉 op 本身
}
```

而 `replaceAllOpUsesWith`（[PatternMatch.cpp:114-120](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp#L114)）先发"即将替换"通知，再对每个结果做 RAUW：

```cpp
// PatternMatch.cpp:114-120
void RewriterBase::replaceAllOpUsesWith(Operation *from, ValueRange to) {
  if (auto *L = dyn_cast_if_present<Listener>(listener))
    L->notifyOperationReplaced(from, to);        // ① 先通知（驱动据此转发给 config.listener）
  replaceAllUsesWith(from->getResults(), to);    // ② 真正改 use-def 边
}
```

**`replaceAllUsesWith` 在数据结构上做什么？** 就是上篇 §6 Part C 讲的那套——对结果的 use-list 逐个改 value：

```cpp
// UseDefLists.h:213（摘）—— IRObjectWithUseList::replaceAllUsesWith
template <typename ValueT>
void replaceAllUsesWith(ValueT &&newValue) {
  while (!use_empty())
    use_begin()->set(newValue);    // 每个 set() = removeFromCurrent(旧) + value=新 + insertIntoCurrent(新)
}
```

> **这正是上篇 Step 6 的 RAUW 机理**：`set()` 内部 = 从旧 value 的 use-list 摘掉自己（`removeFromCurrent`，即 `*back = nextUse` 那一行）+ 把 `value` 改成新 value + 把自己头插进新 value 的 use-list（`insertInto`）。**改 use-def 边 = 摘一根 `OpOperand`、改它的 `value` 字段、插到另一条链上。零额外分配。**

### 4.2 eraseOp：断言无残留使用者，再后序删除

`eraseOp`（[PatternMatch.cpp:161-231](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp#L161)）第一步是个**硬断言**——这是新手头号崩溃源：

```cpp
// PatternMatch.cpp:161-162
void RewriterBase::eraseOp(Operation *op) {
  assert(op->use_empty() && "expected 'op' to have no uses");   // ★ 结果还有人用？直接崩！
  ...
}
```

**所以正确的顺序永远是：先 `replaceOp`/`replaceAllUsesWith`（把使用者引走），再 `eraseOp`。** 绝不能对一个结果还有使用者的 op 直接 `eraseOp`。

接下来，如果 op 有嵌套 region，`eraseOp` 会**按后序（post-order，先删被支配者/后定义者）逐个删嵌套 op**（[PatternMatch.cpp:194-228](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp#L194) `eraseTree`），保证 listener 每次看到的都是"一致的 IR"；然后 `dropAllUses()` + `op->erase()`：

```cpp
// PatternMatch.cpp:184-189（摘，单个 op 的删除）
rewriteListener->notifyOperationErased(op);   // 通知（驱动据此更新工作表）
op->dropAllUses();                            // 显式断掉所有入边（图区域里可能有环）
op->erase();                                  // 从 block 的 iplist 摘除并释放
```

而 `dropAllUses`（[UseDefLists.h:202](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h#L202)）= `while(!use_empty()) use_begin()->drop()`，`drop()` 调 `removeFromCurrent()`——**又是上篇那套 use-list 摘除**。

> 🏗️ **数据结构 · eraseOp 在删除什么？**
> 1. **op 的入边（它作为操作数引用的 value）**：op 的 `OperandStorage`（尾部 `OpOperand[]`）析构时，每个 `OpOperand` 自动从对应 value 的 use-list 摘掉（上篇 Step 6 的"生命周期"）。
> 2. **op 的出边（它的结果被谁用）**：`dropAllUses()` 主动把结果 use-list 上的每个 `OpOperand` 摘掉。**但因为 §4.2 的断言，走到 `dropAllUses` 时 use-list 应该已经空了**（你先 RAUW 过了）——这一步是给"图区域（无 SSA 支配）"兜底。
> 3. **op 自身**：`op->erase()`（[Operation.cpp:539](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp#L539)）把它从 block 的 `iplist<Operation>` 摘除（挂接③反向），iplist 随即用 `destroy()` 释放这块尾分配内存。
>
> **一句话：erase = 摘入边（自动）+ 摘出边（dropAllUses）+ 出 block（iplist erase）。全部是上篇挂接③⑦的反向操作。**

### 4.3 「替换/删除」本质：一张总表（接续上篇 §4 的挂接表）

把本篇的"改图动作"和上篇的"挂接动作"对照——**重写就是反向拆除构建时的挂接**：

| 改图动作 | 反向拆除的是上篇哪类挂接 | 数据结构层面 | rewriter API |
|---|---|---|---|
| **改某条 use-def 边的指向** | 挂接⑦（Operand→Value）的部分修改 | `OpOperand.set(newValue)` = `removeFromCurrent` + 改 value + `insertInto` | `replaceAllUsesWith(from, to)` / `replaceUsesWithIf` |
| **把 op 的结果整体替换** | 改挂接⑦（所有引用旧结果的边）+ 删挂接⑥③（旧 op 的结果、op 自身） | RAUW 每条结果 + `eraseOp` | `replaceOp(op, newValues)` |
| **删一个死 op** | 挂接③（op 出 block）+ 挂接⑦（它的入边） | `dropAllUses` + `iplist::erase` | `eraseOp(op)` |
| **原地改 op 的属性/操作数** | 修改挂接②（属性）/ 挂接⑦（操作数指向） | 改字段后通知 | `modifyOpInPlace(op, lambda)`（[PatternMatch.h:629](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L629)） |
| **挪 op 到别的位置** | 重接挂接③（op→block） | `iplist::splice` + 通知 | `moveOpBefore/After` |

> **核心结论（和上篇呼应）**：上篇说"`OpBuilder::create<>` 一次完成若干挂接"；本篇说**"`rewriter.replaceOp/eraseOp` 一次反向拆除若干挂接"**。构建和重写共用同一套 use-def / iplist 原语，只是方向相反。这就是为什么理解了上篇，本篇的"改图"几乎没有新数据结构要学。

---

## 5. 一个完整例子逐步追踪（Toy `transpose(transpose(x)) → x`）

把前几节串起来，逐帧看 `SimplifyRedundantTranspose` 怎么改掉 §2 那段 IR。

### 5.0 起点：IR + 工作表初始状态

```mlir
func.func @transpose_transpose(%arg0 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %t0 = toy.transpose %arg0 : tensor<2x3xf32>
  %t1 = toy.transpose %t0 : tensor<2x3xf32>
  return %t1 : tensor<2x3xf32>
}
```

驱动首轮把 region 里所有 op 入表（默认 postorder，[GreedyPatternRewriteDriver.cpp:854](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L854)）。use-def 初始状态（接续上篇 §6 Part B 的画法）：

```
%arg0 : def=块参数   uses={transpose0}   firstUse → [OpOperand{owner=transpose0,value=%arg0}]
%t0  : def=transpose0   uses={transpose1}  firstUse → [OpOperand{owner=transpose1,value=%t0}]
%t1  : def=transpose1   uses={return}      firstUse → [OpOperand{owner=return,value=%t1}]

工作表（LIFO，postorder 入表后约）：[ return, transpose1, transpose0 ]   ← 栈顶在右
```

### 5.1 处理 `transpose1`（命中 `SimplifyRedundantTranspose`）

假设栈顶 pop 出 `transpose1`（`return`、`transpose0` 这两个 op 没有匹配的规则，pop 出来试 fold/规则都失败，跳过）。驱动调 `matcher.matchAndRewrite(transpose1, rewriter, ...)`，进入我们的 pattern（[ToyCombine.cpp:38-52](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch4/mlir/ToyCombine.cpp#L38)）：

**① match（沿 use-def 往上看）**：
```cpp
Value transposeInput = op.getOperand();                       // op=transpose1，操作数 = %t0
TransposeOp transposeInputOp =
    transposeInput.getDefiningOp<TransposeOp>();              // %t0 的定义 op = transpose0，是 TransposeOp ✓
if (!transposeInputOp) return failure();                      // 命中！不 return
```
> **这一步纯 use-def**：`getOperand()` 拿 op 尾部的 `OpOperand[0].value`；`getDefiningOp<>()` 顺着这个 value 找它的定义 op（`%t0` 是 `OpResult`，靠地址算术反查到 transpose0）。匹配 = 沿上篇织好的 use-def 链走两步。

**② rewrite（`replaceOp` → RAUW + erase）**：
```cpp
rewriter.replaceOp(op, {transposeInputOp.getOperand()});      // op=transpose1，换成 {%arg0}
return success();
```
`replaceOp(transpose1, {%arg0})` 内部做（§4.1）：
- `replaceAllOpUsesWith(transpose1, {%arg0})` → 对 `transpose1` 的唯一结果 `%t1` 的每个使用者做 RAUW。`%t1` 的使用者是 `return`（`OpOperand{owner=return, value=%t1}`），`set(%arg0)` 把这条边改成 `OpOperand{owner=return, value=%arg0}`——**`return` 现在用 `%arg0` 而非 `%t1`**。
  - 此过程中 `return` 被原地改 → `notifyOperationModified(return)` → 驱动把 `return` 重塞工作表。
- `eraseOp(transpose1)` → 此时 `%t1` 已无使用者（return 被引走了），断言 `use_empty()` 通过 → 摘掉 transpose1 的入边（`OpOperand{value=%t0}` 从 `%t0` 的 use-list 摘除）→ `op->erase()` 把 transpose1 出 block。
  - `notifyOperationErased(transpose1)` → `addOperandsToWorklist`：transpose1 的操作数是 `%t0`，`%t0` 现在的使用者数？原本只有 transpose1，删后变 **0 个**（≤2 ✓）→ 把 `%t0` 的定义 op `transpose0` 重塞工作表。

**改完后的 use-def + 工作表**：

```
%arg0 : uses={return}        ← ★ 边改了：原本 {transpose0}，现在 return 也直接用 %arg0
%t0  : uses={} (空！)         ← ★ transpose1 删了，%t0 没人用了
%t1  : （随 transpose1 一起消失）

IR：
  %t0 = toy.transpose %arg0   ← 还在，但成孤儿（结果无人用）
  return %arg0                ← ★ 改用 %arg0

工作表：[..., transpose0(重塞), return(重塞)]   ← 两个受影响的 op 回到表里
```

### 5.2 处理 `transpose0`（死代码，被 Step B 删掉）

下一轮 pop 出 `transpose0`。它没有匹配的规则，但 **Step B 的 `isOpTriviallyDead` 命中**——`%t0` 已无使用者、transpose 无副作用 → `rewriter.eraseOp(transpose0)`：

- `eraseOp` 断言 `%t0.use_empty()` ✓ → 摘入边（`OpOperand{value=%arg0}` 从 `%arg0` use-list 摘除）→ 出 block。
- `notifyOperationErased` → `addOperandsToWorklist`：操作数 `%arg0`，`%arg0` 现在有 1 个使用者（return），≤2 ✓ → `%arg0` 是块参数没有定义 op（`defOp == nullptr`），不入表（[GreedyPatternRewriteDriver.cpp:704-706](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L704)）。

**最终 IR**：

```mlir
func.func @transpose_transpose(%arg0 : tensor<2x3xf32>) -> tensor<2x3xf32> {
  return %arg0 : tensor<2x3xf32>
}
```

工作表继续跑，剩下的 `return` 等都改不动了 → 收敛。**两个 transpose 消失，这正是 §2 的目标。**

### 5.3 这一例的"接力"链

注意整件事是**两条规则接力**完成的：

```
SimplifyRedundantTranspose(transpose1)  →  改 return 的 use-def 边 + 删 transpose1
        ↓ （notifyOperationErased 把 transpose0 重塞工作表，因 %t0 变成 ≤2 使用者）
isOpTriviallyDead(transpose0)           →  删 transpose0（它因上一步变成孤儿）
```

**第一条规则删 transpose1，"顺手"让 transpose0 变成死代码；驱动通过 listener 听见删除、把 transpose0 重塞工作表；第二轮 Step B 自动清掉它。** pattern 作者完全不需要知道"删完会让别的 op 变死"——驱动的工作表机制替你兜了。这就是 §3.1 "driver-as-listener" 的威力。

---

## 6. 设计契约与陷阱

### 6.1 契约①：match-before-mutate（先确认能换，再动手）

**这是 matchAndRewrite 最重要的一条规矩**：在 `matchAndRewrite` 里，**一旦你调了任何会改图的 rewriter 方法（create/replace/erase/modifyOpInPlace），就视为你已"承诺"本次一定成功**。不要"改了一半发现不行又 `return failure()`"。

为什么？看下一条。

### 6.2 契约②：greedy 驱动没有回滚（这是它和 CH-12 的本质区别）

`PatternRewriter` 唯一新增的钩子 `canRecoverFromRewriteFailure()` **默认返回 `false`**（[PatternMatch.h:794](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L794)）。含义是：**如果 pattern 改了图又返回失败，驱动无法撤销**——因为 greedy 用的 `replaceOp/eraseOp` 是立即生效的（§4），没留撤销日志。

对比一下 CH-12 会登场的 **`ConversionPatternRewriter`**（dialect conversion 框架，[DialectConversion.h:723-726](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/Transforms/DialectConversion.h#L723)）：

| | greedy `PatternRewriter`（本篇，CH-9） | `ConversionPatternRewriter`（CH-12） |
|---|---|---|
| `canRecoverFromRewriteFailure()` | **`false`** | **`true`** |
| 改图方式 | 立即生效、不记账 | 记"动作日志"，可整体回滚（undo） |
| pattern 改一半失败 | IR 处于半改状态（bug） | 自动回滚到改之前 |
| 适用 | 局部化简、peephole、同方言优化 | 跨方言 lowering（需要"全做或全不做"的原子性） |

> **所以 §6.1 的规矩可以这样理解**：greedy 驱动把"无回滚"作为既定约束，把责任交给 pattern——**你要么先把条件查清楚再动手（推荐两步 `match`/`rewrite` 写法，见 §7.1，结构性保证不会"改了又失败"），要么在 `matchAndRewrite` 里把所有可能失败的前置检查放在任何 rewriter 调用之前**。这是把"一个看似的限制"变成了"一条清晰的纪律"。
>
> **(承上启下)**：等到了 NorthStar CH-12（Conversion），你会看到一个**有回滚**的 rewriter——同样的 pattern 写法，但底层记了 undo 日志，pattern 可以放心地"边试边改"。本篇先把无回滚的 greedy 讲透，正是为了让你到 CH-12 时能体会"为什么 conversion 要另造一个 rewriter"。

#### 6.2.1 失败的正确姿势：notifyMatchFailure

放弃匹配时，**不要写 `return failure()`**，要写：

```cpp
return rewriter.notifyMatchFailure(op, "operands not all from the same defining op");
// 或带详细诊断的回调形式
return rewriter.notifyMatchFailure(op, [&](Diagnostic &d){ d << "..."; });
```

`notifyMatchFailure`（[PatternMatch.h:716-740](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L716)）做两件事：① 把失败原因（懒构造的诊断）转发给 listener（驱动据此在 `-debug` 下打印"为什么这条规则没套上"，[GreedyPatternRewriteDriver.cpp:764-773](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L764)）；② **返回 `failure()` 给你**。所以它等价于 `return failure()` 但额外送你一条可观测的诊断。

> **这是从第一天就该养成的调试习惯**：当你的 pattern 在 CH-13 的 lit 测试里"该匹配却没匹配"时，`-debug-only=greedy-rewriter` 配合 `notifyMatchFailure` 能直接告诉你每条规则在每个 op 上失败的原因。NorthStar 的 `DeviceRegionFusion.cpp:187` 用的是裸 `return llvm::failure()`——能跑，但丢了诊断，可当反例对照。

### 6.3 契约③：benefit 是静态的、决定尝试顺序

- **范围**：0..65534，65535 是 `impossibleToMatch` 哨兵（[PatternMatch.h:35](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L35)）。**越大越先试**。
- **静态**：一个 pattern 实例的 benefit 在构造时定死，**不会随 IR 变化**。源码注释（[PatternMatch.h:118-123](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L118)）说：若一条规则的"收益"取决于运行时 IR，你应该把它拆成多个 pattern 实例（不同 benefit）、用不同的前置谓词守护。
- **平手按注册顺序**：benefit 相同时，`stable_sort` 保持注册顺序（[PatternApplicator.cpp:106](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L106)）——别依赖语义无关的顺序。

> **NorthStar 为什么 100 vs 2**（[DeviceRegionFusion.cpp:241/248](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L241)）：`BufferCastOpDeviceRegionFusion`（benefit=100，把算子融合进 device kernel）和 `BufferCastOpFold`（benefit=2，化简 buffer_cast）。看起来"融合"收益更大该先做，但其实两个 pattern 在 Pass 里是**分两次** `applyPatternsAndFoldGreedily` 跑的（[DeviceRegionFusion.cpp:270/280](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L270)），benefit 在各自集合内才比。benefit 主要用于"同一批里多条规则都匹配同一个 op 时谁先上"。

### 6.4 契约④：改图必须走 rewriter（别绕过）

在 pattern 里**永远不要**直接调 `op->erase()`、`op->setAttr(...)`、`op->setOperand(...)`——它们不发通知，驱动的工作表会失同步（指向已删 op、或漏掉该重访的 op）。要用：

| 想做的事 | ✅ 走 rewriter | ❌ 别直接调 |
|---|---|---|
| 删 op | `rewriter.eraseOp(op)` | `op->erase()` |
| 改 op 属性/操作数（原地） | `rewriter.modifyOpInPlace(op, [&]{ op->setAttr(...); })` | `op->setAttr(...)` |
| 建 op | `rewriter.create<>(...)` | `OpBuilder`（除非你手动挂 listener） |

`modifyOpInPlace`（[PatternMatch.h:629](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L629)）是 `startOpModification` + 你的 lambda + `finalizeOpModification` 的 RAII 包装，`finalizeOpModification` 会发 `notifyOperationModified`（[PatternMatch.cpp:248](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp#L248)）——所以原地改的 op 会正确地被重塞工作表。

### 6.5 终止性：两条"不会无限循环"的保险

pattern 可能"换出来的 op 又匹配自己"（如 `reshape(reshape(x))→reshape(x)` 换完还是 reshape）。MLIR 用两条保险防止无限循环：

1. **driver 层**：`config.maxIterations`（默认 10，[GreedyRewriteConfig:66](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h#L66)）限制外层 do-while 轮数；`config.maxNumRewrites`（默认无限制）限制总重写次数。NorthStar `DeviceRegionFusion.cpp:268` 显式设了 `maxIterations = 10`。
2. **pattern 层**：若你的 pattern 确实会递归套用自己的输出、但**已知有界**，可在 pattern 里调 `setHasBoundedRewriteRecursion()`（[PatternMatch.h:202](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L202)）告诉驱动"放心，我会收敛"。否则驱动对某些自递归会保守处理。

---

## 7. NorthStar 实战

把目光收回你正在学的 NorthStar 教程（CH-9 起引入 Rewrite Pattern）。你打开的那个文件 [DeviceRegionFusion.cpp](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp) 恰好同时含两种写法，是绝佳的对照样本。

### 7.1 两步写法：`BufferCastOpFold`（推荐初学者）

[DeviceRegionFusion.cpp:203-234](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L203)——**分别 override `match` 和 `rewrite`**：

```cpp
// DeviceRegionFusion.cpp:203-234（精简）
struct BufferCastOpFold : public OpRewritePattern<north_star::BufferCastOp> {
  LogicalResult match(BufferCastOp op) const {           // ← 只判断，不动 IR
    Operation *above_cast = nullptr;
    for (auto [i, operand] : enumerate(op->getOperands())) {
      if (isa<BlockArgument>(operand)) return failure();               // 操作数是块参数 → 不匹配
      if (!above_cast) above_cast = operand.getDefiningOp();
      else if (operand.getDefiningOp() != above_cast) return failure(); // 操作数不全来自同一 op
      if (!above_cast->getResult(i).hasOneUse()) return failure();      // 必须唯一使用者
    }
    return success();
  }
  void rewrite(BufferCastOp op, PatternRewriter &rewriter) const {    // ← 只改，假设已 match
    Operation *above_cast = op->getOperand(0).getDefiningOp();
    for (auto [i, res] : enumerate(op->getResults()))
      rewriter.replaceAllUsesWith(res, above_cast->getOperand(i));   // ★ RAUW 每个结果
    rewriter.eraseOp(op);                                            // ★ 先引走使用者，再删
    rewriter.eraseOp(above_cast);                                    // ★ above_cast 现在也无使用者了
  }
};
```

**为什么这个顺序正确**（呼应 §4.2 断言）：
1. 先 `replaceAllUsesWith` 每个 `res`——把所有引用 `buffer_cast` 结果的边改指向 `above_cast` 的对应操作数；
2. 此时 `op`（buffer_cast）`use_empty()` ✓，才 `eraseOp(op)`；
3. `above_cast` 的结果原本只被 `op` 用（`match` 里 `hasOneUse` 保证了），`op` 删了它也就 `use_empty()`，才能 `eraseOp(above_cast)`。

> **两步写法的安全性**：基类默认 `matchAndRewrite` 是"match 成功才调 rewrite"（[PatternMatch.h:342-349](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L342)）。所以**你物理上无法"在 rewrite 里改了图、却因为 match 失败而回退"——match 失败时 rewrite 根本不进**。这天然满足 §6.1 的 match-before-mutate 契约，是初学者最稳的写法。

### 7.2 单步写法：`BufferCastOpDeviceRegionFusion`

[DeviceRegionFusion.cpp:170-201](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L170)——**一个 `matchAndRewrite` 搞定**：

```cpp
// DeviceRegionFusion.cpp:170-201（精简）
struct BufferCastOpDeviceRegionFusion : public OpRewritePattern<BufferCastOp> {
  LogicalResult matchAndRewrite(BufferCastOp op, PatternRewriter &rewriter) const override {
    SmallVector<SetVector<Operation*>> op_list;
    for (auto res : op->getResults()) {
      rewriter.setInsertionPointAfterValue(res);
      SetVector<Operation*> ops;
      for (auto use : res.getUsers()) addops(ops, use);   // 递归收集"使用链"上的 DistributeParallelOp
      if (!ops.empty()) op_list.push_back(ops);
    }
    if (op_list.empty()) return failure();                 // ← 注意：这里用了裸 failure()（§6.2.1 反例）
    for (auto ops : op_list) FusionOps(rewriter, ops.takeVector(), op->getLoc());  // ★ 你选中的那行！
    return success();
  }
};
```

`FusionOps`（[DeviceRegionFusion.cpp:111-168](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L111)）干一件更重的事：把一串算子**克隆进一个新建的 `func::FuncOp`（device kernel），在原位插一个 `func::CallOp`**——这就是 [device-region-fusion.mlir](../MLIR-Tutorial/11-lit_for_test/test/NorthStar/device-region-fusion.mlir) 测试里"6 个 softmax 被装进 `device_kernel`、main 里换成 3 个 `call`"的由来。它用 `rewriter.create<func::FuncOp>` 建 kernel、`rewriter.replaceAllUsesWith` 把原算子结果接到 call 结果——**全程走 rewriter，所以新建的 FuncOp/CallOp 自动入工作表、被删的算子自动触发回溯**。

> **你光标停在第 189 行的 `takeVector()`**：`ops` 是个 `SetVector`（去重 + 有序），`.takeVector()` 把它"搬空成一个 `SmallVector`"传给 `FusionOps`。这跟 matchAndRewrite 机制无关，是 LLVM 容器的用法，但它出现在这里正说明：pattern 里大量用 use-def 导航（`res.getUsers()` 顺 `firstUse` 链遍历）+ 集合运算来"圈出要融合的子图"，圈完后交给 rewriter 动刀。

### 7.3 Pass 集成：从 pattern 到驱动

pattern 写完，要在 Pass 里"装弹 + 开火"。看 [DeviceRegionFusionPass::runOnOperation](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L259)：

```cpp
// DeviceRegionFusion.cpp:259-286（精简）
void DeviceRegionFusionPass::runOnOperation() {
  // ① 先跑 buffer_cast 的化简规则（benefit=2 那条）
  RewritePatternSet buffer_cast_patterns(&getContext());
  populateBufferCastOpCanonicalizationPatterns(buffer_cast_patterns);   // 把 pattern 塞进集合
  GreedyRewriteConfig cfg; cfg.maxIterations = 10; cfg.useTopDownTraversal = true;
  applyPatternsAndFoldGreedily(getOperation(),                          // ② 冻结 + 开火（§3 的驱动）
      FrozenRewritePatternSet(std::move(buffer_cast_patterns)), cfg);

  // ③ 再跑融合规则（benefit=100 那条）
  RewritePatternSet patterns(&getContext());
  populateDeviceRegionFusionPatterns(patterns);
  applyPatternsAndFoldGreedily(getOperation(),
      FrozenRewritePatternSet(std::move(patterns)), GreedyRewriteConfig{}, &changed);
}
```

四步范式（所有基于 greedy driver 的 Pass 都长这样）：

| 步 | 做什么 | 涉及的类 |
|---|---|---|
| ① 建 pattern 集合 | `RewritePatternSet patterns(ctx);` 然后 `populateXxxPatterns(patterns);` | `RewritePatternSet`（[PatternMatch.h:808](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L808)） |
| ② 冻结 | `FrozenRewritePatternSet frozen(std::move(patterns));` | `FrozenRewritePatternSet`（不可变快照，驱动拿的是它） |
| ③ 配置 | `GreedyRewriteConfig cfg;`（maxIterations / useTopDownTraversal / strictMode / scope） | `GreedyRewriteConfig`（[GreedyPatternRewriteDriver.h:43](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h#L43)） |
| ④ 开火 | `applyPatternsAndFoldGreedily(op, frozen, cfg);` | 进入 §3 的驱动主循环 |

> **`populateXxxPatterns` 是什么**：约定俗成的"把一族 pattern 注册进集合"的辅助函数（[DeviceRegionFusion.cpp:237-249](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L237)），内部就是 `patterns.addWithLabel<BufferCastOpDeviceRegionFusion>("...", ctx, 100);`（`add` 系列方法见 [PatternMatch.h:847](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L847)）。`add<T>` 会调 `RewritePattern::create<T>`（顺带调 pattern 的 `initialize()` 若有，[PatternMatch.h:275-285](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L275)）。

---

## 8. 延伸

### 8.1 fold 先于 pattern：更便宜的"局部化简"

§3.3 Step C 里，**每个 op 在套用 rewrite pattern 之前，会先被 `fold` 试一遍**（[GreedyPatternRewriteDriver.cpp:491-563](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L491)）。fold 和 pattern 是两种不同的化简机制：

| | `fold()` | rewrite pattern |
|---|---|---|
| 写在哪 | Op 的成员（ODS 生成骨架，如 `MyOp::fold(FoldAdaptor)`） | 独立的 `OpRewritePattern` |
| 输入/输出 | 看 op 自身 + 操作数的"已知值"，返回 `OpFoldResult`（Attribute 或 Value） | 任意逻辑，用 rewriter 改图 |
| 改图能力 | 弱：只能"把结果换成常量/已知值"或"原地小改" | 强：能建任意新 op、重组结构 |
| 在循环里的位置 | **先**（[GreedyPatternRewriteDriver.cpp:493](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L493)） | 后（[GreedyPatternRewriteDriver.cpp:614](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L614)） |
| 专属陷阱 | `ConstantLike` 的 op 不 fold（否则常量折叠成 Attribute 又被物化成常量 op，无限循环，[GreedyPatternRewriteDriver.cpp:491](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp#L491) 注释） | 自递归要声明 bounded（§6.5） |

> **NorthStar CH-14（fold/canonicalize）专门讲 fold**，本篇只定位它为"更便宜、更局部、先于 pattern 跑的兄弟"。两者在同一驱动里协作：fold 搞不定的，才轮到 rewrite pattern。

### 8.2 同一条规则的三种写法：C++ / DRR / PDL

Toy 教程里同一条 `reshape(reshape(x)) → reshape(x)` 规则，展示了三种表达力递增的写法：

**① 手写 C++（本篇主线）**：`OpRewritePattern` + `matchAndRewrite`（如 `SimplifyRedundantTranspose`）。最灵活，能写任意控制流（如 NorthStar 的 `FusionOps` 克隆整个 region）。

**② DRR（Declarative Rewrite Rules，TableGen）**——[ToyCombine.td:33-34](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch4/mlir/ToyCombine.td#L33)：

```tablegen
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```

一行 `Pat<左模式, 右模式>`，`mlir-tblgen` 生成等价的 C++ `OpRewritePattern`（编译进 [ToyCombine.inc](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch4/mlir/ToyCombine.cpp#L23)，在 `ToyCombine.cpp:22-24` 被 include）。**结构化的 1:1 替换用 DRR 极简洁**，还支持 `NativeCodeCall`（内联 C++）和 `Constraint`（条件约束），见 [ToyCombine.td:44-61](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch4/mlir/ToyCombine.td#L44)。

**③ PDL（Pattern Definition Language）**：一种运行时解释的小字节码，能描述更动态的多 root 匹配。`PatternApplicator` 甚至会**先匹配 PDL 字节码、再和 C++ pattern 一起按 benefit 交错**（[PatternApplicator.cpp:138-141](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp#L138)）。

> **何时用哪种**：结构 1:1 替换 → DRR（最省）；需要运行时动态匹配多 op → PDL；需要任意控制流/克隆 region/复杂判定 → 手写 C++。三者最终都变成 `RewritePattern`，被同一套 greedy driver 调度。

### 8.3 与上篇、后续章节的衔接

```
上篇 CH-5~7：建 IR（OpBuilder::create → 七类挂接）        ← 已掌握
本篇 CH-9：  改 IR（matchAndRewrite → RAUW + erase，driver-as-listener）  ← 你在这里
CH-10~11：  ns-opt 工具 + lit/FileCheck 测试（验证 pattern 效果，device-region-fusion.mlir 即此）
CH-12：     Conversion（ConversionPatternRewriter——有回滚的 rewriter，跨方言 lowering）
CH-13：     PassManager（把多个 Pass 串成流水线）
CH-14：     fold / canonicalize（§8.1 的 fold 正篇）
CH-15：     落到 LLVM IR
```

本篇讲透的"`PatternRewriter` 无回滚、`replaceOp`=RAUW+erase、driver-as-listener"是 CH-12 的对照基线——到 CH-12 你会看到"同一个 pattern 接口，底层换成了带 undo 日志的 `ConversionPatternRewriter`"，那时回头看 §6.2 这张对比表，整个 progressive lowering 的设计就全通了。

---

## 附录 A：Rewriter API 速查

| 你想做的事 | 调用 | 出处 |
|---|---|---|
| 建新 op（自动插当前位置） | `rewriter.create<OpTy>(loc, args...)` | 继承自 `OpBuilder`，[Builders.h:508](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L508) |
| 建 + 换一步到位 | `rewriter.replaceOpWithNewOp<NewOpTy>(oldOp, args...)` | [PatternMatch.h:535](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L535) |
| 把 op 的结果换成一组值（并删 op） | `rewriter.replaceOp(op, newValues)` | [PatternMatch.h:525](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L525) |
| 把一个 value 的所有使用者改指（不删任何 op） | `rewriter.replaceAllUsesWith(from, to)` | [PatternMatch.h:638](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L638) |
| 条件替换部分使用者 | `rewriter.replaceUsesWithIf(from, to, pred)` | [PatternMatch.h:670](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L670) |
| 删一个 use_empty 的 op | `rewriter.eraseOp(op)` | [PatternMatch.h:543](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L543) |
| 原地改 op（属性/操作数） | `rewriter.modifyOpInPlace(op, [&]{...})` | [PatternMatch.h:629](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L629) |
| 合并/移动 block | `rewriter.mergeBlocks` / `inlineBlockBefore` / `moveOpBefore` | [PatternMatch.h:557-606](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L557) |
| 放弃匹配（带诊断） | `return rewriter.notifyMatchFailure(op, "reason")` | [PatternMatch.h:716](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h#L716) |
| 保存/恢复插入点 | `OpBuilder::InsertionGuard g(rewriter);` | [Builders.h:351](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/Builders.h#L351) |

---

## 附录 B：关键源码索引

| 文件 | 内容 | 重要行号 |
|---|---|---|
| [`PatternMatch.h`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/PatternMatch.h) | 两个家族的全部声明 | :34-63（PatternBenefit），:73-231（Pattern），:246-312（RewritePattern + match/rewrite/matchAndRewrite 默认），:318-389（OpRewritePattern/Interface/Trait），:400-512（RewriterBase + Listener），:522-707（改图 API），:716-740（notifyMatchFailure），:766-795（IRRewriter/PatternRewriter），:808-1009（RewritePatternSet）|
| [`PatternMatch.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/PatternMatch.cpp) | 改图 API 的默认实现 | :114-128（replaceAllOpUsesWith），:133-157（replaceOp = RAUW + erase），:161-231（eraseOp + 后序 eraseTree），:248（finalizeOpModification）|
| [`GreedyPatternRewriteDriver.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp) | THE 贪心驱动 | :198-273（Worklist），:322（driver IS-A Listener），:411-434（ctor + setListener(this)），:436-633（processWorklist 主循环），:694-725（addOperandsToWorklist ≤2 启发式），:754-762（notifyOperationReplaced 不动工作表），:826-894（外层 do-while 收敛），:896-925（applyPatternsAndFoldGreedily）|
| [`PatternApplicator.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/Rewrite/PatternApplicator.cpp) | 按 benefit 选规则 | :56-115（applyCostModel 排序），:130-240（matchAndRewrite：按名查表 + 按 benefit 交错 + 逐条试）|
| [`UseDefLists.h`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/IR/UseDefLists.h) | use-list 原语（承上篇） | :202（dropAllUses），:213（replaceAllUsesWith），:96-103（insertInto），:86-93（removeFromCurrent）|
| [`Operation.cpp`](../MLIR-Tutorial/third_party/llvm-project/mlir/lib/IR/Operation.cpp) | Operation::erase | :539-544（erase = 出 iplist + destroy）|
| Toy [`ToyCombine.cpp` (Ch4)](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch4/mlir/ToyCombine.cpp) | matchAndRewrite 教科书例 | :28-53（SimplifyRedundantTranspose），:57-68（getCanonicalizationPatterns 注册）|
| Toy [`ToyCombine.td` (Ch4)](../MLIR-Tutorial/third_party/llvm-project/mlir/examples/toy/Ch4/mlir/ToyCombine.td) | DRR 声明式写法 | :33-61（Pat / NativeCodeCall / Constraint）|
| NorthStar [`DeviceRegionFusion.cpp` (CH-9]](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp) | split + combined 两种写法 + Pass | :170-201（单步 matchAndRewrite：融合），:203-234（两步 match/rewrite：fold），:237-249（populate*），:259-286（Pass 集成）|
| NorthStar [`device-region-fusion.mlir` (CH-11]](../MLIR-Tutorial/11-lit_for_test/test/NorthStar/device-region-fusion.mlir) | 融合前后的 lit 测试 | :11-25（softmax 链 → device_kernel + call）|
| [`GreedyPatternRewriteDriver.h`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h) | GreedyRewriteConfig | :43-90（maxIterations/maxNumRewrites/useTopDownTraversal/strictMode/scope）|
| [`DialectConversion.h`](../MLIR-Tutorial/third_party/llvm-project/mlir/include/mlir/Transforms/DialectConversion.h) | CH-12 对照 | :723-726（ConversionPatternRewriter canRecoverFromRewriteFailure=true，有回滚）|

---

## 参考资料

- [MLIR 官方文档 — Pattern Rewriting / Greedy Pattern Rewrite Driver](https://mlir.llvm.org/docs/PatternRewriter/)
- [MLIR 官方文档 — Declarative Rewrite Rules (DRR)](https://mlir.llvm.org/docs/DeclarativeRewrites/)
- [MLIR Toy 教程 — Chapter 4: Enriching Toy with Transformations](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/)（canonicalization patterns / SimplifyRedundantTranspose 出处）
- 续集关系：上篇 [从零构建一棵 MLIR IR 树：AST → Module → 挂接 → 遍历 lowering](MLIR-IR-树的构建过程教程_精品.md)
- [LLVM 19.1.7 源码（本仓库 submodule）](../MLIR-Tutorial/third_party/llvm-project/mlir/)
- NorthStar Tutorial (violetDelia/MLIR-Tutorial, Apache 2.0) —— CH-9 起
