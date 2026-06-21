# 第 13 章 综合：BufferCastOpFold 闭环

> **本章位置**　全书压轴。前十二章分别讲了 IR 的各个机制——结构（Ch2-3）、数据流边（Ch4）、遍历（Ch5）、构建（Ch6-8）、重写（Ch9-11）、序列化（Ch12）。本章把这些机制串成一个真实的优化 pass：**BufferCastOpFold**——一个冗余 cast 链折叠，把 walk → match → RAUW → erase 串成完整的"发现→变换→清理"闭环。
>
> **前置依赖**　全书。本章综合运用 Ch4（use-def）、Ch5（walk/遍历）、Ch9（matchAndRewrite）、Ch10（RAUW/erase）、Ch11（driver/工作表）。
>
> **编译原理切入**　本章从**端到端优化 pass 的全景**立论。一个真实优化（冗余 cast 折叠）如何把发现（walk/match）、变换（RAUW）、清理（erase）串成闭环——这正是编译器优化 pass 的标准结构，对应 Dragon Book 第 8.5 节的公共子表达式消除与规范化的工程化实现。本章回扣全书所有机制，让读者看到它们如何协作。

---

## 13.1 编译器优化 pass 的标准结构

Dragon Book 第 8-9 章描述的优化 pass 都遵循一个共同结构：

```text
发现（detect）  →  变换（transform）  →  清理（cleanup）
```

- **发现**：遍历 IR，识别可优化的模式（如公共子表达式、死代码、冗余运算）。
- **变换**：用某种等价改写把识别出的模式换成更优的形式。
- **清除**：删除变换产生的死代码，整理 IR。

MLIR 的优化 pass 完美对应这个结构——发现用 walk/遍历（Ch5）+ match（Ch9），变换用 rewriter 改图（Ch10），清理用 erase + 工作表自动回溯（Ch10-11）。本章用一个具体例子（BufferCastOpFold）把这个闭环展示出来。

## 13.2 场景：冗余的 cast 链

NorthStar 教程 CH-9/CH-14 有一个贯穿匹配-重写-删除的经典案例——`BufferCastOpFold`。场景是一段冗余的 cast→cast 链：

```mlir
// 输入：冗余的 cast→cast 链条
%t0 = "north_star.buffer_cast"(%a) : (!ns.tensor<...>) -> !ns.tensor<...>
%out = "north_star.buffer_cast"(%t0) : (!ns.tensor<...>) -> !ns.tensor<...>
                                          ↑ 这个 cast 是冗余的（输入就是上面 cast 的输出）
```

这种模式在 lowering 过程中很常见——不同阶段插入的 buffer_cast 可能叠加成冗余链。BufferCastOpFold 的任务是把这种冗余链折叠掉。

源码在 [DeviceRegionFusion.cpp:203-234](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L203)（CH-9）和 [NorthStarCanonicalize.cpp:53-82](../MLIR-Tutorial/14-fold_and_canonicalization/src/Dialect/NorthStar/IR/NorthStarCanonicalize.cpp#L53)（CH-14）。

## 13.3 逐步走读：match → rewrite → driver 调度

### 13.3.1 第一步：match() — 遍历 use-def 链检查前置条件

`BufferCastOpFold` 采用两步写法（Ch11 §11.3.1 推荐的初学者写法）——分别 override `match`（只判断）和 `rewrite`（只改）。match 阶段（[DeviceRegionFusion.cpp:203-234](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L203)）：

```cpp
LogicalResult BufferCastOpFold::match(BufferCastOp op) const {
  // ① 遍历每个 operand，验证它们来自同一个 defining op（而不是 BlockArgument）
  Operation *above_cast = nullptr;
  for (OpOperand &operand : op->getOpOperands()) {
    // ← 通过 use-def 的反向：operand.get() → value → getDefiningOp()
    Operation *defOp = operand.get().getDefiningOp();
    if (!defOp || (above_cast && defOp != above_cast))
      return failure();               // operand 来自不同 op 或直接来自 BlockArgument
    above_cast = defOp;
  }

  // ② 类型匹配
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    OpOperand &aboveOperand = above_cast->getOpOperand(index);
    if (result.getType() != aboveOperand.get().getType())
      return failure();
  }

  // ③ 每个中间结果只有一个 use（否则跨越会破坏另外的边）
  for (OpResult result : above_cast->getResults()) {
    if (!result.hasOneUse())          // ← 遍历 uses 检查
      return failure();
  }

  return success();
}
```

`match()` 中的图操作全是**数据流遍历**（Ch4-5）：
- `getDefiningOp()`（operand → value → 定义 op）：O(1)，UD chain 方向。
- `hasOneUse()`（value → 检查 firstUse 链表恰好一个元素）：O(1)，DU chain 方向。
- `getOpOperand(index)`（op → 通过 OperandStorage 的尾分配数组）：O(1)。

这三步验证确认"两个 cast 能安全折叠"——操作数同源、类型匹配、中间结果唯一使用（折叠不会破坏其他边）。

> **match 本质是沿 use-def 链导航**：第 9 章讲过，match 阶段不改图，只沿着 use-list 读。这一步全部是"看"——`getDefiningOp`（UD）、`hasOneUse`（DU）、`getOpOperand`（读 operand 数组）。没有任何写入。

### 13.3.2 第二步：rewrite() — RAUW + erase

rewrite 阶段（[DeviceRegionFusion.cpp:203-234](../MLIR-Tutorial/9-rewrite_pattern/src/Dialect/NorthStar/Transforms/DeviceRegionFusion.cpp#L203)）：

```cpp
void BufferCastOpFold::rewrite(BufferCastOp op,
                                PatternRewriter &rewriter) const {
  Operation *above_cast = op->getOperand(0).getDefiningOp();

  // ① 将所有结果全部重新引到上面 cast 的输入 (RAUW)
  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    Value replacement = above_cast->getOperand(index).get();
    rewriter.replaceAllUsesWith(result, replacement);
    //   在每个 use 上调用 set(newValue)：
    //     removeFromCurrent()   — 从旧 results 链表移除
    //     value = newValue
    //     insertIntoCurrent()   — 插入新 replacement value 的链表
    //   now all uses → replacement
  }

  // ② 删除两个 op（顺序重要 — 先删外层，再删里层）
  rewriter.eraseOp(op);         // op 此时已无 uses（我们刚 RAUW 掉）
  rewriter.eraseOp(above_cast); // above_cast 也已无 uses（match 时 hasOneUse 验证过）
}
```

这一步全是第 10 章讲的两个原子动作的组合：
- **RAUW**（`replaceAllUsesWith`）：第 10 章 §10.2 的三原子动作（removeFromCurrent + 改 value + insertInto）的循环。每次把一条 use 边从旧 result 搬到新 replacement。
- **erase**（`eraseOp`）：第 10 章 §10.3 的安全删除——先 RAUW 保证 use_empty，再 erase。

**为什么这个顺序正确**（呼应 Ch10 §10.3.3 的"先 RAUW 再 erase"）：

1. 先 `replaceAllUsesWith` 每个 `result`——把所有引用 buffer_cast 结果的边改指向 above_cast 的对应操作数；
2. 此时 `op`（buffer_cast）`use_empty()` ✓，才 `eraseOp(op)`；
3. `above_cast` 的结果原本只被 `op` 用（match 里 `hasOneUse` 保证了），`op` 删了它也就 `use_empty()`，才能 `eraseOp(above_cast)`。

如果顺序反了（先 erase op 再 RAUW），op 已经被销毁，它的 result 容器也没了，RAUW 找不到东西搬——典型的 use-after-free。

### 13.3.3 第三步：driver 调度与工作表滚雪球

rewrite 执行时，每次 `rewriter.eraseOp` 都触发 `notifyOperationErased`（Ch9 §9.3、Ch11 §11.2）。驱动听见后：

- `addOperandsToWorklist(op)`：把被删 op 的操作数定义 op 重塞工作表。op 的操作数来自 above_cast，above_cast 已被删，它的操作数（%a）的 defOp 可能也被重塞——如果 %a 的使用者 ≤ 2（Ch9 §9.5.2 的启发式）。
- `worklist.remove(op)`：把被删的 op 自己摘掉，避免悬空指针。

这个"向上回溯"让重写能滚雪球——删除一个 cast 后，可能触发上游 op 的进一步化简。在 BufferCastOpFold 场景里，above_cast 被删后，它的操作数定义 op（产生 %a 的那个）如果也是冗余的，driver 下一轮会再尝试匹配。

## 13.4 过程中的图状态

用图把整个折叠过程的状态变化画出来（接续 Ch10 §10.2.4 的画法）：

```text
应用前：
  %a ──→ buffer_cast(above) ──→ %t0 ──→ buffer_cast(op) ──→ %out ──→ (下游的 softmax)

match() 后确认可折叠：
  %a ──→ [buffer_cast_above] ──→ %t0 ──→ [buffer_cast_op] ──→ %out ──→ (下游)

rewrite 后（RAUW + erase 完成）：
  %a ──→ (下游的 softmax)  ← 两个 buffer_cast 消失了
  use-def 链上的所有 use 从 %out 重新指向了 %a
```

具体追踪 use-def 边的变化：

**rewrite 开始前**（%out 的 use-list）：
```text
%out firstUse ──► [OpOperand{owner=softmax, value=%out}] ──► nullptr
```

**RAUW(%out, %a) 后**（Ch10 §10.2.4 的搬迁过程）：
```text
%out firstUse ──► nullptr           ← 搬空了
%a   firstUse ──► [OpOperand{owner=softmax, value=%a}] ──► ...（原本 %a 的使用者）
```
softmax 的那个 OpOperand 从 %out 的链上摘下、改 value 指向 %a、头插进 %a 的链上。**零 new、零 erase op**——只是搬运 OpOperand 对象（Ch10 的核心论断）。

**eraseOp(op) 后**：op（外层 cast）从 block 摘出、销毁。它的 operand（指向 %t0 的 OpOperand）由 ~OperandStorage 析构自动从 %t0 的 use-list 摘下。

**eraseOp(above_cast) 后**：above_cast 也销毁。它的 operand（指向 %a 的 OpOperand）自动从 %a 的 use-list 摘下。

**最终**：两个 cast op 消失，softmax 直接用 %a。整条链从 `%a → cast → %t0 → cast → %out → softmax` 变成 `%a → softmax`。

## 13.5 模式注册与 pass 管道集成

BufferCastOpFold 写好后，要注册进 pattern 集合并由 pass 调度。两种集成方式（Ch11 §11.7）：

**方式一：显式 RewritePatternSet（CH-9）**。在 Pass 的 runOnOperation 里：

```cpp
void DeviceRegionFusionPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateBufferCastOpCanonicalizationPatterns(patterns);   // 注册 BufferCastOpFold
  GreedyRewriteConfig cfg; cfg.maxIterations = 10;
  applyPatternsAndFoldGreedily(getOperation(),
      FrozenRewritePatternSet(std::move(patterns)), cfg);
}
```

`populateBufferCastOpCanonicalizationPatterns` 内部就是 `patterns.add<BufferCastOpFold>(ctx, /*benefit=*/2);`。

**方式二：ODS 的 hasCanonicalizer（CH-14）**。通过 ODS 的 `hasCanonicalizer = 1`，`--canonicalize` pass 自动拾取：

```tablegen
def NorthStar_BufferCastOp : NorthStar_Op<"buffer_cast", ...> {
    let hasCanonicalizer = 1;   // 声明这个 op 有 canonicalization pattern
}
```

然后实现 `BufferCastOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context)`，把 BufferCastOpFold 注册进去。`--canonicalize` pass 会自动调用所有 op 的 getCanonicalizationPatterns，收集成一个大的 pattern 集合，再跑 greedy driver。

> **两种方式对应"局部优化"与"通用规范化"的区分**：方式一（显式 pattern set）用于"某个特定 pass 想做的优化"（如 device region fusion 顺带做 cast 折叠）；方式二（hasCanonicalizer）用于"这个 op 的通用规范化规则"（任何跑 `--canonicalize` 的地方都自动应用）。Dragon Book 第 8.5 节讨论的"local optimization"对应方式一，"canonical form"对应方式二。

## 13.6 把全书串起来：BufferCastOpFold 用到的所有机制

BufferCastOpFold 这个看似简单的折叠，其实调用了全书几乎每一个机制。逐一回扣：

| 机制 | 在 BufferCastOpFold 里的作用 | 本书章节 |
|---|---|---|
| **use-list 数据结构** | match 沿 use-def 链读、RAUW 沿 use-list 搬 | Ch4 |
| **back 指针的指针** | RAUW 的零分支搬迁靠它 | Ch4 |
| **walk / 遍历** | driver 用 walk 把所有 op 入工作表 | Ch5 |
| **use_empty / hasOneUse** | match 的前置条件检查 | Ch4-5 |
| **OpRewritePattern** | BufferCastOpFold 继承它 | Ch9 |
| **两步 match/rewrite 写法** | 结构性保证 match-before-mutate | Ch9, Ch11 |
| **PatternRewriter** | rewrite 里调 replaceAllUsesWith / eraseOp | Ch9-10 |
| **RAUW 三原子动作** | rewrite ① 把 use 边搬迁 | Ch10 |
| **erase 安全（先 RAUW 再 erase）** | rewrite ② 删 op 前保证 use_empty | Ch10 |
| **driver 作为监听者** | eraseOp 触发 notifyOperationErased → 工作表更新 | Ch9, Ch11 |
| **工作表滚雪球** | 删 cast 后上游 op 重塞工作表，可能触发更多化简 | Ch9, Ch11 |
| **benefit** | BufferCastOpFold benefit=2（在它的 pattern 集合内） | Ch11 |
| **DCE（死代码消除）** | 删除产生的死 cast 被 driver 自动清理 | Ch10 |
| **不动点收敛** | driver 反复跑直到 IR 不再变 | Ch11 |

**这就是为什么本章是压轴**——它不是一个新机制，而是所有机制的综合应用。理解了 BufferCastOpFold 的每一步，就理解了 MLIR 优化 pass 的完整工作方式。

> **编译原理浸润点：端到端优化 pass 的全景**　Dragon Book 第 8.5 节（局部优化、公共子表达式消除）、第 9.4 节（死代码消除）描述的优化 pass 结构——发现→变换→清理——在 BufferCastOpFold 里完美体现：
> - **发现**：match 沿 use-def 链检查前置条件（Ch4-5 的遍历）。
> - **变换**：RAUW 把 use 边改指向新值（Ch10 的原子动作）。
> - **清理**：erase 删掉冗余 op + driver 自动清理连带死代码（Ch10-11 的 DCE）。
>
> 这三步是任何优化 pass 的标准结构。MLIR 把它工程化为"pattern + driver"——pattern 描述"发现+变换"（matchAndRewrite），driver 负责"清理+调度"（工作表 + DCE + 不动点收敛）。这个分工让优化规则的编写变得极其局部——你只管"这个模式换成什么"，全局调度交给框架。

## 13.7 全书收束：MLIR 的完整图景

走到这里，我们已经把 MLIR IR 的完整生命周期讲完了：

```text
           ┌─────────────────────────────────────────────────┐
           │  Ch0  MLIR 在编译器学科中的位置                  │
           │  Ch1  MLIRContext：对象的所有权与唯一化          │
           └─────────────────────────────────────────────────┘
                              │ 拥有者
                              ▼
           ┌─────────────────────────────────────────────────┐
           │  Ch2  Operation：一切皆操作                      │  结构
           │  Ch3  Block、Region 与控制流结构                │  (长什么样)
           │  Ch4  def-use chain：数据流边                   │
           └─────────────────────────────────────────────────┘
                              │ 静态结构
                              ▼
           ┌─────────────────────────────────────────────────┐
           │  Ch5  遍历 IR：walk、变更契约                    │  读
           └─────────────────────────────────────────────────┘
                              │ 能读了
                              ▼
           ┌─────────────────────────────────────────────────┐
           │  Ch6  从 AST 到 IR：两类挂接                     │  写
           │  Ch7  Operation::create 与尾分配                │  (怎么造)
           │  Ch8  use-def 链自动形成                        │
           └─────────────────────────────────────────────────┘
                              │ 造好了
                              ▼
           ┌─────────────────────────────────────────────────┐
           │  Ch9  matchAndRewrite：图重写                    │  改
           │  Ch10 RAUW 与 erase：重写原子动作                │  (怎么改)
           │  Ch11 驱动、工作表与监听者                       │
           └─────────────────────────────────────────────────┘
                              │ 改完了
                              ▼
           ┌─────────────────────────────────────────────────┐
           │  Ch12 parse/print 往返（文本形态）               │  专题
           │  Ch13 BufferCastOpFold（全书机制综合）           │  综合
           └─────────────────────────────────────────────────┘
```

贯穿这十三章的，是几个编译器学科的经典概念——它们是 MLIR 的"血液"：

- **SSA 形式**（Ch0 引入，Ch4 物理实现，Ch8 构造维护，Ch10 重写维护）。
- **def-use chain / 数据流分析**（Ch4-5）。
- **支配关系**（Ch0 定义，Ch3 block argument，Ch8 构造维护）。
- **项重写系统与不动点**（Ch9-11）。
- **IR 表示理论**（Ch0 三派对照，Ch2 一切皆 Operation）。

读者合上本书时，应该不仅能"用 MLIR"，更要理解——MLIR 在编译器学科谱系中处于什么位置、它的每个机制回应了哪些经典编译原理问题、相比 LLVM/GCC/Sea of Nodes 做了什么取舍。这是本书的立意：**以编译器视角理解 MLIR**。

---

## 编译原理浸润点回顾

1. **端到端优化 pass 结构**：本章主题。发现→变换→清理，Dragon Book 第 8.5 节与第 9.4 节。
2. **数据流分析（综合）**：match 沿 use-def 链导航，是全书 use-list 机制的综合应用。
3. **项重写（综合）**：BufferCastOpFold 是一条重写规则，由 driver 系统化套用。
4. **死代码消除（综合）**：删除产生的死 op 被 driver 自动清理。
5. **等价/规范化**：BufferCastOpFold 是规范化规则（canonicalization），对应 Dragon Book 第 8.5 节。

---

## 本章关键结论

1. **优化 pass 的标准结构是发现→变换→清理**。BufferCastOpFold 完美体现：match（发现）+ RAUW（变换）+ erase（清理）。
2. **match 沿 use-def 链导航**：getDefiningOp（UD）、hasOneUse（DU）、getOpOperand。全是 Ch4-5 的遍历，不改图。
3. **rewrite = RAUW + erase**：Ch10 的两个原子动作组合。先 RAUW 保证 use_empty，再 erase。
4. **driver 自动清理连带的死代码**：eraseOp 触发 notifyOperationErased → addOperandsToWorklist 向上回溯 → DCE。
5. **两种集成方式**：显式 RewritePatternSet（局部 pass）vs hasCanonicalizer（通用规范化）。
6. **BufferCastOpFold 综合了全书 14 个机制**：use-list、back 指针、walk、OpRewritePattern、PatternRewriter、RAUW、erase、driver、工作表、benefit、DCE、不动点收敛。理解它就理解了 MLIR 优化 pass 的完整工作方式。

---

## 全书结语

本书从 MLIR 在编译器学科中的位置讲起（Ch0），经 IR 的拥有者（Ch1）、逻辑结构（Ch2-3）、数据流边（Ch4）、遍历（Ch5）、构建（Ch6-8）、重写（Ch9-11）、序列化（Ch12），最后以 BufferCastOpFold 闭环（Ch13）收束。这条路径——**表示→读→写→改→序列化→综合**——正是一个编译器 IR 的完整生命周期，也是编译器教科书的经典叙事。

希望读者合上本书时，不仅掌握了 MLIR 的具体机制，更建立起一个编译器学科的视角——用这个视角看任何编译器（LLVM、GCC、Graal、Cranelift），都能更快地理解它的设计。这是本书的最终目的。

---

## 原文对照

本章素材主要来自：
- `docs/MLIR-IR-Node组织与遍历插入删除教程.md` §6（BufferCastOpFold 的 match/rewrite 逐步走读、模式注册、pass 管道、图状态变化）——**全文保留**
- 全书机制的综合回扣为本书新增
- 编译原理铺垫（端到端优化 pass 结构、DCE、规范化）为本书新增，对应 Dragon Book Ch.8.5, Ch.9.4

## 参考文献

- **[Aho 2006]** Dragon Book，第 8.5 节（局部优化、公共子表达式消除、规范化）、第 9.4 节（死代码消除）。
- **[Lattner 2020]** Lattner et al. "MLIR"，canonicalization pattern 与 driver 的设计。
- **[Baader 1998]** Baader & Nipkow. *Term Rewriting and All That*.（重写规则系统化套用的理论基础）

---

*全书完*
