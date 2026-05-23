# 第九章：端到端综合追踪 —— 一个 torch.compile 的完整旅程

> "纸上得来终觉浅，绝知此事要躬行。前面八章我们拆解了每一个齿轮、每一根弹簧，现在是时候把它们组装起来，按下启动按钮，看整台机器如何运转。"

本章是全书的收官之章。我们将追踪一个具体的神经网络计算块，从 Python 函数定义出发，经过 TorchDynamo 捕获、AOTAutograd 分解、GraphLowering 逐节点 lowering、Scheduler 融合决策、TritonKernel 代码生成，直到最终由 PythonWrapperCodegen 组装成可执行的 Python 函数。每一个阶段，我们都会标明哪些类被创建、哪些 V 上下文被切换、哪些 Handler 被安装——将前八章散落的知识点编织成一张完整的网。

---

## 9.1 问题描述与 FX Graph 生成

### 9.1.1 待编译的计算块

考虑如下简化的神经网络前向块：

```python
@torch.compile
def model(x, w, b):
    # x: [B, D], w: [D, H], b: [H]
    h = torch.matmul(x, w)              # [B, H]
    h = torch.batch_norm(h, ...)        # [B, H] — 简化，真实 BN 需要 running stats
    h = torch.nn.functional.relu(h)     # [B, H]
    return h
```

这个例子虽小，却覆盖了三类典型的计算模式：

| 算子 | 计算模式 | Inductor 处理策略 |
|------|---------|------------------|
| `matmul` | 外部库调用（GEMM） | ExternKernel → 委托给 cuBLAS |
| `batch_norm` | 分解为 mean/var/norm/mul/add | Reduction + Pointwise |
| `relu` | 逐元素运算 | Pointwise，可融合 |

正是这种混合模式，让我们能观察到 Inductor 处理不同类型算子时的差异化策略。

### 9.1.2 Step 1: TorchDynamo — 从 Python 到 FX Graph

当 Python 解释器执行到 `@torch.compile` 装饰的函数时，TorchDynamo 介入执行流程。它通过 PEP 523 的 Frame Evaluation API 拦截 Python 字节码的执行，将逐行解释运行转化为符号追踪（symbolic tracing）。

TorchDynamo 的核心工作是：对函数中的每一条语句，不执行真实计算，而是记录"做了什么操作"——在 FX Graph 中创建一个对应的 `Node`。这个过程类似于"抄账"：原本是花钱买东西（执行计算），现在是把每一笔花销记在账本上（记录 FX Node）。

生成的 FX Graph 如下：

```
graph():
    %x = placeholder[target=x]           # 输入占位符
    %w = placeholder[target=w]           # 输入占位符
    %b = placeholder[target=b]           # 输入占位符
    %matmul = call_function[target=aten.matmul](args=(%x, %w))
    %bn = call_function[target=aten.batch_norm](args=(%matmul, ...))
    %relu = call_function[target=aten.relu](args=(%bn,))
    %output = output[args=(%relu,)]
```

这个 FX Graph 有 6 个节点：3 个 `placeholder`（输入）、2 个 `call_function`（计算）、1 个 `output`（输出）。每个节点携带 `target`（调用目标）和 `args`（参数引用），形成一张有向无环图（DAG）。

**关键认识**：FX Graph 是 TorchDynamo 的产出物，但还不是 Inductor 的输入。在进入 Inductor 之前，还需要 AOTAutograd 进行一层预处理。

### 9.1.3 Step 2: AOTAutograd — 分解与前向/反向分离

AOTAutograd 承担两个职责：

**第一，前向/反向分离**。对于训练场景，AOTAutograd 利用 PyTorch 的 `torch.autograd` 机制，将原始 FX Graph 拆分为 forward 和 backward 两个子图。本文聚焦 forward 部分，backward 的流程是对称的。

**第二，算子分解（Decomposition）**。AOTAutograd 应用预定义的分解表，将复合算子拆解为更基础的 ATen 操作。对于我们的例子，最关键的分解是 `batch_norm`：

```
batch_norm(input, weight, bias, running_mean, running_var, ...)
    → mean = aten.mean(input, dim=[0])
    → diff = aten.sub(input, mean)
    → var = aten.mean(aten.pow(diff, 2), dim=[0])
    → norm = aten.div(diff, aten.sqrt(aten.add(var, eps)))
    → output = aten.add(aten.mul(norm, weight), bias)
```

为什么需要分解？因为 `batch_norm` 对于 Inductor 来说"颗粒度太粗"。分解后变成 `mean`、`sub`、`pow`、`div`、`sqrt`、`add`、`mul` 等基础操作，Inductor 就可以对它们逐个分析、做融合优化。这就像把一道复杂的菜谱拆解为"切菜"、"炒菜"、"调味"等基本步骤，让厨师可以灵活安排工序。

分解后的 forward 子图（简化表示）：

```
graph():
    %x, %w = inputs                        # [B, D], [D, H]

    %matmul = aten.matmul(%x, %w)          # GEMM → [B, H]

    %mean = aten.mean(%matmul, dim=[0])    # BN 分解: 均值 → [H]
    %diff = aten.sub(%matmul, %mean)       # BN 分解: 差值 → [B, H]
    %pow = aten.pow(%diff, 2)              # BN 分解: 平方 → [B, H]
    %var = aten.mean(%pow, dim=[0])        # BN 分解: 方差 → [H]
    %vareps = aten.add(%var, 1e-5)         # BN 分解: 加 eps → [H]
    %std = aten.sqrt(%vareps)              # BN 分解: 标准差 → [H]
    %norm = aten.div(%diff, %std)          # BN 分解: 归一化 → [B, H]
    %scaled = aten.mul(%norm, weight)      # BN 分解: 缩放 → [B, H]
    %bn_out = aten.add(%scaled, bias)      # BN 分解: 偏移 → [B, H]

    %relu_out = aten.relu(%bn_out)         # 激活 → [B, H]

    return %relu_out
```

原本 3 个算子变成了 12 个——节点数增加了，但每个节点的语义更简单，Inductor 的优化空间反而更大了。

---

## 9.2 Phase 1: GraphLowering 的逐节点 Lowering

### 9.2.1 GraphLowering 的创建与上下文设置

分解后的 FX Graph 被传递给 `compile_fx_inner()`（`compile_fx.py`）。这个函数是 Inductor 编译流水线的入口，它首先创建 `GraphLowering` 实例：

```python
# compile_fx.py: compile_fx_inner()
graph = GraphLowering(decomposed_graph)    # 创建 GraphLowering
V.set_graph_handler(graph)                 # 安装 V.graph
```

`V.set_graph_handler` 做了什么？它在 `Virtualized` 动态作用域中注册 `V.graph`、`V.kernel` 等上下文。回顾第二章的内容，这意味着从这一刻起，任何代码通过 `V.graph` 访问的都是这个 `graph` 实例。

```python
# virtualized.py 中的上下文注册
def set_graph_handler(g):
    return V.graph._set_handler(g), V.kernel._set_handler(NullKernelHandler())
```

此时 V 的状态快照：

| V 上下文 | 当前值 | 用途 |
|---------|-------|------|
| `V.graph` | `GraphLowering` 实例 | 全局计算图持有者 |
| `V.ops` | `MockHandler`（默认） | 尚未开始 lowering |
| `V.kernel` | `NullKernelHandler` | 尚未开始 codegen |
| `V.fake_mode` | `FakeTensorMode` | Fake tensor 支持 |

### 9.2.2 `graph.run()` — 触发 Interpreter 遍历

`GraphLowering` 继承自 `torch.fx.Interpreter`。调用 `graph.run()` 时，Interpreter 按拓扑顺序遍历 FX Graph 的每个节点，对每个节点调用对应的 dispatch 方法。`GraphLowering` 重写了这些 dispatch 方法，将"执行节点"变为"lowering 节点"。

这是 Interpreter 模式的经典应用：基类定义遍历框架，子类重写行为。就像一个音乐播放器框架——基类负责按顺序播放每一首曲目，子类决定"播放"是真正发声（eager 模式）还是记录乐谱（lowering 模式）。

### 9.2.3 节点逐一 Lowering

下面我们逐节点追踪整个 lowering 过程。

#### 节点 1-2: `placeholder(x)`, `placeholder(w)`

```python
# GraphLowering.placeholder() 的核心逻辑
def placeholder(self, target):
    # 为每个输入创建一个 InputBuffer IR 节点
    buffer = InputBuffer(target, shape=shape, dtype=dtype)
    self.buffers[target] = buffer
    return buffer
```

产出：

| IR 节点 | 类型 | 形状 | 含义 |
|--------|------|------|------|
| `InputBuffer("x")` | `InputBuffer` | `[B, D]` | 原始输入 |
| `InputBuffer("w")` | `InputBuffer` | `[D, H]` | 原始权重 |

**Handler 状态**：`placeholder` lowering 不需要切换 Handler，因为创建输入缓冲区只是简单的数据结构操作，不涉及 IR 闭包。

#### 节点 3: `matmul(x, w)`

```python
# GraphLowering.call_function() 的核心逻辑
def call_function(self, target, args, kwargs):
    # 查找 lowering rule
    lowering_rule = get_lowering_rule(target)  # → matmul 的 lowering rule
    return lowering_rule(args, kwargs)
```

`matmul` 的 lowering rule 发现它是一个 GEMM 操作，Inductor 目前不会自己生成 GEMM kernel（这需要高度优化的库如 cuBLAS），因此将其标记为外部 kernel 调用：

```python
# matmul lowering rule（简化）
def matmul_lowering(x, w):
    result = ComputedBuffer("matmul", shape=[B, H],
        data=ExternKernel(aten.matmul, args=[x, w])
    )
    V.graph.register_buffer(result)
    return result
```

产出：

| IR 节点 | 类型 | 数据 | 是否可融合 |
|--------|------|------|----------|
| `ComputedBuffer("matmul")` | `ComputedBuffer` | `ExternKernel(aten.matmul)` | 否（外部库调用） |

**关键决策**：`ExternKernel` 意味着这个操作将在运行时调用 cuBLAS，而不是生成 Triton kernel。这是 Inductor 的 "知道自己不知道什么" 的智慧——GEMM 有大量专用优化（tiling、流水线、tensor core），与其自己写一个次优的实现，不如直接委托给专家库。

#### 节点 4: `mean(matmul, dim=[0])`

`mean` 是一个 reduction 操作（沿 batch 维度求均值）。Lowering 过程：

```python
# mean lowering rule（简化）
def mean_lowering(input, dim):
    # 先做 sum reduction，再除以元素数量
    sum_result = make_reduction(
        "sum", input, dim=dim,
        inner_fn=lambda idx: V.ops.load("matmul", idx)
    )
    # 然后做 pointwise 的除法
    return make_pointwise(
        lambda idx: V.ops.truediv(V.ops.load("sum_buf", idx), B)
    )
```

但更准确地说，Inductor 内部会将 `mean` 直接 lowering 为一个带平均语义的 Reduction IR 节点：

```python
ComputedBuffer("mean", shape=[H],
    data=Reduction(
        inner_fn=lambda idx: ops.load("matmul", idx),  # 读取 matmul 结果
        reduction_type="avg",                           # 平均（而非求和）
        dim=[0]                                         # 沿 batch 维
    )
)
```

**核心概念：`inner_fn` 闭包**。这里的 `lambda idx: ops.load("matmul", idx)` 是一个 Python callable，它描述了 reduction 的循环体——"对于每个索引 `idx`，从 `matmul` 缓冲区读取一个值"。这个闭包此时不会被真正执行，它只是被存储在 IR 节点中，等待后续阶段被不同的 Handler "解释"执行。

这正是第二章和第三章所述的 **define-by-run** 设计精髓：IR 不是一种静态数据结构，而是一个 Python callable，它的语义取决于执行时安装的 Handler。

#### 节点 5: `sub(matmul, mean)`

```python
ComputedBuffer("diff", shape=[B, H],
    data=Pointwise(
        inner_fn=lambda idx: ops.sub(
            ops.load("matmul", idx),      # 加载 matmul 结果
            ops.load("mean", idx_0)       # 加载 mean 结果（广播）
        )
    )
)
```

注意 `idx` 和 `idx_0` 的区别：`matmul` 的形状是 `[B, H]`，`mean` 的形状是 `[H]`。`mean` 需要广播（broadcast）到 `[B, H]`，这在索引操作中体现为忽略 batch 维度的索引。Inductor 的 IR 层会自动处理这种广播语义。

#### 节点 6: `pow(diff, 2)`

```python
ComputedBuffer("pow", shape=[B, H],
    data=Pointwise(
        inner_fn=lambda idx: ops.pow(
            ops.load("diff", idx),
            ops.constant(2, torch.float32)
        )
    )
)
```

#### 节点 7: `mean(pow, dim=[0])` → `var`

与节点 4 类似，生成另一个 Reduction：

```python
ComputedBuffer("var", shape=[H],
    data=Reduction(
        inner_fn=lambda idx: ops.load("pow", idx),
        reduction_type="avg",
        dim=[0]
    )
)
```

#### 节点 8-11: `add(var, eps)` → `sqrt` → `div(diff, ...)` → `norm`

这一组操作形成一条 pointwise 链：

```python
# add(var, eps)
ComputedBuffer("vareps", shape=[H],
    data=Pointwise(
        inner_fn=lambda idx: ops.add(
            ops.load("var", idx_0),
            ops.constant(1e-5, torch.float32)
        )
    )
)

# sqrt(vareps)
ComputedBuffer("std", shape=[H],
    data=Pointwise(
        inner_fn=lambda idx: ops.sqrt(ops.load("vareps", idx_0))
    )
)

# div(diff, std) → norm
ComputedBuffer("norm", shape=[B, H],
    data=Pointwise(
        inner_fn=lambda idx: ops.truediv(
            ops.load("diff", idx),
            ops.load("std", idx_0)
        )
    )
)
```

#### 节点 12-13: `mul(norm, weight)` → `add(scaled, bias)`

```python
# mul(norm, weight)
ComputedBuffer("scaled", shape=[B, H],
    data=Pointwise(
        inner_fn=lambda idx: ops.mul(
            ops.load("norm", idx),
            ops.load("weight", idx_0)
        )
    )
)

# add(scaled, bias)
ComputedBuffer("bn_out", shape=[B, H],
    data=Pointwise(
        inner_fn=lambda idx: ops.add(
            ops.load("scaled", idx),
            ops.load("bias", idx_0)
        )
    )
)
```

#### 节点 14: `relu(bn_out)`

```python
ComputedBuffer("relu_out", shape=[B, H],
    data=Pointwise(
        inner_fn=lambda idx: ops.relu(ops.load("bn_out", idx))
    )
)
```

### 9.2.4 Phase 1 产出物总览

经过逐节点 lowering，`V.graph` 中积累了以下 IR 缓冲区：

| # | 缓冲区名称 | IR 数据类型 | 形状 | inner_fn 依赖 |
|---|-----------|-----------|------|--------------|
| 0 | `x` | `InputBuffer` | `[B, D]` | 无 |
| 1 | `w` | `InputBuffer` | `[D, H]` | 无 |
| 2 | `matmul` | `ExternKernel` | `[B, H]` | x, w |
| 3 | `mean` | `Reduction` | `[H]` | matmul |
| 4 | `diff` | `Pointwise` | `[B, H]` | matmul, mean |
| 5 | `pow` | `Pointwise` | `[B, H]` | diff |
| 6 | `var` | `Reduction` | `[H]` | pow |
| 7 | `vareps` | `Pointwise` | `[H]` | var |
| 8 | `std` | `Pointwise` | `[H]` | vareps |
| 9 | `norm` | `Pointwise` | `[B, H]` | diff, std |
| 10 | `scaled` | `Pointwise` | `[B, H]` | norm, weight |
| 11 | `bn_out` | `Pointwise` | `[B, H]` | scaled, bias |
| 12 | `relu_out` | `Pointwise` | `[B, H]` | bn_out |

**依赖关系图**（简化，仅展示关键边）：

```
x ──┐
    ├──→ matmul ──┬──→ mean ──────→ (broadcast)
    │             │                      │
    │             ├──→ diff ←────────────┘
    │             │      │
    │             │      ├──→ pow ──→ var ──→ vareps ──→ std
    │             │      │                                       │
    │             │      ├──→ norm ←────────────────────────────┘
w ──┘             │      │        │
                  │      │        ├──→ scaled ──→ bn_out ──→ relu_out → output
                  │      │        │
                  └──(reduction)──┘
```

此时，所有 IR 节点都已创建完毕，但 **没有任何优化决策被做出**——每个算子还是一个独立的缓冲区，没有融合。优化决策是下一阶段 Scheduler 的工作。

---

## 9.3 Phase 2: Scheduler 的融合决策

### 9.3.1 Scheduler 的创建

Phase 1 完成后，`GraphLowering` 调用 `_update_scheduler()` 创建 Scheduler：

```python
# GraphLowering._update_scheduler()
self.scheduler = Scheduler(self.operations)  # 传入所有 IR 操作
V.graph.scheduler = self.scheduler           # 注册到全局上下文
```

Scheduler 的构造函数触发 `_init()`，执行三个核心步骤：

```python
# scheduler.py: Scheduler._init()
def _init(self, nodes):
    # Step 1: 为每个 IR buffer 创建 SchedulerNode
    self.nodes = [self._create_node(buf) for buf in nodes]

    # Step 2: 拓扑排序 + 依赖分析
    self._topological_sort()

    # Step 3: 融合决策
    self._apply_fusion()
```

### 9.3.2 Step 1: 创建 SchedulerNode

每个 IR buffer 被包装为一个 `SchedulerNode`（或其子类 `ExternKernelSchedulerNode`）：

| IR buffer | SchedulerNode 类型 | 原因 |
|-----------|-------------------|------|
| `matmul` | `ExternKernelSchedulerNode` | 外部库调用，特殊处理 |
| `mean` | `SchedulerNode` | Reduction 类型 |
| `diff` | `SchedulerNode` | Pointwise 类型 |
| `pow` | `SchedulerNode` | Pointwise 类型 |
| `var` | `SchedulerNode` | Reduction 类型 |
| `vareps` | `SchedulerNode` | Pointwise 类型 |
| `std` | `SchedulerNode` | Pointwise 类型 |
| `norm` | `SchedulerNode` | Pointwise 类型 |
| `scaled` | `SchedulerNode` | Pointwise 类型 |
| `bn_out` | `SchedulerNode` | Pointwise 类型 |
| `relu_out` | `SchedulerNode` | Pointwise 类型 |

### 9.3.3 Step 2: 依赖分析与拓扑排序

这是 Scheduler 进行融合决策的前置步骤。Scheduler 利用 `_RecordLoadStoreInner` Handler（详见第六章和第三章）分析每个节点读取和写入了哪些缓冲区。

**机制回顾**：`_RecordLoadStoreInner` 是一个分析型 Handler。当它被安装为 `V.ops` 时，重新执行每个 IR 节点的 `inner_fn`，记录所有 `load` 调用（读依赖）和 `store` 调用（写依赖）。

```
安装 V.ops = _RecordLoadStoreInner(node="diff")
执行 inner_fn: lambda idx: ops.sub(ops.load("matmul", idx), ops.load("mean", idx_0))
    → ops.load("matmul", ...)  → 记录 reads = {"matmul"}
    → ops.load("mean", ...)    → 记录 reads = {"matmul", "mean"}
结果: node.diff.reads = {"matmul", "mean"}
```

对所有节点执行完依赖分析后，得到完整的读写关系表：

| 节点 | reads | writes |
|------|-------|--------|
| `matmul` | {x, w} | {matmul} |
| `mean` | {matmul} | {mean} |
| `diff` | {matmul, mean} | {diff} |
| `pow` | {diff} | {pow} |
| `var` | {pow} | {var} |
| `vareps` | {var} | {vareps} |
| `std` | {vareps} | {std} |
| `norm` | {diff, std} | {norm} |
| `scaled` | {norm, weight} | {scaled} |
| `bn_out` | {scaled, bias} | {bn_out} |
| `relu_out` | {bn_out} | {relu_out} |

基于读写关系，Scheduler 做拓扑排序，确保生产者（producer）排在消费者（consumer）前面。排序结果已经隐含在上述表格的顺序中。

**Handler 切换时序**：

```
对于每个 SchedulerNode:
    with V.set_ops_handler(_RecordLoadStoreInner(node)):
        node.inner_fn()    # 执行闭包，记录 reads/writes
    # Handler 自动恢复
```

`_RecordLoadStoreInner` 是短命的——每个节点分析完后立即被销毁，替换为下一个节点的分析实例。这体现了 Handler 的 "即插即用" 特性。

### 9.3.4 Step 3: 融合决策

融合决策是 Scheduler 最核心的工作。它由后端调度器（`BaseScheduling` 子类）驱动，根据设备特性决定哪些节点可以合并为一个 kernel。

对于 Triton GPU 后端（`TritonScheduling`），融合遵循以下启发式规则：

**纵向融合（Vertical Fusion）**：如果节点 B 的输入是节点 A 的输出（或传递依赖），且两者都是 Pointwise 类型，它们可以融合为一个 kernel。融合的好处是中间结果不需要写回全局内存，可以直接在寄存器中传递。

**Reduction 节点**：Reduction 是融合的天然屏障。一个 reduction kernel 的输出必须写回全局内存（因为结果维度比输入小），所以 reduction 之后的所有节点必须开始一个新的融合组。

**ExternKernel 节点**：外部库调用无法融合。`matmul` 由 cuBLAS 执行，其输出是全局内存中的缓冲区，后续操作必须重新读取。

基于这些规则，融合分析如下：

#### 融合组 1: matmul（不可融合）

```
matmul → ExternKernelSchedulerNode
理由: 外部库调用，委托给 cuBLAS，不能融合
```

#### 融合组 2: mean（独立 reduction kernel）

```
mean → 单独的 reduction kernel
理由: reduction 操作，不能与下游 pointwise 融合（reduction 输出需要写回全局内存）
```

#### 融合组 3: diff + pow（纵向融合）

```
diff + pow → FusedSchedulerNode
理由: diff 是 pointwise，pow 也是 pointwise，且 pow 直接消费 diff 的输出
      融合后，diff 的中间结果不需要写回全局内存
```

为什么 `diff` 不和 `mean` 融合？因为 `mean` 是 reduction，reduction 节点作为 producer 时无法与下游 pointwise 融合。`diff` 必须读取 `mean` 的全局内存输出，所以它们不能在同一个 kernel 中。

#### 融合组 4: var（独立 reduction kernel）

```
var → 单独的 reduction kernel
理由: reduction 操作
```

#### 融合组 5: vareps + std + norm + scaled + bn_out + relu_out（大融合）

```
vareps + std + norm + scaled + bn_out + relu_out → FusedSchedulerNode
理由: 这是一条纯 pointwise 链:
  vareps = var + eps           (pointwise)
  std = sqrt(vareps)          (pointwise)
  norm = diff / std           (pointwise, 但依赖 diff — 已在融合组3中计算)
  scaled = norm * weight      (pointwise)
  bn_out = scaled + bias      (pointwise)
  relu_out = relu(bn_out)     (pointwise)

  每一步都是 pointwise，且形成一条直线依赖链，完全可以纵向融合
```

注意 `norm` 依赖 `diff`（融合组 3 的输出），这意味着融合组 5 需要从全局内存中读取 `diff` 缓冲区。但融合组 5 内部的所有中间值（`vareps` → `std` → `norm` → `scaled` → `bn_out` → `relu_out`）都在寄存器中传递，不需要中间全局内存写入。

### 9.3.5 最终调度方案

```
╔══════════════════════════════════════════════════════════════════════════╗
║                        最终调度方案：5 个 kernel 调用                       ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Kernel 1: cuBLAS matmul                                               ║
║  ┌─────────────────────────────────┐                                   ║
║  │ buf_matmul = cublas_mm(x, w)   │  → ExternKernel                   ║
║  └─────────────────────────────────┘                                   ║
║                         ↓                                              ║
║  Kernel 2: Triton reduction (mean)                                     ║
║  ┌─────────────────────────────────┐                                   ║
║  │ buf_mean = reduce_avg(matmul)   │  → ReductionKernel                ║
║  └─────────────────────────────────┘                                   ║
║                         ↓                                              ║
║  Kernel 3: Triton pointwise (diff + pow)                               ║
║  ┌─────────────────────────────────────┐                               ║
║  │ diff = matmul - mean                │                               ║
║  │ pow  = diff²                        │  → FusedPointwiseKernel        ║
║  └─────────────────────────────────────┘                               ║
║                         ↓                                              ║
║  Kernel 4: Triton reduction (var)                                      ║
║  ┌─────────────────────────────────┐                                   ║
║  │ buf_var = reduce_avg(pow)       │  → ReductionKernel                ║
║  └─────────────────────────────────┘                                   ║
║                         ↓                                              ║
║  Kernel 5: Triton pointwise (norm + bn + relu)                         ║
║  ┌──────────────────────────────────────────────┐                      ║
║  │ vareps  = var + eps                           │                      ║
║  │ std     = sqrt(vareps)                        │                      ║
║  │ norm    = diff / std                          │                      ║
║  │ scaled  = norm * weight                       │  → FusedPointwise    ║
║  │ bn_out  = scaled + bias                       │                      ║
║  │ relu_out = relu(bn_out)                       │                      ║
║  └──────────────────────────────────────────────┘                      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**融合效果统计**：

| 指标 | 融合前 | 融合后 |
|------|-------|-------|
| Kernel 调用次数 | 11 | 5 |
| 全局内存写入次数 | 11 | 5 |
| 全局内存读取次数（中间值） | ~22 | ~8 |
| 中间缓冲区数量 | 10 | 5 |

融合将 kernel 调用次数减半，全局内存 I/O 更是大幅减少。在 GPU 上，全局内存访问是性能的头号杀手，融合优化直接转化为显著的加速比。

---

## 9.4 Phase 3: TritonScheduling + TritonKernel 的代码生成

Phase 2 做出了"谁和谁融合"的决策，Phase 3 则是执行这个决策——为每个融合组生成具体的 kernel 代码。本阶段的主角是 `TritonScheduling`（继承自 `SIMDScheduling`）和 `TritonKernel`。

### 9.4.1 代码生成的触发

Scheduler 在 `_codegen()` 中按拓扑顺序遍历融合后的节点列表，将每个节点分派给对应的后端调度器：

```python
# scheduler.py: Scheduler._codegen()
for node in self.nodes:
    backend = self._get_backend(node.device)  # → TritonScheduling (for CUDA)
    if isinstance(node, ExternKernelSchedulerNode):
        backend.codegen_template(node, ...)    # 外部 kernel
    elif isinstance(node, FusedSchedulerNode):
        backend.codegen_node(node)             # 融合节点
    else:
        backend.codegen_node(node)             # 普通节点
```

### 9.4.2 Kernel 1: cuBLAS matmul（ExternKernel）

对于 `ExternKernelSchedulerNode`，TritonScheduling 调用 `codegen_template()`，其处理逻辑非常简单：

```python
# codegen_template for ExternKernel
def codegen_template(self, template_node, ...):
    # 不创建 TritonKernel，直接在 wrapper 中生成外部调用代码
    self.wrapper_request_codegen(
        "buf_matmul = torch._C._blas_matmul(arg0, arg1, out=buf_matmul)"
    )
```

没有 Triton kernel 被创建，没有 Handler 被安装。这只是一个"转发调用"——在最终生成的 wrapper 函数中直接调用 `torch._C._blas_matmul()`。

### 9.4.3 Kernel 2: Reduction mean（TritonKernel）

这是第一个真正的 Triton kernel。`SIMDScheduling.codegen_node()` 的核心流程如下：

#### 步骤 1: 创建 TritonKernel 实例

```python
kernel = TritonKernel(
    name="reduction_mean",
    schedule=self.generate_node_schedule(node),
    ...
)
```

`TritonKernel` 继承自 `SIMDKernel`，后者继承自 `Kernel`（第七章详述）。创建 kernel 实例时，它会初始化代码缓冲区、CSE 缓存和参数列表。

#### 步骤 2: `kernel.__enter__()` — 安装 Handler

这是关键的上下文切换时刻。`TritonKernel` 作为 context manager，`__enter__` 方法执行以下操作：

```python
# Kernel.__enter__()（简化）
def __enter__(self):
    # 1. 将自己注册为 V.kernel
    self._kernel_context = V.set_kernel_handler(self)

    # 2. 创建 CSEProxy 包装 TritonKernelOverrides
    self._ops_context = V.set_ops_handler(
        CSEProxy(TritonKernelOverrides(self))
    )

    return self
```

此时 V 的状态变为：

| V 上下文 | 进入前 | 进入后 |
|---------|-------|-------|
| `V.kernel` | `NullKernelHandler` | `TritonKernel` 实例 |
| `V.ops` | `MockHandler` | `CSEProxy(TritonKernelOverrides)` |

Handler 链为：

```
V.ops → CSEProxy → TritonKernelOverrides → TritonKernel 的代码缓冲区
```

回顾第三章的分析：`CSEProxy` 是代码生成型 Handler 的粘合层。它在调用 `TritonKernelOverrides` 生成代码的同时，叠加 CSE（公共子表达式消除）和类型传播。

#### 步骤 3: 执行 inner_fn — 生成代码

现在，`mean` 节点的 `inner_fn` 被执行：

```python
# Reduction 的 codegen 过程
inner_fn = lambda idx: ops.load("matmul", idx)
result = inner_fn(index)  # 触发 V.ops.load("matmul", index)
```

追踪 `ops.load("matmul", index)` 的调用路径：

```
V.ops.load("matmul", index)
    → CSEProxy.load("matmul", index)
        → 检查 CSE 缓存（是否已生成相同加载）
        → 若未命中：调用 TritonKernelOverrides.load("matmul", index)
            → 生成代码: "tl.load(matmul_ptr + offsets, mask=mask)"
            → 返回 CSEVariable("tmp0")
        → 缓存结果
```

接下来是 reduction 语义的代码生成。TritonKernel 为 reduction 操作生成特定的 Triton 归约原语：

```python
# TritonKernel 中 reduction 的代码生成（简化）
# 生成: acc = tl.zeros([BLOCK_H], dtype=tl.float32)
# 生成: acc += tl.load(matmul_ptr + offsets, mask=mask)  # serial reduction
# 生成: tl.store(out_ptr + h, acc / B)
```

#### 步骤 4: `kernel.__exit__()` — 收集代码

```python
def __exit__(self, *args):
    # 1. 收集代码缓冲区中的内容
    self.finalize()

    # 2. 恢复 V.kernel 和 V.ops
    self._ops_context.__exit__()     # 恢复 V.ops
    self._kernel_context.__exit__()  # 恢复 V.kernel
```

#### 生成的 Triton kernel 代码

```python
@triton.jit
def triton_reduction_mean(ptr_in, ptr_out, B, H, BLOCK_H: tl.constexpr):
    # 每个 program 处理 H 维度中的一个元素
    h = tl.program_id(0)

    # 累加器初始化
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    # 串行遍历 batch 维度
    for b in range(0, B, 1):
        ptr = ptr_in + b * H + h
        val = tl.load(ptr)              # 从 matmul 结果中读取
        acc += val                       # 累加

    # 写回均值
    tl.store(ptr_out + h, acc / B)       # 存储均值结果
```

### 9.4.4 Kernel 3: Fused pointwise (diff + pow)

`diff` 和 `pow` 被融合为一个 kernel。融合后，`inner_fn` 被组合：

```python
# 融合后的 inner_fn（概念性）
def fused_inner_fn(idx):
    # diff = matmul - mean
    diff = ops.sub(ops.load("matmul", idx), ops.load("mean", idx_0))
    # pow = diff^2
    pow_result = ops.pow(diff, ops.constant(2, torch.float32))
    # store pow_result
    ops.store("pow_buf", idx, pow_result)
    # 注意: diff 不被 store（它是中间值，融合后不需要写回）
```

代码生成过程与 Kernel 2 类似，但这次 `CSEProxy` 发挥了关键作用：

1. `ops.load("matmul", idx)` → 生成 `tl.load(matmul_ptr + offsets, mask=mask)` → 返回 `CSEVariable("tmp0")`
2. `ops.load("mean", idx_0)` → 生成 `tl.load(mean_ptr + h_offset)` → 返回 `CSEVariable("tmp1")`
3. `ops.sub(tmp0, tmp1)` → 生成 `tmp2 = tmp0 - tmp1` → 返回 `CSEVariable("tmp2")`
4. `ops.pow(tmp2, 2)` → 生成 `tmp3 = tmp2 * tmp2` → 返回 `CSEVariable("tmp3")`
5. `ops.store("pow_buf", idx, tmp3)` → 生成 `tl.store(pow_ptr + offsets, tmp3, mask=mask)`

如果同一个 `ops.load("matmul", idx)` 被调用两次（例如，`diff` 和后续的 `norm` 都需要它），CSEProxy 会在第二次调用时直接返回缓存的 `CSEVariable("tmp0")`，避免生成重复的加载指令。

生成的 Triton kernel：

```python
@triton.jit
def triton_fused_diff_pow(
    matmul_ptr, mean_ptr, pow_ptr,
    B, H, total_elements: tl.int32,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements

    # 计算索引
    b_idx = offsets // H
    h_idx = offsets % H

    # 加载输入
    matmul_val = tl.load(matmul_ptr + offsets, mask=mask)
    mean_val = tl.load(mean_ptr + h_idx)           # [H]，广播

    # diff = matmul - mean
    diff = matmul_val - mean_val

    # pow = diff^2（不需要写回 diff，diff 留在寄存器中）
    pow_val = diff * diff

    # 只存储 pow 结果（diff 是中间值，融合后不写全局内存）
    tl.store(pow_ptr + offsets, pow_val, mask=mask)
```

**融合的收益**：对比不融合的情况——如果 `diff` 和 `pow` 分别是两个 kernel，那么 `diff` 的结果需要写入全局内存，然后 `pow` kernel 再从全局内存中读取。融合后，`diff` 的结果直接在寄存器中传递给 `pow`，省去了一次全局内存写 + 一次全局内存读。

### 9.4.5 Kernel 4: Reduction var

与 Kernel 2 完全对称，只是输入从 `matmul` 变成了 `pow`：

```python
@triton.jit
def triton_reduction_var(ptr_in, ptr_out, B, H, BLOCK_H: tl.constexpr):
    h = tl.program_id(0)
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for b in range(0, B, 1):
        val = tl.load(ptr_in + b * H + h)
        acc += val
    tl.store(ptr_out + h, acc / B)
```

### 9.4.6 Kernel 5: Fused pointwise (vareps + std + norm + scaled + bn_out + relu_out)

这是最大的融合组，包含 6 个 pointwise 操作。

#### 融合后的 inner_fn

```python
def fused_inner_fn(idx):
    # vareps = var + eps
    vareps = ops.add(ops.load("var", idx_0), ops.constant(1e-5, torch.float32))
    # std = sqrt(vareps)
    std = ops.sqrt(vareps)
    # norm = diff / std
    norm = ops.truediv(ops.load("diff", idx), std)
    # scaled = norm * weight
    scaled = ops.mul(norm, ops.load("weight", idx_0))
    # bn_out = scaled + bias
    bn_out = ops.add(scaled, ops.load("bias", idx_0))
    # relu_out = relu(bn_out)
    relu_out = ops.relu(bn_out)
    ops.store("output_buf", idx, relu_out)
```

#### 代码生成追踪

逐行追踪 `fused_inner_fn` 中每个 `ops.xxx` 调用经过 Handler 链的转换：

```
1. ops.load("var", idx_0)
   → CSEProxy → TritonOverrides.load()
   → 生成: "var_val = tl.load(var_ptr + h_offset)"
   → 返回: CSEVariable("var_val")

2. ops.constant(1e-5, torch.float32)
   → CSEProxy → TritonOverrides.constant()
   → 生成: 无需代码（常量折叠为字面量）
   → 返回: CSEVariable("1e-5")

3. ops.add(var_val, 1e-5)
   → CSEProxy → TritonOverrides.add()
   → 生成: "vareps = var_val + 1e-5"
   → 返回: CSEVariable("vareps")

4. ops.sqrt(vareps)
   → CSEProxy → TritonOverrides.sqrt()
   → 生成: "std = tl.sqrt(vareps)"
   → 返回: CSEVariable("std")

5. ops.load("diff", idx)
   → 生成: "diff_val = tl.load(diff_ptr + offsets, mask=mask)"
   → 返回: CSEVariable("diff_val")

6. ops.truediv(diff_val, std)
   → 生成: "norm = diff_val / std"
   → 返回: CSEVariable("norm")

7. ops.load("weight", idx_0)
   → 生成: "weight_val = tl.load(weight_ptr + h_offset)"
   → 返回: CSEVariable("weight_val")

8. ops.mul(norm, weight_val)
   → 生成: "scaled = norm * weight_val"
   → 返回: CSEVariable("scaled")

9. ops.load("bias", idx_0)
   → 生成: "bias_val = tl.load(bias_ptr + h_offset)"
   → 返回: CSEVariable("bias_val")

10. ops.add(scaled, bias_val)
    → 生成: "bn_out = scaled + bias_val"
    → 返回: CSEVariable("bn_out")

11. ops.relu(bn_out)
    → CSEProxy → TritonOverrides.relu()
    → 生成: "result = tl.where(bn_out > 0, bn_out, 0.0)"
    → 返回: CSEVariable("result")

12. ops.store("output_buf", idx, result)
    → 生成: "tl.store(out_ptr + offsets, result, mask=mask)"
```

#### 最终生成的 Triton kernel

```python
@triton.jit
def triton_fused_norm_bn_relu(
    diff_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N: tl.int32, eps: tl.float32,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N

    # 计算广播索引（var/weight/bias 是 [H]，需要广播到 [B, H]）
    h_idx = offsets % H

    # --- vareps = var + eps ---
    var_val = tl.load(var_ptr + h_idx)
    vareps = var_val + eps

    # --- std = sqrt(vareps) ---
    std = tl.sqrt(vareps)

    # --- norm = diff / std ---
    diff_val = tl.load(diff_ptr + offsets, mask=mask)
    norm = diff_val / std

    # --- scaled = norm * weight ---
    weight_val = tl.load(weight_ptr + h_idx)
    scaled = norm * weight_val

    # --- bn_out = scaled + bias ---
    bias_val = tl.load(bias_ptr + h_idx)
    bn_out = scaled + bias_val

    # --- relu_out = relu(bn_out) ---
    result = tl.where(bn_out > 0, bn_out, 0.0)

    # --- store output ---
    tl.store(out_ptr + offsets, result, mask=mask)
```

**代码生成阶段的 Handler 切换时序**：

```
Kernel 1 (cuBLAS matmul):
    无 Handler 切换（ExternKernel，直接委托）

Kernel 2 (reduction mean):
    __enter__: V.kernel ← TritonKernel, V.ops ← CSEProxy(TritonOverrides)
    执行 inner_fn (1 个 load + reduction 语义)
    __exit__: V.kernel ← NullKernelHandler, V.ops ← MockHandler

Kernel 3 (fused diff + pow):
    __enter__: V.kernel ← TritonKernel, V.ops ← CSEProxy(TritonOverrides)
    执行 fused inner_fn (2 loads + sub + pow + store)
    __exit__: V.kernel ← NullKernelHandler, V.ops ← MockHandler

Kernel 4 (reduction var):
    __enter__: V.kernel ← TritonKernel, V.ops ← CSEProxy(TritonOverrides)
    执行 inner_fn (1 load + reduction 语义)
    __exit__: V.kernel ← NullKernelHandler, V.ops ← MockHandler

Kernel 5 (fused norm + bn + relu):
    __enter__: V.kernel ← TritonKernel, V.ops ← CSEProxy(TritonOverrides)
    执行 fused inner_fn (5 loads + 6 ops + 1 store)
    __exit__: V.kernel ← NullKernelHandler, V.ops ← MockHandler
```

---

## 9.5 Phase 4: PythonWrapperCodegen 的模块组装

前三个阶段生成了 5 个 kernel 的代码。但这些 kernel 代码目前是独立的字符串片段，还需要一个"组装车间"将它们整合为一个可调用的 Python 函数。这个车间就是 `PythonWrapperCodegen`。

### 9.5.1 PythonWrapperCodegen 的角色

`PythonWrapperCodegen` 继承自 `CodeGen`（第八章），负责生成最终的 wrapper 函数。这个 wrapper 函数是一个普通的 Python 函数，它：

1. 接收输入张量
2. 分配中间缓冲区
3. 按顺序调用各个 kernel
4. 返回输出张量

Wrapper 函数是编译器的最终产出物——它是用户代码经过 `torch.compile` 后实际运行的东西。

### 9.5.2 生成的 Wrapper 函数

以下是 `PythonWrapperCodegen` 为我们的例子生成的完整 wrapper 函数：

```python
# 由 PythonWrapperCodegen 自动生成（简化版）
def call(args):
    """
    Compiled forward function for model(x, w, b)
    args[0]: x [B, D] float32
    args[1]: w [D, H] float32
    args[2]: weight [H] float32  (BN 缩放因子)
    args[3]: bias [H] float32    (BN 偏移)
    """
    # 解包参数
    arg0 = args[0]   # x: [B, D]
    arg1 = args[1]   # w: [D, H]
    arg2 = args[2]   # weight: [H]
    arg3 = args[3]   # bias: [H]

    # 获取维度信息（符号化）
    B = arg0.size(0)
    D = arg0.size(1)
    H = arg1.size(1)
    total = B * H

    # ═══════════════════════════════════════
    # Kernel 1: cuBLAS matmul
    # ═══════════════════════════════════════
    buf_matmul = torch.empty([B, H], device='cuda', dtype=torch.float32)
    torch._C._blas_matmul(arg0, arg1, out=buf_matmul)
    # buf_matmul: [B, H] — matmul(x, w) 的结果

    # ═══════════════════════════════════════
    # Kernel 2: Triton reduction mean
    # ═══════════════════════════════════════
    buf_mean = torch.empty([H], device='cuda', dtype=torch.float32)
    triton_reduction_mean[(H,)](
        buf_matmul, buf_mean,
        B, H,
        BLOCK_H=64
    )
    # buf_mean: [H] — 沿 batch 维度的均值

    # ═══════════════════════════════════════
    # Kernel 3: Triton fused diff + pow
    # ═══════════════════════════════════════
    buf_pow = torch.empty([B, H], device='cuda', dtype=torch.float32)
    triton_fused_diff_pow[(total // 1024,)](
        buf_matmul, buf_mean, buf_pow,
        B, H, total,
        BLOCK=1024
    )
    # buf_pow: [B, H] — (x - mean)^2 的结果
    # 注意: diff 是中间值，未写入全局内存

    # ═══════════════════════════════════════
    # Kernel 4: Triton reduction var
    # ═══════════════════════════════════════
    buf_var = torch.empty([H], device='cuda', dtype=torch.float32)
    triton_reduction_var[(H,)](
        buf_pow, buf_var,
        B, H,
        BLOCK_H=64
    )
    # buf_var: [H] — 沿 batch 维度的方差

    # ═══════════════════════════════════════
    # Kernel 5: Triton fused norm + bn + relu
    # ═══════════════════════════════════════
    buf_out = torch.empty([B, H], device='cuda', dtype=torch.float32)
    triton_fused_norm_bn_relu[(total // 1024,)](
        buf_matmul,  # diff 存储在 buf_matmul... 不对
        # 实际上 diff 没有被单独存储！
        # 回顾融合组 3：diff + pow 被融合，但 diff 不被 store
        # 所以 norm 需要 diff 时，需要从哪里读？
        # 答案：融合组 3 中 diff 实际上被 store 到了 buf_diff
        # 让我们修正：
        buf_diff, buf_var, arg2, arg3, buf_out,
        total, 1e-5,
        BLOCK=1024
    )
    # buf_out: [B, H] — 最终输出 relu(norm * weight + bias)

    return (buf_out,)
```

等一下——上面的注释揭示了一个重要细节。融合组 3 中 `diff` 是中间值不需要被 store，但融合组 5 的 `norm` 操作又需要 `diff` 的值。这该如何处理？

实际上，Inductor 的 Scheduler 在做融合决策时会检测这种跨融合组的依赖。当发现融合组 5 需要读取 `diff`（融合组 3 的中间值），但 `diff` 没有被 store 时，Scheduler 会调整策略：**让融合组 3 也 store `diff`**，或者将 `diff` 也纳入融合组 5。

在我们的例子中，最可能的调整是让融合组 3 同时输出 `diff` 和 `pow`（两个输出缓冲区），这样融合组 5 可以读取 `diff`。这对应着 Triton kernel 中多输出（tuple output）的模式。

修正后的 Kernel 3：

```python
@triton.jit
def triton_fused_diff_pow(
    matmul_ptr, mean_ptr, diff_ptr, pow_ptr,
    B, H, total: tl.int32,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total

    h_idx = offsets % H
    matmul_val = tl.load(matmul_ptr + offsets, mask=mask)
    mean_val = tl.load(mean_ptr + h_idx)

    diff = matmul_val - mean_val
    pow_val = diff * diff

    # 同时输出 diff 和 pow
    tl.store(diff_ptr + offsets, diff, mask=mask)
    tl.store(pow_ptr + offsets, pow_val, mask=mask)
```

修正后的 Kernel 5：

```python
@triton.jit
def triton_fused_norm_bn_relu(
    diff_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    N: tl.int32, eps: tl.float32,
    BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    h_idx = offsets % H

    # 加载所有输入
    diff_val = tl.load(diff_ptr + offsets, mask=mask)
    var_val = tl.load(var_ptr + h_idx)

    # 内联所有计算
    std = tl.sqrt(var_val + eps)
    norm = diff_val / std
    scaled = norm * tl.load(weight_ptr + h_idx)
    bn_out = scaled + tl.load(bias_ptr + h_idx)
    result = tl.where(bn_out > 0, bn_out, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)
```

### 9.5.3 Wrapper 函数中的内存管理

Wrapper 函数中每个 `torch.empty(...)` 调用分配一个中间缓冲区。在我们的例子中：

| 缓冲区 | 大小 | 生命周期 | 由谁写入 | 由谁读取 |
|--------|------|---------|---------|---------|
| `buf_matmul` | `B * H * 4` bytes | Kernel 1 → Kernel 3 | cuBLAS | Kernel 2, 3 |
| `buf_mean` | `H * 4` bytes | Kernel 2 → Kernel 3 | Kernel 2 | Kernel 3 |
| `buf_diff` | `B * H * 4` bytes | Kernel 3 → Kernel 5 | Kernel 3 | Kernel 5 |
| `buf_pow` | `B * H * 4` bytes | Kernel 3 → Kernel 4 | Kernel 3 | Kernel 4 |
| `buf_var` | `H * 4` bytes | Kernel 4 → Kernel 5 | Kernel 4 | Kernel 5 |
| `buf_out` | `B * H * 4` bytes | Kernel 5 → 返回 | Kernel 5 | 返回值 |

总临时内存：约 `3 * B * H * 4 + 2 * H * 4` bytes。对于 `B=128, H=512`，大约 750 KB——微不足道。

### 9.5.4 从 Wrapper 到执行

生成的 wrapper 函数被 `PyCodeCache` 加载为真正的 Python callable。当用户调用编译后的模型时：

```python
compiled_model = torch.compile(model)
output = compiled_model(x, w, b)  # 触发 wrapper 调用
```

实际执行路径：

```
用户调用 compiled_model(x, w, b)
    → Dynamo dispatch
        → AOTAutograd forward wrapper
            → Inductor 生成的 call(args)
                → Kernel 1: cuBLAS matmul
                → Kernel 2: Triton reduction mean
                → Kernel 3: Triton fused diff + pow
                → Kernel 4: Triton reduction var
                → Kernel 5: Triton fused norm + bn + relu
            → 返回 (buf_out,)
```

---

## 9.6 全流程类实例生命周期回顾

### 9.6.1 完整生命周期时间线

下表展示了编译过程中每一个核心类实例的创建与销毁时机，以及它们之间的关键交互。这是全书的"类关系总表"——将前八章分散介绍的各类串联为一条时间线。

| 阶段 | 类实例 | 创建时机 | 销毁时机 | 生命周期 | 关键交互 |
|------|-------|---------|---------|---------|---------|
| **Dynamo** | FX Graph | `@torch.compile` 首次调用 | 编译完成 | 整个编译过程 | 被 GraphLowering 遍历 |
| **AOT** | Decomposition Table | AOTAutograd 初始化 | 编译完成 | 整个编译过程 | 分解复合算子 |
| **Setup** | `GraphLowering` | `compile_fx_inner()` | 编译完成 | 整个编译过程 | 持有 `V.graph`，遍历 FX Graph |
| **Setup** | `FakeTensorMode` | `compile_fx_inner()` | 编译完成 | 整个编译过程 | 通过 `V.fake_mode` 访问 |
| **Phase 0** | `GraphLowering`（const） | 常量折叠开始 | 常量折叠结束 | 短暂 | 处理编译期常量 |
| **Phase 1** | `InputBuffer` | 每个 placeholder 节点 | 编译完成 | 长 | 存入 `V.graph.buffers` |
| **Phase 1** | `ComputedBuffer` | 每个 call_function 节点 | 编译完成 | 长 | 持有 `inner_fn` 闭包 |
| **Phase 1** | `ExternKernel` | matmul lowering | 编译完成 | 长 | 嵌入在 ComputedBuffer 中 |
| **Phase 1** | `Reduction` | mean/var lowering | 编译完成 | 长 | 嵌入在 ComputedBuffer 中 |
| **Phase 1** | `Pointwise` | diff/pow/... lowering | 编译完成 | 长 | 嵌入在 ComputedBuffer 中 |
| **Phase 2** | `Scheduler` | `_update_scheduler()` | 编译完成 | 长 | 持有 nodes 列表和 backends 字典 |
| **Phase 2** | `SchedulerNode` | 每个 IR buffer | 编译完成 | 长 | 包装 IR buffer，参与排序 |
| **Phase 2** | `_RecordLoadStoreInner` | 依赖分析每个节点 | 该节点分析完毕 | 极短（每节点） | 安装为 `V.ops`，记录 reads/writes |
| **Phase 2** | `FusedSchedulerNode` | 融合决策 | 编译完成 | 长 | 替换多个 SchedulerNode |
| **Phase 2** | `TritonScheduling` | 首个 CUDA 节点 | 编译完成 | 长 | 由 `Scheduler.backends` 持有 |
| **Phase 3** | `TritonKernel` | 每个融合组 codegen | codegen 完毕 | 短暂（每融合组） | 安装为 `V.kernel`，生成代码 |
| **Phase 3** | `CSEProxy` | `kernel.__enter__` | `kernel.__exit__` | 极短（每 kernel） | 安装为 `V.ops`，CSE + 代码生成 |
| **Phase 3** | `TritonKernelOverrides` | `kernel.__enter__` | `kernel.__exit__` | 极短（每 kernel） | 被 CSEProxy 包装 |
| **Phase 4** | `PythonWrapperCodegen` | wrapper 生成开始 | 编译完成 | 长 | 持有所有 kernel 代码片段 |
| **Phase 4** | `PyCodeCache` entry | wrapper 加载 | 进程退出 | 永久 | 缓存编译结果 |

### 9.6.2 Handler 生命周期时序图

Handler 的安装和切换是 Inductor 编译过程中最精妙的机制。下图展示了完整的 Handler 切换时序：

```
时间 ──────────────────────────────────────────────────────────────────→

V.ops Handler 切换序列:

[NullHandler]                  ← 初始状态
    │
    │ Phase 1: GraphLowering
    │ (lowering 过程中 V.ops 保持默认，因为 inner_fn 只是被存储不被执行)
    │
[MockHandler]                  ← GraphLowering 完成后的默认
    │
    │ Phase 2: Scheduler._init()
    │
    ├── [_RecordLoadStoreInner(matmul)]  ──→ 分析 matmul 的依赖
    ├── [_RecordLoadStoreInner(mean)]    ──→ 分析 mean 的依赖
    ├── [_RecordLoadStoreInner(diff)]    ──→ 分析 diff 的依赖
    ├── [_RecordLoadStoreInner(pow)]     ──→ 分析 pow 的依赖
    ├── [_RecordLoadStoreInner(var)]     ──→ 分析 var 的依赖
    ├── [_RecordLoadStoreInner(vareps)]  ──→ ...
    ├── [_RecordLoadStoreInner(std)]     ──→ ...
    ├── [_RecordLoadStoreInner(norm)]    ──→ ...
    ├── [_RecordLoadStoreInner(scaled)]  ──→ ...
    ├── [_RecordLoadStoreInner(bn_out)]  ──→ ...
    └── [_RecordLoadStoreInner(relu_out)]──→ 分析 relu_out 的依赖
    │
    │ Phase 2 续: ValueRangeAnalysis（可选）
    │
    ├── [ValueRangeAnalysis]  ──→ 对所有节点做值域推断
    │
[MockHandler]                  ← Phase 2 完成
    │
    │ Phase 3: Kernel Codegen
    │
    │ Kernel 1 (cuBLAS): 无 Handler 切换
    │
    │ Kernel 2 (reduction mean):
    ├── [CSEProxy(TritonOverrides)]  ──→ __enter__
    │       │ 执行 inner_fn
    │       └─→ __exit__
    │
    │ Kernel 3 (fused diff+pow):
    ├── [CSEProxy(TritonOverrides)]  ──→ __enter__
    │       │ 执行 fused inner_fn
    │       └─→ __exit__
    │
    │ Kernel 4 (reduction var):
    ├── [CSEProxy(TritonOverrides)]  ──→ __enter__
    │       │ 执行 inner_fn
    │       └─→ __exit__
    │
    │ Kernel 5 (fused norm+bn+relu):
    ├── [CSEProxy(TritonOverrides)]  ──→ __enter__
    │       │ 执行 fused inner_fn
    │       └─→ __exit__
    │
[MockHandler]                  ← Phase 3 完成
    │
    │ Phase 4: WrapperCodegen（不需要 Handler）
    │
[NullHandler]                  ← 编译结束，恢复初始状态
```

**关键观察**：

1. **同一个 `inner_fn` 被执行了多次**，每次安装不同的 Handler。例如 `diff` 的 `inner_fn = lambda idx: ops.sub(ops.load("matmul", idx), ops.load("mean", idx_0))` 被执行了至少 3 次：
   - 依赖分析阶段：`_RecordLoadStoreInner` 记录 reads = {matmul, mean}
   - 值域分析阶段：`ValueRangeAnalysis` 推断输出范围
   - 代码生成阶段：`CSEProxy(TritonOverrides)` 生成 `tmp = matmul_val - mean_val`

2. **Handler 的嵌套深度最多 2 层**：`CSEProxy` 包装 `TritonOverrides`。这不像洋葱模型那样可以无限嵌套，保持了调用的简洁性。

3. **`_RecordLoadStoreInner` 是"即用即弃"的**：每个节点创建一个新实例，分析完毕后立即销毁。这避免了状态污染——不同节点的依赖信息不会混淆。

### 9.6.3 V 上下文的完整状态机

除了 `V.ops`，其他 V 上下文也在编译过程中经历状态变化：

| V 上下文 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|---------|---------|---------|---------|---------|
| `V.graph` | `GraphLowering` | `GraphLowering` | `GraphLowering` | `GraphLowering` |
| `V.kernel` | `NullKernelHandler` | `NullKernelHandler` | 交替：`TritonKernel` ↔ `NullKernelHandler` | `NullKernelHandler` |
| `V.ops` | `MockHandler` | 交替：`_RecordLoadStore`/`ValueRange`/`Mock` | 交替：`CSEProxy`/`Mock` | `MockHandler` |
| `V.fake_mode` | `FakeTensorMode` | `FakeTensorMode` | `FakeTensorMode` | `FakeTensorMode` |
| `V.scheduler` | 未创建 | `Scheduler` | `Scheduler` | `Scheduler` |

---

## 9.7 全景回顾：从用户代码到 GPU 执行的七层抽象

### 9.7.1 数据流变换追踪

让我们追踪一个具体的张量值——输入 `x` 的第一行 `x[0, :]`——在编译流水线中的形态变化：

| 阶段 | 形态 | 表示方式 | 所在类 |
|------|------|---------|--------|
| 用户代码 | Python 张量 | `torch.Tensor` | 用户空间 |
| Dynamo | FX Node | `placeholder[target=x]` | `fx.Node` |
| AOT | 分解后 FX Node | `placeholder[target=x]`（不变） | `fx.Node` |
| Phase 1 | IR Buffer | `InputBuffer("x", ...)` | `InputBuffer` |
| Phase 2 | SchedulerNode | `SchedulerNode(buf=x)` | `SchedulerNode` |
| Phase 3 | Triton 参数 | `arg0 = args[0]` → `ptr_in` | `TritonKernel.args` |
| Phase 4 | Wrapper 参数 | `arg0 = args[0]` | `PythonWrapperCodegen` |
| 运行时 | GPU 内存 | CUDA device pointer | cuBLAS / Triton |

每个阶段，同一个数据有不同的表示形式。这就是编译器的本质——**在不改变语义的前提下，不断变换数据的表示形式，直到达到最适合目标硬件的形式**。

### 9.7.2 "同一个闭包，三种命运"

本书反复强调的核心设计——`inner_fn` 闭包被同一个 Handler 体系多次执行——在我们的端到端例子中得到了完美体现。以 `diff` 的 `inner_fn = lambda idx: ops.sub(ops.load("matmul", idx), ops.load("mean", idx_0))` 为例：

**命运 1：依赖分析**

```python
# V.ops = _RecordLoadStoreInner(node=diff)
inner_fn(idx)
    → ops.load("matmul", idx) → 记录 "diff 读取 matmul"
    → ops.load("mean", idx_0) → 记录 "diff 读取 mean"
    → ops.sub(...)            → 记录 "diff 写入 diff"
结果: diff.reads = {"matmul", "mean"}, diff.writes = {"diff"}
```

**命运 2：值域推断**

```python
# V.ops = ValueRangeAnalysis
inner_fn(idx)
    → ops.load("matmul", idx) → 返回 ValueRanges([min_matmul, max_matmul])
    → ops.load("mean", idx_0) → 返回 ValueRanges([min_mean, max_mean])
    → ops.sub(r1, r2)         → 返回 ValueRanges([min_matmul - max_mean, max_matmul - min_mean])
结果: diff 的输出值域 = [min_matmul - max_mean, max_matmul - min_mean]
```

**命运 3：代码生成**

```python
# V.ops = CSEProxy(TritonKernelOverrides)
inner_fn(idx)
    → ops.load("matmul", idx) → 生成 "tmp0 = tl.load(matmul_ptr + offsets, mask=mask)"
    → ops.load("mean", idx_0) → 生成 "tmp1 = tl.load(mean_ptr + h_offset)"
    → ops.sub(tmp0, tmp1)     → 生成 "tmp2 = tmp0 - tmp1"
结果: Triton kernel 中的减法代码
```

三次执行，同一个 callable，三种完全不同的效果——这正是第三章所述的抽象解释框架的工程实践。

---

## 9.8 工厂类比：PyTorch Inductor 编译全景

作为全书的总结，让我们用一个完整的工厂类比来描绘 Inductor 的整体架构。

### 9.8.1 类比映射表

| 工厂角色 | Inductor 类/模块 | 职责 |
|---------|-----------------|------|
| **客户订单** | 用户 Python 函数 | 描述"我要什么"（计算逻辑） |
| **订单录入员** | TorchDynamo | 抄录订单为标准格式（FX Graph） |
| **生产规划师** | AOTAutograd | 拆分前向/反向产线，分解复杂工序 |
| **蓝图设计师** | `GraphLowering` | 将每个工序翻译为标准蓝图（IR + inner_fn） |
| **生产调度员** | `Scheduler` | 决定生产顺序，合并可并行的工序 |
| **车间主任** | `BaseScheduling` / `TritonScheduling` | 管理特定车间（GPU/CPU）的生产方法 |
| **数控机床** | `TritonKernel` / `CppKernel` | 执行加工（代码生成），产出零件（kernel 代码） |
| **刀具附件** | Handler (`CSEProxy`, `_RecordLoadStore`, ...) | 同一台机床安装不同刀具做不同加工 |
| **质检员** | `CSEProxy` | 去重（CSE）、类型检查、值域验证 |
| **总装线** | `PythonWrapperCodegen` | 将所有零件组装为最终产品（wrapper 函数） |
| **仓库** | `PyCodeCache` | 缓存成品，下次相同订单直接取货 |

### 9.8.2 订单的完整旅程

```
客户下单: "我要一个 matmul → batch_norm → relu 的计算"
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  订单录入: TorchDynamo                                          │
│  "好的，我抄录一下：placeholder(x), placeholder(w),              │
│   matmul(x,w), batch_norm(...), relu(...), output(...)"        │
│  产出: FX Graph (6 个节点的 DAG)                                │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  生产规划: AOTAutograd                                          │
│  "batch_norm 太复杂了，拆成 mean/sub/pow/div/sqrt/mul/add。     │
│   另外，训练模式的话还需要规划反向产线。"                          │
│  产出: 分解后 FX Graph (12 个节点)                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  蓝图设计: GraphLowering                                        │
│  "每个工序我都画一张蓝图：                                       │
│   - matmul: 外包给 cuBLAS 专业厂 (ExternKernel)                 │
│   - mean: reduction 类型，蓝图是 λidx.load(matmul,idx)          │
│   - diff: pointwise，蓝图是 λidx.sub(load(matmul),load(mean))   │
│   - ... 以此类推"                                               │
│  产出: 10 个 IR Buffer（每个携带 inner_fn 闭包）                 │
│                                                                 │
│  [关键机制] inner_fn 是 Python callable，不是数据结构             │
│  设计师只画蓝图，不负责解读蓝图——解读交给后续阶段                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  生产调度: Scheduler                                            │
│                                                                 │
│  Step 1: 依赖分析                                               │
│  "让我看看哪些工序有依赖关系...                                  │
│   装上 _RecordLoadStoreInner 刀具，把每张蓝图跑一遍"             │
│                                                                 │
│  Step 2: 拓扑排序                                               │
│  "matmul → mean → diff → pow → var → ... → relu_out"           │
│                                                                 │
│  Step 3: 融合决策                                               │
│  "这些 pointwise 工序可以合并：                                  │
│   - diff + pow → 同一台机器                                     │
│   - vareps + std + norm + scaled + bn_out + relu → 同一台机器    │
│   mean 和 var 是 reduction，必须单独开机器"                      │
│                                                                 │
│  产出: 5 个融合组，每组一个 kernel 调用                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  车间加工: TritonScheduling + TritonKernel                       │
│                                                                 │
│  对每个融合组:                                                   │
│    1. 开启机床 (TritonKernel.__enter__)                          │
│       安装刀具: V.ops = CSEProxy(TritonOverrides)                │
│    2. 执行蓝图 (inner_fn)                                       │
│       每个 ops.xxx 调用经过 CSEProxy → TritonOverrides → 代码   │
│    3. 关闭机床 (TritonKernel.__exit__)                           │
│       收集生成的代码                                             │
│                                                                 │
│  Kernel 1: cuBLAS (外包，不开机床)                               │
│  Kernel 2: reduction mean (开机床，生成 Triton reduction 代码)   │
│  Kernel 3: fused diff+pow (开机床，生成 Triton pointwise 代码)  │
│  Kernel 4: reduction var (开机床，生成 Triton reduction 代码)    │
│  Kernel 5: fused norm+bn+relu (开机床，生成 6-op 融合代码)      │
│                                                                 │
│  产出: 4 个 Triton kernel 源码 + 1 个 cuBLAS 调用               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  总装线: PythonWrapperCodegen                                   │
│                                                                 │
│  "把所有零件按顺序组装：                                         │
│                                                                 │
│   def call(args):                                               │
│       buf_matmul = empty(...)                                   │
│       cublas_matmul(x, w, out=buf_matmul)        # 零件 1       │
│       buf_mean = empty(...)                                     │
│       triton_reduction_mean[...](...)             # 零件 2       │
│       buf_diff = empty(...)                                     │
│       buf_pow = empty(...)                                      │
│       triton_fused_diff_pow[...](...)             # 零件 3       │
│       buf_var = empty(...)                                      │
│       triton_reduction_var[...](...)              # 零件 4       │
│       buf_out = empty(...)                                      │
│       triton_fused_norm_bn_relu[...](...)         # 零件 5       │
│       return (buf_out,)"                                        │
│                                                                 │
│  产出: 一个可调用的 Python 函数                                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  入库: PyCodeCache                                              │
│  "存好了，下次同样的模型直接取"                                  │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
              用户调用: output = compiled_model(x, w, b)
              GPU 执行: 5 个 kernel 调用 → 最终输出
```

### 9.8.3 全书知识地图

至此，我们已经走完了从用户代码到 GPU 执行的完整旅程。以下是全书八章的知识地图，以及本章如何将它们串联：

```
┌─────────────────────────────────────────────────────────────────┐
│                    第九章：端到端综合追踪                          │
│                   （本章：串联所有知识点）                          │
└──────────┬──────────┬──────────┬──────────┬──────────┬─────────┘
           │          │          │          │          │
     ┌─────┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
     │ 第二章    │ │ 第三章  │ │ 第四~六章│ │ 第七章  │ │ 第八章  │
     │Virtualized│ │ Handler│ │ Scheduler│ │ Kernel  │ │ Wrapper │
     │          │ │ 协议    │ │ 融合决策 │ │ 代码生成│ │ 模块组装│
     │ V.graph  │ │ OpsHandler│ │ 拓扑排序│ │ TritonKernel│ │PythonWrapper│
     │ V.ops    │ │ CSEProxy │ │ FusedNode│ │ CSEProxy│ │ CodeGen│
     │ 动态作用域│ │ 抽象解释 │ │ 依赖分析 │ │ 代码缓冲│ │ PyCodeCache│
     └──────────┘ └────────┘ └────────┘ └────────┘ └────────┘
```

**第二章（Virtualized）** 提供了全局状态管理的基石。在本章的追踪中，`V.graph`、`V.kernel`、`V.ops` 的每一次切换都是 Virtualized 动态作用域的实际应用。

**第三章（Handler 协议）** 定义了 IR 闭包的"解释器"。在本章中，同一个 `inner_fn` 被三种不同的 Handler 解释执行——`_RecordLoadStoreInner`（依赖分析）、`ValueRangeAnalysis`（值域推断）、`CSEProxy(TritonOverrides)`（代码生成）——完美体现了抽象解释的思想。

**第四至六章（Scheduler）** 解决了"如何组织计算"的问题。在本章中，Scheduler 将 11 个独立 IR 节点优化为 5 个融合 kernel，展示了融合启发式的实际效果。

**第七章（Kernel）** 是代码生成的"引擎"。在本章中，`TritonKernel` 作为 context manager 接管 `V.ops`，将 IR 闭包翻译为 Triton GPU kernel 代码，展示了 Kernel 的生命周期管理。

**第八章（WrapperCodegen）** 完成了最后的组装。在本章中，`PythonWrapperCodegen` 将 5 个 kernel 代码片段整合为一个可调用的 Python 函数，展示了从编译产物到运行时执行的桥梁。

---

## 9.9 总结：Inductor 的设计哲学

通过这个端到端的追踪，我们可以提炼出 PyTorch Inductor 的三大设计哲学：

### 哲学一：闭包即 IR（Closure-as-IR）

传统编译器（如 LLVM、XLA）用数据结构表示 IR——每个指令是一个对象，每个基本块是一个列表。Inductor 选择了一条不同的路：**用 Python callable（闭包）表示 IR 的循环体**。

这个选择的代价是：IR 不是静态可分析的（你不能"看"一个闭包就知道它做了什么），必须执行它才能获取信息。但收益是巨大的——**闭包可以自然地被不同的 Handler "解释"执行**，无需维护 IR 的数据结构表示和遍历逻辑。这是 Inductor 能用相对简洁的代码实现多遍分析的关键。

### 哲学二：Handler 即抽象域（Handler-as-Abstract-Domain）

`OpsHandler[T]` 的泛型参数 `T` 不是一个简单的类型标注——它定义了一个完整的语义域。每个 Handler 实现都是在这个语义域上的一组抽象转移函数。

这个设计直接来自编译理论中的抽象解释（Abstract Interpretation）。Inductor 的贡献在于将这一理论以极其 Pythonic 的方式落地——不是构建复杂的框架，而是利用 Python 的动态分派和上下文管理器，让同一个闭包在不同语义域中无缝切换。

### 哲学三：延迟决策，后端主导（Late Binding, Backend-Driven）

Inductor 的另一个核心哲学是：**尽可能推迟优化决策，让最了解硬件的后端做最终决定**。

- Phase 1（GraphLowering）不做融合决策——它只是忠实地为每个算子创建 IR
- Phase 2（Scheduler）做融合决策，但融合启发式由后端调度器（`TritonScheduling`、`CppScheduling`）定义
- Phase 3（Kernel）做代码生成，但具体生成什么代码由后端的 `OpOverrides` 决定

这意味着，同一个 IR 可以在不同后端上产生完全不同的融合策略和代码。GPU 后端可以激进地融合（因为全局内存昂贵），CPU 后端可能更保守（因为缓存层次不同）。后端只需要实现自己的 `BaseScheduling` 子类和 `Kernel` 子类，就能完整地接入 Inductor 的编译流水线。

---

### 结语

从第二章到第八章，我们逐个拆解了 Inductor 的核心类——`Virtualized`、`OpsHandler`、`Scheduler`、`Kernel`、`WrapperCodegen`。本章通过一个端到端的追踪，将这些散落的部件组装为一台完整的编译器机器。

回顾整本书的旅程：

1. **第二章**中，我们看到了 `Virtualized` 如何用动态作用域管理全局状态——这是 Inductor 的"神经系统"。
2. **第三章**中，我们理解了 `Handler` 协议如何让同一个 IR 闭包产生多种分析结果——这是 Inductor 的"免疫系统"。
3. **第四至六章**中，我们追踪了 `Scheduler` 如何将独立 IR 节点融合为高效 kernel 组——这是 Inductor 的"大脑"。
4. **第七章**中，我们深入了 `Kernel` 的代码生成机制——这是 Inductor 的"双手"。
5. **第八章**中，我们见证了 `WrapperCodegen` 如何将所有产物组装为可执行函数——这是 Inductor 的"最后一公里"。
6. **本章**中，我们将这一切串联为一条完整的流水线，看到了从用户代码到 GPU 执行的全景。

PyTorch Inductor 的设计证明了一件事：**编译器不一定需要庞大的 IR 框架和复杂的多遍管理器。用 Python 的动态特性——闭包、上下文管理器、元编程——可以构建出一个既灵活又高效的工业级编译器。** 这也许不是编译器设计的唯一道路，但它无疑是一条充满 Python 智慧的道路。
