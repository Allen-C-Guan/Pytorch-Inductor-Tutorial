# 阶段四：理解调度与融合 —— 从 IR 图到优化 Kernel 组的映射

> **定位**：本文档深入 Inductor 编译管线中最具性能影响力的阶段——**调度与融合（Scheduling & Fusion）**。读完本文档，你应当理解：IR 图如何被转换为 SchedulerNode、依赖分析的原理与三类依赖类型、贪心融合算法的核心机制（can_fuse + score_fusion）、内存规划如何减少峰值内存，以及从 Lowering 到代码生成的完整桥梁。
>
> **权威参考**：
> - PyTorch 2 论文 (ASPLOS 2024): *"PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"* — Section 4.4 (Scheduling), Section 6.4 (Sources of Speedups), Table 4 (Ablation Study)
> - TorchInductor 设计帖: [dev-discuss.pytorch.org/t/torchinductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
> - Inductor 文件结构讨论: [dev-discuss.pytorch.org/t/inductor-file-structure-explanation](https://dev-discuss.pytorch.org/t/inductor-file-structure-explanation/1860)
>
> **源码版本**：基于 `main` 分支截取（最近变更: commit `d63aab0`, 2026-04-07），行号可能随代码演进偏移，请以实际源码为准。
>
> **系列导航**：[全景总览](inductor_overview.md) | [← 阶段一：全局观](phase1_global_view.md) | [← 阶段二：FX 优化](phase2_fx_optimization.md) | [← 阶段三：Lowering](phase3_lowering.md) | **阶段四：调度与融合** | [阶段五：代码生成 →](phase5_codegen.md)

---

## 一、设计思想 / 设计哲学

### 1.1 为什么需要调度与融合？

Phase 3 中，Lowering 将 FX Graph 翻译成了 Inductor IR 图——一组独立的 IR 节点（Pointwise、Reduction、ExternKernel 等）。如果直接将每个 IR 节点生成一个独立 kernel，性能会很差：

1. **内存带宽瓶颈**：每个 kernel 的输出必须写回全局内存，下一个 kernel 再从全局内存读取。如果连续两个 pointwise 操作（如 `relu(x) + y`）各生成一个 kernel，中间结果 `relu(x)` 要经历一次 store + 一次 load，白白浪费 2 倍内存带宽。
2. **Kernel Launch 开销**：每次 kernel launch（GPU 上约 5-10μs）是固定开销。如果 100 个小操作变成 100 个 kernel，launch 开销可能超过计算本身。
3. **内存浪费**：每个中间结果分配独立 buffer，峰值内存膨胀。

**调度与融合的核心使命**：将多个 IR 节点组合为少量高效 kernel，消除不必要的内存流量和 launch 开销。

### 1.2 Inlining vs Fusion：两个不同阶段的互补优化

论文 Section 6.4 明确指出，kernel 组合发生在**两个不同阶段**：

> "The biggest speedups in TorchInductor come from combining pointwise, reduction, and scatter kernels together into a smaller number of fused kernels. In TorchInductor, these kernel combinations happen in two places: 1) **Inlining** happens during lowering, and duplicates the body of pointwise kernels into all their consumers when thresholds are met. 2) **Fusion** happens during scheduling, and combines remaining kernels together, and also does **horizontal consumer/consumer fusions**."

```
Lowering 阶段 (Phase 3)：Inlining
├── 将 pointwise body 复制到消费者闭包中
├── 消除中间结果的物化（realize）
├── 只处理 producer-consumer 垂直组合
└── 结果：部分 IR 节点被内联消失，剩余节点仍是独立的

Scheduling 阶段 (Phase 4)：Fusion
├── 将剩余独立节点组合为 FusedSchedulerNode
├── 支持垂直融合（producer-consumer）和水平融合（consumer-consumer）
├── 包含依赖分析、内存规划、执行排序
└── 结果：IR 节点被分组为 kernel 组，准备代码生成
```

**消融实验数据**（论文 Table 4，HuggingFace float16, A100 GPU）：

| 优化 | 推理加速 | 训练加速 | 移除后损失 |
|------|---------|---------|-----------|
| 全部优化 | 1.91x | 1.45x | — |
| 去 fusion | 1.68x | 1.27x | -0.23 / -0.18 |
| 去 inlining | 1.58x | 1.31x | -0.33 / -0.14 |
| **去 fusion + inlining** | **0.80x** | **0.59x** | **-1.11 / -0.86（减速！）** |

关键洞察：**去掉 fusion + inlining 后 Inductor 反而比 eager 慢**。因为分解（decomposition）将大算子拆成了很多小算子，必须通过 Inlining + Fusion 重新组合才能恢复性能。两者是**协同增效**的——联合效果远大于各自效果之和。

### 1.3 调度器的三根支柱

调度与融合由三个紧密协作的子系统组成：

| 支柱 | 核心文件 | 职责 |
|------|---------|------|
| **Scheduler** | `scheduler.py` | 将 IR 节点转换为 SchedulerNode，执行融合、排序、触发代码生成 |
| **Dependencies** | `dependencies.py` | 分析 IR 节点之间的内存读写依赖，为融合决策提供基础 |
| **Memory** | `memory.py` | 分析 buffer 生命周期，规划内存复用，优化峰值内存 |

加上后端注册系统：

| 辅助系统 | 核心文件 | 职责 |
|----------|---------|------|
| **Backend Registration** | `codegen/common.py` | 后端注册机制，让不同后端（Triton/C++/CUTLASS）实现各自的融合和代码生成策略 |
| **Graph Bridge** | `graph.py` | 从 Lowering 到 Scheduling 的桥梁，触发 Scheduler 创建和代码生成 |

### 1.4 核心设计理念：贪心融合 + 后端策略

论文 Section 4.4 描述了调度器的贪心融合算法：

> "In a loop, until no additional fusions remain (since some fusions can open additional fusion opportunities), TorchInductor will perform the following greedy algorithm: 1) find all fusion opportunities; 2) score each of the fusion opportunities and sort by that score; 3) for each fusion opportunity, check if that fusion remains legal and if so apply it."

**两个关键函数**（论文原文）：

- `Scheduler.can_fuse(node1, node2)` → "returns True if two nodes can be fused together. This checks dependency edges, and also checks many other properties to ensure correctness of a fusion."
- `Scheduler.score_fusion(node1, node2)` → "used to order different fusion possibilities. The fusion score orders fusions by: 1) the category of the fusion (e.g. pointwise/reduction/template); 2) estimated bytes of memory traffic saved by the fusion; and 3) shorter distance between nodes in the original graph."

**后端策略模式**：不同后端对融合有不同约束。例如：
- **Triton（GPU）**：支持 reduction-broadcast-reduction 融合
- **C++（CPU）**：不支持上述融合
- 每个后端通过 `can_fuse_vertical()`、`can_fuse_horizontal()` 实现自己的策略

---

## 二、主体核心调用栈

### 2.1 从 Lowering 到 Scheduling 的完整调用链

```
compile_fx.py:1473  GraphLowering(gm, example_inputs, ...)
    │
    ▼ compile_fx.py:1508
    graph.run(*example_inputs)
    │  ← Lowering 阶段（Phase 3），产出 IR 图
    │
    ▼ compile_fx.py:1538 / 1552
    graph.codegen() 或 graph.codegen_with_cpp_wrapper()
    │
    ├── graph.py:2546  GraphLowering.codegen()
    │       ├── graph.py:2535  _update_scheduler()
    │       │       └── scheduler.py:3084  Scheduler(self.operations)
    │       │           │  ← 核心入口：从 IR 操作创建 Scheduler
    │       │           │
    │       │           └── scheduler.py:3088  Scheduler._init(nodes)
    │       │               ├── [Step 1] 创建 SchedulerNode (L3103)
    │       │               │   └── create_scheduler_node(n)
    │       │               │       ├── ir.ComputedBuffer/TemplateBuffer → SchedulerNode
    │       │               │       ├── ir.ExternKernel → ExternKernelSchedulerNode
    │       │               │       └── No-op → NopKernelSchedulerNode
    │       │               │
    │       │               ├── [Step 2] 计算依赖 (L3155)
    │       │               │   └── compute_dependencies()
    │       │               │       └── dependencies.py:659  extract_read_writes()
    │       │               │           └── RecordLoadStore 作为 V.ops handler
    │       │               │               拦截 ops.load → 记录 MemoryDep (reads)
    │       │               │               拦截 ops.store → 记录 MemoryDep (writes)
    │       │               │
    │       │               ├── [Step 3] 拓扑排序 (L3156)
    │       │               │   └── topological_sort_schedule()
    │       │               │
    │       │               ├── [Step 4] 死代码消除 (L3157)
    │       │               │   └── dead_node_elimination()
    │       │               │
    │       │               ├── [Step 5] 计算 ancestors (L3159)
    │       │               ├── [Step 6] 计算 input distances (L3160)
    │       │               ├── [Step 7] 创建 Foreach 节点 (L3168)
    │       │               ├── [Step 8] 分配 CUDA streams (L3182)
    │       │               │
    │       │               ├── [Step 9] ★ 融合 (L3189)
    │       │               │   └── fuse_nodes()
    │       │               │       └── [最多 10 轮迭代]
    │       │               │           └── fuse_nodes_once()
    │       │               │               ├── get_possible_fusions()
    │       │               │               │   └── can_fuse() → 合法性检查
    │       │               │               ├── _try_fusion_pairs()
    │       │               │               │   └── score_fusion_key() → 评分排序
    │       │               │               │   └── speedup_by_fusion() → 利润评估
    │       │               │               │   └── fuse_two_nodes() → 执行融合
    │       │               │               └── _finish_pending_fusions()
    │       │               │
    │       │               ├── [Step 10] 合并循环 (L3202)
    │       │               ├── [Step 11] 创建 Combo Kernel (L3209)
    │       │               │
    │       │               └── [Step 12] ★ 内存优化排序 (L3219)
    │       │                   └── reorder_for_peak_memory()
    │       │                       └── 尝试 LPMF/BFS/DFS 三种排序
    │       │                       └── 选择峰值内存最小的方案
    │       │
    │       ├── graph.py:2554  scheduler.codegen()
    │       │   └── scheduler.py:7325  Scheduler.codegen()
    │       │       └── [遍历所有 SchedulerNode]
    │       │           ├── Template → backend.codegen_template()
    │       │           ├── Extern → codegen_extern_call()
    │       │           ├── Foreach → backend.codegen_combo_kernel()
    │       │           └── Fused/Scheduler → backend.codegen_node()
    │       │               └── [后端生成 Triton/C++ kernel 代码]
    │       │
    │       └── graph.py:2561  wrapper_code.generate()
    │           └── 生成 Python/C++ wrapper 代码
    │
    ▼ 编译产物：CompiledFxGraph (可执行的 Python Callable)
```

### 2.2 关键调用节点说明

| 调用 | 文件:行号 | 作用 |
|------|----------|------|
| `Scheduler(operations)` | scheduler.py:3084 | 将 IR 操作列表转换为调度图 |
| `create_scheduler_node()` | scheduler.py:3103 | IR Operation → SchedulerNode 映射 |
| `extract_read_writes()` | dependencies.py:659 | 通过 V.ops handler 分析内存依赖 |
| `compute_dependencies()` | scheduler.py:3476（调用在 L3155） | 构建 SchedulerNode 间依赖边 |
| `fuse_nodes()` | scheduler.py:4023 | 最多 10 轮贪心融合 |
| `can_fuse()` | scheduler.py:5785 | 融合合法性检查（多阶段） |
| `score_fusion_key()` | scheduler.py:6536 | 融合优先级评分 |
| `reorder_for_peak_memory()` | memory.py:913 | 内存优化排序 |
| `Scheduler.codegen()` | scheduler.py:7325 | 触发后端代码生成 |

---

## 三、主体流程梳理

### 3.1 阶段总览：IR 图 → 优化 Kernel 组

```
输入（来自 Phase 3）
┌─────────────────────────────────────────────────────┐
│ GraphLowering.operations: [ir.Operation, ...]        │
│ GraphLowering.buffers: [ir.Buffer, ...]              │
│ 每个 Operation 包含：inner_fn, ranges, read/writes    │
└─────────────────────────────────────────────────────┘
                        │
    ┌───────────────────┼───────────────────┐
    ▼                   ▼                   ▼
 [Step 1-2]          [Step 3-8]          [Step 9]
 IR → SchedulerNode  依赖分析+排序       贪心融合
    │                   │                   │
    ▼                   ▼                   ▼
 BaseSchedulerNode    依赖边建立          FusedSchedulerNode
 SchedulerNode        拓扑排序            垂直+水平融合
 ExternKernelNode     死代码消除          Template prologue/epilogue
 NopKernelNode        Foreach 分组        Benchmark 评估
                        │                   │
                        └───────┬───────────┘
                                ▼
                          [Step 10-12]
                          后处理优化
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
                 循环合并    Combo Kernel  内存排序
                                │
                                ▼
                    ┌─────────────────────────────────────┐
                    │ 优化后的 SchedulerNode 列表            │
                    │ 每个 node 对应一个 kernel 或 kernel 组  │
                    │ 准备送入 Phase 5（代码生成）            │
                    └─────────────────────────────────────┘
```

### 3.2 Step 1-2：IR Operation → SchedulerNode 映射

**核心函数**：`create_scheduler_node()` (scheduler.py:3427-3438)

```python
def create_scheduler_node(self, node: ir.Operation) -> BaseSchedulerNode:
    assert node.get_origins() is not None
    if node.is_no_op():
        return NopKernelSchedulerNode(self, node)
    elif isinstance(node, (ir.ComputedBuffer, ir.TemplateBuffer)):
        return SchedulerNode(self, node)
    elif isinstance(node, ir.ExternKernel):
        return ExternKernelSchedulerNode(self, node)
    else:
        raise NotImplementedError(node)
```

**映射关系**：

| IR 类型 | SchedulerNode 类型 | 用途 |
|---------|-------------------|------|
| `ir.ComputedBuffer` | `SchedulerNode` | 标准 kernel（Inductor 会生成循环体代码） |
| `ir.TemplateBuffer` | `SchedulerNode`（含 template 标记） | 模板 kernel（mm、conv 等，使用 CUTLASS/Triton 模板） |
| `ir.ExternKernel` | `ExternKernelSchedulerNode` | 外部库调用（cuBLAS、cuDNN） |
| No-op 操作 | `NopKernelSchedulerNode` | 空操作，仅用于维护依赖顺序 |

**SchedulerNode 初始化关键步骤** (scheduler.py:1559-1600)：

```python
class SchedulerNode(BaseSchedulerNode):
    def __init__(self, scheduler, node):
        super().__init__(scheduler)
        self._init_from_node(node)  # 设置 node, ancestors, distances 等
        self._compute_attrs()       # 提取 sizes, body, 计算依赖, 设置 group

    def _compute_attrs(self):
        # 1. 从 IR 节点提取循环大小和循环体
        self._sizes, body = self.node.simplify_and_reorder(...)
        self._body = body

        # 2. 根据后端的 group_fn 设置分组
        device = self.node.get_device_or_error()
        group_fn = self.scheduler.get_backend(device).group_fn
        self.group = (device, group_fn(self._sizes))

        # 3. 提取内存读写依赖
        self.set_read_writes(
            dependencies.extract_read_writes(
                self._body, *self._sizes, normalize=should_normalize
            )
        )
```

**`group` 的含义**：`(device, sizes_tuple)` 用于分组——相同 group 的节点可能被 aggressive fusion 考虑融合。

### 3.3 Step 3-4：依赖分析与拓扑排序

依赖分析是融合决策的基础。其核心机制是利用 V.ops 虚拟化系统：

```python
# dependencies.py:659-698
def extract_read_writes(fn, *argsizes, normalize=False, prefix="d", hidden_args=()):
    args, var_ranges = index_vars_squeeze(*argsizes, prefix=prefix)

    # 创建 RecordLoadStore handler
    rw = RecordLoadStore(var_ranges, normalize=normalize)

    # 将其安装为 V.ops handler，执行循环体函数
    with V.set_ops_handler(rw):
        fn(*args, *hidden_args)

    # 收集结果
    inner = rw.parent_handler
    return ReadWrites(
        OrderedSet(inner._reads),    # 读依赖集合
        OrderedSet(inner._writes),   # 写依赖集合
        inner._index_exprs,
        range_vars,
        var_ranges,
    )
```

**原理**：`RecordLoadStore` 继承自 `V.MockHandler`，拦截 `ops.load()` 和 `ops.store()` 调用，将它们转换为 `MemoryDep` 对象记录到 `_reads` 和 `_writes` 集合中。这就是 define-by-run IR 的威力——**同一个 inner_fn 函数，安装不同 handler 就能做不同的事**：代码生成时生成代码，依赖分析时记录读写模式。

依赖分析完成后，`compute_dependencies()` 将 SchedulerNode 的 `unmet_dependencies` 设置为其所有读取依赖——这些是需要被上游节点满足的依赖。然后 `topological_sort_schedule()` 按拓扑序排列节点，确保生产者先于消费者。

### 3.4 Step 9：贪心融合算法（核心）

融合是调度器最核心的步骤，分为**最多 10 轮**迭代：

```python
# scheduler.py:4023-4056
def fuse_nodes(self, nodes):
    for i in range(10):  # 最多 10 轮
        old_len = len(nodes)
        nodes = self.fuse_nodes_once(nodes, is_reorder_round=False)
        new_len = len(nodes)
        if new_len == old_len or new_len == 1:
            break  # 无法再融合或只剩一个节点

    # 可选的重排序轮
    if config.loop_ordering_after_fusion or config.loop_index_inversion_in_fusion:
        nodes = self.fuse_nodes_once(nodes, is_reorder_round=True)
    return nodes
```

**每轮 `fuse_nodes_once()` 的步骤** (scheduler.py:4957-5024)：

```
1. prune_redundant_deps()       ← 清理冗余依赖
2. get_possible_fusions()       ← 找到所有合法融合机会
   └── can_fuse(node1, node2)   ← 多阶段合法性检查
3. _try_fusion_pairs()          ← 贪心选择最佳融合
   └── score_fusion_key()       ← 评分排序
   └── speedup_by_fusion()      ← 利润评估（可能含 benchmark）
   └── fuse_two_nodes()         ← 执行融合
4. _finish_pending_fusions()    ← 完成异步 benchmark 的融合
```

**融合机会的发现** (scheduler.py:5078-5132)：
- 遍历所有共享 buffer 的节点对（通过 `used_buffer_names()` 分组）
- 对每对调用 `can_fuse()` 检查合法性
- 如果 `config.aggressive_fusion`，还检查同 group 的节点对
- 按分数降序排序后返回

**融合执行** (scheduler.py:4721-4743)：
```python
def fuse_two_nodes(self, node1, node2, fused_nodes):
    device = node1.get_device()
    node3 = self.get_backend(device).fuse(node1, node2)  # 后端创建 FusedSchedulerNode
    fused_nodes.remove(node1)
    fused_nodes.remove(node2)
    fused_nodes.add(node3)
    self.name_to_fused_node.update({n.get_name(): node3 for n in node3.get_nodes()})
    return node3
```

### 3.5 Step 12：内存优化排序

融合完成后，`reorder_for_peak_memory()` (memory.py:913) 尝试三种拓扑排序算法，选择峰值内存最小的方案：

```python
def reorder_for_peak_memory(nodes, name_to_buf, name_to_fused_node,
                            graph_inputs, graph_outputs,
                            methods=[topological_sort_lpmf,
                                     topological_sort_bfs,
                                     topological_sort_dfs]):
    # 尝试每种排序方法，选择峰值内存最小的
    best = nodes
    best_peak = float("inf")
    for method in methods:
        reordered = method(nodes, ...)
        peak = compute_peak_memory(reordered, ...)
        if peak < best_peak:
            best, best_peak = reordered, peak
    return best
```

三种排序策略：
1. **LPMF（Least Peak Memory First）**：贪心选择使当前内存最小的节点
2. **BFS**：广度优先，减少 buffer 存活时间
3. **DFS**：深度优先，优先处理大 buffer 的消费者

### 3.6 触发代码生成

`Scheduler.codegen()` (scheduler.py:7325-7330) 遍历最终节点列表，根据节点类型分派到后端：

```python
def codegen(self):
    if config.graph_partition:
        return self._codegen_partitions()
    else:
        return self._codegen(self.nodes)
```

`_codegen()` 中对每个节点的分派逻辑 (scheduler.py:7587-7621)：

| 节点类型 | 分派方法 | 说明 |
|---------|---------|------|
| Template | `backend.codegen_template()` | 带 prologue/epilogue 的模板代码 |
| Extern | `codegen_extern_call()` | 外部库调用 |
| Foreach | `backend.codegen_combo_kernel()` | 并行操作合并 |
| MixOrder | `backend.codegen_mix_order_reduction()` | 混序归约 |
| Fused/Scheduler | `backend.codegen_node()` | 标准 kernel 代码生成（详见阶段五） |

---

## 四、UML 图 / 架构设计

### 4.1 SchedulerNode 类层次

```
BaseSchedulerNode (scheduler.py:539)
├── SchedulerNode (scheduler.py:1550)
│   ├── 包装 ir.ComputedBuffer（Pointwise/Reduction/Scatter/Scan/Sort）
│   └── 包装 ir.TemplateBuffer（mm/conv 等模板 kernel）
│
├── ExternKernelSchedulerNode (scheduler.py:1506)
│   └── 包装 ir.ExternKernel（cuBLAS/cuDNN/用户自定义 kernel）
│
├── NopKernelSchedulerNode (scheduler.py:1543)
│   └── 空操作，仅维护执行顺序
│
├── FusedSchedulerNode (scheduler.py:1938)
│   ├── 包含 snodes: list[BaseSchedulerNode]
│   ├── 代表融合后的节点组
│   └── union 所有子节点的依赖和输出
│
├── ForeachKernelSchedulerNode (scheduler.py:2331)
│   ├── 继承 FusedSchedulerNode
│   ├── 将并行操作合并为批量执行
│   └── 支持 combo kernel 模式
│
├── GroupedSchedulerNode (scheduler.py:2715)
│   ├── 临时分组，阻止外部融合
│   └── temp_grouping 标记
│
└── FusedMixOrderReductions (scheduler.py:2176)
    └── 融合不同循环顺序的 Reduction
```

### 4.2 依赖类型层次

```
Dep (基类, dependencies.py)
├── MemoryDep (dependencies.py:76)
│   ├── name: str          ← buffer 名称
│   ├── index: sympy.Expr  ← 符号索引表达式（如 768*d0 + d1）
│   ├── var_names: tuple   ← 循环变量（如 (d0, d1)）
│   ├── size: tuple        ← 各维度大小
│   └── mode: str | None   ← 存储模式（"atomic_add", "tma" 等）
│
├── StarDep (dependencies.py:320)
│   ├── name: str          ← buffer 名称
│   └── mode: str | None
│   └── 语义：对整个 buffer 的依赖，不区分具体索引
│
└── WeakDep (dependencies.py:380)
    ├── name: str              ← 被读取的 buffer
    ├── mutating_buf: str      ← 执行修改的 buffer
    └── is_fake: bool          ← 是否仅用于排序（不延长生命周期）
    └── 语义：mutation 排序依赖，不表示实际数据依赖
```

### 4.3 调度器组件交互图

```
                    ┌─────────────────────────────────────────┐
                    │            GraphLowering                 │
                    │  (graph.py)                              │
                    │  ├── operations: [ir.Operation]          │
                    │  ├── buffers: [ir.Buffer]                │
                    │  └── codegen() → _update_scheduler()     │
                    └───────────────┬─────────────────────────┘
                                    │
                      Scheduler(operations)
                                    │
                    ┌───────────────▼─────────────────────────┐
                    │              Scheduler                    │
                    │  (scheduler.py)                           │
                    │                                           │
                    │  ┌─── _init() ──────────────────────┐    │
                    │  │ 1. create_scheduler_node()        │    │
                    │  │ 2. compute_dependencies()  ◄─────┐│    │
                    │  │ 3. topological_sort_schedule()   ││    │
                    │  │ 4. dead_node_elimination()        ││    │
                    │  │ 5. compute_ancestors()            ││    │
                    │  │ 6. fuse_nodes()                   ││    │
                    │  │ 7. reorder_for_peak_memory()      ││    │
                    │  └───────────────────────────────────┘│    │
                    │                                       │    │
                    │  ┌─── codegen() ──────────────────┐   │    │
                    │  │ backend.codegen_node()          │   │    │
                    │  │ backend.codegen_template()      │   │    │
                    │  │ backend.codegen_extern_call()   │   │    │
                    │  └────────────────────────────────┘   │    │
                    └──────────┬──────────────┬──────────────┘    │
                               │              │                   │
                    ┌──────────▼──┐  ┌────────▼───────┐          │
                    │dependencies │  │    memory       │          │
                    │   (deps)    │  │                 │          │
                    │             │  │ reorder_for_    │          │
                    │ MemoryDep   │  │ peak_memory()   │          │
                    │ StarDep     │  │                 │          │
                    │ WeakDep     │  │ compute_memory_ │          │
                    │             │  │ timeline()      │          │
                    │ extract_    │  │                 │          │
                    │ read_writes │  │ BufferInfo      │          │
                    └──────┬──────┘  └────────┬────────┘          │
                           │                  │                   │
                           └──────────────────┼───────────────────┘
                                              │
                    ┌─────────────────────────▼──────────────────┐
                    │          Backend (codegen/)                 │
                    │                                             │
                    │  ┌──────────┐ ┌─────────┐ ┌──────────┐    │
                    │  │ Triton   │ │  C++     │ │ CUTLASS  │    │
                    │  │Scheduling│ │Scheduling│ │Scheduling│    │
                    │  │          │ │         │ │          │    │
                    │  │can_fuse_ │ │can_fuse_│ │codegen_  │    │
                    │  │vertical()│ │vertical│ │template()│    │
                    │  │codegen_  │ │codegen_│ │          │    │
                    │  │node()    │ │node()  │ │          │    │
                    │  └──────────┘ └─────────┘ └──────────┘    │
                    │                                             │
                    │  register_backend_for_device()              │
                    │  (common.py:400)                            │
                    └─────────────────────────────────────────────┘
```

### 4.4 融合类型分类

```
                    融合方向
                    ────────
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  垂直融合 (Vertical / Producer-Consumer)                     │
│  ─────────────────────────────────────                       │
│  Producer → Consumer                                         │
│                                                              │
│  [SchedulerNode A] ──write──▶ buf0 ──read──▶ [SchedulerNode B]│
│                          │                     │              │
│                          └─── 融合为 ──────────┘              │
│                          [FusedSchedulerNode(A, B)]           │
│                                                              │
│  条件：B 的读依赖能被 A 的写完全满足                          │
│  （can_fuse_vertical, scheduler.py:6031）                    │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  水平融合 (Horizontal / Consumer-Consumer)                   │
│  ─────────────────────────────────────                       │
│  共享输入的独立消费者                                        │
│                                                              │
│  [Node A] ──read──▶ buf0 ◄──read── [Node B]                 │
│                     │                   │                     │
│                     └─── 融合为 ────────┘                    │
│                     [FusedSchedulerNode(A, B)]                │
│                                                              │
│  条件：A 和 B 无依赖关系，共享内存访问                        │
│  （score_fusion_memory 评估收益）                            │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Template 融合 (Epilogue / Prologue)                         │
│  ─────────────────────────────────────                       │
│                                                              │
│  Epilogue:  [Template mm] ──▶ [Pointwise relu]              │
│             relu 融入 mm kernel 的 epilogue                  │
│                                                              │
│  Prologue:  [Pointwise exp] ──▶ [Template conv]             │
│             exp 融入 conv kernel 的 prologue                 │
│                                                              │
│  条件：Template 启用对应融合标志                              │
│  （config.epilogue_fusion / config.prologue_fusion）         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 五、关键思想代码讲解

### 5.1 can_fuse()：多阶段合法性检查

`can_fuse()` (scheduler.py:5785-6029) 是融合决策的核心，包含多个检查阶段：

```
can_fuse(node1, node2)
│
├── Stage 1: 基本检查
│   ├── 不能融合自身
│   ├── 不能跨 CUDA stream
│   └── 特殊节点类型检查（Grouped, Nop）
│
├── Stage 2: ExternKernel 特殊处理
│   ├── UserDefinedTritonKernel 只能做 epilogue 融合
│   └── 检查 can_fuse_epilogue()、布局兼容性
│
├── Stage 3: Template Prologue 融合
│   ├── node2 是 Template + node1 是 Pointwise
│   ├── 检查 config.prologue_fusion
│   ├── 检查无 aliasing/mutation
│   └── 检查 prologue 融合启发式
│
├── Stage 4: Template Epilogue 融合
│   ├── node1 是 Template + node2 是 Pointwise
│   ├── 检查 config.epilogue_fusion
│   └── 检查 consumer 是 functional pointwise
│
├── Stage 5: Buffer 级检查
│   ├── 检查 no_fuse_buffer_names 黑名单
│   └── 检查设备兼容性
│
├── Stage 6: 内存评分与变换尝试
│   ├── score_fusion_memory() → 评估共享内存
│   ├── 如果分数低于阈值，尝试循环重排序
│   ├── 尝试维度扩展（expand_dimension）
│   └── 尝试循环索引反转（loop_index_inversion）
│
└── Stage 7: 后端特定检查
    ├── V.choices.can_fuse() → 全局策略
    ├── 垂直融合：can_fuse_vertical() + backend.can_fuse_vertical()
    └── 水平融合：backend.can_fuse_horizontal()
```

**为什么这么复杂？** 因为融合的正确性涉及多个层面：数据依赖、内存布局、设备约束、后端能力。任何一个层面的违规都会导致计算结果错误或 kernel crash。

### 5.2 can_fuse_vertical()：精确的依赖匹配

`can_fuse_vertical()` (scheduler.py:6031-6090) 检查 producer-consumer 融合的合法性：

```python
def can_fuse_vertical(self, node1, node2):
    # node1 = producer, node2 = consumer
    remaining_deps_by_name = defaultdict(list)

    # 收集 node2 的未满足依赖（需要从上游读取的 buffer）
    for dep in node2.unmet_dependencies:
        name = self.mutation_renames.get(dep.name, dep.name)
        if isinstance(dep, WeakDep) and self.fusable_weak_dep(dep, node1, node2):
            continue  # 弱依赖可特殊处理
        remaining_deps_by_name[name].append(dep)

    # 尝试将 node1 的写依赖与 node2 的读依赖配对
    for cd in node1.read_writes.writes:
        if not isinstance(cd, (MemoryDep, StarDep)):
            continue
        remaining = remaining_deps_by_name.get(
            self.mutation_renames.get(cd.name, cd.name)
        )
        if remaining:
            for rd in remaining:
                if self.fusable_read_and_write(rd, cd):
                    remaining.remove(rd)  # 匹配成功！
```

**关键洞察**：融合合法性不仅看 buffer 名称是否匹配，还要看**索引表达式**是否兼容。论文原文：

> "Symbolic memory addresses are important in determining which fusions are legal. For example, if one kernel writes buf0 in forwards order, but a consumer reads in reverse order (using ops.load('buf0', s0 - 1 - i0)), then those nodes cannot be fused."

### 5.3 score_fusion_memory()：融合收益评估

`score_fusion_memory()` (scheduler.py:6207-6305) 估算融合节省的内存流量：

```python
def score_fusion_memory(self, node1, node2):
    # 策略 1: 标准情况 —— 计算共享依赖的总字节数
    common_deps = (node1.reads | node1.writes) & (node2.reads | node2.writes)
    score = sum(dep_size for dep in common_deps)

    # 策略 2: 如果没有精确匹配，检查同 buffer 不同索引的重叠
    if score == 0:
        # 给予 buffer 重叠奖励（高重叠率意味着好的缓存局部性）
        common_buffer_size = ...
        total_read_size = ...
        overlap_ratio = common_buffer_size / total_read_size
        if overlap_ratio > 0.5:
            score = overlap_ratio * ...
```

**评分维度**（论文原文）：
1. **融合类别**：pointwise/reduction/template 优先级不同
2. **节省的内存流量字节数**：精确估算融合消除了多少 load/store
3. **原始图中的节点距离**：距离越短越好（减少中间 buffer）

### 5.4 RecordLoadStore：V.ops 的依赖分析化身

`RecordLoadStore` (dependencies.py:615-620) 是 define-by-run IR 在依赖分析中的体现：

```python
class RecordLoadStore(V.KernelFormatterHandler):
    def __init__(self, var_ranges, normalize):
        parent_handler = _RecordLoadStoreInner(var_ranges, normalize)
        super().__init__(parent_handler=parent_handler)
```

其内部 handler `_RecordLoadStoreInner` (dependencies.py:509) 继承 `V.MockHandler`：

```python
class _RecordLoadStoreInner(V.MockHandler):
    def load(self, name, index):
        self._reads.add(MemoryDep(name, *self.canonicalize(index)))

    def store(self, name, index, value, mode=None):
        self._writes.add(MemoryDep(name, *self.canonicalize(index), mode=mode))
```

**工作原理**：
1. 创建 `RecordLoadStore` 实例
2. 用 `V.set_ops_handler(rw)` 安装为当前 ops handler
3. 执行循环体函数 `fn(*args)`
4. 函数内部的 `ops.load()` 被拦截 → 记录 `MemoryDep` 到 `_reads`
5. 函数内部的 `ops.store()` 被拦截 → 记录 `MemoryDep` 到 `_writes`
6. 执行完毕后提取 `ReadWrites(reads, writes, ...)`

**为什么这样设计？** 因为 define-by-run IR 的循环体是一个普通 Python 函数，不知道自己的依赖。通过替换 V.ops handler，同一个函数可以做代码生成、依赖分析、类型传播等不同事情——**这是 Inductor IR 架构的精髓**。

### 5.5 FusedSchedulerNode：融合产物的数据结构

`FusedSchedulerNode` (scheduler.py:1938-2137) 代表融合后的节点组：

```python
class FusedSchedulerNode(BaseSchedulerNode):
    snodes: list[BaseSchedulerNode]  # 包含的子节点

    @staticmethod
    def fuse(node1, node2):
        # 合并两个节点的属性
        fused = FusedSchedulerNode(node1.scheduler)
        fused.snodes = node1.get_nodes() + node2.get_nodes()
        # union 未满足的依赖
        fused.unmet_dependencies = node1.unmet_dependencies | node2.unmet_dependencies
        # 但减去内部已满足的依赖
        fused.unmet_dependencies -= satisfied_deps
        return fused
```

**核心属性**：
- `snodes`：包含的所有子节点列表
- `unmet_dependencies`：合并后仍需上游满足的依赖
- `group`：取子节点中最大的 group（决定 kernel 参数）
- `min_order` / `max_order`：子节点在原始序列中的位置范围

---

## 六、关键源码讲解

### 6.1 scheduler.py — Scheduler._init() 全流程

`scheduler.py:3088-3280`，完整初始化流程：

```python
def _init(self, nodes: list[ir.Operation]):
    super().__init__()
    V.graph.scheduler = self
    self.backends = {}
    self.completed_operations = OrderedSet()
    self.available_buffer_names = OrderedSet([
        *V.graph.graph_inputs.keys(),
        *V.graph.constants.keys(),
        *V.graph.torchbind_constants.keys(),
    ])

    # [Step 1] 创建调度节点
    self.nodes = [self.create_scheduler_node(n) for n in nodes]

    # [Step 1b] 构建查找映射
    for node in self.nodes:
        self.name_to_node[node.get_name()] = node
        for buf_name in node.get_buffer_names():
            self.name_to_fused_node[buf_name] = node
        ...

    # [Step 1c] 处理 mutation 重命名
    for node in self.nodes:
        for buf_name in node.get_mutated_names():
            ...

    # [Step 2] 计算依赖
    self.compute_dependencies()

    # [Step 3] 拓扑排序
    self.nodes = self.topological_sort_schedule(self.nodes)

    # [Step 4] 死代码消除
    if config.use_dce:
        self.dead_node_elimination()

    # [Step 5-6] 计算祖先和输入距离
    self.compute_ancestors()
    self.compute_input_distances()

    # [Step 7] 创建 Foreach 节点
    self.create_foreach_nodes()

    # [Step 8] 分配 CUDA streams
    self._populate_stream_assignments()

    # [Step 9] 融合！
    self.nodes = self.fuse_nodes(self.nodes)

    # [Step 10-11] 循环合并、Combo Kernel
    self.merge_loops()
    self.finalize_multi_template_buffers()
    if config.combo_kernels:
        self.create_combo_kernel_nodes()

    # [Step 12] 内存优化排序
    if config.reorder_for_peak_memory:
        self.nodes = reorder_for_peak_memory(self.nodes, ...)

    # [Step 13] 计算/通信重叠
    if config.reorder_for_compute_comm_overlap:
        self.nodes = reorder_for_compute_comm_overlap(self.nodes, ...)
```

### 6.2 dependencies.py — 三类依赖详解

**MemoryDep** (dependencies.py:76-317) — 精确内存访问：

```python
@dataclasses.dataclass(frozen=True)
class MemoryDep(Dep):
    name: str                    # buffer 名称
    index: sympy.Expr            # 符号索引（如 768*d0 + d1）
    var_names: tuple             # 循环变量（如 (d0, d1)）
    size: tuple                  # 各维度大小
    mode: str | None = None      # 存储模式
```

关键方法：
- `normalize()` (L167)：合并循环、简化索引表达式
- `normalize_with_stride_order()` (L179)：按 stride 重新排序循环，使不同循环顺序的依赖可比较
- `decide_loop_order_to_match()` (L104)：判断两个依赖的循环顺序是否可通过重排来匹配

**StarDep** (dependencies.py:320-369) — 全 buffer 依赖：

```python
@dataclasses.dataclass(frozen=True)
class StarDep(Dep):
    name: str
    mode: str | None = None
```

使用场景：
- 无法确定具体索引的操作（如 `bucketize`）
- 对 unbacked 符号变量的依赖
- 输出 mutation

语义：对整个 buffer 的依赖，不与 MemoryDep 精确匹配。

**WeakDep** (dependencies.py:380-422) — 排序依赖：

```python
@dataclasses.dataclass(frozen=True)
class WeakDep(Dep):
    name: str              # 被读取的 buffer
    mutating_buf: str      # 执行修改的 buffer
    is_fake: bool = False  # 是否仅用于排序
```

使用场景：
- 跟踪 mutation 排序：如果 A 读 buffer，B 修改 buffer，B 必须在 A 之后
- `is_fake=True`：纯排序依赖，不延长 buffer 生命周期（如 clone）
- `is_fake=False`：真实依赖，延长生命周期（如 view 共享存储）

### 6.3 memory.py — 峰值内存优化

**BufferInfo** (memory.py:337-344) — buffer 生命周期：

```python
@dataclasses.dataclass
class BufferInfo:
    buffer: SchedulerBuffer | FreeableInputBuffer
    size_alloc: int   # 分配时的字节数
    size_free: int    # 释放时的字节数
    start_step: int   # 分配步（哪个节点创建它）
    end_step: int     # 释放步（最后一个消费者之后）
```

**compute_memory_timeline()** (memory.py:346-432) 计算每个 buffer 的生命周期：

```
1. 对每个节点按执行顺序编号 → node_to_step 映射
2. 对每个 buffer：
   ├── start_step = 定义该 buffer 的节点的 step
   └── end_step = 该 buffer 的所有后继节点中最大的 step
3. 返回 [BufferInfo] 列表
```

**topological_sort_lpmf()** (memory.py:540-680) — 最小峰值内存排序：

```python
def topological_sort_lpmf(nodes, name_to_freeable_input_buf, ...):
    # 贪心算法：每步选择使当前内存最小的节点
    while nodes_to_schedule:
        selected_node = min(
            nodes_to_schedule,
            key=lambda node: (
                # 优先级 1: 新增内存减去释放内存
                node.mpi_node.size - node_info[node]["memory_to_free"],
                # 优先级 2: 原始索引
                node.mpi_node.index,
            ),
        )
        result.append(selected_node)
        # 更新可调度节点集合
        ...
    return result
```

### 6.4 codegen/common.py — 后端注册机制

**register_backend_for_device()** (codegen/common.py:400-424)：

```python
def register_backend_for_device(
    device: str,
    device_scheduling: SchedulingConstructor,   # 如 TritonScheduling
    device_wrapper_codegen: WrapperConstructor,  # 如 PythonWrapperCodegen
    ...
):
    device_codegens[device] = DeviceCodegen(
        scheduling=device_scheduling,
        wrapper_codegen=device_wrapper_codegen,
        ...
    )
```

**init_backend_registration()** (codegen/common.py:491-612) 注册所有后端：

```python
# CPU 后端
register_backend_for_device("cpu",
    lambda scheduling: cpu_backends[config.cpu_backend](scheduling),
    PythonWrapperCodegen, ...)

# CUDA 后端
register_backend_for_device("cuda",
    CUDACombinedScheduling,   # 组合 Triton + CUTLASS 调度
    PythonWrapperCodegen, ...)
```

**`DeviceCodegen` 数据结构** (codegen/common.py:304-314)：

```python
@dataclasses.dataclass
class DeviceCodegen:
    scheduling: SchedulingConstructor    # 调度策略（融合、代码生成分派）
    wrapper_codegen: WrapperConstructor  # Wrapper 代码生成
    cpp_wrapper_codegen: ...             # C++ Wrapper（可选）
    fx_wrapper_codegen: ...              # FX IR Wrapper（可选）
```

### 6.5 graph.py — 从 Lowering 到 Scheduling 的桥梁

**_update_scheduler()** (graph.py:2535-2544)：

```python
def _update_scheduler(self):
    from .scheduler import Scheduler
    # 创建 Scheduler 时不生成 CUBIN（避免影响融合决策）
    with config.patch("triton.store_cubin", False):
        self.scheduler = Scheduler(self.operations)
```

**codegen()** (graph.py:2546-2563)：

```python
def codegen(self):
    with dynamo_timed("GraphLowering.codegen"):
        self.init_wrapper_code()        # 初始化 wrapper 代码生成器
        self._update_scheduler()         # 创建/更新 Scheduler
        self.wrapper_code.push_codegened_graph(self)
        self.scheduler.codegen()         # 触发调度 + 代码生成
        result = self.wrapper_code.generate(...)  # 生成 wrapper
        return result
```

---

## 七、核心技术

### 7.1 贪心融合算法

**算法伪代码**：

```
function GREEDY_FUSION(nodes):
    for iteration in 1..10:
        # 1. 发现所有合法融合机会
        fusions = []
        for (node1, node2) in pairs_sharing_buffers(nodes):
            if can_fuse(node1, node2):
                fusions.append((node1, node2))

        # 2. 按分数降序排序
        fusions.sort(key=score_fusion, reverse=True)

        # 3. 贪心应用
        for (node1, node2) in fusions:
            if still_legal(node1, node2):  # 可能被之前的融合影响
                fused = FusedSchedulerNode(node1, node2)
                replace(node1, node2, fused)

        if len(nodes) unchanged:
            break
    return nodes
```

**复杂度考量**：最坏情况 O(10 × N² × M)，其中 N 是节点数，M 是每次 can_fuse 的检查开销。实际中通过 buffer 分组将配对搜索限制在同 buffer 的节点之间。

### 7.2 符号依赖匹配

融合合法性的核心挑战是**符号索引匹配**。例如：

```python
# Producer 写 buf0 使用正向索引
ops.store("buf0", d0 * N + d1, value)

# Consumer 读 buf0 使用反向索引
ops.load("buf0", (M - 1 - d0) * N + d1)
```

这两个 MemoryDep 的索引表达式不同，但通过 `normalize_with_stride_order()` 可以标准化后比较。如果标准化后仍不匹配，则**不能融合**——因为无法生成同时满足两种访问模式的统一循环。

### 7.3 死代码消除

`dead_node_elimination()` (scheduler.py:3824) 移除没有用户的节点：

1. 从输出节点开始反向标记可达节点
2. 未被标记的节点被视为死代码
3. 移除死节点并释放其 buffer

### 7.4 水平融合 (Horizontal Fusion)

水平融合将**共享输入但无依赖关系**的节点合并：

```
原始：                         水平融合后：
buf0 ──▶ [relu] ──▶ buf1      buf0 ──▶ [relu + sigmoid] ──▶ buf1, buf2
buf0 ──▶ [sigmoid] ──▶ buf2   （单次读 buf0，两个操作在同一 kernel）
```

**收益**：减少对共享 buffer 的读取次数，利用 GPU 缓存局部性。

### 7.5 Template Prologue/Epilogue 融合

**Epilogue 融合**：将 pointwise 操作融入模板 kernel 的尾部

```
原始：                                        融合后：
[A @ B = buf0] ──▶ [relu(buf0) = buf1]      [A @ B + relu] ──▶ buf1
    mm kernel           pointwise kernel         单个 mm kernel + epilogue
```

**Prologue 融合**：将 pointwise 操作融入模板 kernel 的头部

```
原始：                                        融合后：
[exp(x) = buf0] ──▶ [conv(buf0, w) = buf1]  [conv(exp(x), w)] ──▶ buf1
  pointwise kernel       conv kernel            单个 conv kernel + prologue
```

### 7.6 内存规划：三层优化

> **注意**：前两层（调度重排 + 复用判断）在 Scheduling 阶段完成，属于本阶段（Phase 4）。第三层池化优化在 Wrapper 代码生成阶段由 `MemoryPlanner` 完成（详见阶段五 6.4 节）。

1. **调度层**（Phase 4）：`reorder_for_peak_memory()` — 重排节点执行顺序以降低峰值内存
2. **复用层**（Phase 4）：`AllocateLine.should_reuse_buffer()` — 判断 freed buffer 是否可复用
   - 条件：设备、dtype、存储大小、对齐匹配，且不增加峰值内存
3. **池化层**（Phase 5）：`AllocationPool` — 多个 buffer 共享一块 `torch.empty()` 分配（详见阶段五）

---

## 八、自主学习 Debug 路线

### Step 1: 观察从 Lowering 到 Scheduler 的过渡

**目标**：验证 IR 操作如何转换为 SchedulerNode。

**操作**：

```python
import torch

@torch.compile(mode="default")
def test_simple(x, y):
    return torch.relu(x + y)

x = torch.randn(4, 4, device="cuda")
y = torch.randn(4, 4, device="cuda")
result = test_simple(x, y)
```

**断点**：

```
断点 1: graph.py:2544  self.scheduler = Scheduler(self.operations)
    → 观察 self.operations：有多少个 ir.Operation？
    → 类型是什么？（应该只有 Pointwise）

断点 2: scheduler.py:3103  self.nodes = [self.create_scheduler_node(n) for n in nodes]
    → 每个 n 被转换为哪种 SchedulerNode？
    → 预期：1 个 SchedulerNode（因为 Inlining 将 add+relu 合并为一个）
```

**关注点**：
- Inlining 后还剩几个 operation？
- 每个 SchedulerNode 的 read_writes 包含哪些 buffer？

**输入**：简单的 relu(x+y) 模型
**输出**：理解 IR → SchedulerNode 的映射

### Step 2: 观察依赖分析的 V.ops 拦截

**目标**：理解 RecordLoadStore 如何通过 V.ops 拦截提取依赖。

**断点**：

```
断点 1: dependencies.py:682  with V.set_ops_handler(rw):
    → rw 是 RecordLoadStore 实例
    → fn 是什么？（应该是 inner_fn 闭包）

断点 2: dependencies.py:581  def load(self, name, index):
    → name 是什么 buffer？
    → index 是什么 SymPy 表达式？
    → self._reads 增加了什么 MemoryDep？

断点 3: dependencies.py:588  def store(self, name, index, value, mode):
    → 写入了什么 buffer？
    → self._writes 增加了什么？
```

**关注点**：
- `ops.load` 调用被替换为了 `MemoryDep` 记录
- 索引表达式包含 SymPy 符号变量（如 `d0*4 + d1`）
- reads 和 writes 的区别

**输入**：带断点的 debug 环境
**输出**：理解 V.ops 在依赖分析中的化身

### Step 3: 观察 can_fuse 的多阶段检查

**目标**：理解融合合法性检查的完整流程。

**操作**：使用更复杂的模型触发融合：

```python
@torch.compile(mode="default")
def test_fusion(x, y):
    a = x + y              # Pointwise
    b = torch.relu(a)      # Pointwise（通常被 Inlining）
    c = b * 2.0            # Pointwise
    d = c.sum()            # Reduction
    return d
```

**断点**：

```
断点: scheduler.py:5785  can_fuse()
    → node1, node2 分别是什么类型？
    → 走了哪些 Stage？
    → 返回 True 还是 False？
    → 如果是垂直融合，看 can_fuse_vertical() (L6031)
```

**关注点**：
- 哪些节点对被考虑融合？
- `can_fuse_vertical()` 中依赖匹配的具体过程
- Score 如何计算？

**输入**：包含多类型算子的模型
**输出**：理解融合决策的完整检查链

### Step 4: 观察融合的执行

**目标**：理解 `fuse_two_nodes()` 和 `FusedSchedulerNode` 的创建。

**断点**：

```
断点 1: scheduler.py:4721  fuse_two_nodes()
    → node1, node2 是什么？
    → backend.fuse() 返回什么？
    → node3 (FusedSchedulerNode) 包含哪些子节点？

断点 2: scheduler.py:1948  FusedSchedulerNode.fuse()
    → snodes 列表的内容
    → unmet_dependencies 的合并逻辑
```

**关注点**：
- 融合后的 FusedSchedulerNode 如何维护子节点关系
- 依赖如何合并（union - 满足的内部依赖）
- name_to_fused_node 映射如何更新

**输入**：会触发融合的模型
**输出**：理解 FusedSchedulerNode 的数据结构

### Step 5: 观察三类依赖的使用场景

**目标**：区分 MemoryDep、StarDep、WeakDep 的使用场景。

**断点**：

```
断点 1: dependencies.py:581  MemoryDep 创建 (load)
断点 2: dependencies.py:588  MemoryDep 创建 (store)
断点 3: dependencies.py:610  StarDep 创建 (bucketize 等)
断点 4: memory.py:247        WeakDep 检查 (is_fake 分支)
```

**操作**：使用包含不同操作的模型：

```python
@torch.compile(mode="default")
def test_deps(x):
    a = x * 2          # 产生 MemoryDep
    b = a.sum()         # 产生 MemoryDep (read) + MemoryDep (write)
    return b
```

**关注点**：
- MemoryDep 的 index 表达式如何反映访问模式
- StarDep 何时出现
- WeakDep 在 mutation 场景下的作用

**输入**：包含多种操作的模型
**输出**：理解三类依赖的语义区别

### Step 6: 观察内存规划

**目标**：理解 buffer 生命周期计算和峰值内存优化。

**断点**：

```
断点 1: memory.py:913  reorder_for_peak_memory()
    → 三种排序方法分别产生什么结果？
    → 最终选择了哪种？

断点 2: memory.py:540  topological_sort_lpmf()
    → 每步选择了哪个节点？
    → 当前内存使用量如何变化？
```

**操作**：

```python
import torch._inductor.config as config

# 启用内存优化排序
with config.patch(reorder_for_peak_memory=True):
    @torch.compile
    def test_memory(x):
        a = x * 2
        b = a + 1
        c = b.sum()
        d = a * 3    # a 被再次使用，生命周期长
        return c, d

    x = torch.randn(32, 32)
    test_memory(x)
```

**关注点**：
- buffer `a` 的生命周期有多长？（start_step 到 end_step）
- LPMF 如何重排以尽早释放 buffer
- 峰值内存的减少量

**输入**：包含长生命周期 buffer 的模型
**输出**：理解内存规划的机制和效果

### Step 7: 观察 Template 融合（Epilogue）

**目标**：理解 mm/conv 等 Template 节点的 epilogue 融合。

**操作**：

```python
@torch.compile(mode="default")
def test_epilogue(x, w):
    a = torch.mm(x, w)     # Template (mm kernel)
    b = torch.relu(a)      # Pointwise → epilogue fusion
    return b

x = torch.randn(32, 64, device="cuda")
w = torch.randn(64, 32, device="cuda")
result = test_epilogue(x, w)
```

**断点**：

```
断点 1: scheduler.py:5848  Epilogue 融合检查
    → node1 是 Template？node2 是 Pointwise？
    → config.epilogue_fusion 是否启用？
    → 融合成功了吗？

断点 2: scheduler.py:7587  Template codegen
    → prologue, template_node, epilogue 分别是什么？
    → backend.codegen_template() 如何处理这三部分？
```

**关注点**：
- Epilogue 融合的条件检查
- Template 节点和 Pointwise 节点的交互
- 最终生成的 kernel 是否包含 relu 操作

**输入**：mm + relu 模型
**输出**：理解 Template epilogue 融合的完整流程

---

## 九、数据流加工过程重点

### 9.1 IR 图在调度过程中的形态演变

```
阶段 0: Lowering 产物（来自 Phase 3）
│  形态：
│  ├── graph.operations: [ir.Operation, ...]
│  │   ├── 每个 Operation 包含 inner_fn（闭包）和 ranges（符号形状）
│  │   ├── 类型：ComputedBuffer(data=Pointwise/Reduction/...)
│  │   │         TemplateBuffer（mm/conv 模板）
│  │   │         ExternKernel（外部库调用）
│  │   └── 部分已被 Inlining 消除（多个 pointwise 合并为一个闭包）
│  │
│  ├── graph.buffers: [ir.Buffer, ...]
│  │   ├── 每个 Buffer 有名称、布局（Layout）、dtype
│  │   └── 部分中间 buffer 已被 Inlining 消除
│  │
│  └── 延迟求值已全部 resolve（所有 TensorBox/StorageBox 已 realize）
│
▼ Step 1: IR → SchedulerNode 映射
SchedulerNode 图
│  变化：
│  ├── [类型转换] ir.ComputedBuffer → SchedulerNode
│  │             ir.ExternKernel → ExternKernelSchedulerNode
│  │             ir.TemplateBuffer → SchedulerNode（带 template 标记）
│  │
│  ├── [新增] group 属性：(device, group_fn(sizes)) — 用于融合分组
│  ├── [新增] read_writes 属性：通过 extract_read_writes() 提取
│  ├── [新增] unmet_dependencies：初始等于 reads（所有读依赖）
│  └── [新增] _body: LoopBody — 从 IR 节点提取的循环体
│
▼ Step 2-3: 依赖分析与拓扑排序
带依赖边的 SchedulerNode DAG
│  变化：
│  ├── [新增] 依赖边：每个 SchedulerNode 的 read_writes 被分析
│  │   ├── reads: OrderedSet[MemoryDep/StarDep/WeakDep]
│  │   └── writes: OrderedSet[MemoryDep]
│  │
│  ├── [结构] 节点按拓扑序排列（生产者先于消费者）
│  └── [新增] ancestors 和 input_distance（用于融合评分）
│
▼ Step 4: 死代码消除
精简的 SchedulerNode DAG
│  变化：
│  ├── [删除] 没有用户且非输出的节点
│  └── [释放] 对应的 buffer
│
▼ Step 9: 贪心融合（最多 10 轮）
FusedSchedulerNode 图
│  变化：
│  ├── [合并] 多个 SchedulerNode → FusedSchedulerNode
│  │   ├── 垂直融合：producer-consumer（依赖边满足）
│  │   ├── 水平融合：consumer-consumer（共享输入，无依赖）
│  │   └── Template 融合：prologue/epilogue 融入 mm/conv
│  │
│  ├── [依赖更新] FusedSchedulerNode 的 unmet_dependencies =
│  │              union(子节点 deps) - 内部已满足的 deps
│  │
│  ├── [group 更新] 取子节点中最大的 group
│  └── [评分] 每个融合机会按 memory saving + distance 评分
│
▼ Step 12: 内存优化排序
最终调度方案
│  变化：
│  ├── [重排序] 节点执行顺序被优化以减少峰值内存
│  │   ├── 尝试 LPMF/BFS/DFS 三种排序
│  │   └── 选择峰值内存最小的方案
│  │
│  └── [Buffer 生命周期] 每个活的 buffer 的 start/end step 已确定
│
▼ 最终产品：优化后的 SchedulerNode 列表
   特性：
   ├── 融合分组：多个 IR 操作被组合为少量 FusedSchedulerNode
   ├── 依赖完整：所有节点的读写依赖已分析完毕
   ├── 执行顺序：拓扑排序 + 内存优化排序
   ├── 内存规划：buffer 生命周期已计算，复用方案已确定
   └── 准备送入代码生成阶段（Phase 5）
```

### 9.2 关键转变点分析

**转变 1：IR Operation → SchedulerNode（IR 翻译）**

```
原始（ir.Operation）：                    翻译后（SchedulerNode）：
ir.ComputedBuffer(                       SchedulerNode(
  name="buf5",                             node=ir.ComputedBuffer(...),
  layout=FlexibleLayout(...),              _sizes=[s0, s1],
  data=Pointwise(                          _body=LoopBody(inner_fn),
    inner_fn=<closure>,                    group=("cuda", (s0, s1)),
    ranges=[s0, s1]                        read_writes=ReadWrites(
  )                                          reads={MemoryDep("arg0", ...)},
)                                            writes={MemoryDep("buf5", ...)}
                                           ),
                                           unmet_dependencies={MemoryDep("arg0", ...)}
                                         )

新增能力：
├── group：用于融合分组（同 device 同 size 的节点可考虑融合）
├── read_writes：精确的内存访问模式（用于依赖匹配）
├── unmet_dependencies：等待上游满足的依赖（融合时逐个消除）
└── _body：循环体的调度器版本（可被后端变换）
```

**转变 2：独立节点 → FusedSchedulerNode（融合）**

```
融合前：                              融合后：
[SchedulerNode A]                    [FusedSchedulerNode]
  reads: {buf0}                        snodes: [A, B, C]
  writes: {buf1}                       reads: {buf0}          ← 只保留外部依赖
                                       writes: {buf3}
[SchedulerNode B]
  reads: {buf1}              →        消除的中间 buffer：
  writes: {buf2}                        buf1（A 的输出，B 的输入）
                                        buf2（B 的输出，C 的输入）
[SchedulerNode C]
  reads: {buf2}
  writes: {buf3}

收益：
├── 消除 buf1、buf2 的 store + load（节省内存带宽）
├── 减少 2 次 kernel launch（3 → 1）
├── GPU 缓存局部性提升（中间结果在寄存器/共享内存中传递）
└── 减少 2 个 buffer 的内存分配
```

**转变 3：原始拓扑序 → 内存优化排序**

```
原始排序（拓扑序）：             优化排序（LPMF）：
Step 1: [A] alloc=1MB          Step 1: [A] alloc=1MB
Step 2: [B] alloc=1MB          Step 2: [C] alloc=1MB
Step 3: [C] alloc=1MB          Step 3: [B] alloc=1MB
Step 4: [D] reads A,C          Step 4: [D] reads A,C
Peak: 3MB                       Peak: 2MB
（B 的 buffer 直到很晚才释放）   （C 先执行，其 buffer 可被 B 复用）
```

### 9.3 每种融合类型的数据流对比

```
                    融合类型         条件                      收益
                    ────────         ────                      ────
垂直融合        Producer→Consumer  依赖边精确匹配            消除中间 buffer
(Pointwise→       can_fuse_vertical()                      减少 kernel launch
 Pointwise)       读索引 == 写索引                         寄存器传递中间值

垂直融合        Producer→Consumer  Reduction 输出被          消除 reduction 输出
(Pointwise→       can_fuse_vertical()     Pointwise 消费    的全局内存写入
 Reduction)       支持 reduction 类型                      但 reduction 输出
                                                                            仍需 store

水平融合        Consumer↔Consumer  无依赖关系                共享输入的一次读取
(Pointwise∥       共享输入 buffer   score_fusion_memory     GPU 缓存局部性
 Pointwise)       > 0                                     减少总内存流量

Epilogue       Template→Pointwise config.epilogue_fusion   relu/sigmoid 等融入
融合             Template 允许                             mm/conv kernel 尾部
                  consumer 是 functional                   无额外 kernel launch

Prologue       Pointwise→Template config.prologue_fusion   exp/sigmoid 等融入
融合             Pointwise producer                        mm/conv kernel 头部
                  Template 允许                            无额外 kernel launch
```

---

## 十、交叉校验报告

> 校验时间：2026-04-15
> 校验方法：对比 PyTorch 2 论文 (ASPLOS 2024)、TorchInductor 设计帖 (dev-discuss #747)、PyTorch 源码 (main 分支)

### 校验结果汇总

| 校验项 | 来源 | 结果 |
|--------|------|------|
| Scheduler._init() 包含 12+ 步骤（创建节点→依赖→排序→融合→内存优化） | 源码 scheduler.py:3088-3280 | **通过** |
| create_scheduler_node() 将 IR 操作映射为 4 种 SchedulerNode | 源码 scheduler.py:3427-3438 | **通过** |
| fuse_nodes() 最多 10 轮迭代 | 源码 scheduler.py:4025 | **通过** |
| can_fuse() 多阶段检查（基本→Template→Buffer→内存→后端） | 源码 scheduler.py:5785-6029 | **通过** |
| can_fuse_vertical() 通过依赖边精确匹配判断融合合法性 | 源码 scheduler.py:6031-6090 | **通过** |
| score_fusion_key() 委托给 V.choices.score_fusion() | 源码 scheduler.py:6536-6542 | **通过** |
| 融合评分按类别→内存流量→距离排序 | 论文 Section 4.4 + 源码 | **通过** |
| MemoryDep 包含 name/index/var_names/size/mode | 源码 dependencies.py:76 | **通过** |
| StarDep 用于全 buffer 依赖，不与 MemoryDep 精确匹配 | 源码 dependencies.py:320-369 | **通过** |
| WeakDep 用于 mutation 排序，is_fake 控制是否延长生命周期 | 源码 dependencies.py:380-422 | **通过** |
| RecordLoadStore 继承 V.KernelFormatterHandler，拦截 load/store | 源码 dependencies.py:615-620 | **通过** |
| extract_read_writes() 通过 V.set_ops_handler(rw) 执行循环体 | 源码 dependencies.py:679-682 | **通过** |
| reorder_for_peak_memory() 尝试 LPMF/BFS/DFS 三种排序 | 源码 memory.py:913-999 | **通过** |
| BufferInfo 包含 start_step 和 end_step 计算生命周期 | 源码 memory.py:337-344 | **通过** |
| _update_scheduler() 在创建 Scheduler 时禁用 cubin 生成 | 源码 graph.py:2535-2544 | **通过** |
| register_backend_for_device() 注册调度和 wrapper 策略 | 源码 codegen/common.py:400-424 | **通过** |
| 消融实验数据：去 fusion -0.23，去 inlining -0.33，去两者 -1.11 | 论文 Table 4 | **通过** |
| FusedSchedulerNode 通过 union 子节点依赖减去内部满足的依赖 | 源码 scheduler.py:1938-2137 | **通过** |

### 修正记录

| 修正内容 | 修正原因 |
|----------|----------|
| 论文中 "score_fusion" 在源码中实际委托给 V.choices.score_fusion()（通过 score_fusion_key() 方法），最终由 LookupTableChoices 实现 | 源码确认 score_fusion_key() 是 list.sort 的 key 函数，实际评分逻辑在 V.choices 中 |
| 论文提到 "horizontal consumer/consumer fusions"，源码中水平融合通过 score_fusion_memory() 评估收益，但不总是需要显式 can_fuse_horizontal() | 源码确认水平融合主要通过共享 buffer 评估，后端特定检查是可选的 |
| 论文提到 "checks dependency edges"，源码中 can_fuse_vertical() 的实现涉及 MemoryDep 索引精确匹配，比论文描述更精确 | 源码确认 fusable_read_and_write() (scheduler.py:6149) 检查 index == index |

### 权威出处

- PyTorch 2 论文 (ASPLOS 2024): [ACM DL](https://dl.acm.org/doi/10.1145/3620665.3640366) | [PDF](https://docs.pytorch.org/assets/pytorch2-2.pdf)
- TorchInductor 设计帖: [dev-discuss #747](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- Inductor 文件结构讨论: [dev-discuss #1860](https://dev-discuss.pytorch.org/t/inductor-file-structure-explanation/1860)
- PyTorch 源码: `torch/_inductor/` 目录，main 分支 (2026-04)

---

## 附录 A：关键源码文件索引

| 文件 | 核心行号 | 核心内容 |
|------|---------|---------|
| `scheduler.py` | L539 | `BaseSchedulerNode` 基类 |
| `scheduler.py` | L1550 | `SchedulerNode` 标准 kernel 节点 |
| `scheduler.py` | L1938 | `FusedSchedulerNode` 融合节点 |
| `scheduler.py` | L3084 | `Scheduler.__init__()` 入口 |
| `scheduler.py` | L3427 | `create_scheduler_node()` IR 映射 |
| `scheduler.py` | L4023 | `fuse_nodes()` 融合主循环 |
| `scheduler.py` | L4721 | `fuse_two_nodes()` 执行融合 |
| `scheduler.py` | L4957 | `fuse_nodes_once()` 单轮融合 |
| `scheduler.py` | L5078 | `get_possible_fusions()` 发现融合机会 |
| `scheduler.py` | L5785 | `can_fuse()` 合法性检查 |
| `scheduler.py` | L6031 | `can_fuse_vertical()` 垂直融合检查 |
| `scheduler.py` | L6207 | `score_fusion_memory()` 内存评分 |
| `scheduler.py` | L6536 | `score_fusion_key()` 评分 key 函数 |
| `scheduler.py` | L7325 | `Scheduler.codegen()` 代码生成入口 |
| `scheduler.py` | L7587 | 节点类型分派（Template/Extern/Foreach/Fused） |
| `dependencies.py` | L76 | `MemoryDep` 精确内存依赖 |
| `dependencies.py` | L320 | `StarDep` 全 buffer 依赖 |
| `dependencies.py` | L380 | `WeakDep` 排序依赖 |
| `dependencies.py` | L432 | `ReadWrites` 读写集合 |
| `dependencies.py` | L509 | `_RecordLoadStoreInner` V.ops 拦截器 |
| `dependencies.py` | L615 | `RecordLoadStore` 依赖分析 handler |
| `dependencies.py` | L659 | `extract_read_writes()` 依赖提取 |
| `memory.py` | L40 | `MemoryPlanningInfoForBuffer` buffer 内存信息 |
| `memory.py` | L337 | `BufferInfo` buffer 生命周期 |
| `memory.py` | L540 | `topological_sort_lpmf()` 最小峰值内存排序 |
| `memory.py` | L913 | `reorder_for_peak_memory()` 内存优化入口 |
| `graph.py` | L2535 | `_update_scheduler()` Scheduler 创建 |
| `graph.py` | L2546 | `codegen()` 代码生成入口 |
| `codegen/common.py` | L304 | `DeviceCodegen` 后端数据结构 |
| `codegen/common.py` | L400 | `register_backend_for_device()` 后端注册 |
| `codegen/common.py` | L491 | `init_backend_registration()` 初始化所有后端 |

---

## 附录 B：Phase 4 检验清单

完成以上 7 步 debug 路线后，你应当能够回答以下问题：

- [ ] `Scheduler._init()` 包含哪些主要步骤？融合在第几步？
- [ ] `create_scheduler_node()` 将每种 IR 操作映射为什么 SchedulerNode？
- [ ] `RecordLoadStore` 如何利用 V.ops 虚拟化拦截 load/store？提取出的是什么数据结构？
- [ ] `MemoryDep`、`StarDep`、`WeakDep` 分别在什么场景下使用？语义有何不同？
- [ ] `can_fuse()` 有哪些检查阶段？为什么需要这么多阶段？
- [ ] `can_fuse_vertical()` 如何判断 producer-consumer 融合的合法性？索引匹配为什么重要？
- [ ] 贪心融合算法为什么需要多轮迭代？每轮做什么？
- [ ] 水平融合和垂直融合的区别是什么？各自的收益来源是什么？
- [ ] Template epilogue/prologue 融合如何将 pointwise 操作融入 mm/conv kernel？
- [ ] `reorder_for_peak_memory()` 尝试哪三种排序策略？如何选择最优方案？
- [ ] Buffer 生命周期如何计算？`start_step` 和 `end_step` 分别由什么决定？
- [ ] 你能否在不查看本文档的情况下，画出从 Lowering 产物到优化 SchedulerNode 列表的完整数据流？
