# Scheduler._init 调试级深度解读

> 文件路径：`torch/_inductor/scheduler.py:3088-3323`
>
> 本文档遵循 Inductor Teaching Code 的 9 段式框架，以调试视角逐行剖析 `Scheduler._init` 的完整执行流程。

---

## Section 1：设计哲学与编译器对应

### What（一句话）

`Scheduler._init` 将 IR 层的计算图节点列表转化为一个**经过依赖分析、死代码消除、拓扑排序、算子融合和内存优化的调度图**，为后续代码生成做好准备。

### Why（为什么需要这个抽象）

在 Inductor 的编译管线中，IR 层（`torch/_inductor/ir.py`）产出的 `ir.Operation` 节点列表是一个**平坦的、未优化的计算描述**。它存在以下问题：

1. **没有显式依赖边**：节点之间的执行顺序依赖是隐含在 buffer 的读写关系中的
2. **没有经过任何优化**：冗余节点未消除、可融合的 kernel 未合并
3. **没有内存感知**：不考虑峰值内存、buffer 生命周期

Scheduler 就像一个**编译器后端的指令调度器**（Instruction Scheduler），它接收 SSA 形式的中间表示，完成调度决策后交给代码生成阶段。

### 编译器对应

| Inductor 概念 | 传统编译器对应 |
|---|---|
| `Scheduler._init` | 指令调度（Instruction Scheduling）+ 循环优化（Loop Optimization） |
| `compute_dependencies` | 依赖图构建（Dependency Graph Construction）—— 类似于构建 DAG |
| `topological_sort_schedule` | 拓扑排序（Topological Sort）—— 确保合法执行序 |
| `dead_node_elimination` | 死代码消除（Dead Code Elimination, DCE） |
| `fuse_nodes` | 指令融合（Instruction Fusion / Kernel Fusion） |
| `reorder_for_peak_memory` | 寄存器/内存压力优化（Register Pressure Aware Scheduling） |
| `compute_last_usage` | 活跃变量分析（Liveness Analysis）—— 确定变量最后使用点 |

### 类比

把 Scheduler 想象成一个**建筑工地的总调度员**：

- **IR 节点** = 设计图纸上的工序（"打地基"、"砌墙"、"装窗"）
- **Buffer** = 建材（水泥、砖头、玻璃）
- **Scheduler._init** = 总调度员拿到图纸后做的一系列事情：
  1. 确定哪些工序必须先做（依赖分析）
  2. 去掉不需要的工序（死代码消除）
  3. 把能同时做的工序合并（算子融合）
  4. 安排最优顺序，避免材料堆积（内存优化）
  5. 确定每种材料什么时候可以清场（活跃变量分析）

---

## Section 2：核心数据结构

在进入代码之前，先理解 `Scheduler._init` 操作的核心"零件"——类型层次和协作关系。

### 2.1 调度节点类型层次

```
BaseSchedulerNode                          ← 所有调度节点的基类
├── SchedulerNode                          ← 包装 ir.ComputedBuffer / ir.TemplateBuffer
│   └── 持有 LoopBody，参与融合、循环合并
├── ExternKernelSchedulerNode              ← 包装 ir.ExternKernel（ATen/BLAS 等）
│   └── 通常是不可融合的外部调用
├── NopKernelSchedulerNode                 ← 包装 no-op 节点（占位用）
├── FusedSchedulerNode                     ← 融合后的节点，包含多个 SchedulerNode
│   └── ForeachKernelSchedulerNode         ← foreach 批量操作的融合节点
├── GroupedSchedulerNode                   ← 分组节点（combo kernel 等）
└── FusedExternTritonKernelSchedulerNode   ← 用户自定义 Triton kernel + epilogue 融合
```

### 2.2 Buffer 管理类型

```
SchedulerBuffer                            ← 调度层面的 buffer 视图
├── .node: ir.Buffer                       ← 持有底层 IR buffer
├── .users: list[NodeUser]                 ← 谁在使用这个 buffer（出度）
├── .defining_op: BaseSchedulerNode        ← 谁生产了这个 buffer（入度）
└── SchedulerDonatedBuffer                 ← 可被原地修改的外部输入 buffer
    └── defining_op = None                 ← 不是图内节点产出的
```

### 2.3 关键 API 语义分解

```
API: Scheduler.create_scheduler_node(node: ir.Operation) → BaseSchedulerNode
├── 语义：将 IR 层节点包装为调度层节点（工厂方法）
├── 分发逻辑：
│   ├── node.is_no_op()        → NopKernelSchedulerNode
│   ├── ComputedBuffer/TemplateBuffer → SchedulerNode（最常见）
│   └── ExternKernel           → ExternKernelSchedulerNode
├── WHY 工厂方法：不同 IR 节点有截然不同的调度语义
│   （SchedulerNode 可融合、ExternKernel 不可融合、Nop 可跳过）
└── 输出：调度节点已初始化 read_writes、group、outputs 等属性
```

```
API: Scheduler.compute_dependencies()
├── 语义：构建节点间的依赖图，处理别名和 mutation
├── 核心数据流：
│   ├── name_to_users: defaultdict[str, DedupList[NodeUser]] ← 临时大账本
│   ├── 为每个 buffer 维护使用者列表
│   ├── 处理 WAR（Write-After-Read）依赖 → WeakDep
│   ├── 处理 mutation 依赖 → StarDep
│   └── 处理 unbacked symint 依赖 → StarDep
├── 终局产物：
│   ├── Buffer.users: list[NodeUser]    ← 每块内存的出度
│   └── V.graph.mutated_inputs: set     ← 被修改的输入
└── 编译器概念：活跃变量分析 + 依赖图构建
```

```
API: NodeUser(node, can_inplace, is_weak)
├── 语义：记录一个节点对某块 buffer 的使用关系
├── 字段：
│   ├── node: BaseSchedulerNode | OutputNode ← 使用者
│   ├── can_inplace: bool             ← 是否可以原地操作（节省内存）
│   └── is_weak: bool                 ← 是否是弱依赖（仅排序用，不影响生命周期）
└── WHY is_weak：mutation 场景中，读者只需要等待写者完成，
  但不意味着读者的 buffer 需要存活到写者之后
```

```
Dep（依赖类型）层次：
├── MemoryDep(name, index)     ← 常规内存依赖：读写特定的 buffer
├── StarDep(name)              ← 全量依赖：依赖整个 buffer（mutation/符号依赖）
└── WeakDep(name, ...)         ← 弱依赖：仅用于排序，不影响 buffer 生命周期
```

---

## Section 3：全局数据流图

以下是 `_init` 方法从输入到最终输出的**完整数据流转**。

```
[ 阶段 0 | Bootstrap Phase ]
(建立基础设施，包装 IR 节点)

(1) 初始化调度器核心字段
    │  ├── 数据流 1 (输入扫描)：从 V.graph.graph_inputs / constants / torchbind_constants 收集名称
    │  └── 数据流 2 (节点包装)：nodes: list[ir.Operation] → create_scheduler_node → self.nodes
    ▼
【 self.completed_operations / self.available_buffer_names / self.nodes (调度节点列表) 】
    │
(2) 零维张量标记与依赖修剪
    │  ├── 数据流 3 (零维检测)：遍历 GPU 节点的 reads，检测 CPU 零维 buffer
    │  └── 数据流 4 (依赖修剪)：每个 node.prune_deps() 清理已满足的依赖
    ▼
【 V.graph.zero_dim_cpu_tensor_list / 修剪后的 node.unmet_dependencies 】
    │
    ========== (Bootstrap 结束：所有节点已包装、基础索引已建立) ==========
    ▼

[ 阶段 1 | Index & Donation Phase ]
(建立名称索引，收集可捐赠 buffer)

(3) 构建名称查找索引
    │  ├── 数据流 5 (节点索引)：self.nodes → {name: node} → self.name_to_node
    │  ├── 数据流 6 (buffer 索引)：self.nodes 的 outputs → {name: buf} → self.name_to_buf
    │  └── 数据流 7 (捐赠收集)：V.graph.graph_inputs_original → SchedulerDonatedBuffer → self.name_to_donated_buffer
    ▼
【 self.name_to_node / self.name_to_buf / self.name_to_donated_buffer (三张核心索引表) 】
    │
    ========== (索引建立完成：后续所有 pass 通过索引查找) ==========
    ▼

[ 阶段 2 | Dependency Analysis Phase ]
(依赖图构建——整个调度器最核心的 pass)

(4) 通信排序 + 依赖计算
    │  ├── 数据流 8 (通信排序)：self.nodes → comms.decide_global_ordering_of_comms → 重排序节点
    │  └── 数据流 9 (依赖计算)：遍历 nodes 的 read_writes → name_to_users → Buffer.users
    ▼
【 self.nodes (含完整依赖) / self.mutation_renames / self.mutation_real_name / V.graph.mutated_inputs 】
    │
    ========== (依赖分析完成：每个节点知道自己的前驱和后继) ==========
    ▼

[ 阶段 3 | Graph Optimization Phase ]
(图优化：排序、DCE、融合)

(5) 拓扑排序 → DCE → 祖先计算 → 距离计算
    │  ├── 数据流 10 (拓扑排序)：unmet_dependencies → DFS → 合法执行序
    │  ├── 数据流 11 (DCE)：Buffer.users → 识别无用户节点 → 移除 → V.graph.removed_buffers/operations
    │  ├── 数据流 12 (祖先)：unmet_dependencies 传递闭包 → node.ancestors
    │  └── 数据流 13 (距离)：依赖跳数 → node.min/max_input_distance
    ▼
【 拓扑有序、无死代码的 self.nodes / node.ancestors / node.min/max_input_distance 】
    │
(6) Foreach 节点创建 + 自定义 pass
    │  ├── 数据流 14 (Foreach)：V.graph.lists → ForeachKernelSchedulerNode → 替换原子节点
    │  └── 数据流 15 (自定义)：config._pre_fusion_custom_pass → 自定义变换
    ▼
【 含 Foreach 节点的 self.nodes 】
    │
    ========== (融合前优化完成) ==========
    ▼

[ 阶段 4 | Fusion Phase ]
(算子融合——调度器最重要的优化)

(7) Stream 分配 + 融合
    │  ├── 数据流 16 (Stream)：FX node metadata → node_to_stream / buff_to_stream
    │  ├── 数据流 17 (融合)：遍历节点对 → 判断可融合性 → FusedSchedulerNode
    │  └── 数据流 18 (自定义后 pass)：config._post_fusion_custom_pass → 自定义变换
    ▼
【 融合后的 self.nodes / self.name_to_fused_node 更新 】
    │
(8) 融合后清理
    │  ├── 数据流 19 (DCE 回访)：ExternTriton fusion 后的 Nop 消除
    │  ├── 数据流 20 (循环合并)：FusedSchedulerNode 内的 loop body 合并
    │  └── 数据流 21 (Template 定稿)：MultiTemplateBuffer 选择最优 backend
    ▼
【 最终融合图 / self.node_to_stream 】
    │
    ========== (融合完成：节点数大幅减少) ==========
    ▼

[ 阶段 5 | Memory & Ordering Phase ]
(内存优化和最终排序)

(9) Combo kernel + 峰值内存 + 通信重叠
    │  ├── 数据流 22 (Combo)：config.combo_kernels → 合并多个 kernel
    │  ├── 数据流 23 (峰值内存)：reorder_for_peak_memory → 按内存压力重排节点
    │  └── 数据流 24 (通信重叠)：reorder_compute_and_comm_for_overlap → 计算通信并行
    ▼
【 优化后执行序的 self.nodes 】
    │
(10) 分区优化 + 活跃变量分析
    │  ├── 数据流 25 (分区)：maybe_reorder_for_minimizing_partition → 减少 CUDA graph 分区
    │  └── 数据流 26 (活跃变量)：reversed self.nodes → node.last_usage → buffer 释放时机
    ▼
【 self.nodes (最终态) / node.last_usage (每个节点的 buffer 释放清单) 】
    │
    ========== (所有优化 pass 完成) ==========
    ▼

[ 阶段 6 | Finalization Phase ]
(收尾：日志、调试、度量)

(11) 日志输出 + 调试图 + 度量上报
    │  ├── 数据流 27 (IR 日志)：log_ir_post_fusion → 写出融合后 IR
    │  ├── 数据流 28 (调试图)：debug_draw_graph → 可视化 PNG
    │  └── 数据流 29 (度量)：graph_stats 表 → 记录节点数变化
    ▼
【 V.debug 日志 / 度量表 / self.buffer_names_to_free / self.origin_to_index (代码生成阶段使用) 】
```

---

## Section 4：粗粒度功能拆解表

| 阶段 | 行范围 | 功能 | 一句话总结 |
|---|---|---|---|
| 0a. Bootstrap | 3088-3106 | 初始化核心字段 + 包装 IR 节点 | 建立调度器骨架，把 IR 节点转化为调度节点 |
| 0b. 零维 & 修剪 | 3106-3111 | 零维 CPU tensor 标记 + 依赖修剪 | 处理边界情况，清理无用依赖 |
| 1. 索引构建 | 3112-3126 | 构建名称到节点/buffer 的索引 | 为后续所有 pass 建立 O(1) 查找表 |
| 2a. 通信排序 | 3149-3153 | 通信操作的固定排序 | 确保分布式通信操作的全局顺序一致 |
| 2b. 依赖分析 | 3155 | `compute_dependencies()` | **最核心**：构建节点间的依赖图 |
| 2c. 拓扑排序 | 3156 | `topological_sort_schedule()` | 确保合法执行顺序 |
| 2d. DCE | 3157 | `dead_node_elimination()` | 移除无用户的死节点 |
| 2e. 祖先 & 距离 | 3159-3160 | `compute_ancestors()` + `compute_input_distances()` | 计算依赖传递闭包和距离度量 |
| 3a. Foreach | 3168 | `create_foreach_nodes()` | 将批量操作合并为 ForeachKernel |
| 3b. 融合前自定义 | 3171-3172 | `_pre_fusion_custom_pass` | 用户自定义的融合前变换 |
| 4a. Stream 分配 | 3182-3187 | `_populate_stream_assignments()` | 为多流并行分配 CUDA stream |
| 4b. 融合 | 3189 | `fuse_nodes()` | **关键优化**：将可融合的节点合并为一个 kernel |
| 4c. 融合后自定义 | 3190-3191 | `_post_fusion_custom_pass` | 用户自定义的融合后变换 |
| 4d. 融合后清理 | 3193-3203 | DCE + loop merge + template finalize | 融合后的收尾清理 |
| 5a. Combo kernel | 3209-3215 | `create_combo_kernel_nodes()` | 将多个小 kernel 合并为 combo kernel |
| 5b. 峰值内存 | 3219-3228 | `reorder_for_peak_memory()` | 按内存压力重排节点以降低峰值 |
| 5c. 通信重叠 | 3232-3279 | `reorder_compute_and_comm_for_overlap()` | 计算和通信并行化 |
| 5d. 分区优化 | 3282-3291 | CUDA graph 分区最小化 | 减少 CUDA graph 的分区数 |
| 6. 活跃变量 | 3293 | `compute_last_usage()` | 确定每个 buffer 的最后使用点 |
| 7. 收尾 | 3295-3323 | 日志 + 调试 + 度量 + 初始化代码生成字段 | 为代码生成阶段准备最终数据 |

---

## Section 5：逐行详解

### 阶段 0a：Bootstrap

```python
# torch/_inductor/scheduler.py:3088-3106

def _init(self, nodes: list[ir.Operation]) -> None:
    super().__init__()
```

**功能与数据流起点：** 初始化调度器。`super().__init__()` 调用 `object.__init__`，这是 Python 的标准操作。

```python
    V.graph.scheduler = self
```

**设计哲学：** 将调度器实例挂载到全局虚拟化对象 `V.graph` 上，使得任何代码路径都能通过 `V.graph.scheduler` 访问当前调度器。这是 Inductor 中常见的**全局上下文模式**——通过 `V` (Virtualized) 实现类似 thread-local storage 的效果。

```python
    self.backends: dict[torch.device, BaseScheduling] = {}
    self.post_grad_graph_id = next(_post_grad_graph_counter)
    self._graph_partition_counter = itertools.count()
```

**功能与数据流起点：**
- `backends`：设备到调度后端的映射，延迟初始化
- `post_grad_graph_id`：全局递增的图 ID，用于区分不同的编译图
- `_graph_partition_counter`：分区计数器，用于 CUDA graph 分区

```python
    self.completed_operations: OrderedSet[str] = OrderedSet()
    self.available_buffer_names = OrderedSet(
        [
            *V.graph.graph_inputs.keys(),
            *V.graph.constants.keys(),
            *V.graph.torchbind_constants.keys(),
        ]
    )
```

**API 语义与底层机制：**
- `completed_operations`：已完成的操作名集合，调度执行期用来追踪进度
- `available_buffer_names`：当前可用的 buffer 名称集合

拆解 `available_buffer_names` 的初始化：
```
V.graph.graph_inputs.keys()        → 图输入张量的名称（如 "arg0_1", "arg1_1"）
V.graph.constants.keys()           → 常量的名称（如 "primals_1"）
V.graph.torchbind_constants.keys() → torchbind 常量的名称
```

**设计哲学：** 为什么需要 `available_buffer_names`？在调度执行期，当某个节点执行完毕后，需要判断其输出的 buffer 是否可以被后续节点使用。这个集合是一个**动态增长的可用资源池**。

```python
    self.nodes = [self.create_scheduler_node(n) for n in nodes]
    self.previous_node: BaseSchedulerNode | None = None
    self.current_node: BaseSchedulerNode | None = None
```

**[CORE] API 语义与底层机制（核心逻辑）：**

`create_scheduler_node` 的拆解：
```
self.create_scheduler_node(n) 的分发逻辑：
  n.is_no_op()                → NopKernelSchedulerNode(self, n)
  isinstance(n, ComputedBuffer | TemplateBuffer) → SchedulerNode(self, n)
  isinstance(n, ExternKernel) → ExternKernelSchedulerNode(self, n)
```

**设计哲学：** 为什么用工厂方法而非统一类型？因为不同 IR 节点有**完全不同的调度语义**：
- `SchedulerNode`：可以通过 Triton 后端融合，有 LoopBody，参与循环优化
- `ExternKernelSchedulerNode`：通常是 cuBLAS/cuDNN 调用，不可与 Triton kernel 融合
- `NopKernelSchedulerNode`：空操作，几乎可以跳过

**HOW：** 每个 `SchedulerNode.__init__` 内部会调用：
1. `_init_from_node(node)` → 初始化 outputs、read_writes 等
2. `_compute_attrs()` → 提取循环结构、计算 group key（设备 + 迭代空间）

```python
    self.update_zero_dim_cpu_tensor()
```

**[AUX]** 处理边界情况：当 GPU kernel 读取 CPU 上的零维 tensor 时，需要特殊标记（因为零维 tensor 的数据指针处理不同）。

```python
    # some new constants could have been created above
    self.available_buffer_names.update(V.graph.constants.keys())
```

**数据流转变：** `update_zero_dim_cpu_tensor` 可能创建新的常量 buffer，所以这里刷新可用名称集合。

```python
    for node in self.nodes:
        node.prune_deps()
```

**[CORE] 功能与数据流起点：** 修剪每个节点的依赖——移除已被满足的依赖。

**HOW：** `prune_deps()` 检查 `node.unmet_dependencies` 中的每个依赖，如果该依赖对应的 buffer 在 `available_buffer_names` 中（即图输入或常量），则将其从 `unmet_dependencies` 中移除，并加入 `node.read_writes.reads`。

**WHY：** 图输入和常量在调度开始时就已可用，对应的依赖自然已满足。不修剪的话，后续拓扑排序会错误地认为这些节点有未满足的前驱。

---

### 阶段 1：索引构建

```python
    # torch/_inductor/scheduler.py:3112-3146

    # See [Note: Graph Partition Device Contexts]
    self.default_device_context: torch.device | None = None

    self.name_to_donated_buffer: dict[str, SchedulerDonatedBuffer] = (
        self.get_donated_buffers()
    )
```

**[CORE] API 语义与底层机制：**

`get_donated_buffers()` 的拆解：
```
V.graph.graph_inputs_original  → 原始输入的有序字典
  遍历每个输入：
    isinstance(value, ir.DonatedBuffer)? → 是：创建 SchedulerDonatedBuffer(self, value, defining_op=None)
                                         → 否：跳过
```

**背景补充（重要高阶概念）：什么是 Donated Buffer？**

在 JAX/XLA 中引入的概念。当用户通过 `torch.compile` 编译函数时，如果某个输入张量在函数执行后不再被外部使用（即没有其他引用），Inductor 可以将其标记为 "donated"——意味着这块内存可以被**原地修改**，复用为输出 buffer。这消除了额外的内存分配，是峰值内存优化的关键手段之一。

在代码中，`SchedulerDonatedBuffer` 的 `defining_op=None` 表示它不是由图内节点产出的，而是从外部"捐赠"进来的。

```python
    self.name_to_node: dict[str, BaseSchedulerNode] = {
        n.get_name(): n for n in self.nodes
    }
```

**数据流转变：** 从列表 `self.nodes` → 字典 `{name: node}`。O(n) 遍历 → O(1) 按名查找。

**WHY：** 后续所有 pass 都需要频繁地通过名称查找节点。线性扫描是不可接受的。

```python
    self.name_to_buf: dict[str, SchedulerBuffer] = {
        buf.get_name(): buf for node in self.nodes for buf in node.get_outputs()
    }
```

**[CORE] 逻辑下钻：** 两层循环——外层遍历节点，内层遍历每个节点的输出 buffer。

拆解：
```
node.get_outputs()  → 返回 list[SchedulerBuffer]
                       每个 SchedulerBuffer 持有一个 ir.Buffer
buf.get_name()      → buf.node.get_name()，即底层 IR buffer 的名称（如 "buf0"）
```

**设计哲学：** 为什么需要 `name_to_buf` 而不只是 `name_to_node`？因为**一个节点可以产出多个 buffer**（如 MultiOutput 节点）。`name_to_node` 只能通过节点名查找，而 `name_to_buf` 可以通过任意 buffer 名查找其对应的 SchedulerBuffer。

```python
    self.name_to_fused_node: dict[str, BaseSchedulerNode] = self.name_to_node.copy()
```

**设计哲学：** `name_to_fused_node` 是 `name_to_node` 的**可变副本**。初始化时它们完全相同。当融合发生时，被融合的子节点的名称会指向新的 `FusedSchedulerNode`。这个字典在整个生命周期中不断更新，始终反映"当前每个名称归属哪个融合节点"。

```python
    self.mutation_real_name: dict[str, str] = {}
    self.mutation_renames: dict[str, str] = {}
```

**背景补充（重要）：为什么需要两个 mutation 字典？**

mutation（原地修改）是 Inductor 中最棘手的问题之一。当一个 kernel 原地修改了某个 buffer，所有后续对该 buffer 的引用都需要理解这个名字已经被"重命名"了。

- `mutation_renames`：正向映射。`{新名: 旧名}`。例如 `buf1` 在 `buf0` 的 kernel 内部被修改 → `mutation_renames = {"buf1": "buf0"}`
- `mutation_real_name`：反向映射。`{新名: 原始名}`。用于代码生成时追溯到原始名称。

**WHY 两个字典：** `mutation_renames` 处理链式 mutation（A 修改 B，C 又修改 B → 需要 rename 链），`mutation_real_name` 确保 codegen 始终使用用户可见的原始名称。

```python
    self.seen_template_fusions: OrderedSet[
        tuple[BaseSchedulerNode, BaseSchedulerNode]
    ] = OrderedSet()
```

**[AUX]** 去重集合：记录已经尝试过的 template fusion 配对，避免重复计算。

---

### 阶段 2a：通信排序

```python
    # torch/_inductor/scheduler.py:3147-3153

    # Must run first to correctly set dependencies, before all other passes that rely on
    # reading from .read_writes.reads or .unmet_dependencies
    self.nodes = comms.decide_global_ordering_of_comms(
        self.nodes,
        self.name_to_buf,
        self.name_to_fused_node,
    )
```

**[CORE] 功能与数据流起点：** 在分布式训练中，确保所有 rank 上的通信操作（AllReduce、AllGather 等）按照**相同的全局顺序**执行。

**WHY 必须最先运行：** 注释说得很清楚——`compute_dependencies` 及后续所有 pass 都依赖 `.read_writes.reads` 和 `.unmet_dependencies`。如果通信操作顺序不同步，依赖图就会不同步，导致不同 rank 生成不同的 kernel。

**HOW：** `decide_global_ordering_of_comms` 通过全局共识算法（通常基于节点名的确定性排序），在所有分布式 rank 之间统一通信操作的执行顺序。

---

### 阶段 2b：依赖分析（最核心的 pass）

```python
    # torch/_inductor/scheduler.py:3155

    self.compute_dependencies()
```

这是整个 `_init` 中**最复杂、最重要的单个调用**。详见 [compute_dependencies 调试指南](./compute_dependencies_debug_guide.md)。这里简述其核心逻辑：

**数据流：**
```
输入：self.nodes (每个 node 有 .read_writes.reads/.writes)
     + V.graph.graph_inputs (图输入)
     + node.get_outputs() (输出 buffer 及其别名/mutation 信息)

处理：
  1. 构建别名指针合并 → name_to_users 别名共享
  2. 扫描 unbacked symbol 的 def/use
  3. 对每个 node：
     - 处理 mutation → StarDep + WeakDep
     - 处理常规 reads → add_user()
     - 更新 mutation_renames 链
  4. 标记 outputs 为 "不可消除"
  5. 资产转移：name_to_users → Buffer.users

输出：每个 Buffer 有 .users 列表
     每个 Node 有完整的 .unmet_dependencies
     mutation_renames / mutation_real_name 已填充
     V.graph.mutated_inputs 已标记
```

---

### 阶段 2c-2e：排序、DCE、祖先计算

```python
    # torch/_inductor/scheduler.py:3156-3160

    self.nodes = self.topological_sort_schedule(self.nodes)
```

**[CORE] API 语义与底层机制：**

`topological_sort_schedule` 使用 **DFS 后序遍历**实现拓扑排序。

拆解：
```
对每个 node 调用 visit(n):
  如果 n 未被访问：
    标记 n 为已访问
    对 n.unmet_dependencies 按 name 排序后递归 visit
    将 n 追加到 result
```

**WHY DFS 而非 BFS/Kahn：** DFS 天然产生深度优先的拓扑序，这使得依赖链上的节点尽量相邻——有利于后续的算子融合（融合启发式倾向于融合相邻节点）。

**WHY 对依赖按 name 排序：** 确保排序的**确定性**（determinism）。相同的图总是产生相同的调度。

```python
    self.dead_node_elimination()
```

**[CORE] 功能：** 移除没有用户的节点（死代码消除）。

**HOW：** 逆序遍历拓扑排序后的节点（先访问用户，再访问生产者）：
1. 检查 node 的每个 output buffer 的 users
2. 如果所有 user 都是 weak 或已被移除 → 该 buffer 可消除
3. 如果 node 没有活跃 buffer 且无副作用 → 整个 node 可消除
4. 从 `name_to_buf` 中清理被移除 node 的依赖

**WHY 逆序遍历：** 拓扑序保证消费者在生产者之前被访问。如果消费者被移除了，生产者可能也变成死代码了。逆序遍历使得这种级联消除自然发生。

**编译器概念：** 这就是经典的 **Dead Code Elimination (DCE)**，等价于编译器中的 `ADCE`（Aggressive Dead Code Elimination）——不仅消除不可达代码，还消除结果未被使用的计算。

```python
    self.name_to_fused_node = {n.get_name(): n for n in self.nodes}
    self.compute_ancestors()
    self.compute_input_distances()
```

**`compute_ancestors`** 拆解：
```
对拓扑序中的每个 node：
  ancestors = ∅
  对每个 unmet_dependency：
    dep_node_name = name_to_buf[dep.name].defining_op_name()
    ancestors.add(dep_node_name)
    ancestors |= name_to_ancestors[dep_node_name]  ← 传递闭包
  node.ancestors = ancestors
```

**设计哲学：** `ancestors` 是依赖图的**传递闭包**。融合启发式用这个信息判断：如果两个节点有共同的祖先，它们可能共享数据，融合更有价值。

**`compute_input_distances`** 拆解：
1. 宏观设计哲学：为什么编译器需要知道“距离（Distance）”？

在执行期调度时，如果有多个节点同时处于“就绪（Ready）”状态（入度均为 0），调度器应该先执行哪一个？
随意挑一个吗？绝对不行！

Max Distance（最长距离 / 关键路径深度）：如果一个节点的 max_input_distance 很大，说明它是关键路径上的一环。它必须被优先执行！ 因为如果不赶紧算它，下游有一大串嗷嗷待哺的节点都会被卡死（暴露了计算延迟）。这在体系结构里叫“隐藏延迟（Latency Hiding）”。

Min Distance（最短距离）：如果两个节点的 min_dist 差不多，说明它们在图的“同一层”。调度器（如 Triton Codegen）会倾向于把同一层的无依赖节点**融合（Fusion）**在一起，以最大化指令级并行（ILP）并减少寄存器溢出（Register Spilling）。

这段代码的目标：就是为后续的融合器（Fuser）和调度器，贴上用于排序和决策的“距离标签”
算法：使用了DP算法，通过将所有父节点距离input的最短/最长距离+1的方式，更新本节点的长度
```
对拓扑序中的每个 node：
  如果无 unmet_dependencies → dist = 0（直接依赖图输入）
  否则 → min/max of (所有前驱的距离 + 1)
```

**WHY：** `min_input_distance` 和 `max_input_distance` 度量节点离图输入有多"远"。这影响融合决策：距离相近的节点更可能被融合（它们操作的数据在缓存中更热）。

---

### 阶段 3：Foreach 节点 + 自定义 Pass

```python
    # torch/_inductor/scheduler.py:3163-3172

    metrics.ir_nodes_pre_fusion += len(self.nodes)
    from torch._inductor.debug import log_ir_post_fusion, log_ir_pre_fusion

    log_ir_pre_fusion(self.nodes)
    self.num_orig_nodes = len(self.nodes)
```

**[AUX]** 记录融合前的节点数，用于度量和调试。

```python
    self.create_foreach_nodes()
```

**[CORE] 功能：** 将 `torch.foreach_*` 系列操作（如 `torch.stack`、`torch.cat` 列表版本）合并为单个 `ForeachKernelSchedulerNode`。

拆解 `create_foreach_nodes`：
```
对 V.graph.lists 中的每个名称列表（代表一个 foreach 操作组）：
  过滤掉已消除的节点和 Nop 节点
  创建 ForeachKernelSchedulerNode(self, snodes)
  在 name_to_fused_node 中将这些名称指向新的 foreach 节点
将 foreach 节点追加到 self.nodes 末尾
```

**WHY：** foreach 操作在语义上就是"对一组 tensor 做相同操作"。将它们合并为一个 kernel 可以减少 kernel launch overhead，并允许更激进的内存优化。

```python
    self.nodes = self.topological_sort_schedule(self.nodes)
    self.logged_slow_fusion = OrderedSet[tuple[str, str]]()
    if config._pre_fusion_custom_pass is not None:
        self.nodes = config._pre_fusion_custom_pass(self.nodes)
```

**[AUX]** 创建 foreach 节点后重新拓扑排序（因为新节点加入可能打乱序）。应用用户自定义的融合前 pass。

---

### 阶段 4a：Stream 分配

```python
    # torch/_inductor/scheduler.py:3180-3187

    # Stream assignments must be populated BEFORE fusion
    # to prevent fusing nodes across stream boundaries
    self.node_to_stream: dict[BaseSchedulerNode, int] = {}
    self.buff_to_stream: dict[str, int] = {}
    self._multi_stream_nodes: bool = False
    self.stream_idx_to_user_obj_idx: dict[int, int] = {}
    self._populate_stream_assignments()
```

**背景补充（中等重要性）：什么是 CUDA Stream？**

CUDA Stream 是 GPU 上的任务队列。不同 stream 上的操作可以并行执行。用户通过 `with torch.cuda.stream(s):` 指定操作在哪个 stream 上执行。

**WHY 必须在融合前运行：** 注释明确说明——不同 stream 上的节点**不能被融合**，因为融合后的 kernel 只能在一个 stream 上执行。如果先融合再分配 stream，可能会错误地跨越 stream 边界。

**HOW：** `_populate_stream_assignments` 从 FX 节点的 metadata 中读取 `custom.stream` 信息，映射到内部的 stream 索引。

---

### 阶段 4b：算子融合（最重要的优化）

```python
    # torch/_inductor/scheduler.py:3189

    self.nodes = self.fuse_nodes(self.nodes)
```

**[CORE] 这是整个调度器中影响性能最大的单个 pass。**

**编译器概念：** Kernel Fusion。将多个独立的 GPU kernel 合并为一个，消除中间结果的 global memory 读写，将其保留在 GPU 的快速缓存/寄存器中。

**类比：** 想象你有 10 个包裹要送到隔壁楼。不融合 = 跑 10 趟；融合 = 一次拿 10 个跑 1 趟。

`fuse_nodes` 的核心逻辑（简化）：
```
对每对相邻节点 (node1, node2)：
  如果 should_fuse(node1, node2)：
    创建 FusedSchedulerNode 包含两者
    更新 name_to_fused_node
重复直到无法继续融合
```

融合判断考虑：设备相同、迭代空间相同、无 stream 边界、无循环携带依赖等。

---

### 阶段 4d：融合后清理

```python
    # torch/_inductor/scheduler.py:3193-3203

    if any(
        isinstance(node, FusedExternTritonKernelSchedulerNode)
        for node in self.nodes
    ):
        self.dead_node_elimination()
```

**[AUX]** 如果有用户 Triton kernel 被 epilogue 融合，其原始输出 buffer 可能变成了 NopKernel（因为融合后的 kernel 直接写入目标 buffer）。这里再次运行 DCE 清理这些空壳节点。

```python
    self.merge_loops()
```

**[CORE] 功能：** 在融合后的 `FusedSchedulerNode` 内部，合并循环嵌套。

**编译器概念：** Loop Merging / Loop Fusion（循环融合）。将两个独立的循环体合并为一个循环：
```
# 融合前：
for i in range(N): A[i] = B[i] + C[i]
for i in range(N): D[i] = A[i] * E[i]

# 融合后：
for i in range(N):
    A[i] = B[i] + C[i]
    D[i] = A[i] * E[i]
```

**WHY：** 合并后 `A[i]` 不需要写入 global memory 再读回来，直接在寄存器中传递。

```python
    self.finalize_multi_template_buffers()
```

**[CORE] 功能：** `MultiTemplateBuffer` 是一种特殊的 IR 节点，它在编译期维护多个候选实现（如 Triton template vs cuBLAS），通过 benchmarking 选择最优的。这个 pass 确保所有未融合的 MultiTemplateBuffer 都做了最终选择。

---

### 阶段 5：内存优化与最终排序

```python
    # torch/_inductor/scheduler.py:3209-3215

    if config.combo_kernels:
        with dynamo_timed(
            "Scheduler.create_combo_kernel_nodes",
            log_pt2_compile_event=True,
            log_waitcounter=True,
        ):
            self.create_combo_kernel_nodes(num_ck_nodes=None)
```

**背景补充（中等重要性）：什么是 Combo Kernel？**

Combo Kernel 是一种将多个**不同形状**的 kernel 合并为一个大型 kernel 的技术。不同于普通 fusion（要求相同迭代空间），combo kernel 通过在同一个 kernel 中顺序执行多个独立操作来减少 kernel launch overhead。适用于大量小 kernel 的场景。

```python
    # torch/_inductor/scheduler.py:3219-3228

    if config.reorder_for_peak_memory:
        from .memory import reorder_for_peak_memory

        self.nodes = reorder_for_peak_memory(
            self.nodes,
            self.name_to_buf,
            self.name_to_fused_node,
            OrderedSet(V.graph.graph_inputs.keys()),
            OrderedSet(V.graph.get_output_names()),
        )
```

**[CORE] 编译器概念：Register Pressure Aware Scheduling**

在指令调度中，编译器可以重排指令来减少同时存活的寄存器数量。类比到 GPU，就是减少同时存活的 buffer 数量，降低峰值显存使用。

**HOW：** `reorder_for_peak_memory` 通过模拟 buffer 的分配/释放时间线，找到使峰值内存最小的执行顺序。

**WHY 必须最后运行（注释说）：** "Peak memory pass and overlap pass must run last, otherwise other reordering passes could undo their effects." 其他重排 pass 不考虑内存压力，可能会破坏内存最优的排序。

```python
    # torch/_inductor/scheduler.py:3232-3279

    if not config.deterministic and config.reorder_for_compute_comm_overlap:
        ...
        self.nodes = comms.reorder_compute_and_comm_for_overlap(self.nodes)
```

**背景补充（中等重要性）：计算-通信重叠**

在分布式训练中，通信操作（如 AllReduce）会在 GPU 和网络之间传输数据。如果能让计算和通信并行执行，就能隐藏通信延迟。这个 pass 将计算节点移到通信节点附近，使得通信进行时 GPU 不空闲。

```python
    self.process_grouped_nodes()
```

**[AUX]** 处理分组节点的后处理。

```python
    # torch/_inductor/scheduler.py:3282-3291

    if (
        config.graph_partition
        and config.triton.cudagraphs
        and config.triton.reorder_for_reducing_graph_partitions
    ):
        self.nodes = self.maybe_reorder_for_minimizing_partition(self.nodes)
        self.nodes = self.reorder_for_partition_with_simple_dependency(self.nodes)
```

**背景补充（低重要性）：CUDA Graph Partition**

CUDA Graph 将多个 GPU 操作录制为一个图，减少 CPU-GPU 同步开销。但 CUDA Graph 要求所有操作在同一个 CUDA stream 上，且不能有动态控制流。当 stream 切换时必须分割图。这个 pass 通过重排节点来减少分割次数。

---

### 阶段 6：活跃变量分析 + 收尾

```python
    # torch/_inductor/scheduler.py:3293

    self.compute_last_usage()
```

**[CORE] 编译器概念：Liveness Analysis（活跃变量分析）**

这是编译器后端的经典 pass。对每个变量，确定其**最后一次被使用的位置**。在最后一次使用后，该变量占用的内存可以被释放。

拆解 `compute_last_usage`：
```
future_used_buffers = 所有输出 buffer 名称（这些必须存活到最后）

逆序遍历 self.nodes：
  node.set_last_usage(future_used_buffers, mutation_real_name)
  # set_last_usage 做的事：
  #   对于 node 读取的每个 buffer：
  #     如果该 buffer 不在 future_used_buffers 中
  #     → 它的最后使用就是当前 node → 加入 node.last_usage
  #   将 node 的所有输出 buffer 加入 future_used_buffers
  future_used_buffers.update(node.last_usage)
```

**WHY 逆序遍历：** 和 DCE 相同的道理——从消费者向生产者遍历，自然确定"谁最后一次使用了我"。

**数据流转变：** 从全局视角的 `future_used_buffers`（正向集合） → 每个节点局部的 `node.last_usage`（该节点负责释放哪些 buffer）。

```python
    # torch/_inductor/scheduler.py:3295-3301

    if torch._inductor.config.test_configs.track_memory_lifecycle:
        self.insert_memory_check_nodes()

    log_ir_post_fusion(self.nodes)
    V.debug.graph_diagram(self.nodes)
    self.debug_draw_graph()
```

**[AUX]** 测试模式下的内存检查、IR 日志输出、调试图可视化。

```python
    # torch/_inductor/scheduler.py:3303-3323

    # used during codegen:
    self.buffer_names_to_free: OrderedSet[str] = OrderedSet()

    # fx graph node to the position it appears in the graph
    # for debug attribution
    self.origin_to_index: dict[torch.fx.Node, int] = {}

    # The only source of which stream context we are currently in during the codegen phase.
    self._current_stream_ctx: EnterCudaStreamContextLine | None = None
```

**功能与数据流起点：** 初始化代码生成阶段将使用的字段：
- `buffer_names_to_free`：代码生成期间待释放的 buffer 集合
- `origin_to_index`：FX 节点到图位置的映射（用于调试追踪）
- `_current_stream_ctx`：当前 CUDA stream 上下文（代码生成期间跟踪）

```python
    get_metric_table("graph_stats").add_row(
        lambda: {
            "graph_id": self.post_grad_graph_id,
            "num_nodes_before_fusion": self.num_orig_nodes,
            "num_nodes_after_fusion": len(self.nodes),
        }
    )

    self.removed_ops: OrderedSet[str] = OrderedSet()
```

**[AUX]** 记录融合前后的节点数到度量表。`removed_ops` 记录被移除但 buffer 仍需通过其他方式生成的操作。

---

## Section 6：数据结构生命周期总览

```
[Bootstrap 阶段]
1. 创建 → self.nodes (list[BaseSchedulerNode])
2. 创建 → self.available_buffer_names (OrderedSet[str])
3. 创建 → self.completed_operations (OrderedSet[str])
     ↓
==================== (索引阶段) ====================

[索引阶段]
4. 创建 → self.name_to_node (dict[str, BaseSchedulerNode])
5. 创建 → self.name_to_buf (dict[str, SchedulerBuffer])
6. 创建 → self.name_to_fused_node (dict[str, BaseSchedulerNode])
7. 创建 → self.name_to_donated_buffer (dict[str, SchedulerDonatedBuffer])
8. 创建 → self.mutation_renames (dict[str, str]) — 空
9. 创建 → self.mutation_real_name (dict[str, str]) — 空
     ↓
==================== (依赖分析) ====================

[依赖分析阶段]
10. 填充 → name_to_users (临时 defaultdict) ← 计算完即销毁
11. 填充 → Buffer.users (list[NodeUser]) ← 从 name_to_users 转移
12. 填充 → node.unmet_dependencies (OrderedSet[Dep])
13. 填充 → self.mutation_renames / self.mutation_real_name
14. 填充 → V.graph.mutated_inputs / V.graph.mutated_input_idxs
     ↓
==================== (拓扑排序 & DCE) ====================

[优化阶段]
15. 重排 → self.nodes (拓扑序)
16. 缩减 → self.nodes (移除死节点)
17. 填充 → node.ancestors (OrderedSet[str])
18. 填充 → node.min/max_input_distance (int)
19. 更新 → self.name_to_fused_node (Foreach 节点)
20. 填充 → self.node_to_stream / self.buff_to_stream
     ↓
==================== (融合) ====================

[融合阶段]
21. 缩减 → self.nodes (融合后大幅减少)
22. 更新 → self.name_to_fused_node (指向 FusedSchedulerNode)
23. 修改 → node.loop_body (循环合并)
     ↓
==================== (内存优化) ====================

[内存 & 排序阶段]
24. 可能缩减 → self.nodes (combo kernel)
25. 重排 → self.nodes (峰值内存优化 / 通信重叠)
26. 填充 → node.last_usage (OrderedSet[str]) ← 活跃变量分析
     ↓
==================== (收尾) ====================

[收尾阶段]
27. 创建 → self.buffer_names_to_free (空，代码生成期使用)
28. 创建 → self.origin_to_index (空，代码生成期使用)
29. 创建 → self._current_stream_ctx (None，代码生成期使用)
30. 创建 → self.removed_ops (空，代码生成期使用)

[最终存活的数据结构]
  self.nodes                  — 优化后的调度节点列表（主产物）
  self.name_to_buf            — buffer 名称 → SchedulerBuffer
  self.name_to_fused_node     — buffer 名称 → 所属融合节点
  self.mutation_renames       — mutation 重命名映射
  self.mutation_real_name     — mutation 原始名映射
  self.node_to_stream         — 节点 → CUDA stream
  node.last_usage             — 每个节点负责释放的 buffer
  node.unmet_dependencies     — 每个节点的未满足依赖
  Buffer.users                — 每个 buffer 的使用者列表
```

---

## Section 7：执行结果示例

### 用户代码

```python
import torch

@torch.compile
def f(x, y):
    a = x + y        # element-wise add
    b = a * 2        # element-wise mul
    c = torch.relu(b) # element-wise relu
    return c
```

### 执行模拟

```
# => 阶段 0：Bootstrap
#    nodes 输入: [ComputedBuffer("buf0", add), ComputedBuffer("buf1", mul), ComputedBuffer("buf2", relu)]
#    create_scheduler_node 对每个产生 SchedulerNode:
#      buf0: SchedulerNode(group=(cuda:0, ((s0,),)))
#      buf1: SchedulerNode(group=(cuda:0, ((s0,),)))
#      buf2: SchedulerNode(group=(cuda:0, ((s0,),)))
#
#    available_buffer_names = {"arg0_1", "arg1_1"} (图输入 x, y)
#
# => 阶段 1：索引构建
#    name_to_node = {"buf0": SN0, "buf1": SN1, "buf2": SN2}
#    name_to_buf  = {"buf0": SB0, "buf1": SB1, "buf2": SB2}
#    name_to_fused_node = name_to_node 的副本
#    name_to_donated_buffer = {} (无捐赠)
#
# => 阶段 2：依赖分析
#    name_to_users 构建:
#      "arg0_1": [NodeUser(SN0, can_inplace=False)]
#      "arg1_1": [NodeUser(SN0, can_inplace=False)]
#      "buf0":   [NodeUser(SN1, can_inplace=True)]    ← buf0 只被 SN1 读
#      "buf1":   [NodeUser(SN2, can_inplace=True)]    ← buf1 只被 SN2 读
#      "buf2":   [NodeUser(OutputNode, is_weak=False)] ← buf2 是输出
#
#    node.unmet_dependencies:
#      SN0: {MemoryDep("arg0_1"), MemoryDep("arg1_1")}  → prune 后 → ∅ (输入已可用)
#      SN1: {MemoryDep("buf0")}
#      SN2: {MemoryDep("buf1")}
#
# => 阶段 3：排序 & DCE
#    拓扑排序: [SN0, SN1, SN2] (已经有序)
#    DCE: 无死节点（所有节点最终被输出使用）
#    ancestors: SN0={}, SN1={SN0}, SN2={SN0, SN1}
#    distances: SN0=(0,0), SN1=(1,1), SN2=(2,2)
#
# => 阶段 4：融合
#    fuse_nodes 判断:
#      SN0 + SN1? → 可融合 (相同设备, 相同迭代空间, 无依赖冲突)
#      结果 → FusedSN0 = FusedSchedulerNode([SN0, SN1])
#      FusedSN0 + SN2? → 可融合
#      结果 → FusedSN1 = FusedSchedulerNode([SN0, SN1, SN2])
#
#    self.nodes = [FusedSN1]   ← 3 个节点融合为 1 个！
#    name_to_fused_node = {"buf0": FusedSN1, "buf1": FusedSN1, "buf2": FusedSN1}
#
# => 阶段 5：内存优化
#    (对于简单示例，无需重排)
#
# => 阶段 6：活跃变量分析
#    future_used_buffers = {"buf2"} (输出)
#
#    逆序遍历 FusedSN1:
#      SN2 读取 buf1 → buf1 不在 future_used_buffers → SN2.last_usage = {"buf1"}
#      SN1 读取 buf0 → buf0 不在 future_used_buffers → SN1.last_usage = {"buf0"}
#      SN0 读取 arg0_1, arg1_1 → 这些是输入，不参与释放
#      FusedSN1.last_usage = {"buf0", "buf1"} ← 执行完后可释放中间 buffer
#
# => 最终 self.nodes:
#    [FusedSchedulerNode([add, mul, relu])]  — 1 个融合 kernel

# => 度量:
#    num_nodes_before_fusion = 3
#    num_nodes_after_fusion = 1
```

---

## Section 8：断点速查表

| 断点位置 | 阶段 | 观察重点 |
|---|---|---|
| `scheduler.py:3088` — `_init` 入口 | 0 | `nodes` 参数的内容和数量 |
| `scheduler.py:3103` — `create_scheduler_node` 循环 | 0a | 每个 IR 节点的类型，被包装为什么 SchedulerNode |
| `scheduler.py:3119` — `name_to_node` 构建 | 1 | 索引是否完整 |
| `scheduler.py:3122` — `name_to_buf` 构建 | 1 | 是否有一个节点产出多个 buffer 的情况 |
| `scheduler.py:3149` — `decide_global_ordering_of_comms` | 2a | 通信节点顺序变化（分布式场景） |
| `scheduler.py:3155` — `compute_dependencies` | 2b | **最核心断点**。观察 `name_to_users` 的构建过程、mutation 处理 |
| `scheduler.py:3476` — `compute_dependencies` 内部 | 2b | 别名合并、WAR 依赖生成、unbacked symbol 处理 |
| `scheduler.py:3728` — `Buffer.users` 赋值 | 2b | 依赖从临时字典转移到 Buffer 对象 |
| `scheduler.py:3156` — `topological_sort_schedule` | 2c | 排序前后节点顺序变化 |
| `scheduler.py:3157` — `dead_node_elimination` | 2d | 被移除的节点名和原因 |
| `scheduler.py:3159` — `compute_ancestors` | 2e | 传递闭包的计算结果 |
| `scheduler.py:3189` — `fuse_nodes` | 4b | **关键优化断点**。观察哪些节点被融合 |
| `scheduler.py:3202` — `merge_loops` | 4d | 循环合并前后的 LoopBody 结构 |
| `scheduler.py:3293` — `compute_last_usage` | 6 | 每个节点的 `last_usage` 集合 |

### 关键断点详解

**调试依赖问题：**
```
🔴 断点位置：scheduler.py:3675 — 常规 reads 依赖注册
观察变量：
  - node.get_name() → 当前节点名
  - read.name → 被读取的 buffer 名
  - name_to_users.keys() → 当前已知的所有 buffer
  - node.unmet_dependencies → 不断增加的依赖集合
```

**调试融合问题：**
```
🔴 断点位置：scheduler.py:4023 — fuse_nodes 入口
观察变量：
  - nodes 的数量和类型
  - self.node_to_stream → 哪些节点在不同 stream（不可融合）
  - name_to_fused_node → 融合前的归属关系
```

**调试内存问题：**
```
🔴 断点位置：scheduler.py:6544 — compute_last_usage
观察变量：
  - future_used_buffers → 从后往前追踪活跃集合
  - node.last_usage → 每个节点负责释放的 buffer
```

---

## Section 9：核心概念速查表

| 概念 | 代码中的体现 | 编译器术语 |
|---|---|---|
| 节点依赖图 | `node.unmet_dependencies` / `Buffer.users` | Data Dependence Graph (DDG) |
| 拓扑排序 | `topological_sort_schedule()` — DFS 后序 | Topological Sort |
| 死代码消除 | `dead_node_elimination()` — 逆序扫描无用户节点 | Dead Code Elimination (DCE) |
| 依赖传递闭包 | `compute_ancestors()` — 累加前驱的 ancestors | Transitive Closure |
| 算子融合 | `fuse_nodes()` — 合并为 FusedSchedulerNode | Kernel/Instruction Fusion |
| 循环合并 | `merge_loops()` — 合并 FusedSchedulerNode 内的循环体 | Loop Fusion |
| 活跃变量分析 | `compute_last_usage()` — 逆序追踪 buffer 使用 | Liveness Analysis |
| 内存压力调度 | `reorder_for_peak_memory()` — 按峰值内存重排 | Register Pressure Scheduling |
| 别名分析 | `compute_dependencies()` 中的 `buf.get_aliases()` | Alias Analysis |
| WAR 依赖 | `WeakDep(other_name)` — 写后读的排序约束 | Anti-Dependence |
| WAW 依赖 | `StarDep(alt_name)` — 写后写（mutation） | Output Dependence |
| 原地操作 | `NodeUser.can_inplace` + `ir.DonatedBuffer` | In-Place Optimization |
| 通信-计算重叠 | `reorder_compute_and_comm_for_overlap()` | Latency Hiding / Overlapping |
| CUDA Graph 分区 | `maybe_reorder_for_minimizing_partition()` | Graph Partitioning |
| Mutation 重命名 | `mutation_renames` / `mutation_real_name` | SSA Renaming (for mutations) |
| 传递闭包距离 | `min/max_input_distance` — 依赖跳数 | Loop Depth / Dependence Distance |
| 去重列表 | `DedupList` — 带去重的 append | Unique List / Ordered Set with list semantics |
| 多流并行 | `node_to_stream` / `buff_to_stream` | Multi-Stream / Concurrent Execution |
| Unbacked SymInt | `unbacked_symbol_to_origin_node` — 动态形状符号 | Symbolic Execution / Dynamic Shapes |

---

> **交叉引用：** 本文涉及的 `compute_dependencies` 详细分析见 [compute_dependencies 调试指南](./compute_dependencies_debug_guide.md)。Scheduler 的整体设计哲学详见 `inductor-teaching-docs: scheduler/` 系列文档。
