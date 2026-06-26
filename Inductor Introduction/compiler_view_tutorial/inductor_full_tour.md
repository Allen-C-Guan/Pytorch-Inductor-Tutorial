# 一文看穿 Inductor：`torch.compile` 五层编译全记录

## —— 循环与寄存器在最末才出生的编译之旅

> 跟着两个极简例子 —— `exp(A + B)`(广播加法)和 `softmax(x)` —— 走完 PyTorch Inductor 编译栈的**全部 5 层**。每一层都配**真实运行的中间产物**(不是手写伪代码),并讲清"**为什么这么做,而不是 LLVM 那样做**"。
>
> 目的:**先建立全景心智模型**,再去翻海量代码。

---

## §0 这份文档怎么读

### Provenance(产物真实性)

**本文每一段产物都来自一次真实运行**,不是阅读或猜想出来的:

```bash
source env.sh new && python repro.py 2> artifacts/torch_logs.stderr.log
```

- **后端**:`TORCHINDUCTOR_NPU_BACKEND=new`(走 `torch_npu._inductor_new`,PyTorch GPU Triton 后端的干净扩展)
- **环境**:`torch 2.7.1+cpu` · `torch_npu 2.7.1.post4` · `Triton-Ascend 3.2.0` · `Ascend910B4` · 设备 `npu:0`
- **捕获日期**:2026-06-24
- 数值验证:`addexp` 与 `softmax` 的 compiled 输出 vs eager 全部 `torch.allclose = True`
- 复现脚本:[repro.py](repro.py) · 原始产物:[artifacts/](artifacts/)

> **本机无 CUDA**(`+cpu`/CANN 环境),所以第⑤层的 Triton 是 **Triton-Ascend** 编译到昇腾的 Triton(本机唯一能真实跑的 Triton)。前 4 层产物 GPU/NPU **完全一致**(设备无关)—— "换芯片只换 codegen 一片叶子"是全文压轴洞察(第⑤层揭晓)。

### 五层全景

| 层 | 一句话(方法 → 结果) | 核心产物 | 本例的数 |
|---|---|---|---|
| ① Dynamo | 符号追踪:把 Python 的动态执行录成静态 FX graph | `fx_graph_readable.py` | addexp 2 节点;softmax 1 个 `_softmax` |
| ② 分解 | 一张 decomp 表:把 ~2000 ATen 归一到 ~250 core-aten | `output_code.py` 头注 | addexp 不变;softmax `_softmax` → 5 个 |
| ③ Lowering | `inner_fn` 闭包嵌套:把 producer 直接嵌进 consumer 闭包 | `ir_pre_fusion.txt` + `inner_fn` | addexp `add+exp` → 1 个 buf0;softmax → 3 个 buffer |
| ④ 调度 + 融合 | `FusedSchedulerNode` 分组 + `can_fuse` 4 道闸判定 | `ir_post_fusion.txt` + `+fusion` | addexp no-op;softmax 3 → 1 组 |
| ⑤ Codegen | 两趟重放成 `@triton.jit`,中间值走片上 SSA | `output_code.py` 核 | addexp 1 核;softmax 1 核(中间 2 个不落 GM) |

### Lowering 与 Tracing 是同一个 pass

很多材料把 "Lowering" 和 "Tracing" 画成两个独立阶段。代码里它们是**同一个 pass** —— Inductor 本质是一个 `torch.fx.Interpreter`,每跑一个 FX 节点就去查 `lowerings[]` 表构造 IR 闭包,"符号执行"和"建 IR"是同一个动作的两面。真正的"独立产物边界"是 5 个,不是 4 个(对照本目录的[蓝本草图](../inductor_deep_dive.md))。

### 章节约定

每一层先连贯讲完真实产物与具体机制,**章末**再用三点收束:**这套机制是什么 → LLVM 会怎么做 → 一句话心智模型**。

---

## §1 两个例子 + 三范式预告

```python
# 广播 add+exp(纯 pointwise)
@torch.compile
def f_addexp(A, B):           # A:(64,1)  B:(1,48)
    return torch.exp(A + B)

# softmax(含 reduction)
@torch.compile
def f_softmax(x):             # x:(64,48)
    return torch.softmax(x, dim=-1)
```

`addexp` 是**纯 pointwise**(广播 + 逐元素),走完整 5 层最干净;`softmax` 含 **reduction**(沿一维折叠),用来揭开 inductor 设计真正发光的地方。

读到最后你会抓住三个范式(它们在不同层反复出现):

1. **去控制流** —— 没有 `for`/`if`,控制流被抽象成多维空间。
2. **符号依赖** —— 依赖关系用 sympy 代数式表达,不是 use-def / phi。
3. **延迟具象化** —— 循环、寄存器、寻址全部推迟到最后一层才生成。

---

## §2 第①层 · Dynamo 捕获

> **Dynamo 用符号追踪(interpreter 实跑一遍 Python),把 `f_addexp` 录成 2 个 ATen 节点(`add`/`exp`)的静态 FX graph。**

`addexp` 被捕获成的 FX graph([来源:artifacts/addexp/fx_graph_readable.py](artifacts/addexp/fx_graph_readable.py)):

```python
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 1]", arg1_1: "f32[1, 48]"):
        # repro.py:155  return torch.exp(A + B)
        add: "f32[64, 48]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        exp: "f32[64, 48]" = torch.ops.aten.exp.default(add);  add = None
        return (exp,)
```

两个 ATen 算子:`add.Tensor`、`exp.default`。**广播已经体现在 shape 上**:`(64,1)+(1,48)→(64,48)`,但此刻还不知道它要怎么实现。`softmax` 捕获后已经是分解形态(第②层),5 个算子:

```python
amax: "f32[64, 1]" = torch.ops.aten.amax.default(arg0_1, [-1], True)
sub:  "f32[64, 48]" = torch.ops.aten.sub.Tensor(arg0_1, amax)
exp:  "f32[64, 48]" = torch.ops.aten.exp.default(sub)
sum_1:"f32[64, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
div:  "f32[64, 48]" = torch.ops.aten.div.Tensor(exp, sum_1)
```

**机制**:`torch._dynamo` **符号追踪** Python 函数,把执行路径"录"成一张静态 FX graph。每个 FX 节点是一个 `torch.ops.aten.*` 调用,带 `tensor_meta`(shape/dtype/stride)。源码:`torch/_dynamo/` + `compile_fx.py:957` `V.debug.fx_graph(...)` 落盘。
**LLVM 对照**:LLVM 的前端**解析 AST / 字节码**(gcc、clang 那套)。Dynamo 不解析语法树,而是**真的去"运行"一遍你的 Python**,在每一步把遇到的算子记下来 —— 所以能处理 Python 动态性(只要追踪时没碰到它无法符号化的东西)。
**心智模型**:Python 的动态性,在这一层被一次性**烧成一张静态图**。从这往后,Inductor 只看这张图,不再碰 Python。

> FX graph 的原始可运行复现脚本在 `fx_graph_runnable.py`;post-grad pattern-matcher 改写后的图在 `fx_graph_transformed.py`(本案例无改写)。

---

## §3 第②层 · 分解

> **一张 decomp 表把 ~2000 ATen 归一到 ~250 core-aten:`_softmax` 1 → 5;addexp 的 `add/exp` 不变,仍是 2。**

你写的是**一个** `torch.softmax`,但 Inductor 收到的是**五个**算子。证据直接写在生成代码的 provenance 注释里([来源:artifacts/softmax/output_code.py](artifacts/softmax/output_code.py)):

```
# Topologically Sorted Source Nodes: [softmax], Original ATen: [aten._softmax]
# Source node to ATen node mapping:
#   softmax => amax, div, exp, sub, sum_1
```

即 `aten._softmax`(1 个)⇒ `amax, sub, exp, sum, div`(5 个 core-aten)。`addexp` 的 `torch.exp(A + B)` 本来就是 `add` + `exp`,没有进一步分解。

**机制**:在 AOTAutograd 阶段,一张分解表(`torch/_inductor/decomposition.py:99` = `core_aten_decompositions() ∪ inductor_decompositions`)把 ~2000 个 ATen 算子**归一**到 ~250 个 core-aten。复杂算子被拆成一组简单原语,后续 lowering 只需认识这 ~250 个。触发:`compile_fx` 里 `decompositions=decompositions`。
**LLVM 对照**:LLVM **没有对应物**。这是 ML 编译器特有的"前端归一" —— 在真正的优化器看到图之前,先把方言缩成一个小而全的字母表。
**心智模型**:进真正的编译器前,先把算子**缩成一个小字母表**,让后面所有阶段只需面对少量稳定的原语。这也解释了为什么 `addexp` 只会产生一个 IR 节点,而 `softmax` 会产生三个。

---

## §4 第③层 · Lowering(含 Tracing)→ IR

> **`make_loader` 返回 `inner_fn` 闭包本身(而非 `ops.load(name)`),把 producer 直接嵌进 consumer 闭包 —— `add+exp` 编进单个 `buf0`,`add` 从不物化。**

这是 Inductor 与传统编译器**分道扬镳**的一层 —— 这里产生的 IR 长得跟任何教科书里的 IR 都不一样。

### 真实产物:符号闭包 `inner_fn_str`

`addexp` 的输出 buffer `buf0`,其**符号闭包体**([来源:artifacts/addexp/inner_fn.txt](artifacts/addexp/inner_fn.txt),由 `Loops.inner_fn_str()` 打印,`ir.py:860`):

```python
def inner_fn(index):
    i0, i1 = index
    tmp0 = ops.load(arg0_1, i0)      # A[i0]   ← 广播!A 只用第 0 维(行)
    tmp1 = ops.load(arg1_1, i1)      # B[i1]   ← 广播!B 只用第 1 维(列)
    tmp2 = tmp0 + tmp1
    tmp3 = ops.exp(tmp2)
    return tmp3
```

**广播的全部奥秘在这一段里**:`A:(64,1)` 只用索引 `i0`(行),`B:(1,48)` 只用 `i1`(列)。没有数据搬运、没有对齐 —— 广播就是**索引维度的省略**,一道代数上的 index remapping。

### 真实产物:`ir_pre_fusion.txt`(LoopBody 物化形态)

把闭包在 `MockHandler` 下重放、物化成 FX 图后([来源:artifacts/addexp/ir_pre_fusion.txt](artifacts/addexp/ir_pre_fusion.txt)):

```
op0: SchedulerNode(ComputedBuffer)
op0.group.iteration = (3072, 1)            # 64*48=3072 个迭代点
op0.sizes = ([64, 48], [])
class op0_loop_body:
    var_ranges = {p0: 64, p1: 48}
    index0 = p0          # ← A 的索引(广播:只用 p0)
    index1 = p1          # ← B 的索引(广播:只用 p1)
    index2 = 48*p0 + p1  # ← 输出 buf0 的线性地址
    def body(self, ops):
        load   = ops.load('arg0_1', get_index('index0'))
        load_1 = ops.load('arg1_1', get_index('index1'))
        add    = ops.add(load, load_1)
        exp    = ops.exp(add)
        store  = ops.store('buf0', get_index('index2'), exp, None)
```

`index0=p0`、`index1=p1`、`index2=48*p0+p1` —— **广播落地成 sympy 线性地址公式**。注意 `add` 和 `exp` 已经在**同一个** `op0_loop_body` 里(为什么?见下节)。

### `add+exp` 为何是 1 个 IR:闭包嵌套,逐步打开

`add` 和 `exp` 明明是两个算子,怎么就成了**一个** IR 节点(`buf0`)?因为从头到尾只有 1 个 IR 节点 —— `add` 是个"幽灵":一个被引用、但从未物化的懒闭包。逐步打开:

**Step 1 · lowering `add(A,B)`** —— `register_pointwise`→`make_pointwise`(`lowering.py:574`)。广播在 `transform_args`(`broadcast=True`)里把 A:(64,1)/B:(1,48) `expand` 成 (64,48),包成 `ExpandView`(不分配存储,只记 reindexer)。然后:

```python
loaders = [A.make_loader(), B.make_loader()]                       # lowering.py:597
inner_fn = lambda index: ops.add(loaders[0](index), loaders[1](index))
return TensorBox(StorageBox(Pointwise(inner_fn, ranges=(64,48))))  # ir.py:826 / 7105
```

此刻 `add` 是一个**未物化**的 Pointwise:没有 name、没有 buffer、没注册进 `V.graph`,只有一段闭包。

**Step 2 · lowering `exp(add结果)`** —— 同样走 `make_pointwise`,关键一行 `loaders = [add结果.make_loader()]`(`lowering.py:597`)。`add结果` 是 `TensorBox(StorageBox(Pointwise))`,它的 `.make_loader()` 链式解析:`TensorBox.make_loader`(`ir.py:6971`)→ `StorageBox.data.make_loader()` → **`Pointwise.make_loader` 直接 `return self.inner_fn`**(`ir.py:921-926`)。

> 🔑 **合并的唯一秘密**:`Pointwise.make_loader()` 不返回"从 buffer 读",而是**返回计算闭包本身**。所以 `exp` 的 loader 就是 `add` 的 `inner_fn` —— 同一个 Python 闭包对象的引用,不是拷贝、不是 `ops.load(name)`。

于是 `exp.inner_fn(index) = ops.exp( add.inner_fn(index) ) = ops.exp( ops.add(load_A, load_B) )`。整条 `D=f(A,B)` 在 lowering 时就编进**一个嵌套闭包**,只是还没展开。

**Step 3 · 为什么 add 不物化?** 没有任何触发条件逼它 realize(单消费者、非输出、闭包很小 —— 见下文 realize 机制)。`add` 永远停在"懒闭包"形态,只被 `exp` 唯一引用。最终只有 `exp`(输出)被物化 → **全图只有 1 个 buffer `buf0`**,它的闭包里(经由 loader 引用)包含了 `add` 的逻辑。

**Step 4 · "展开"发生在 trace 时** —— 即"在 OpsHandler 下重放闭包"。当 `buf0` 的 `ComputedBuffer` 要被 trace 成 LoopBody 时,`store_output`(`ir.py:934`)调 `make_loader()(index)`,**深度优先地**把 `ops.exp(...)` 里的 `add.make_loader()(index)` 也跑了一遍 → 在同一个 handler 下、产生同一个 LoopBody,里面既有 `add` 又有 `exp`。这就是 `ir_pre_fusion.txt` 里 `op0_loop_body` 同时含 add+exp 的由来。

**真实插桩证据**([artifacts/merge_trace/run.log](artifacts/merge_trace/run.log),脚本 [merge_trace.py](artifacts/merge_trace/merge_trace.py),patched `Pointwise.make_loader` / `StorageBox.realize` / `ComputedBuffer.__init__`):

```
BUFFER name=buf0 data_type=Pointwise origins=[add, exp]
        tmp2 = tmp0 + tmp1          # ← add
        tmp3 = ops.exp(tmp2)        # ← exp  (同一个闭包)
V.graph.buffers 总数 = 1
StorageBox.realize 总调用次数 = 4    # 真正 Pointwise→ComputedBuffer 晋升只有 1 次(buf0);add 从未晋升
Pointwise.make_loader 触发次数 = 3   # add 的 loader 被内联进 exp 的信号
结论: addexp 的 add 中间值被合并为单个 Pointwise(仅 1 个核, 无独立 add 核) = True
```

### Reduction 的 IR 形态

`softmax` 的 `amax` 被降级成一个 **Reduction 节点**([来源:artifacts/softmax/ir_pre_fusion.txt](artifacts/softmax/ir_pre_fusion.txt)):

```
op0.sizes = ([64], [48])          # 外层 64 个点,折叠维度 48
class op0_loop_body:
    var_ranges = {p0: 64, p1: 48}
    index0 = 48*p0 + p1           # 读:扫过整行 48 个
    index1 = p0                   # 写:折叠成 1 个
    def body(self, ops):
        load     = ops.load('arg0_1', get_index('index0'))
        reduction = ops.reduction(torch.float32, torch.float32, 'max', load)   # ← 折叠算子
        store_reduction = ops.store_reduction('buf0', get_index('index1'), reduction)  # ← 写折叠结果
```

`ops.reduction('max', ...)` + `ops.store_reduction` —— reduction 在 IR 里不是一个 for 循环累加,而是"**多一个会折叠的维度**"(`sizes=([64],[48])`:64 个外层点,每个点把 48 维折叠成 1)。

### realize 机制:从"懒"到"实"的边界

物化只有**一个原语**:`StorageBox.realize()`(`ir.py:7123`)。它把懒的 `Pointwise/Reduction` 换成一个**有 name(buf0)、有 layout、注册进 `V.graph`** 的 `ComputedBuffer`。

```
lowering 产物(全懒,无 name / 无 layout):
  TensorBox → StorageBox → Pointwise/Reduction(inner_fn 闭包)
                │
                │  某个触发调用 .realize()
                ▼  StorageBox.realize()   ir.py:7123
  self.data = ComputedBuffer(name=None, layout=FlexibleLayout, data=<原 Pointwise>)
  self.data.name = V.graph.register_buffer(...)     # 拿到 buf0 / buf1 ...
                │
                ▼  之后 scheduler 把每个 ComputedBuffer 包成 SchedulerNode
```

**🔑 "冻结"边界 —— 合并的分水岭**:realize 之后,后续读者拿到的 loader 变成 `ComputedBuffer.make_loader()` → `ops.load(name, index)`(**从具名 buffer 读**),而不再是原闭包。所以闭包内联合并(免费、lowering 级、无判定)**只发生在物化之前**;物化之后,合并只能靠 scheduler 的 `can_fuse`(有判定)。这就是 `add+exp`(1 个节点,闭包内联)与 softmax 三个独立 buffer(要走 scheduler)的根本区别 —— 也是第③层通往第④层的交接。

**什么会触发 realize?**(真实代码,非穷尽)

| 触发 | 位置 | 为什么 |
|---|---|---|
| 图输出 | `graph.py:1297` | 输出必须是有名 buffer 才能返回 |
| extern 核输入(matmul/conv/fallback) | `ir.py:5071` | extern 要真实指针,吃不了闭包 |
| reduction 的**结果** | `lowering.py:5768` | reduction 输出物化(**输入保持懒**,被内联进 reduction 闭包) |
| mutation / in-place | `ir.py:3724` 等 | 被改写的 buffer 要有稳定 name |
| mutation 的**读者** | `graph.py:937` | 被改写前,所有读过旧值的懒闭包要先冻结 |
| 闭包过大(防栈溢出) | `graph.py:1693` | inner_fn 太大时强制物化,避免 Inductor 自己求值时爆栈 |

**多消费者:复制,不是缓存** —— 一个懒 pointwise 若有 N 个消费者(如 softmax 的 `exp` 同时喂 `sum` 和 `div`),默认是把闭包**复制进每个消费者的 loader**(重复计算),而不是物化+缓存。唯一的缓存启发式是 `mark_reuse`(`ir.py:7172`):仅当 `users>1` 且(读次数 > 4 或闭包过大 或 CPU+含 exp/sigmoid)才物化缓存。softmax 的 `exp`(2 消费者、读 1 次)没踩阈值 → 被复制进 op1 和 op2 —— 这正是 `ir_pre_fusion.txt` 里 op1、op2 都重复 `sub+exp` 的根因(设计如此,非 bug)。

> **scheduler 从不调 `realize()`** —— 物化纯粹是 lowering 阶段的概念。

### 机制 · LLVM 对照 · 心智模型

**机制**:`GraphLowering`(`graph.py:266`)本身是一个 `torch.fx.Interpreter`。它逐个"运行" FX 节点,每遇到一个算子就去 `lowerings[]`(`lowering.py:104`)查表,构造一个**延迟的 IR 节点**:`TensorBox → StorageBox → Pointwise/Reduction`(`ir.py`),节点内部持有一个**闭包 `inner_fn`**。"Tracing" 就是"把闭包在某个 `OpsHandler`(`ops_handler.py`)下重放一遍":重放时 `ops.load/add/exp` 被拦截,从而读出依赖、生成代码。同一个闭包会被重放多次(分析时用 `MockHandler`,生成代码时用 `TritonKernel` 的 handler)。
**LLVM 对照**:LLVM **解析 AST**,然后做 use-def 分析、插 phi 节点、建 SSA。Inductor **不解析任何源码** —— IR 是**把闭包"跑"出来的**。广播在 LLVM 范式里要插 broadcast 节点 / 做数据布局对齐;在 Inductor 里它只是一道索引公式。
**心智模型**:**维度即空间,索引即代数**。一个算子 = "在一个 (M,N) 空间的每个点 (i,j),按这几道索引公式 load 几个值、算一下、store 回去"。广播不是一个动作,而是一道索引公式的省略。

---

## §5 第④层 · 调度 + 融合

> **`FusedSchedulerNode` 把可融合的 SchedulerNode 分组到一个组节点(softmax: `op0+op1→op0_op1→+op2`),body 原样不动;`can_fuse` 的 4 道闸判定谁能同组。**

### 真实产物:addexp 是 no-op

`addexp` 融合前 = 融合后,都只有 `op0` 一个节点(因为 lowering 已把 add+exp 内联进一个闭包):

```
BEFORE fuse_nodes:  [SchedulerNode(name='op0'), Pointwise(origins=[exp, add])]
AFTER  fuse_nodes:  [SchedulerNode(name='op0'), Pointwise(origins=[exp, add])]   # 不变
```

### 真实产物:softmax 真正发生融合(3 → 1)

`softmax` 融合前有 **3 个独立节点**(各自物化了 buffer):`op0`(max reduction)、`op1`(sum reduction)、`op2`(div pointwise)。融合决策日志([来源:artifacts/torch_logs.stderr.log](artifacts/torch_logs.stderr.log),`TORCH_LOGS=+fusion`):

```
attempting fusion (1/10): 3 nodes
fuse_nodes_once, candidates:
  op0: Reduction([48], max,  origins=[amax])
  op1: Reduction([48], sum,  origins=[sum_1, exp, sub])
  op2: Pointwise([64,48],    origins=[div, exp, sub])
cannot fuse op0 with op2: intermediate nodes between node1 & node2   # op1 夹在中间,不能跳着合
found 2 possible fusions
fusing op0 with op1            # 先合 op0+op1
fusing op0_op1 with op2        # 再合 (op0_op1)+op2
completed fusion round (1/10): fused 3 nodes into 1 nodes
```

融合后产物 —— 一个 `FusedSchedulerNode`,**递归**打印出全部 3 个成员及各自的 LoopBody([来源:artifacts/softmax/ir_post_fusion.txt](artifacts/softmax/ir_post_fusion.txt)):

```
op0_op1_op2: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode)
op0_op1_op2.snodes[0] = op0: ... class op0_loop_body: ...(max reduction)...
op0_op1_op2.snodes[1] = op1: ... class op1_loop_body: ...(sum reduction)...
op0_op1_op2.snodes[2] = op2: ... class op2_loop_body: ...(div pointwise)...
```

### `can_fuse` 的 4 道闸

`can_fuse`(`scheduler.py:3528`)是**四道串行闸**,任意一道挂掉就 `return False`(并记一条 `cannot fuse X with Y: <原因>`):

| 闸 | 检查 | 代码 |
|---|---|---|
| ① 身份/类型/**序** | node1≠node2;非 extern/nop;**node2 的算子不能是 node1 的祖先**(node1 必须能排在 node2 前) | `scheduler.py:3534-3643` |
| ② **方向** | node1 ∈ node2.ancestors → **垂直**(生产→消费);否则**水平**(兄弟/共读) | `scheduler.py:3663-3673` |
| ③ **迭代空间 group 兼容** | 两节点 `group=(device, 各层 size 之积)` 的 numel/rnumel 要匹配(pointwise 全等;reduce+reduce 全等) | `codegen/simd.py:1071-1205` |
| ④ **依赖序可行(无中间节点)** | node2 的未满足依赖能否被 node1 的写满足;剩余依赖若指向"夹在中间的第三个节点"→ `intermediate nodes` 拒绝 | `scheduler.py:3675-3726` |

闸④里的 `fusable_read_and_write`(`scheduler.py:3783`:`read.index == write.index`)是判断"某个读能否被某个写满足"的子步骤 —— 常被简化说成"判融合 = 解 sympy 偏移方程(write−read==0)",其实那只是闸④的一个子检查,不是判定的全部。

**softmax 的真实判决**(patched `Scheduler.can_fuse`,[artifacts/merge_trace/run.log](artifacts/merge_trace/run.log)):

```
can_fuse 总调用次数 = 5
ACCEPT  op0 + op1     same_group=True                       # 两个 reduction,group 兼容(闸③),无中间节点(闸④)
REJECT  op0 + op2     same_group=False   why=intermediate nodes between node1 & node2   # op1 夹在 op0 与 op2 之间(闸④)
ACCEPT  op1 + op2                                            # 下一轮
ACCEPT  op0_op1 + op2                                        # op0+op1 合体后 buffer_names={op0,op1} 同时满足 op2 的两个读 → 闸④空了,通过
```

**为什么 op0+op2 被拒、op0_op1+op2 又通过?** op2 同时读 op0 的 max 和 op1 的 sum。直接合 op0+op2:op1 夹在中间(闸④"intermediate nodes")。先合 op0+op1 成 `FusedSchedulerNode`(`get_buffer_names`=并集 {op0,op1},`scheduler.py:1400`),它同时产出 op2 要的两个值 → op2 的依赖全被满足,闸④再无中间节点 → 通过。

> 和 [fuse_principle.md](../fuse_principle.md) 的对应:那里的"原则一 迭代空间同构"≈ 闸③(group numel/rnumel);"原则二 访存依赖局域性"≈ 闸④;`Expr_write − Expr_read == 0` 是闸④的子检查。概念版抓住主干,代码版补上了"序 / 方向 / group / 中间节点"几道同样关键的闸。

### fuse 动作:只分组,不碰 body

判定通过后,fuse 怎么把两个节点合起来?调用链(`scheduler.py`):`fuse_nodes_once` 主循环(`:3048`)→ `fuse_two_nodes`(`:3007`)→ `get_backend(device).fuse`(`:3014`)→ `BaseScheduling.fuse`(`:4436`)→ **`FusedSchedulerNode.fuse`(`:1316`)**。核心就一句(`:1346`):

```python
nodes = list(itertools.chain(node1.get_nodes(), node2.get_nodes()))   # 把两边的成员摊平拼接
return cls(node1.scheduler, nodes)                                     # 造一个新的 FusedSchedulerNode
```

**只把成员塞进新节点,不碰任何 body。** 真正的"merge"是**簿记重连** —— `refresh_group_node_dependencies`(`scheduler.py:1266`):

```python
group_snode.unmet_dependencies = (
    OrderedSet(dep for dep in 成员 unmet 并集
               if dep.name not in group_snode.get_buffer_names())   # − 内部依赖
    - group_snode.read_writes.writes)                                # − 自己的写
```

即 `unmet = 成员并集 − 内部 buffer − 自己的写`。真实插桩([artifacts/fuse_trace/run.log](artifacts/fuse_trace/run.log)):softmax `init_group_node` 调用 2 次,`op0+op1→op0_op1`(成员 unmet 并集 `[buf0]` → 组 unmet `[]`,buf0 变内部依赖被消),再 `+op2→op0_op1_op2`(`[buf0,buf1]` → `[]`)。addexp 是 `init_group_node 调用 0 次`(没有 scheduler 融合)。

**🔑 `FusedSchedulerNode` 没有 `_body` 字段**(只有 `SchedulerNode` 有,`scheduler.py:1011`);各成员自带各自的 LoopBody。它的 docstring 原话:*"a 'fake' scheduler node that represents a group ... by maintaining its unmet dependencies as the union of its constituent nodes."* —— **fuse 的即时产物是个"分组簿记节点",不是合并的计算体**。body 的真正合并发生在第⑤层 codegen。

### 节点 → 核的映射 + 逐 pass

融合组最终怎么变成核的执行顺序?`+schedule` 打印 codegen 时的节点调度([来源:artifacts/torch_logs.stderr.log](artifacts/torch_logs.stderr.log)):

```
softmax Schedule:
  [SchedulerNode(op0), DisableReduction, EnableReduction,
   SchedulerNode(op1), DisableReduction, EnableReduction,
   SchedulerNode(op2)]
```

`DisableReduction/EnableReduction` 是 reduction 边界哨兵 —— 说明 op0/op1/op2 会在同一个核里顺序执行(reduction 处插入边界)。

`repro.py` 用 mock 包裹了 `create_foreach_nodes / fuse_nodes / merge_loops / reorder_for_peak_memory / compute_last_usage`,记录每个 pass 前后节点列表([来源:artifacts/passes/](artifacts/passes/)):softmax 的 `fuse_nodes` 把 3 个 `SchedulerNode` 变成 1 个 `FusedSchedulerNode(op0_op1_op2)`,其余 pass 不改变节点身份(只调循环顺序/生命周期)。

> SVG 可视化(`graph_diagram.svg`)需 graphviz,本机未装 `dot`,无法真实生成;融合组的文字版(`ir_post_fusion.txt` 的 `FusedSchedulerNode`)已完整呈现同样信息。

### 机制 · LLVM 对照 · 心智模型

**机制**:`scheduler.py` 的 `Scheduler`(`:1983`)把每个 `ComputedBuffer` 包成 `SchedulerNode`,然后跑一串 pass。核心是 `fuse_nodes_once`(`:2995`)对候选对调用 `can_fuse`(`:3528`)(完整 4 闸见上文)。softmax 里 `cannot fuse op0 with op2: intermediate nodes` 就是第④道闸拦下(op1 夹在中间)。
**LLVM 对照**:LLVM 判断两个循环能否融合,要做**循环依赖图遍历 + use-def 分析**,极其复杂(要证明没有依赖冲突、没有数据竞争)。Inductor 因为 IR 天然是"多维空间 + 索引代数",把这件事拆成了**几道结构化、可判定的检查** —— 不是图遍历,是闸门。
**心智模型**:**结构对得上就合**(同迭代空间 + 方向兼容 + 无中间节点),对不上就断。融合不是"把两个 for 循环拼一起",而是"两个计算若能共用同一个循环空间、且中间 buffer 不必落内存,就让它们同组"。

---

## §6 第⑤层 · Codegen → Triton

> **codegen 把每个顶层节点(单 `SchedulerNode` 或 `FusedSchedulerNode`)摊平成成员列表,在同一个 `TritonKernel` 下两趟重放 —— 1 次 call = 1 个 `@triton.jit` 核;成员间中间值经 `store_cache` 走片上 SSA、其 `tl.store` 被 `remove_kernel_local_buffers` 抹除,不落 GM。**

只有到了这一层,你才第一次看到 `for` 循环、`tl.program_id`、指针算术。前三层**根本不存在循环**。

### 真实产物:addexp 的 pointwise 核

`addexp` 生成的核([来源:artifacts/addexp/output_code.py](artifacts/addexp/output_code.py),节选 `@triton.jit` 体):

```python
@triton_heuristics.pointwise(size_hints={'x': 4096}, ...)
@triton.jit
def triton_poi_fused_add_exp_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK               # ← 循环 = program_id × block(去控制流的具体化)
    xindex  = xoffset + tl.arange(0, XBLOCK)[:]
    xmask   = xindex < xnumel
    x1 = xindex // 48        # ← 行索引(由线性地址反解;对应 IR 的 index0=p0)
    x0 = (xindex % 48)       # ← 列索引(对应 IR 的 index1=p1)
    x2 = xindex              # ← 输出地址(对应 IR 的 index2=48*p0+p1)
    tmp0 = tl.load(in_ptr0 + (x1), xmask, ...)         # A[x1]  ← 广播:A 只按行
    tmp1 = tl.load(in_ptr1 + (x0), xmask, ...)         # B[x0]  ← 广播:B 只按列
    tmp2 = tmp0 + tmp1
    tmp3 = tl_math.exp(tmp2)                           # ← NPU 分叉:tl_math,不是 tl.math
    tl.store(out_ptr0 + (x2), tmp3, xmask)
```

把第③层的 sympy 索引和这里的 Triton 一一对照:`index0=p0 → x1=xindex//48`、`index1=p1 → x0=xindex%48`、`index2=48*p0+p1 → x2=xindex`。**同一道索引重映射,在 IR 里是 sympy 公式,在核里是指针算术** —— 广播从头到尾没搬过一次内存。

### 真实产物:softmax 的融合 reduction 核

`softmax` 的 3 个节点融合成**一个**核([来源:artifacts/softmax/output_code.py](artifacts/softmax/output_code.py)):

```python
@triton_heuristics.persistent_reduction(size_hints={'x': 64, 'r0_': 64}, reduction_hint=INNER, ...)
@triton.jit
def triton_per_fused__softmax_0(in_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 64; r0_numel = 48
    xindex  = tl.program_id(0)*XBLOCK + tl.arange(0, XBLOCK)[:, None]   # 行
    r0_index = tl.arange(0, R0_BLOCK)[None, :]                          # 列 = 折叠维
    tmp0 = tl.load(in_ptr0 + (r0_1 + 48*x0), ...)                       # load 整行
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]                        # reduction #1: max 沿列
    tmp5 = tmp0 - tmp4
    tmp6 = tl_math.exp(tmp5)                                            # exp(NPU tl_math)
    tmp10 = tl.sum(tmp9, 1)[:, None]                                    # reduction #2: sum 沿列
    tmp11 = (tmp6 / tmp10)
    tl.store(out_ptr2 + (r0_1 + 48*x0), tmp11, ...)                     # 写回
```

整个 `max→sub→exp→sum→div` 在**一个核、一次 pass** 完成,`num_reduction=2`(max + sum)。中间的 `max`/`sum` 结果留在片上,不落 HBM。(这是真实 `@triton.jit` reduction 核,未 fallback 到 aclnn eager —— `sum`/`softmax` 不在 `_inductor_new/__init__.py` 的 `make_fallback` 列表里。)

### codegen 如何消费融合组:无特判,1 call = 1 核

codegen 怎么消费 `FusedSchedulerNode`?它**根本不为融合组特判**。`_codegen` 的 dispatch(`scheduler.py:4263`)对 `FusedSchedulerNode` 和 `SchedulerNode` 走**同一条** `codegen_node`:

```python
elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
    self.get_backend(device).codegen_node(node)
```

`codegen_node`(`codegen/simd.py:1306`)第一行就把节点摊平:

```python
nodes = node.get_nodes()   # simd.py:1313
```

- `SchedulerNode.get_nodes()` → `[self]`(`scheduler.py:396`,基类默认)
- `FusedSchedulerNode.get_nodes()` → `self.snodes`(`scheduler.py:1447`)

摊平后 `nodes` 就是个普通列表,后续 `generate_node_schedule` / `codegen_node_schedule` **完全看不到"这是不是融合组"** —— 热路径里没有任何 `isinstance(FusedSchedulerNode)`。所以 standalone `SchedulerNode` 就是 **N=1 的同一条路径**:`get_nodes()` 返回 `[self]` → 1 个 body 进 1 个核;`FusedSchedulerNode` 是 N=3 → 3 个 body 进同一个核。**核数 = `codegen_node` 调用次数 = 融合后的顶层节点数**;融合是 scheduler 的决定(`fuse_nodes` 改写 `self.nodes`,`scheduler.py:2063`),codegen 照单全收。

**真实插桩证据**([artifacts/codegen_trace/run.log](artifacts/codegen_trace/run.log),脚本 [codegen_trace.py](artifacts/codegen_trace/codegen_trace.py),hook 了 `codegen_node`):

```
addexp:  codegen_node 调用 1 次 = 1 个核 | 顶层 op0           [SchedulerNode]      get_nodes() → 1 成员 ['op0']
softmax: codegen_node 调用 1 次 = 1 个核 | 顶层 op0_op1_op2   [FusedSchedulerNode] get_nodes() → 3 成员 ['op0','op1','op2']
```

两者**都是 1 次 call、1 个核** —— 唯一区别是 `get_nodes()` 返回 1 个还是 3 个成员。

### 两趟组装:把 N 个 body 装进一个核

`codegen_node_schedule_with_kernel`(`simd.py:1429`)对一个核跑**两趟**(同一份 `node_schedule`):

- **Pass 1**(`:1434`):对每个成员 `decide_inplace_update()` + `_body.indexing_from_args()` **静态枚举所有索引表达式**,再 `finalize_indexing()` —— 在发任何 op 之前,先预算好整个核的 block 指针 / 参数 / mask。
- **Pass 2**(`:1452`):对每个成员调 `node.codegen(index_vars)`(`scheduler.py:1195`)→ `_body(*index_vars)` 在**同一个 `V.kernel`** 下重放:

```python
def codegen(self, index_vars):
    with V.set_ops_handler(SimplifyIndexing(...)), V.kernel.set_current_node(self):
        self._body(*index_vars)        # 重放这个成员的 LoopBody
```

每个成员的 ops 追加进**同一个核的 4 个 buffer**(`indexing_code`/`loads`/`compute`/`stores`)—— 这就是 N 个 body 真正"并"进一个核的地方。第③层在 trace 时展开闭包;第⑤层在 pass 2 重放成员 body —— 同一种"重放"思想。

### 成员间的数据接续:`store_cache`(走片上 SSA,不落 GM)

成员 body 并进同核后,成员之间的数据怎么传?关键在 **`CSEProxy`** 层(`codegen/common.py`,不在 `TritonKernel.load/store`):

- 生产者 `CSEProxy.store`(`:2471`)→ `_update_store_cache[name] = SSA寄存器`(`:2464`);
- 消费者 `CSEProxy.load`(`:2447`):**`if name in self.kernel.cse.store_cache: return store_cache[name]`** —— 命中就返回生产者的寄存器,**不 emit `tl.load`**。

真实插桩([artifacts/fuse_trace/run.log](artifacts/fuse_trace/run.log)):softmax 的 load 命中 `[buf0,buf0,buf1]`(op1/op2 读 max/sum 直接拿寄存器,不 emit `tl.load`),只有 `arg0_1` 真读 GM。

### 中间 store 怎么消失

消费者不读 GM(load 命中 store_cache),那**生产者的 `tl.store`** 呢?它在两趟**之后**被抹掉:

- `Kernel.__exit__`(`common.py:2056`)→ `remove_kernel_local_buffers`:对每个"**所有 user 都在本核内**"的 buffer 调 `can_buffer_be_removed_through_fusion`(`scheduler.py:3985`):

```python
return (all(user.is_weak or user.get_name() in fused_node_names for user in users)
        and name not in self.mutation_renames ...)
```

- 命中 → `remove_buffer`(`common.py:2099`)把 buffer 加入 `removed_buffers`。
- pass 2 里生产者的 store 本是以 `DeferredLine(name, line)`(`triton.py:2293`)形式 emit 的;渲染时 `DeferredLine.__call__`(`common.py:1259`)发现 name 在 `removed_buffers` → 返回 `None` → 那条 `tl.store` 根本不进核源码。

(注意:删除决策发生在两趟之后的 `__exit__`,而非 store 时刻 —— 所以只有 hook 到 `remove_buffer` 才能直接观测到。)

**真实插桩证据**([artifacts/codegen_trace/run.log](artifacts/codegen_trace/run.log),hook 了 `remove_buffer`):

```
softmax: remove_buffer 调用 2 次: ['buf0', 'buf1']   ← max/sum 两个中间 buffer 被删
addexp:  remove_buffer 调用 0 次: []                  ← buf0 是输出,不删
```

合起来:softmax 的 buf0/buf1 **既不被 load(走 store_cache 寄存器)、也不被 store(DeferredLine 抹除)** → 最终核只有 **1 个 `tl.store`(buf2 输出)+ 1 个 `tl.load`(arg0_1 输入)**,`num_reduction=2`。中间值从头到尾是片上 SSA 寄存器(`tmp4`/`tmp10`),**不落 GM** —— 这就是融合省 HBM 带宽的代码落地。

### inplace vs removal —— 别把两个机制混了

- `decide_inplace_update`(pass 1 调,`scheduler.py:438`)是 **inplace 内存复用** —— 把输出写进某个输入 buffer 的分配(省一次显存分配),它**不删 store**。
- 删 store 的是上面那套 `Kernel.__exit__ → remove_kernel_local_buffers`。

两者都在 pass 1 / `__exit__` 决策,但目的不同:**inplace 省显存分配,removal 省显存带宽**。

### 一个核的完整一生(生命周期)

一次 `codegen_node` call 的完整链路(`codegen/simd.py` + `codegen/triton.py`):

```
create_kernel_choices (~1357)    → 造一个空的 TritonKernel(4 个 IndentedBuffer)
with kernel: (~1430)             → 装 CSEProxy 为 ops handler,V.kernel 指向它
  pass 1 (~1434)                 → decide_inplace_update + finalize_indexing(预算索引)
  pass 2 (~1452)                 → 每成员 _body 重放,ops 进 4 buffer;store 以 DeferredLine emit
__exit__ (~2056)                 → remove_kernel_local_buffers:buf0/buf1 进 removed_buffers
codegen_kernel (triton.py:3464)  → codegen_body 拼 4 buffer(抹掉 removed 的 DeferredLine)+ 包 @triton.jit
define_kernel (~1366)            → 把核 def 写进 wrapper
call_kernel (~1388)              → 发射启动调用(grid/args)
```

**一次 `codegen_node` call = 走完这条链 = 产出一个 `@triton.jit` 核。** softmax 1 call → 1 核(装 3 body);addexp 1 call → 1 核(装 1 body)。

### reduction 与 pointwise 同核共存

softmax 核里 max/sum 两个 reduction 和 div(pointwise)共存于一核,靠:

- `generate_node_schedule`(`simd.py:1210`)在成员间插 `EnableReduction`/`DisableReduction` 哨兵;
- `codegen_node_schedule_with_kernel`(`:1436`)用 `kernel.disable_reduction()` 上下文消费 —— reduction 成员被包在 `for roffset in range(...)`(`triton.py:3217`),依赖其输出的 pointwise 成员跑在循环外;
- `num_reduction` 在 `CSEProxy.reduction`(`common.py:2495`)每次 `+1` —— 这就是核名 `triton_per_fused__softmax_0` 的 `num_reduction=2`(3 成员里 2 个 reduction)。

### 三层"延迟合并"对照(capstone)

回看全栈,三层都是**"结构先合、body 后并"**的延迟合并范式 —— 这是 Inductor 的统一设计哲学:

| | 第③层 Lowering 合并(add+exp) | 第④层 Scheduler 融合(op0+op1+op2) | 第⑤层 Codegen 并核 |
|---|---|---|---|
| 合的是什么 | 闭包嵌套(`make_loader` 返回 `inner_fn`) | 分组(`FusedSchedulerNode` 包装成员) | 成员 body 重放进同一个核 |
| 何时并 body | trace 时(展开闭包成 LoopBody) | 不并(fuse 只分组) | codegen pass 2(重放成员 body) |
| 中间值去向 | `add` 从不 realize,内联进 `exp` 闭包 | 各成员自带 body,不合并 | 中间 buffer 走 `store_cache` SSA,**不落 GM** |
| 触发判定 | 无(懒,自动) | `can_fuse` 4 道闸 | 无(每个顶层节点产一核,照单全收) |

### 机制 · LLVM 对照 · 心智模型

**机制**:`codegen/triton.py` 的 `TritonKernel.codegen_kernel`(`:3464`)把(融合组的)闭包重放给 Triton 的 `V.ops` handler,生成 `@triton.jit` 源码。sympy 索引 → `tl.program_id(0)*XBLOCK + tl.arange(...)`(循环具象化)、`tl.load/store`(指针算术)。`codegen_iteration_ranges_entry`(`:3812`)负责把多维 sympy 区间映成 `tl.arange`。
**LLVM 对照**:在 LLVM 里,**循环、寄存器分配、指令选择是编译器的核心**,贯穿全程。在 Inductor 里,**前三层根本没有循环和寄存器** —— 它们全部推迟到这最后一层才"出生"。这就是"延迟具象化"。而且——
- **NPU 只换了 codegen 一片叶子**:看核里 `tl_math.exp`(CANN libdevice)、文件头的 `from torch_npu._inductor_new.runtime.npu_triton_helpers import ... math as tl_math`。`_inductor_new` 复用了上游 100% 的 lowering / scheduler / fusion(第②–④层产物 GPU/NPU 逐字节相同),只在 codegen 叶子把 `tl.math.exp` 换成 `tl_math.exp`、做 32 字节对齐、挂 `PropagateNan`。
- **换芯片,只换这片叶子;IR 与调度设备无关。** 这正是 Inductor 这套范式相对传统编译器的核心优势。
**心智模型**:**前三层是"声明空间",循环与寄存器在最末才出生。** Inductor 是一个"先想清楚在哪个空间、按什么索引读写,最后才翻译成具体指令"的系统 —— 而不是"一开始就在写循环"的系统。

---

## §7 reduction 路径收束

`softmax` 把三层串成一条线,reduction 的专属心智模型是:**reduction = "多一个会折叠的维度",不是嵌套 for 累加。**

- **第③层**:reduction 在 IR 里是"多一个会折叠的维度"(`op0` 的 `sizes=([64],[48])` —— 64 个外层点,每个把 48 维用 `ops.reduction('max', ...)` 折成 1,`store_reduction` 写折叠后的标量)。
- **第④层**:reduction 与 pointwise 可融合 —— 3 个节点(op0 max、op1 sum、op2 div)合成一个 `FusedSchedulerNode`(op0+op1 先合同 group 两个 reduction,op0_op1 再吸收 op2)。
- **第⑤层**:一个核里两个 reduction(`persistent_reduction`,`num_reduction=2`),折叠中间值留片上、不落 HBM。

折叠维度的算子之间只要对齐就能融合,折叠中间值留在片上。

---

## §8 附录

### 产物索引

| 产物 | 路径 | 层 |
|---|---|---|
| Dynamo FX graph | [artifacts/addexp/fx_graph_readable.py](artifacts/addexp/fx_graph_readable.py)、[artifacts/softmax/fx_graph_readable.py](artifacts/softmax/fx_graph_readable.py) | ① |
| FX 可运行/改写脚本 | `artifacts/<case>/fx_graph_runnable.py`、`fx_graph_transformed.py`(在 torchinductor 调试目录内) | ① |
| 分解 provenance | `artifacts/softmax/output_code.py` 头部注释 | ② |
| 融合前 IR | [artifacts/addexp/ir_pre_fusion.txt](artifacts/addexp/ir_pre_fusion.txt)、[artifacts/softmax/ir_pre_fusion.txt](artifacts/softmax/ir_pre_fusion.txt) | ③ |
| 符号闭包 inner_fn | [artifacts/addexp/inner_fn.txt](artifacts/addexp/inner_fn.txt)、[artifacts/softmax/inner_fn.txt](artifacts/softmax/inner_fn.txt) | ③ |
| 融合后 IR | [artifacts/addexp/ir_post_fusion.txt](artifacts/addexp/ir_post_fusion.txt)、[artifacts/softmax/ir_post_fusion.txt](artifacts/softmax/ir_post_fusion.txt) | ④ |
| 融合决策日志 | [artifacts/torch_logs.stderr.log](artifacts/torch_logs.stderr.log)(`__fusion` 段) | ④ |
| 节点→核调度 | [artifacts/torch_logs.stderr.log](artifacts/torch_logs.stderr.log)(`__schedule` 段) | ④ |
| 逐 pass 变化 | [artifacts/passes/](artifacts/passes/) | ④ |
| Triton 核 | [artifacts/addexp/output_code.py](artifacts/addexp/output_code.py)、[artifacts/softmax/output_code.py](artifacts/softmax/output_code.py) | ⑤ |
| 闭包合并 / realize / can_fuse 插桩 | [artifacts/merge_trace/run.log](artifacts/merge_trace/run.log)(脚本 [merge_trace.py](artifacts/merge_trace/merge_trace.py)) | ③④ |
| fuse 动作 / store_cache 插桩 | [artifacts/fuse_trace/run.log](artifacts/fuse_trace/run.log)(脚本 [fuse_trace.py](artifacts/fuse_trace/fuse_trace.py)) | ④⑤ |
| codegen 消费融合组插桩 | [artifacts/codegen_trace/run.log](artifacts/codegen_trace/run.log)(脚本 [codegen_trace.py](artifacts/codegen_trace/codegen_trace.py)) | ⑤ |

### 怎么改例子 / 自己跑

1. 改 [repro.py](repro.py) 里的 `f_addexp` / `f_softmax`(或加新函数),在 `run(...)` 里调用。
2. `source env.sh new && python repro.py 2> artifacts/torch_logs.stderr.log`
3. 产物落在 `artifacts/torch_compile_debug/run_<ts>/torchinductor/<graph>.<N>/`(env 产物)与 `artifacts/addexp`、`artifacts/softmax`、`artifacts/passes/`(repro 自动归拢);Triton 核在 `$TORCHINDUCTOR_CACHE_DIR`(`log/single_test_log/debug/`)。
4. 想看单个 IR 节点的清洁符号体:对任何 `Loops` 实例调 `data.inner_fn_str()`(`ir.py:860`);想看物化形态:对 `SchedulerNode` 调 `snode._body.debug_str()`(`loop_body.py:324`)。

### 已验证

- ✅ 环境:`source env.sh new` → `TORCHINDUCTOR_NPU_BACKEND=new`,`TORCH_COMPILE_DEBUG=1`、`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` 由 env.sh 设定。
- ✅ 数值:`addexp`、`softmax` compiled vs eager 全部 `allclose=True`。
- ✅ reduction 真走 Triton:softmax 生成 `@triton.jit persistent_reduction` 核(`num_reduction=2`),未 fallback。
- ✅ 闭包合并(第③层):插桩证实 addexp 仅 1 个 buffer、`add` 从未 realize、`buf0` 闭包内联 add+exp(`origins=[add,exp]`)。
- ✅ `can_fuse` 判定(第④层):插桩抓到 softmax 的 `ACCEPT op0+op1` / `REJECT op0+op2 (intermediate nodes)` / `ACCEPT op0_op1+op2` 序列。
- ✅ fuse 动作(第④层):插桩证实 softmax `init_group_node` 2 次、`FusedSchedulerNode` 无 `_body`(只分组)。
- ✅ codegen 消费(第⑤层):插桩证实 addexp/softmax **都是 1 次 `codegen_node` call = 1 个核**(无特判),`get_nodes()` 返回 1 vs 3 成员;softmax `remove_buffer`=`[buf0,buf1]`(中间 store 在 `Kernel.__exit__` 被删)、load 命中 store_cache `[buf0,buf0,buf1]`(片上 SSA 不落 GM)。
- ✅ provenance:文中每段产物都能在 [artifacts/](artifacts/) 找到原文逐字对应。

### 不在范围

- `new` vs `npu_inductor` 对照(前 4 层产物相同,冗余)。
- vanilla CUDA Triton(本机无 CUDA,跑不出)。
- 反向图 / autograd(正向已足建立心智模型;可作下一步)。
- SVG 可视化(需 graphviz,本机未装)。

---

> **一句话总结**:Inductor 不是"一个会优化的编译器",而是"一个先用代数把张量计算描述清楚、再在最后一刻翻译成指令的系统"。维度即空间,索引即代数,循环与寄存器在最末才出生,换芯片只换 codegen 一片叶子 —— 这就是它和 LLVM 范式的根本区别。
