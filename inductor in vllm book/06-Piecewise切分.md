---
chapter: 6
part: III
title: Piecewise 切分:为什么 attention 是图的边界
status: done
difficulty: ★★★
depends: [2, 3, 4, 5]
---

# 第 6 章 Piecewise 切分:为什么 attention 是图的边界

```text
I   背景:   [1 为何compile] → [2 V1引擎] → [3 编译栈全景]
II  前端:   [4 配置体系] → [5 Dynamo一次捕获]
III 中端:   [6 Piecewise切分]◄== → [7 算子边界与vllm_ir]
IV  后端:   [8 VllmBackend] → [9 逐range编译] → [10 Pass与融合]
V   运行时: [11 代码生成] → [12 CUDA Graph] → [13 缓存与AOT]
VI  收官:   [14 设计哲学] → [15 动手实践]
```

## 6.0 本章俯瞰

第 5 章结束时,流水线手里有了一张图:dynamo 只 trace 一次、丢掉全部 guards,把整个模型压成单张 FX 图。一个很自然的想法随之而来:把这张图整个交给 Inductor 不就行了吗,为什么 vLLM 反其道而行,默认在每一层 decoder 的 attention 处各切一刀,让几十层的模型散成上百段碎片?何况每切一刀,就多出一批子图的输入输出要管理、多一份编译产物要缓存,看起来全是成本。答案预告:attention 同时戴着三顶帽子——它是 Inductor 根本写不出来的手写 kernel、它是每一步内容都在变的动态信息的入口、它还是把几十个同构层隔成可复用单元的天然隔板;这一刀落下之后,"交给编译器的基本单位"与"交给 CUDA Graph 录制的基本单位"恰好对齐到同一位置,后文的逐段编译、逐段捕获、缓存去重才有各自的着力点。本章就把这一刀讲透:在哪里切、按什么规则切、切完的碎片如何在不改动整张图结构的前提下,被逐个偷换成编译产物。

## 6.1 最小前置知识

- **FX GraphModule**(FX 图模块)与节点线性序:dynamo 捕获的产物,节点按拓扑序线性排列在 `graph.nodes` 里,每个 `call_function` 节点是一次算子调用(详见第 3、5 章)。本章的切分,本质上就是对这条线性序列做一次扫描编号。
- **splitting_ops**(切分算子表):`CompilationConfig` 的字段,一组 `namespace::name` 格式的算子名字符串;None 默认展开为 13 个注意力算子加 2 个 KV cache 写入算子,`[]` 表示整图不切——三档语义详见第 4 章。
- **fake tensor**(假张量)与 **example_value**(示例值):只携带 shape/stride/dtype 元数据、不占真实数据的张量;dynamo 会给每个 FX 节点在 `node.meta["example_value"]` 里存好它,是编译期的通货(详见第 3 章)。
- **AOTAutograd 与子模块边界**:每个切出的可编译片段都会各自走一遍 AOTAutograd 与 Inductor(详见第 9 章)。片段之间传递的值必须是这条流水线能接受的形态——张量与 SymInt 可以,`torch.Size` 元组不行。
- **PIECEWISE cudagraph**(分片 CUDA Graph):每个可编译片段单独捕获一张 CUDA Graph、attention 段以源码执行的运行时模式(详见第 4 章;捕获与重放细节详见第 12 章)。

## 6.2 设计思想与设计哲学

**理由一:attention 是 Inductor 写不出的算子。** 推理引擎里的 attention 由 FlashAttention、FlashInfer 等项目的手写 kernel 实现,它们对 KV cache 的布局、变长序列的组织、tensor core 的占用各有一套精巧设计。Inductor 的代码生成面向 pointwise、reduction、线性代数拼出来的规则图,对 attention 这类算子既无法分解也无从生成——若强行留在编译区内,Inductor 只能把它当原子节点供着,还要为它引入的变长语义搅乱整个 shape 系统。把 attention 连同它周围的动态信息挡在编译区外、以源码方式执行,是对编译器能力边界的诚实承认。算子侧如何配合这道墙,详见第 7 章。

**理由二:cudagraph 需要可控的捕获边界。** capture 的本质是把一段 GPU 工作录死、之后原样重放;而 attention 依赖 **attn_metadata**(注意力元数据)——`query_start_loc`、`seq_lens` 这些每步都变的变长信息。它们不是模型 forward 的入参、从不进图:每步前向之前,由 `set_forward_context` 上下文管理器把 **ForwardContext**(前向上下文)写进模块级全局状态(`vllm/forward_context.py`),attention 的实现在运行时经 `get_forward_context()` 读取。若把整图连同 attention 一起捕获,这些动态信息就被录死在图里,重放读到的永远是录制那一刻的旧值。切成片段之后,attention 段留在捕获之外、以源码执行,两侧的可编译片段各自成为干净的捕获单元——编译单元与捕获单元在同一刀上对齐。

**理由三:不切则图太大,编译时间与缓存粒度都不可接受。** 一个几十层的模型整图有数千节点,任何一个配置变化都让整份缓存失效,编译时间也随时可能膨胀到分钟级。切开后,同构层的同构片段是完全相同的子图,共享同一份 Inductor 编译缓存——Llama 全模型切完实际只有 3 类唯一子图(见 6.3.5)。编译时间与缓存体积从"随层数线性膨胀"降到"按片段种类数计",这是命题 P1(compile once, run forever)能负担得起的前置条件。

【哲学时刻】**切分即边界设计(P3/P1)。** 命题 P3 说 custom op 既是墙也是桥;切分就是把这枚硬币拍在桌上的那个动作。attention 以不透明算子的形态留在 **splitting piece**(切分点片段,不进 Inductor、以源码内联执行),墙内的复杂性对编译器关闭;墙外的可编译区整段交给 Inductor,让融合、调优、代码生成放开手脚。同时,片段化让缓存键从"整图一份"细化到"子图一份",同构片段全部命中同一份产物——切分既是编译的边界,也是缓存的边界。回扣全书叙事主线:decode 是 CPU/launch bound → 必须消灭 Python 开销 → 代价是编译 → 编译代价是启动时间 → 所以全部编译前置并缓存 → 所以敢丢 guards、敢一次 trace、敢去泛函化。切分站在这条链的中间环节:一次 trace 换来的整张图,经这一刀拆成可缓存、可去重、可与捕获对齐的单元;没有这一刀,第 9 章的逐 range 编译、第 12 章的分片捕获都没有作用对象。

## 6.3 具体流程(流水线视角)

### 6.3.1 一刀的输入与输出

> **上游给我**:VllmBackend.__call__ 六步流水线第③步递来的整图 `graph`(单张 FX GraphModule,唯一符号维 num_tokens,来源 `vllm/compilation/backends.py:1179`)→ **我做**:`split_graph` 沿节点线性序扫描,在 `should_split` 命中处开刀 → **产出**:`split_gm`(带 `submod_0..N` 子模块的 GraphModule)与 `piecewise_graphs`(SplitItem 清单),去向 `backends.py:1222` 的 **PiecewiseCompileInterpreter**(逐片段编译解释器)。

**SplitItem**(切分条目,dataclass,`backends.py:406`)只有四个字段:`submod_name`、`graph_id`、`is_splitting_graph`、`graph`。`is_splitting_graph` 为真的条目就是 attention 所在的 splitting piece,不进 Inductor;其余条目是等待编译的可编译 **piece**(片段),切分实现上挂在 `split_gm` 的 submodule 属性上。另有一条旁路:配置 `use_inductor_graph_partition=True` 时,`backends.py:1173-1177` 把切分表置空、跳过 FX 级预切分,分区改由 Inductor 的 scheduler 在 codegen 期完成(`vllm/compilation/partition_rules.py:41` 的 `inductor_partition_rule_context` 把同一份名单注册进 `torch._inductor.config.custom_should_partition_ops`)——同一刀的两种落法,命题 P6 的一个注脚。本章以下讲默认的 FX 级切分。

```mermaid
flowchart LR
    classDef vllm fill:#DBEAFE,stroke:#3B82F6,color:#1E3A8A
    classDef torch fill:#FEF3C7,stroke:#F59E0B,color:#78350F
    classDef artifact fill:#D1FAE5,stroke:#10B981,color:#064E3B
    classDef ext fill:#E5E7EB,stroke:#6B7280,color:#374151
    classDef danger fill:#FEE2E2,stroke:#EF4444,color:#7F1D1D

    BEFORE([切分前 整图 graph<br/>两层 decoder 同在一张 FX 图<br/>含 2 个 attention 节点]):::torch ==> SG[split_graph 线性扫描<br/>should_split 命中处开刀]:::vllm
    SG ==> P0[submod_0 可编译 piece<br/>rms qkv rope<br/>产物 逐 range 编成 Inductor artifact]:::torch
    P0 ==> A1[submod_1 splitting piece<br/>attention 0 与 KV 写入<br/>产物 源码内联 eager 执行]:::vllm
    A1 ==> P2[submod_2 可编译 piece<br/>o_proj add rms MLP add<br/>产物 Inductor artifact]:::torch
    P2 ==> A3[submod_3 splitting piece<br/>attention 1]:::vllm
    A3 ==> P4[submod_4 可编译 piece<br/>末层残差与输出<br/>产物 Inductor artifact]:::torch
    P4 ==> OUT[split_gm 与 piecewise_graphs<br/>SplitItem 清单 标记 is_splitting_graph]:::artifact
    OUT ==> NEXT[PiecewiseCompileInterpreter<br/>把可编译 piece 偷换成编译产物]:::vllm
    NEXT -.-> SERVE([serve 期 查表调用 各 piece]):::vllm
```

### 6.3.2 should_split:一张字符串名单的匹配语义

> **上游给我**:FX 节点(来自 `split_graph` 的扫描循环)与定稿的 splitting_ops 字符串表(来自 `vllm/config/compilation.py:517` 字段,经 `set_splitting_ops_for_v1`(`vllm/config/compilation.py:1133`)三档归一,详见第 4 章)→ **我做**:`should_split` 判断该节点是否命中名单 → **产出**:一个布尔值,驱动 `split_graph` 的分区编号(`backends.py:578`)。

`should_split`(`vllm/compilation/partition_rules.py:14`)只认 `call_function` 节点,且只对 torch 的两种算子对象做匹配:

```python
# vllm/compilation/partition_rules.py:21-38 (v0.27.1)
    if node.op != "call_function":
        return False
    target = node.target
    if isinstance(target, torch._ops.OpOverloadPacket):   # 如 aten::add
        return target._qualified_op_name in splitting_ops
    if isinstance(target, torch._ops.OpOverload):         # 如 aten::add.default
        packet_name = target.name()
        op_overload_name = f"{packet_name}.{target._overloadname}"
        return op_overload_name in splitting_ops or packet_name in splitting_ops
    return False
```

也就是说,配置里写的 `vllm::unified_attention_with_output` 这类字符串,匹配的是 FX 节点 target 的**算子全名**——无论是未选 overload 的 OpOverloadPacket(取 `_qualified_op_name`)还是已选 overload 的 OpOverload(取 `name()`,并额外容忍带 `.overload` 后缀的写法)。`call_method`、`get_attr` 等其他节点形态一概不匹配;普通 Python 函数(例如 `operator.getitem`)也不是 OpOverload,同样落空——这个细节马上会在 6.3.3 的第③步被利用。

### 6.3.3 split_graph 六步算法

> **上游给我**:整图 `graph` 与切分表(同上)→ **我做**:先解引用 size,再线性扫描编号、合并异常分区,最后调用 torch 的 `split_module` 真正落刀 → **产出**:`(split_gm, piecewise_graphs)`(`backends.py:627` 返回)。

**① `_decompose_size_nodes`:先拆 `torch.Size`。** dynamo 把 `x.size()`(无参调用)捕获为 `call_method` target="size" 的节点,它返回一个 `torch.Size` 元组。源码注释说得直白(`backends.py:482-483`):"torch.Size objects cannot cross split boundaries because aot_autograd cannot handle them as submodule outputs"——每个片段要各自过 AOTAutograd,元组无法作为子模块输出跨边界。于是 `_decompose_size_nodes`(`backends.py:479`)借 `tensor_node.meta["example_value"]` 逐维检查:动态维(SymInt)改写为新的 `aten.sym_size.int` 节点,静态维(普通 int)直接内联为字面量;下游若是对 size 取下标(`operator.getitem`),直接换成对应那一维的结果,若是整元组消费(如 `view(x, size)`),则把参数展开成逐维传入(`:525-549`)。

**② 线性扫描编号。** 沿 `graph.graph.nodes` 顺序前进,`placeholder` 与 `output` 跳过;每个节点记入 `node_to_subgraph_id`;`should_split` 命中时 `subgraph_id` 先加一并把该 id 记入 `split_op_graphs`(切分点分区表),随后窥一眼 `node.next`:后继也命中则把 `subgraph_id` 减回来——后继节点自己的"加一"会把它送回同一个 id,**连续 splitting op 合并同区**;否则再加一,开启新区容纳后继节点。

```python
# vllm/compilation/backends.py:578-591 (v0.27.1)
        if should_split(node, splitting_ops):
            subgraph_id += 1
            node_to_subgraph_id[node] = subgraph_id
            split_op_graphs.append(subgraph_id)

            # keep consecutive splitting ops together
            # (we know node.next exists because node isn't the last (output) node)
            if should_split(node.next, splitting_ops):
                # this will get incremented by the next node
                subgraph_id -= 1
            else:
                subgraph_id += 1
        else:
            node_to_subgraph_id[node] = subgraph_id
```

为什么必须合并:KV cache 写入(`unified_kv_cache_update`)默认也在切分表里(第 4 章),它通常紧贴在 attention 调用之前;若两个切分算子各占一区,它们之间一个节点都没有,会凭空产生一个空 piece、空子模块、乃至一次空的 cudagraph 捕获。

**③ `operator.getitem` 跟随其输入所在区。** 扫描循环里,`call_function` 且 target 为 `operator.getitem` 的节点被直接赋成其输入(元组生产者)的分区 id(`backends.py:570-576`)。否则,若某个多值算子的 tuple 输出跨界,整只元组都得作为子模块输入传出——这与 standalone_compile 与 AOTAutograd 的输入要求相悖(注释见 `:566-569`);让取下标动作跟随生产者,跨边界流动的就只剩单个张量。

**④ `_merge_empty_only_subgraphs`:折掉"只含空分配"的分区。** 有些分区里唯一的节点是一次 `torch.empty`/`new_empty`(判定见 `_is_empty_allocation_node`,`backends.py:414`)。`_merge_empty_only_subgraphs`(`backends.py:437`)把这种分区折入前面最近的非切分分区,docstring 写明动机:"avoid generating standalone empty submodules, which can lead to empty cudagraph captures"。折入有安全检查(`:465-473`):该分配节点的每个输入,要么是图的 placeholder,要么产自分区号不大于目标分区的节点——即**移动之后,分配不会越过它任何输入的生产者**;不满足就放弃合并。

**⑤ 真正落刀:`torch.fx.passes.split_module.split_module`(FX 切分工具)。** 上游的通用工具,分区函数就是查表 `lambda node: node_to_subgraph_id[node]`(`backends.py:602-608`)。两个承重参数:`keep_original_order=True`——源码注释(`:595-598`)警告否则 pytorch 可能重排节点、"the semantics of the graph will change when we have mutations in the graph",而 vLLM 的图必然含 mutation(attention 写 output、KV 写 cache);`tuple_return=True`(torch≥2.12 才传入,`:600-601`)让子模块以元组返回多值。外层再包 `_use_lazy_graph_module(True)`(`:599`)纯为提速。分区 id 只是中间记号,`split_module` 按实际产出的子模块重编为 `submod_0..N`。

**⑥ 收集排序。** 遍历 `split_gm.named_modules()`,跳过根模块与嵌套子模块,`graph_id` 取自 `int(name.replace("submod_", ""))`,`is_splitting_graph` 判断该 id 是否在 `split_op_graphs` 里,构造 SplitItem;最后**按整数排序**而非字符串排序(`backends.py:610-625`,否则 `submod_10` 会排在 `submod_2` 之前)。

### 6.3.4 PiecewiseCompileInterpreter:图还是那张图,节点已被偷换

> **上游给我**:`split_gm`(带 submod_N 子模块,`backends.py:1179`)、`submod_names_to_compile`(只含非 splitting 子图名的清单,`backends.py:1202-1206`)与 `fake_args`(把每个真实输入张量替换为对应 placeholder 的 `example_value`,纯 fake 世界零真实计算,`backends.py:1208-1216`)→ **我做**:以 fx.Interpreter 逐节点跑一遍 fake 前向,在每个待编译子模块处构造 PiecewiseBackend 并全量预编译 → **产出**:编译产物替换 `split_gm` 的 submodule 属性(`backends.py:766`),供第⑤步 generate_execution_code 直接引用。

为什么用 **PiecewiseCompileInterpreter**(`backends.py:687`,改造自 `torch.fx.passes.shape_prop.ShapeProp`)而不是静态遍历?因为每个片段被编译时必须知道自己的输入实参是什么——哪些位置是张量、哪些位置是 SymInt、顺序如何。这只能靠让 fake 值沿图**真实流动**才能在每个 `call_module` 处拿到;静态遍历拿不到"运行到这一步时实参长什么样"。`call_module`(`backends.py:730`)并不真正执行子模块,输出直接取自输出节点的 `node.meta["example_value"]`(`:739-740`):

```python
# vllm/compilation/backends.py:742-772 (v0.27.1,节选)
        if target in self.compile_submod_names:
            index = self.compile_submod_names.index(target)
            submod = self.fetch_attr(target)
            sym_shape_indices = [
                i for i, x in enumerate(args) if isinstance(x, torch.SymInt)
            ]
            piecewise_backend = PiecewiseBackend(
                submod, self.vllm_config, index,
                len(self.compile_submod_names), sym_shape_indices,
                self.vllm_backend, graph_returns_tuple(submod),
                submod_name=target)
            # __dict__ 优先于 _modules:getattr 从此拿到编译产物
            self.module.__dict__[target] = wrap_with_cudagraph_if_needed(
                piecewise_backend, self.vllm_config, self.compilation_config,
                piecewise_backend.is_first_graph, piecewise_backend.is_last_graph)
```

三处要害。其一,`sym_shape_indices`(`:746-748`)记下该片段输入里所有 SymInt 的位置——正是 num_tokens 这个唯一符号维进入每个片段的口子,交给 **PiecewiseBackend**(片段后端,`vllm/compilation/piecewise_backend.py:86`)做逐 range 编译(第 9 章展开);其构造函数把所有 range 一次编完,这是「启动期编完一切」这条全书不变量在片段层的执行。其二,`index` 与总数决定 `is_first_graph`/`is_last_graph`(`piecewise_backend.py:131-132`),docstring(`backends.py:693-696`)说明清单顺序承重:首图负责日志,末图有特殊的 cudagraph 输出处理。其三,也是本节标题的那句话:`self.module.__dict__[target] = ...`(`:766`)——FX 的属性查找先查 `__dict__` 再回退 `_modules`(源码注释 `backends.py:1280-1282`),于是在不改动 `split_gm` 任何节点与调用顺序的前提下,`getattr(split_gm, "submod_2")` 拿到的已经是包装好的编译产物。后续的序列化也有对应安排:`VLLM_USE_MEGA_AOT_ARTIFACT` 开启时,`backends.py:1181-1185` 在解释器动手之前 deepcopy 一份"还没被偷换"的 split_gm 专供落盘。

构造完成后顺手过一道 **`wrap_with_cudagraph_if_needed`**(`backends.py:633-684`):没有 piecewise cudagraph、或走 Inductor 分区旁路时原样返回(`:655-659`);否则用 `current_platform.get_static_graph_wrapper_cls()` 解析平台相关的静态图包装类,以 `runtime_mode=PIECEWISE`、`debug_log_enable=首图`、`gc_disable=非首图`、`weak_ref_output=末图` 的选项包上(`:675-683`)——可编译片段从此同时是"编译单元"与"捕获单元",编译期捕获选项在此时此刻一次性定死。

### 6.3.5 手推示例:2 层 LlamaDecoderLayer

取每层的主干节点序(示意图,非真实 dump;残差路径简化为显式张量传递,lm_head 本就在编译图之外,详见第 2 章):`rms → qkv → rope → unified_attention_with_output → o_proj → add → rms → gate_up → silu_and_mul → down → add`。两层共 22 个节点,按 6.3.3 的规则推编号(★ 表示 should_split 命中):

```text
【示意图 · 非真实 dump】2 层 LlamaDecoderLayer,★ = should_split 命中
节点序(按图内线性顺序)                       id   piece
--------------------------------------------------------------------
L0 rms qkv rope                              0    submod_0  可编译
L0 ★unified_attention_with_output            1    submod_1  splitting
L0 o_proj add rms gate_up silu down add      2    submod_2  可编译
L1 rms qkv rope                              2      ‖ 与 post_0 同区合并
L1 ★unified_attention_with_output            3    submod_3  splitting
L1 o_proj add rms gate_up silu down add      4    submod_4  可编译

计数器演化:  起始 id=0
  命中 attn_0 → id=1 且记入 split_op_graphs,后继非切分 → id=2
  (L1 的 rms qkv rope 仍落 id=2,与 post_0 挤进同一分区)
  命中 attn_1 → id=3 且记入 split_op_graphs,后继非切分 → id=4
若 KV 写入紧邻 attention → 第二次命中不抬 id,两者同区(:585-587)
```

注意 `submod_2`:第一层的后处理与第二层的前处理之间没有任何切分算子,扫描器不会在那里抬 id,它们天然落进同一个分区——这正是"相邻 attention 之间的区域各成 piece"的机制来源。每个 piece 的输入输出张量清单(全部以 num_tokens 为第一维;输入里 SymInt 实参出现的位置,则由 interpreter 记入 sym_shape_indices):

| piece | is_splitting_graph | 输入 | 输出 |
|---|---|---|---|
| submod_0 | 否 | hidden_states | q_rope、k_rope、v |
| submod_1 | 是 | q_rope、k_rope、v | attn_output(attn_metadata 运行时从 ForwardContext 取) |
| submod_2 | 否 | attn_output、residual | q1_rope、k1_rope、v1 |
| submod_3 | 是 | q1_rope、k1_rope、v1 | attn_output_1 |
| submod_4 | 否 | attn_output_1、residual_1 | hidden_states_out |

推广到 N 层:可编译片段只有三类——首层 attention 之前的 pre-attention(`submod_0`)、相邻 attention 之间的 per-layer(出现 N−1 次的 `submod_2`)、最后 attention 之后的 post-attention(`submod_4`);attention 本身各成 splitting piece。这与官方文档口径一致:Llama 全模型切分后只有 3 类唯一子图,同构片段共享同一份 Inductor 编译缓存(官方文档口径,机制已与源码核对一致)。附带一个缓存动机的细节:KV cache 写入算子进切分表,除了捕获边界,还因为它携带字符串参数(层名)会阻碍 Inductor 复用分片图(`vllm/config/compilation.py:1157-1161` 的注释)。

## 6.4 本章小结

- 切分三理由:手写 kernel、动态元数据、缓存粒度。(P3、P1)
- 匹配语义只认 call_function 的算子全名字符串。(P5)
- 连续切分算子合并同区,空分配分区向前折。(P2)
- interpreter 靠 fake 值真实流动重建各片输入输出。(P2)
- 偷换 __dict__ 属性:图不动,子模块已是编译产物。(P1)

本机制集中服务于命题 P3(custom op 既是墙也是桥)与 P1(compile once, run forever):切分把"编译器不必懂的"与"编译器该使劲的"分到两侧,又把缓存键细化到子图粒度让同构片段全数去重。下一章(第 7 章)放大切分点本身:attention/vllm_ir 这些算子被设计成什么样,才能既当墙又当桥。本章反复把它们当"切分的锚"来用,下一章将看到锚自身的六个设计决定与一套激活捐赠机制。

### 本章源码索引

| 锚点 | 内容 |
|---|---|
| `vllm/compilation/backends.py:406`(`SplitItem`) | 切分条目:submod_name / graph_id / is_splitting_graph / graph |
| `vllm/compilation/backends.py:414`(`_is_empty_allocation_node`) | 判定"只含一次空分配"的节点(torch.empty / new_empty 系) |
| `vllm/compilation/backends.py:437`(`_merge_empty_only_subgraphs`) | 空分配分区折入前区;安全检查 :465-473 |
| `vllm/compilation/backends.py:479`(`_decompose_size_nodes`) | torch.Size 逐维解引用为 sym_size.int 节点/字面量 |
| `vllm/compilation/backends.py:553`(`split_graph`) | 切分入口;返回 (split_gm, piecewise_graphs) :627 |
| `vllm/compilation/backends.py:562-576` | 线性扫描;operator.getitem 跟随其输入所在区 :570-576 |
| `vllm/compilation/backends.py:578-591` | should_split 命中抬 id;连续切分算子合并(:585-587) |
| `vllm/compilation/backends.py:599-608`(`split_module` 调用) | keep_original_order=True 承重;tuple_return;lazy graph module |
| `vllm/compilation/backends.py:610-625` | 收集 submod_N,按整数 graph_id 排序 |
| `vllm/compilation/backends.py:633`(`wrap_with_cudagraph_if_needed`) | 平台静态图包装类,PIECEWISE 模式与首末图选项 :675-683 |
| `vllm/compilation/backends.py:687`(`PiecewiseCompileInterpreter`) | 改造自 ShapeProp;清单顺序承重 :693-696 |
| `vllm/compilation/backends.py:730`(`call_module`) | fake 输出 :739-740;sym_shape_indices :746-748;偷换 :766 |
| `vllm/compilation/backends.py:1173-1179` | use_inductor_graph_partition 分流;split_graph 调用点 |
| `vllm/compilation/backends.py:1202-1216` | submod_names_to_compile 只含非 splitting;fake_args 构造 |
| `vllm/compilation/backends.py:1280-1286` | "__dict__ 优先于 _modules"注释(generate_execution_code 取产物) |
| `vllm/compilation/partition_rules.py:14`(`should_split`) | 只认 call_function;OpOverloadPacket/OpOverload 两种全名匹配 |
| `vllm/compilation/partition_rules.py:41`(`inductor_partition_rule_context`) | Inductor 分区旁路:注册 custom_should_partition_ops |
| `vllm/config/compilation.py:517`(`splitting_ops` 字段) | 两类用途的 docstring(FX 切分 / Inductor 分区规则) |
| `vllm/config/compilation.py:1133`(`set_splitting_ops_for_v1`) | 三档语义与连带降级;KV 写入入表的缓存动机 :1157-1161 |
| `vllm/compilation/piecewise_backend.py:86`(`PiecewiseBackend`) | 片段后端;is_first/is_last_graph :131-132 |
| `vllm/forward_context.py`(`set_forward_context` / `get_forward_context`) | 模块级全局 + 上下文管理器:attn_metadata 的运行时通道 |

### 参考资料

1. [vLLM design doc — torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/) — 已与 0.27.1 源码核对一致(Llama 3 类唯一子图口径)
2. [vLLM design doc — Optimization Levels](https://docs.vllm.ai/en/latest/design/optimization_levels/) — 已与 0.27.1 源码核对一致(splitting_ops 三档,详见第 4 章)
3. [vLLM design doc — CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) — 已与 0.27.1 源码核对一致(PIECEWISE 捕获边界,详见第 12 章)
4. [vLLM blog — Introduction to torch.compile and How It Works with vLLM (2025-08-20)](https://vllm.ai/blog/2025-08-20-torch-compile) — 仅作背景叙事
5. [vLLM design doc — fusions](https://docs.vllm.ai/en/latest/design/fusions/) — 仅作背景叙事(融合需要可编译区,详见第 10 章)
6. [vLLM design doc — Debug vLLM Compile](https://docs.vllm.ai/en/latest/design/debug_vllm_compile/) — 仅作背景叙事(computation_graph.py 即切分后整图转储,`backends.py:1259`)
7. [RFC issue #6378 — A Graph Optimization System in vLLM using torch.compile](https://github.com/vllm-project/vllm/issues/6378) — 仅作背景叙事
8. [PyTorch blog — PyTorch + vLLM](https://pytorch.org/blog/pytorch-vllm-%E2%99%A5%EF%B8%8F/) — 仅作背景叙事
9. [vLLM v0.8.1 设计文档 — torch_compile(历史版)](https://docs.vllm.ai/en/v0.8.1/design/v1/torch_compile.html) — 仅作背景叙事(演进对比:piecewise 切分思想的最早公开表述)
