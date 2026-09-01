---
part: 附录
title: 附录 A 术语表中英对照
status: done
difficulty: ★
---

# 附录 A 术语表中英对照

> 本表是全书的术语契约:正文首次出现某个术语时加粗并给中文释义,此后一律使用英文原文。
> "首见"列标注该术语在正文中的首次定义章节。按主题分组,便于按需查阅。

## A.1 引擎与调度(vLLM V1)

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `AsyncLLM` | 异步推理前端 | Ch2 | API server 侧入口,接收请求并经 ZMQ 转发给 EngineCore |
| `EngineCore` | 引擎核心进程 | Ch2 | 独立进程,busy loop 驱动 调度→执行→更新 循环 |
| `Scheduler` | 调度器 | Ch2 | 每步为每个请求分配 token 预算,产出 `SchedulerOutput` |
| `SchedulerOutput` | 调度器输出 | Ch2 | 含 `num_scheduled_tokens`(req→数量)与 `total_num_scheduled_tokens` |
| continuous batching | 连续批处理 | Ch2 | 无固定迭代边界,请求随时加入/退出同一个 batch |
| chunked prefill | 分块预填充 | Ch2 | 长 prompt 切成多步调度,与 decode 混跑 |
| ragged batch | 参差批 | Ch2 | 各请求 token 数不同,经 `np.repeat`/cumsum 展平为一维 |
| `GPUModelRunner` | GPU 模型执行器 | Ch2 | 构造输入、调 compiled model、采样;执行侧总管 |
| `InputBatch` | 持久输入批状态 | Ch2 | 常驻结构,增删改请求而非重建,零分配哲学 |
| `ForwardContext` | 前向上下文 | Ch2 | 模块级全局(`vllm/forward_context.py`)贯穿的运行时元数据(attn_metadata、cudagraph 模式等) |
| `attn_metadata` | 注意力元数据 | Ch2 | 每层 attention 所需的变长信息(query_start_loc、seq_lens 等) |
| `logits_indices` | logits 下标 | Ch2 | 每请求最后一个 token 的位置;lm_head 在编译图之外 |

## A.2 编译模式与配置

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `OptimizationLevel` | 优化级别 | Ch4 | CLI `-O0..-O3`;O2 为默认,决定 fusion 与 cudagraph 组合 |
| `CompilationMode` | 编译模式 | Ch4 | NONE / STOCK_TORCH_COMPILE / DYNAMO_TRACE_ONCE / VLLM_COMPILE |
| `VLLM_COMPILE` | vLLM 自定义编译 | Ch4 | mode=3:走 VllmBackend 的 piecewise + Inductor 流水线 |
| `enforce_eager` | 强制 eager | Ch4 | 等价 mode=NONE + cudagraph_mode=NONE,关闭一切编译 |
| `CompilationConfig` | 编译配置 | Ch4 | 编译相关全部字段的容器,`vllm/config/compilation.py` |
| `-cc` / `--compilation-config` | 编译配置参数 | Ch4 | CLI 上以 JSON 或点路径覆写 CompilationConfig 字段 |
| `CUDAGraphMode` | CUDA Graph 模式 | Ch4 | NONE/PIECEWISE/FULL 运行时态 + FULL_DECODE_ONLY/FULL_AND_PIECEWISE 组合态 |

## A.3 Dynamo 前端

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| guard | 守卫条件 | Ch5 | dynamo 记录的"何时不需重编译"的断言集合 |
| `skip_all_guards_unsafe` | 丢弃全部 guards | Ch5 | vLLM 核心 trick:一次 trace 后永不评估 guards |
| `fail_on_recompile` | 重编译即失败 | Ch5 | `torch.compiler.set_stance`:把"永不 re-trace"升级为运行时断言 |
| `mark_dynamic` | 标注动态维 | Ch5 | 显式声明某输入的某维是符号的(BACKED) |
| `mark_unbacked` | 标注无界符号维 | Ch5 | 声明该维符号无 0/1 特化,配合 range 编译兜底 |
| BACKED / UNBACKED symbol | 有界/无界符号 | Ch5 | 两种 SymInt:带上下界 vs 完全自由 |
| `traced_files` | 被 trace 的源文件集 | Ch5 | 劫持 dynamo inline 记录,文件内容进入编译缓存 key |
| bytecode hook | 字节码钩子 | Ch5 | 捕获 dynamo 变换后的 bytecode 并热替换 `forward.__code__` |
| `fullgraph` | 全图模式 | Ch5 | `torch.compile(..., fullgraph=True)`:禁止 graph break |

## A.4 图与切分(中端)

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| FX `GraphModule` | FX 图模块 | Ch3 | dynamo 捕获产物的中间表示,节点为 aten/custom op 调用 |
| `splitting_ops` | 切分算子表 | Ch4 | 默认 = attention 类算子;在这些 op 处把 FX 图切成 pieces |
| piece / piecewise graph | 片段 | Ch3 | 切分后的子图;分"可编译 piece"与"splitting piece"(eager) |
| splitting piece | 切分点片段 | Ch6 | attention 本身所在的 piece,不进 Inductor,以源码内联 |
| `split_module` | FX 切分工具 | Ch6 | `torch.fx.passes.split_module.split_module`,按分区函数切图 |
| `PiecewiseCompileInterpreter` | 逐片段编译解释器 | Ch6 | fx.Interpreter:fake 值传播 + 逐 piece 构造 PiecewiseBackend |
| fake tensor | 假张量 | Ch3 | 只携带元数据(shape/stride/dtype)不占数据的张量,编译期通货 |
| `example_value` | 示例值 | Ch6 | FX 节点 meta 里携带的 fake 值,interpreter 用它传播 |

## A.5 算子层

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `unified_attention_with_output` | 统一注意力算子 | Ch7 | vLLM 核心 custom op;output 由调用方分配传入,可被 dynamo trace |
| `direct_register_custom_op` | 直注册自定义算子 | Ch7 | 绕过 `torch.library.custom_op` 的 dispatch 开销直接注册 |
| fake impl | 假实现 | Ch7 | custom op 的 no-op fake,让 dynamo 不窥视内部即可 trace |
| `mutates_args` | 可变参数声明 | Ch7 | schema 中声明哪些参数被就地写——functionalization 的前提 |
| `vllm_ir` | vLLM 自定义 IR 命名空间 | Ch7 | `Library("vllm_ir","FRAGMENT")`;rms_norm 等"可融合算子"的载体 |
| `maybe_inplace` | 或就地 overload | Ch7 | vllm_ir 算子的就地变体,配合 activation donation |
| activation donation | 激活捐赠 | Ch7 | 证明激活无后继使用者后允许就地写、删防御性 clone |
| provider | 实现提供者 | Ch7 | 同一 vllm_ir 算子的多种实现,按优先级 dispatch |
| `auto_functionalized` | 自动泛函化形态 | Ch10 | in-place op 被 AOTAutograd 转成的函数式等价形态 |
| defunctionalization | 去泛函化 | Ch10 | 故意把 `auto_functionalized` 还原为 in-place 调用(FixFunctionalization) |

## A.6 后端编译

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `VllmBackend` | vLLM 编译后端 | Ch3 | torch.compile 的自定义 backend,全书主角 |
| `PiecewiseBackend` | 片段后端 | Ch6 | 每个可编译 piece 的后端;构造函数里把该 piece 全部 range 一次编完,产物偷换 split_gm 的 submodule 属性 |
| `CompilerInterface` | 编译器接口 | Ch8 | 编译器抽象基类:eager/inductor/standalone 三实现可插拔 |
| `InductorAdaptor` | Inductor 适配器 | Ch9 | 以 `compile_fx` + monkey-patch 栈对接 Inductor |
| `InductorStandaloneAdaptor` | standalone 适配器 | Ch9 | torch≥2.8 `standalone_compile` 路径,缓存更完整 |
| `EagerAdaptor` | eager 适配器 | Ch9 | 不编译直接返回图,调试用 |
| `compile_fx` | Inductor 入口函数 | Ch9 | `torch._inductor.compile_fx.compile_fx`,Inductor 主编译 API |
| `AlwaysHitShapeEnv` | 永真 shape 环境 | Ch9 | 假 shape env 使 Inductor 缓存查找永命中(dynamo 一次 Inductor 多次) |
| `pre_grad_custom_pass` | 前置自定义 pass 钩子 | Ch8 | vLLM 占用 Inductor 的两个钩子之一(AOTAutograd 之前) |
| `post_grad_custom_post_pass` | 后置自定义 pass 钩子 | Ch8 | 两钩子之二;整个 PostGradPassManager 挂在此处 |
| monkey-patch | 运行时补丁 | Ch9 | 对 torch 内部函数的临时替换;vLLM 集成的现实代价 |

## A.7 Shape 机制

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `compile_sizes` | 精确编译尺寸表 | Ch4 | 每个尺寸编一份全静态产物(单尺寸 Range) |
| `compile_ranges` | 区间编译表 | Ch4 | 一份符号 shape 产物覆盖一个 token 数区间 |
| `Range` | 尺寸区间 | Ch9 | `[start, end]`;`Range(n,n)` 即单尺寸 |
| `cudagraph_capture_sizes` | 捕获尺寸表 | Ch4 | cudagraph capture 的尺寸桶,默认 [1,2,4]+8..256(步长8)+..max(步长16) |
| `max_cudagraph_capture_size` | 最大捕获尺寸 | Ch4 | 桶上限;超出走符号 range 产物 |
| bucket padding | 桶填充 | Ch12 | 实际 num_tokens 向上取整到最近的捕获尺寸 |
| `max_autotune` | 最大自动调优 | Ch9 | 静态 shape 产物开启的 Triton 参数 exhaustive 搜索 |
| `coordinate_descent_tuning` | 坐标下降调优 | Ch9 | 在 max_autotune 基础上的进一步参数搜索 |

## A.8 Pass 与融合

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `PostGradPassManager` | 后置 pass 管理器 | Ch8 | vLLM 自定义 pass 的编排器,挂在 post_grad 钩子 |
| `InductorPass` | vLLM pass 基类 | Ch10 | uuid=sha256(pass 源码),进 Inductor 缓存 key |
| `VllmPatternMatcherPass` | 模式匹配 pass 基类 | Ch10 | 封装 `torch._inductor.pattern_matcher.register_replacement` |
| pattern / replacement | 模式/替换体 | Ch10 | 融合的两侧:待匹配子图 / 融合后的子图 |
| `is_applicable_for_range` | 区间适用门控 | Ch10 | 按 compile_range 决定 pass 是否作用于该产物 |
| range gating | 区间门控 | Ch10 | 不同 shape 区间启用不同 fusion 的机制 |
| `FixFunctionalization` | 修复泛函化 pass | Ch10 | 固定序最后一步,故意去泛函化 |

## A.9 运行时执行

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| CUDA Graph capture / replay | 捕获/重放 | Ch1 | 录制一段 GPU 工作流,此后一次 launch 整段重放 |
| `CUDAGraphWrapper` | CUDA Graph 包装器 | Ch3 | 按 BatchDescriptor 键控的 capture/replay 包装 |
| `execution_fn` | 执行函数 | Ch3 | generate_execution_code 生成、compile_execution_fn exec 出的纯 Python 组装函数,绕开 FX 解释开销 |
| liveness 分析 | 活性分析 | Ch11 | 回答"每个值活到哪条语句为止";决定生成源码里 `del` 的插入位置 |
| `BatchDescriptor` | 批描述符 | Ch12 | (num_tokens, num_reqs, uniform, LoRA 计数)——图的键 |
| `cudagraph_runtime_mode` | 运行时图模式 | Ch12 | ForwardContext 携带:NONE/PIECEWISE/FULL,wrapper 据此匹配 |
| FULL cudagraph | 整图 CUDA Graph | Ch4 | 包住整个 forward 的单一 graph |
| PIECEWISE cudagraph | 分片 CUDA Graph | Ch4 | 每个 piece 一个 graph,attention 走 eager |
| `CudagraphDispatcher` | 图分派器 | Ch3 | 运行时按 num_tokens/LoRA 选模式与桶 |
| graph pool | 图内存池 | Ch12 | 全部 graph 共享的静态显存池,复用地址省显存 |
| `weak_ref_output` | 弱引用输出 | Ch12 | 输出不被静态池钉死,内存可回收 |
| `copy_and_call` | 拷贝后调用 | Ch11 | 把符号 shape 输入拷入常驻静态 buffer 再调用产物 |

## A.10 缓存

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `VLLM_CACHE_ROOT` | vLLM 缓存根目录 | Ch13 | 默认 `~/.cache/vllm`;编译缓存在其 `torch_compile_cache/` 下 |
| cache hash | 缓存哈希 | Ch13 | sha256(env 因素+config+traced_files 内容+编译器因素) |
| `VllmSerializableFunction` | 可序列化函数 | Ch11 | 封装产物+FX 图,可 pickle 落盘/加载 |
| `GraphPickler` | FX 图序列化器 | Ch11 | torch 的 FX 专用 pickler,vLLM 加了 reducer 定制 |
| `vllm_compile_cache.py` | 编译管理缓存文件 | Ch13 | (Range, graph_index, compiler)→handle 的映射文件 |
| handle | 缓存句柄 | Ch13 | compile 返回的 (hash, path),load 凭它直接取产物 |
| AOT compile | 前置编译缓存 | Ch13 | `VLLM_USE_AOT_COMPILE`:整个编译产物 pickle,重启秒载 |
| `VLLM_DISABLE_COMPILE_CACHE` | 关闭编译缓存 | Ch13 | 排查缓存问题时使用 |

## A.11 调试

| 英文术语 | 中文释义 | 首见 | 一句话说明 |
|---|---|---|---|
| `debug_dump_path` | 调试转储目录 | Ch15 | CompilationConfig 字段;落盘各阶段 FX 图与 bytecode |
| `computation_graph.py` | 计算图转储 | Ch13 | 切分后整图的 print_readable 产物 |
| `transformed_code.py` | 变换字节码转储 | Ch5 | dynamo 变换后 forward 的 depyf 反编译产物 |
| depyf | dynamo 反编译工具 | Ch15 | 把 dynamo 内部产物转成可读 Python |
| tlparse | 时间线解析 | Ch15 | `TORCH_COMPILE_DEBUG=1` 产物的解析查看器 |
| match table | 匹配统计表 | Ch15 | `VllmPatternMatcherPass.get_match_table`:各 fusion 命中计数 |
| `compilation_counter` | 编译计数器 | Ch8 | 全书各编译事件计数的单例,测试与调试用 |

## A.12 易混词速查

| 词组 | 区别 |
|---|---|
| capture size / compile size | capture size 是 cudagraph 桶;compile size 是精确编译尺寸——v1 默认前者包含于后者 |
| compile size / compile range | size=单点全静态产物(开 max_autotune);range=区间符号产物 |
| splitting op / custom op / vllm_ir 算子 | splitting op=切分点(不编译);custom op=一般自定义算子(可禁用回退 native);vllm_ir 算子=为融合而生的 IR 载体(必然被 lowering) |
| piece / submodule / graph | piece=切分后的子图;切分实现上挂在 split_gm 的 submodule;graph 有时指整个 dynamo 图 |
| guard 评估 / shape guard | 前者是 dynamo 是否重 trace 的判定;后者是符号 shape 的断言,vLLM 可选保留(SHAPE_ENV) |
