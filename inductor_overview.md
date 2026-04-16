# PyTorch Inductor 全景结构与功能指南

> 本文档是 `torch/_inductor/` 的文件级全景地图，用于建立全局视野、理解编译管线各阶段的职责与协作关系，为深入走读代码提供路标。
>
> **权威参考**：
> - PyTorch 2 论文 (ASPLOS 2024): *"PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"* — [PDF](https://docs.pytorch.org/assets/pytorch2-2.pdf) | [ACM](https://dl.acm.org/doi/10.1145/3620665.3640366)
> - TorchInductor 原始设计帖: [TorchInductor: a PyTorch-native Compiler with Define-by-Run IR and Symbolic Shapes](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
> - Inductor 文件结构讨论: [Inductor file structure explanation](https://dev-discuss.pytorch.org/t/inductor-file-structure-explanation/1860)
> - PyTorch 2.0 Conference Talk: [PyTorch 2.0: TorchInductor (YouTube)](https://www.youtube.com/watch?v=ppWKVg-VxmQ)

---

## 〇、Inductor 的设计哲学

理解设计原则是读懂代码的前提。PyTorch 2 论文明确提出了四项设计原则（Section 4.1）：

| 原则 | 含义 |
|------|------|
| **PyTorch Native** | Inductor 的 IR 与 PyTorch eager 共享相似的抽象——张量有暴露的 strides、aliasing views 是常态、数据和元数据都可以 in-place 修改。编译器的翻译层很薄 |
| **Python First** | Inductor 用 Python 实现。PyTorch 社区的 Python 贡献远多于 C++ 贡献。Python 实现让 PyTorch 用户容易理解和修改 |
| **Breadth First** | 早期就专注于覆盖广泛的算子、硬件和优化，而非只针对少数模型（如 ResNet/BERT）。这也是为什么 Inductor 优先支持 training（比 inference 更难） |
| **Reuse State-Of-The-Art Languages** | 不自己发明 kernel 语言，而是生成 **Triton**（GPU）和 **C++/OpenMP**（CPU）作为输出语言。这样可以直接利用这些项目的优化，且输出的代码是 PyTorch 用户可以理解的 |

> 论文实验数据：TorchInductor 在 180+ 真实模型上实现了 **2.27x 推理** 和 **1.41x 训练** 几何平均加速（NVIDIA A100 GPU）。

---

## 一、Inductor 在 PyTorch 编译栈中的位置

```
用户 Python 代码
    │
    ▼
torch._dynamo          ── 字节码追踪，生成 FX Graph + Guards
    │
    ▼
AOTAutograd            ── 用 fake tensor 运行 eager autograd，
                          录制联合前向/反向图，用 min-cut 算法
                          拆分为独立的前向图和反向图，
                          优化重计算以减少内存占用
    │
    ▼
torch._inductor        ── 本目录：FX Graph → 优化 → 后端代码生成
    │
    ├── Decomposition  ── 算子分解（191 个分解，387 含重载）
    ├── FX Passes      ── 图级优化（融合、模式匹配、量化等）
    ├── Lowering       ── FX 算子 → Define-by-Run IR（含 Inlining）
    ├── Scheduler      ── IR 节点调度 + Fusion（贪心融合算法）
    ├── Codegen        ── IR → 目标后端代码（Triton / C++ / MPS 等）
    └── Runtime        ── 编译产物的运行时支撑
    │
    ▼
优化的可执行 Kernel（Triton GPU kernel / CPU C++ kernel / ...）
```

---

## 二、编译管线总览（数据流）

```
FX GraphModule（前向图，来自 Dynamo + AOTAutograd）
    │
    ▼  compile_fx.py (入口)
    │
    ▼  decomposition.py (算子分解：191 个分解，将复杂算子拆为简单组合)
    │
    ▼  fx_passes/ (图优化 Pass：模式匹配、注意力融合、conv-bn 融合等)
    │
    ▼  graph.py :: GraphLowering (FX Interpreter)
    │       └── lowering.py (逐算子翻译：433 个算子 → IR)
    │       └── 内联(Inlining)：将 pointwise body 复制到消费者中
    │
    ▼  ir.py (Define-by-Run Loop-Level IR：TensorBox/Buffer/Pointwise/Reduction)
    │       └── 54 个 ops.* 原语操作
    │       └── SymPy 符号形状
    │
    ▼  scheduler.py (调度 + 融合决策)
    │       └── can_fuse() / score_fusion() 贪心融合
    │       └── dependencies.py (依赖分析：MemoryDep/StarDep/WeakDep)
    │       └── memory.py (内存规划：生命周期分析 + 复用)
    │
    ▼  select_algorithm.py (Autotuning：为 GEMM/Conv 选择最优算法)
    │
    ▼  codegen/ (后端代码生成)
    │       ├── triton.py    → Triton GPU kernel (@pointwise/@reduction 装饰器)
    │       ├── cpp.py       → C++/OpenMP CPU kernel (向量化/非向量化两种变体)
    │       ├── cutlass/     → CUTLASS GEMM 模板 (Jinja + 手写 Triton 混合)
    │       ├── rocm/        → AMD Composable Kernel 模板
    │       └── wrapper.py   → Python/C++ 封装代码 + CUDA Graph 支持
    │
    ▼  output_code.py (编译产物封装：CompiledFxGraph)
    │
    ▼  可执行的 Python Callable
```

---

## 三、核心文件详解

### 3.1 编译入口与编排层

| 文件 | 行数 | 职责 |
|------|------|------|
| `__init__.py` | ~30 | 模块入口。暴露 `compile()` 函数——Inductor 对外的主 API，接受 FX GraphModule 和 example_inputs，返回优化后的 callable |
| `compile_fx.py` | ~3800 | **编译主流程的核心编排器**。包含 `compile_fx_inner()`，是整个 Inductor 编译管线的主入口。协调 FX Passes 执行、GraphLowering 创建、AOT 编译、CUDA Graph 封装、缓存查找等。这是理解"一次编译从头到尾发生了什么"的起点 |
| `compile_fx_async.py` | ~450 | 异步编译版本。将编译任务提交到后台线程池，不阻塞主线程，适用于训练场景中 overlap compile 和 compute |
| `compile_fx_ext.py` | ~750 | 扩展编译功能。支持子图编译、自定义后端注册、region-based 编译（部分图编译、部分 eager 执行） |
| `compile_fx_subproc.py` | ~100 | 子进程编译。将编译过程放到独立子进程中执行，用于隔离编译环境的内存和状态 |
| `standalone_compile.py` | ~550 | 独立编译接口。支持 AOT 编译导出（`torch._export` 路径），生成可序列化的编译产物 `CompiledArtifact` |
| `subgraph_lowering.py` | ~220 | 子图 Lowering。处理 region-based 编译中独立子图的 lowering 和代码生成 |

**学习建议**：从 `compile_fx.py` 的 `compile_fx_inner()` 函数入口开始走读，它是整个管线的总调度。

---

### 3.2 FX 图优化 Pass（fx_passes/）

在 Lowering 之前对 FX Graph 进行图级优化的 Pass 集合。这些 Pass 操作的是 FX Node 级别的图结构。

| 文件 | 职责 |
|------|------|
| `pre_grad.py` | **梯度计算前的优化 Pass 集合**。包含 conv-bn 融合、split-cat 优化、normalization 优化、shape 传播等。是最先执行的一批 Pass |
| `post_grad.py` | **梯度计算后的优化 Pass**。在 AOTAutograd 分离出前向/反向图后，对反向图执行额外的优化 |
| `joint_graph.py` | 联合图优化。处理前向+反向联合图上的优化机会 |
| `fuse_attention.py` | **Scaled Dot-Product Attention (SDPA) 融合**。将手动实现的 attention 子图替换为高效的 `flash_attention` 或 `memory_efficient_attention` kernel。是 Inductor 最重要的优化之一 |
| `efficient_conv_bn_eval.py` | 推理模式下 conv + batchnorm 融合。利用结合律将 BN 参数折叠进 conv 权重，消除 BN 计算 |
| `binary_folding.py` | 二元算子（add/sub/mul）与矩阵乘法的融合。将逐元素操作折叠进 GEMM kernel 中，减少内存带宽开销 |
| `b2b_gemm.py` | Back-to-Back GEMM 融合。将 `(A @ B) @ C` 形式的连续矩阵乘法融合为单个优化 kernel |
| `pad_mm.py` | 矩阵乘法 padding 优化。将非对齐尺寸的 matmul padding 到对齐尺寸，提升 GEMM 效率 |
| `decompose_mem_bound_mm.py` | 分解内存瓶颈的 matmul。对于小尺寸 matmul，分解为逐元素操作可能比调用 GEMM 更快 |
| `group_batch_fusion.py` | 组批融合。将多个小算子合并为批量执行，提高 GPU 利用率 |
| `fusion_regions.py` | 融合区域识别。检测可融合的算子区域，用于 overlap scheduling |
| `ddp_fusion.py` | DDP 通信融合。合并分布式训练中的 allreduce 通信 |
| `fsdp.py` | FSDP 通信优化。针对 Fully Sharded Data Parallel 的 all_gather/reduce_scatter 分桶优化 |
| `bucketing.py` | 通信操作分桶。将多个小通信操作合并为大块通信，减少延迟 |
| `quantization.py` | 量化模式优化。处理量化/反量化相关的图模式 |
| `reinplace.py` | 就地操作转换。将 out-of-place 操作转换为 in-place 操作以节省内存 |
| `split_cat.py` | split/cat 模式优化。消除冗余的 split-cat 操作链 |
| `misc_patterns.py` | 杂项优化模式。收集了各种小规模的模式匹配优化 |
| `control_dependencies.py` | 控制依赖管理。为需要精确排序的操作（如集合通信）插入显式依赖 |
| `overlap_scheduling.py` | 重叠调度优化。将计算和通信重叠执行 |
| `overlap_preserving_bucketer.py` | 保持重叠的分桶器。在 overlap scheduling 中确保分桶策略正确 |
| `overlap_manual_scheduling.py` | 手动重叠调度。支持用户手动指定的重叠调度策略 |
| `micro_pipeline_tp.py` | 微流水线张量并行。支持 pipeline parallelism 中的计算通信重叠 |
| `memory_estimator.py` | 内存估算。估算图节点的内存使用量 |
| `node_runtime_estimation.py` | 节点运行时估算。估算各节点的执行时间，辅助调度决策 |
| `graph_view.py` | 图视图。提供层次化的图结构视图，用于分析和调试 |
| `auto_chunker/` | 自动分块。将大操作拆分为适合缓存的小块（tiling），提高内存局部性 |
| `serialized_patterns/` | 预序列化的模式匹配规则。包含 SDPA（28+ 种模式）、mm、bmm、addmm 的模式 |

**学习建议**：先理解 `pre_grad.py` 和 `post_grad.py` 中的 Pass 注册机制，再深入具体 Pass。`fuse_attention.py` 是理解模式匹配优化的好例子。

---

### 3.3 图解释与 Lowering 层

| 文件 | 行数 | 职责 |
|------|------|------|
| `graph.py` | ~3300 | **GraphLowering：FX Interpreter + 状态管理器**。继承 `torch.fx.Interpreter`，逐节点执行 FX Graph。将每个 FX Node 翻译为 Inductor IR 节点。管理图输入/输出、设备信息、布局优化决策（channels-last）、常量处理、子图缓存等。是 FX Graph → Inductor IR 的桥梁 |
| `lowering.py` | ~8300 | **算子 Lowering 注册表**。为每个 ATen 算子注册"如何翻译为 Inductor IR"的函数。包含数百个算子的 lowering 实现——从简单的 elementwise 到复杂的 matmul/conv/attention。使用 `@lowering_registry.register` 装饰器模式。这是理解"每个算子如何变成 IR"的核心文件 |
| `decomposition.py` | ~1300 | **算子分解**。在 lowering 之前，将复杂算子分解为更基础的算子组合。例如将 `torch.addmm` 分解为 `mm + add`。使用 `core_aten_decompositions` 并添加 Inductor 特有的分解规则 |
| `ops_handler.py` | ~1100 | **操作处理器**。定义 `OpsHandler` 基类，提供 Inductor IR 中各类操作的默认实现。lowering 过程中通过 `V.ops` 虚拟化接口调用这些操作 |
| `inductor_prims.py` | ~400 | Inductor 原语操作。定义 Inductor 内部使用的底层 prim 算子（区别于 ATen 算子） |

**学习建议**：先读懂 `graph.py` 中的 `GraphLowering.run_node()` 和 `call_function()`，理解 FX Node 如何通过 lowering 注册表翻译为 IR。然后从 `lowering.py` 中挑选几个简单算子（如 `add`、`mul`）和一个复杂算子（如 `mm`）走读其 lowering 实现。

---

### 3.4 中间表示（IR）层

| 文件 | 行数 | 职责 |
|------|------|------|
| `ir.py` | ~10600 | **Inductor IR 节点定义**。这是 Inductor 最大的单一文件，定义了所有 IR 节点类型。核心类层次：|

`ir.py` 的关键类层次：

```
IRNode (基类)
├── Operation            ── 计算操作基类
│   └── Loops            ── 循环类操作
│       ├── Pointwise    ── 逐元素操作（add, relu, ...）
│       ├── Scatter      ── scatter 操作
│       ├── Reduction    ── 归约操作（sum, max, ...）
│       ├── Scan         ── 扫描操作（cumsum, ...）
│       └── Sort         ── 排序操作
├── BaseView             ── 视图操作（零拷贝）
│   ├── ExpandView       ── 广播扩展
│   ├── PermuteView      ── 维度重排
│   ├── SqueezeView      ── 维度压缩
│   └── GenericView      ── 通用视图（reshape, slice, ...）
├── BaseConstant         ── 常量
├── Buffer               ── 内存缓冲区（命名存储）
│   ├── OperationBuffer  ── 计算产生的缓冲区
│   ├── InputBuffer      ── 输入缓冲区
│   ├── ComputedBuffer   ── 已计算的缓冲区
│   └── TemplateBuffer   ── 模板 kernel 的缓冲区
├── ExternKernel         ── 外部 kernel 调用（cuBLAS, cuDNN 等）
├── MutableBox           ── 可变容器（lazy evaluation）
│   ├── TensorBox        ── 张量容器（延迟 realize）
│   └── StorageBox       ── 存储容器
└── Subgraph             ── 嵌套子图
```

**核心抽象**：
- **TensorBox / StorageBox**：延迟求值容器。TensorBox 包装数据，在需要时才 realize（物化为 Buffer）。这是 Inductor 的 lazy evaluation 机制
- **Buffer**：命名存储单元，对应最终生成的代码中的一个变量
- **Layout**：描述数据的内存布局（strides, dtype, device）。`FixedLayout`（固定布局）和 `FlexibleLayout`（可优化布局）
- **ExternKernel**：对 ATen / cuBLAS / cuDNN 等外部库的调用封装

**学习建议**：先理解 `TensorBox` → `StorageBox` → `Buffer` → `Layout` 的层次关系和延迟 realize 机制。这是理解 Inductor 如何避免不必要计算的关键。

---

### 3.5 调度与融合层

| 文件 | 行数 | 职责 |
|------|------|------|
| `scheduler.py` | ~9500 | **调度器**。Inductor 的第二个核心大文件。将 IR 节点分组为可融合的 Kernel Group，决定执行顺序、融合策略、内核选择。核心类 `Scheduler` 管理所有 `SchedulerNode`，执行：1) 依赖分析 2) 融合决策 3) 内存规划 4) 代码生成触发 |
| `dependencies.py` | ~1000 | **依赖分析**。分析 IR 节点之间的读写依赖关系。定义 `MemoryDep`（内存依赖）、`StarDep`（通配依赖）、`WeakDep`（弱依赖）等依赖类型。是调度器判断能否融合的基础 |
| `memory.py` | ~1200 | **内存规划**。分析 buffer 的生命周期，进行内存复用。决定哪些 buffer 可以共享同一块物理内存，减少总内存占用 |
| `select_algorithm.py` | ~6800 | **算法/模板选择与 Autotuning**。为 GEMM、卷积等操作选择最优算法。通过 benchmarking 多个候选实现（Triton kernel、cuBLAS、CUTLASS 等），选择最快的。包含 `TritonBenchmarkRequest`、`ExternKernelRequest` 等 |
| `choices.py` | ~850 | **候选算法管理**。管理同一操作可能有的多种实现选择（choices），配合 autotuning 使用 |
| `tiling_utils.py` | ~850 | **分块工具**。计算 reduction 和 pointwise 操作的最优 tiling 参数（block size, num warps 等） |

**学习建议**：从 `scheduler.py` 的 `Scheduler` 类的 `run()` 方法开始，理解节点如何被分组（`FusedSchedulerNode`）和调度。然后看 `dependencies.py` 如何判断两个节点能否融合。

---

### 3.6 后端代码生成（codegen/）

这是 Inductor 的代码生成层，将 IR 翻译为目标后端的实际代码。

#### 3.6.1 代码生成框架

| 文件 | 职责 |
|------|------|
| `common.py` | **代码生成基类与注册框架**。定义 `Kernel`（kernel 基类）、`KernelTemplate`（模板基类）、`DeviceOpOverrides`（设备操作覆盖）、`BackendFeature`（后端能力声明）。提供 `register_backend_for_device()` API 供各后端注册。所有后端共享的 CSE（公共子表达式消除）、索引计算等逻辑也在此 |
| `wrapper.py` | **Python 封装代码生成器**。生成最终运行的 Python 函数代码——分配内存、调用 kernel、处理输入输出。`PythonWrapperCodegen` 是所有后端共用的 wrapper 层 |
| `wrapper_fxir.py` | FX IR 形式的 wrapper 生成。生成 FX Graph 格式的封装代码 |
| `memory_planning.py` | **内存分配规划**。分析 buffer 生命周期，生成内存分配和释放代码。实现 Live Range Analysis 和内存池复用 |
| `block_analysis.py` | 块级分析。分析代码块之间的依赖和调度 |
| `debug_utils.py` | 代码生成调试工具 |
| `simd.py` | **CPU SIMD 向量化**。生成 AVX/AVX2/AVX-512 等 SIMD 指令的 CPU kernel。`SIMDKernel` 是 CPU 向量化 kernel 的基类 |
| `simd_kernel_features.py` | SIMD kernel 特征分析。提取 kernel 所需的 SIMD 操作特征 |
| `multi_kernel.py` | 多 kernel 选择。为同一操作生成多个 kernel 版本，运行时根据输入选择最优版 |
| `segmented_tree.py` | 分段树调度。管理多个 kernel 的分段执行 |
| `subgraph.py` | 子图代码生成。将子图编译为独立的可调用单元 |
| `custom_extern_kernel_codegen.py` | 自定义外部 kernel 的代码生成支持 |

#### 3.6.2 Triton 后端（GPU）

| 文件 | 职责 |
|------|------|
| `triton.py` | **Triton GPU kernel 生成器**。Inductor 最重要的后端。`TritonKernel` 将 IR 翻译为 Triton kernel 代码，`TritonScheduling` 管理融合和调度决策。处理 pointwise、reduction、scan 等所有 IR 类型 |
| `triton_combo_kernel.py` | **组合 Triton Kernel**。将多个独立操作合并到单个 Triton kernel launch 中，减少 kernel launch 开销 |
| `triton_utils.py` | Triton 工具函数。类型映射、配置生成等 |
| `triton_split_scan.py` | 分离扫描的 Triton 实现 |

#### 3.6.3 C++ 后端（CPU）

| 文件 | 职责 |
|------|------|
| `cpp.py` | **C++ CPU kernel 生成器**。`CppKernel` 和 `CppScheduling` 生成 OpenMP 并行的 C++ kernel。支持多种 kernel 类型：`CppVecKernel`（向量化）、`CppTile2DKernel`（2D 分块）、`OuterLoopFusedKernel`（外层循环融合） |
| `cpp_template.py` | CPU 模板抽象。定义 CPU kernel 模板的基类 |
| `cpp_gemm_template.py` | CPU GEMM 模板。生成优化的 CPU 矩阵乘法 kernel |
| `cpp_bmm_template.py` | CPU BMM（Batch MatMul）模板 |
| `cpp_grouped_gemm_template.py` | CPU 分组 GEMM 模板 |
| `cpp_flex_attention_template.py` | CPU Flex Attention 模板 |
| `cpp_micro_gemm.py` | CPU 微型 GEMM。小矩阵乘法的特化优化 |
| `cpp_utils.py` | C++ 代码生成工具函数 |
| `cpp_wrapper_cpu.py` | CPU Python wrapper 生成 |
| `cpp_wrapper_cpu_array_ref.py` | CPU Array Ref wrapper |
| `cpp_wrapper_gpu.py` | GPU C++ wrapper 生成（用于 CUDA C++ kernel） |
| `cpp_wrapper_mps.py` | MPS C++ wrapper 生成 |
| `cpp_template_kernel.py` | C++ 模板 kernel |
| `cpu_device_op_overrides.py` | CPU 设备操作覆盖 |

#### 3.6.4 CUTLASS 后端（NVIDIA GEMM 模板）

| 文件 | 职责 |
|------|------|
| `cutlass/gemm_template.py` | CUTLASS GEMM 模板。使用 NVIDIA CUTLASS 库生成高性能 GEMM kernel |
| `cutlass/kernel.py` | CUTLASS kernel 封装 |
| `cutlass/scheduling.py` | CUTLASS 调度策略 |
| `cutlass/template.py` | CUTLASS 模板基类 |
| `cutlass/cache.py` | CUTLASS 编译缓存 |
| `cutlass/utils.py` | CUTLASS 工具函数 |
| `cutlass/serialization.py` | CUTLASS 配置序列化 |
| `cutlass/python_evt.py` | CUTLASS Python EVT（Epilogue Visitor Tree）支持 |
| `cutlass/lib_extensions/` | CUTLASS 库扩展，包含 mock imports 用于无 CUTLASS 环境的编译 |

#### 3.6.5 ROCm 后端（AMD GPU）

| 文件 | 职责 |
|------|------|
| `rocm/ck_template.py` | Composable Kernel (CK) 模板。AMD GPU 的高性能 GEMM 模板 |
| `rocm/ck_conv_template.py` | CK 卷积模板 |
| `rocm/ck_tile_template.py` | CK Tile 模板（新架构） |
| `rocm/rocm_kernel.py` | ROCm kernel 基类 |
| `rocm/rocm_template.py` | ROCm 模板基类 |
| `rocm/rocm_cpp_scheduling.py` | ROCm C++ 调度 |
| `rocm/rocm_utils.py` | ROCm 工具函数 |
| `rocm/compile_command.py` | ROCm 编译命令构建 |

#### 3.6.6 其他后端

| 目录/文件 | 职责 |
|-----------|------|
| `mps.py` | **Apple MPS (Metal) 后端**。为 Apple Silicon GPU 生成 Metal shader 代码 |
| `mps_device_op_overrides.py` | MPS 设备操作覆盖 |
| `halide.py` | Halide 后端。编译为 Halide DSL 的 CPU 优化代码 |
| `pallas.py` | TPU Pallas 后端。编译为 JAX/Pallas 格式用于 Google TPU |
| `xpu/` | Intel XPU (GPU) 后端。SYCL + Triton 混合调度 |
| `mtia/` | MTIA (Meta Training and Inference Accelerator) 后端 |
| `cuda/` | CUDA 设备辅助工具（环境检测、编译选项） |
| `cutedsl/` | CuTe DSL 后端。NVIDIA 的高级 GPU 编程 DSL |
| `nv_universal_gemm/` | NVIDIA 通用 GEMM。统一的 GEMM 接口支持多种 GEMM 变体 |
| `cuda_combined_scheduling.py` | CUDA 组合调度。将 Triton + CUTLASS + C++ 的调度统一管理 |

**学习建议**：先读懂 `common.py` 中的后端注册机制和 `Kernel` 基类，再选择一个后端深入（推荐 `triton.py`）。通过 `register_backend_for_device()` 理解后端如何被选择。

---

### 3.7 运行时支撑（runtime/）

编译产物的运行时辅助代码。这些代码在编译好的 kernel 被调用时执行。

| 文件 | 职责 |
|------|------|
| `runtime_utils.py` | 运行时通用工具。Buffer 分配、dtype 处理等 |
| `triton_helpers.py` | Triton 运行时辅助。kernel 加载和调用 |
| `triton_heuristics.py` | Triton 启发式参数。根据输入大小选择最优 kernel 配置（block size, num warps 等） |
| `triton_compat.py` | Triton 兼容性处理 |
| `triton_lazy_compile.py` | Triton 延迟编译。kernel 在第一次使用时才编译 |
| `static_triton_launcher.py` | 静态 Triton launcher。预编译 kernel 的启动器 |
| `benchmarking.py` | Benchmarking 工具。运行 kernel benchmark 用于 autotuning |
| `coordinate_descent_tuner.py` | 坐标下降调优器。使用坐标下降法搜索最优 kernel 配置 |
| `autotune_cache.py` | Autotune 结果缓存。避免重复 benchmark |
| `hints.py` | 运行时提示。`ReductionHint`、`DeviceProperties` 等，辅助调度决策 |
| `compile_tasks.py` | 编译任务管理 |
| `halide_helpers.py` | Halide 运行时辅助 |
| `debug_utils.py` | 运行时调试工具 |
| `proton_utils.py` | Proton profiler 集成 |
| `cache_dir_utils.py` | 缓存目录管理 |
| `caching/` | 缓存框架。包含缓存配置、上下文、编码器、接口、锁等子模块 |

---

### 3.8 编译缓存与 Autotuning

| 文件 | 行数 | 职责 |
|------|------|------|
| `codecache.py` | ~5500 | **代码缓存系统**。管理编译产物的缓存和加载。`PyCodeCache` 将生成的 Python 代码缓存到磁盘，下次相同输入时直接加载。`FxGraphCache` 缓存整个 FX Graph 的编译结果。是 Inductor 增量编译的核心 |
| `cache.py` | ~450 | 编译缓存辅助。CacheKey 计算、缓存有效性检查 |
| `autotune_process.py` | ~1500 | **Autotune 子进程管理**。`TuningProcess` 在独立子进程中运行 kernel benchmark，避免影响主进程。支持并行 benchmark 多个 kernel 变体 |
| `async_compile.py` | ~900 | **异步编译框架**。将 kernel 编译任务分发到线程池，与主流程并行执行。`AsyncCompile` 管理编译任务队列 |
| `select_algorithm.py` | ~6800 | 算法选择（见 3.5 节） |
| `remote_cache.py` | ~400 | 远程缓存。支持将编译缓存存放到远程存储，在多机器间共享 |
| `mock_cache.py` | ~250 | 模拟缓存。用于测试 |
| `triton_bundler.py` | ~500 | Triton 打包器。将 Triton kernel 及其依赖打包为独立可分发单元 |

---

### 3.9 分析与传播

| 文件 | 职责 |
|------|------|
| `analysis/` | 分析工具目录 |
| `analysis/device_info.py` | 设备信息分析。收集 GPU/CPU 的能力信息 |
| `analysis/profile_analysis.py` | Profile 数据分析 |
| `bounds.py` | **值范围分析**。使用 `sympy` 对 IR 表达式进行上下界分析，用于优化索引计算和消除边界检查 |
| `dtype_propagation.py` | **数据类型传播**。在 IR 中传播 dtype 信息，确保类型一致性 |
| `shape_propagation.py` | **形状传播**。在 lowering 过程中传播张量形状信息 |
| `index_propagation.py` | **索引传播**。分析和传播数组索引表达式 |
| `invert_expr_analysis.py` | 表达式求逆分析。用于 view 操作的反向映射 |
| `analyze_preserves_zero_mask.py` | 零掩码保持分析。判断操作是否保持零值掩码不变 |

---

### 3.10 辅助基础设施

| 文件 | 行数 | 职责 |
|------|------|------|
| `config.py` | ~3200 | **配置系统**。`config` 对象提供数百个配置项控制 Inductor 行为。可通过 `torch._inductor.config.patch()` 临时修改。包含 triton、cpp、debug、优化等各方面的配置 |
| `config_comms.py` | ~100 | 分布式通信配置 |
| `virtualized.py` | ~500 | **虚拟化全局变量系统**。`V` 对象提供线程局部、动态作用域的全局变量。核心机制：`V.ops`（操作处理器）、`V.graph`（当前图）、`V.kernel`（当前 kernel）。代码通过 `V.set_*` 安装处理器，通过 `V.*` 访问。这是 Inductor 中 "define-by-run" IR 的基础——loop body 是可调用函数，通过替换 `V.ops` 来改变执行语义 |
| `utils.py` | ~4600 | **通用工具函数集**。包含 `sympy` 工具、条件表达式简化、dtype 处理、设备检测、内存对齐等。是 Inductor 的"瑞士军刀" |
| `sizevars.py` | ~1350 | **符号大小变量管理**。`SimplifyIndexing` 处理动态形状（symbolic shapes）。使用 `sympy` 简化索引表达式、生成长度断言。是 Inductor 支持动态形状的核心组件 |
| `loop_body.py` | ~800 | **循环体表示**。`LoopBody` 将 IR 的循环体表示为可调用的函数。通过 `V.ops` 挂钩执行语义（代码生成、分析、优化等） |
| `fx_utils.py` | ~400 | FX Graph 工具函数。FLOP 计算、节点分析等 |
| `pattern_matcher.py` | ~2650 | **模式匹配引擎**。在 FX Graph 上进行子图模式匹配。支持多行模式、自定义检查函数。是 fx_passes 中大多数优化的基础工具 |
| `constant_folding.py` | ~470 | 常量折叠。在编译时计算常量表达式的值 |
| `freezing.py` | ~340 | 模型冻结。将训练模式下的参数固定为常量，用于推理优化 |
| `freezing_utils.py` | ~40 | 冻结工具函数 |
| `custom_graph_pass.py` | ~330 | **自定义图 Pass 接口**。允许用户注册自己的图优化 Pass。`CustomGraphModulePass` 和 `CustomInferenceAwareGraphPass` 是用户扩展点 |
| `debug.py` | ~1400 | 调试工具。图可视化、编译日志、中间产物保存等 |
| `metrics.py` | ~440 | 性能指标收集。统计编译时间、kernel 数量、fusion 命中率等 |
| `exc.py` | ~150 | Inductor 异常定义。`InvalidCxxDeviceError`、`GPUTooOldForTriton`、`TritonMissing` 等 |

---

### 3.11 Kernel 模板库（kernel/）

预定义的高性能 kernel 模板，为特定操作提供专家级实现。

| 文件 | 职责 |
|------|------|
| `mm.py` | **MatMul kernel 模板**。矩阵乘法的 Triton kernel 实现 |
| `mm_common.py` | MatMul 公共逻辑 |
| `mm_grouped.py` | 分组 MatMul |
| `mm_plus_mm.py` | `A @ B + C @ D` 融合模板 |
| `bmm.py` | Batch MatMul 模板 |
| `conv.py` | 卷积 kernel 模板 |
| `custom_op.py` | 自定义操作 kernel |
| `kernel_inputs.py` | Kernel 输入管理 |
| `kernel_template_choice.py` | Kernel 模板选择 |
| `flex/` | Flex Attention kernel 系列 |
| `flex/flex_attention.py` | Flex Attention 主实现 |
| `flex/flex_flash_attention.py` | Flash Attention 变体 |
| `flex/flex_decoding.py` | 解码阶段 Attention |
| `flex/flex_cpu.py` | CPU Flex Attention |
| `vendored_templates/` | 第三方 kernel 模板 |

---

### 3.12 分布式通信

| 文件 | 行数 | 职责 |
|------|------|------|
| `comms.py` | ~3200 | **分布式通信 Lowering**。将 `all_reduce`、`all_gather`、`reduce_scatter` 等集合通信操作翻译为 Inductor IR。处理通信与计算的调度 |
| `comm_lowering.py` | ~870 | 通信 lowering 辅助 |
| `comm_analysis.py` | ~440 | 通信性能分析。估算 NCCL 集合通信的运行时间 |
| `comms_debug.py` | ~130 | 通信调试工具 |
| `distributed_autotune.py` | ~370 | 分布式 Autotuning。在多卡环境下进行 kernel 调优 |

---

### 3.13 其他专用模块

| 文件 | 职责 |
|------|------|
| `cudagraph_trees.py` | **CUDA Graph 树管理**。CUDA Graph 将多个 kernel 调用录制为单个 GPU 提交，减少 launch 开销。管理 CUDA Graph 的缓存、复用和更新 |
| `cudagraph_utils.py` | CUDA Graph 工具函数 |
| `cpp_builder.py` | **C++ 编译器构建器**。管理 C++ kernel 的编译命令构建和执行（跨平台） |
| `cpu_vec_isa.py` | CPU 向量指令集检测。检测 AVX/AVX2/AVX-512 等 ISA 支持情况 |
| `mkldnn_lowerings.py` | **oneDNN Lowering**。将操作翻译为 oneDNN (MKLDNN) 的调用，用于 Intel CPU 优化 |
| `mkldnn_ir.py` | oneDNN IR 节点定义 |
| `quantized_lowerings.py` | 量化算子 Lowering |
| `jagged_lowerings.py` | Jagged tensor（不规则张量）Lowering |
| `rocm_multiarch_utils.py` | ROCm 多架构工具 |
| `output_code.py` | **编译产物封装**。`CompiledFxGraph` 和 `CompiledAOTI` 封装编译结果，管理缓存序列化、CUDA Graph 配置、常量绑定等 |
| `compile_worker/` | 编译工作进程 |
| `compile_worker/subproc_pool.py` | 子进程池管理 |
| `compile_worker/tracked_process_pool.py` | 带追踪的进程池 |
| `autoheuristic/` | **自动启发式学习**。使用历史数据学习最优 kernel 配置。包含 A100/H100 的 MatMul 排名模型 |
| `template_heuristics/` | **模板启发式规则**。为 GEMM、Conv 等模板操作提供启发式配置选择。包含 Triton、CUTLASS、CuteDSL 等后端的启发式规则 |
| `lookup_table/` | 查找表优化。将计算替换为预计算的查找表 |
| `package/` | AOT 编译打包。将编译产物打包为可分发文件 |
| `hooks.py` | 编译钩子。提供插入自定义逻辑的扩展点 |
| `fuzzer.py` | Fuzzer 工具。生成随机 IR 用于测试代码生成的正确性 |
| `wrapper_benchmark.py` | Wrapper benchmark 工具 |
| `await_utils.py` | 异步等待工具 |
| `augmented_graph_helper.py` | 增强图辅助工具 |
| `stream_utils.py` | CUDA 流管理工具 |
| `stream_constants.py` | 流常量 |
| `extern_node_serializer.py` | 外部节点序列化 |
| `aoti_eager.py` | AOTI eager 模式 |
| `compiler_bisector.py` | 编译器二分法调试。自动定位导致编译失败的具体 Pass |
| `test_case.py` | Inductor 测试基类 |
| `test_operators.py` | Inductor 算子测试 |

---

## 四、关键机制深入

### 4.1 Define-by-Run IR：核心理念

Inductor IR 最大的创新是 **Define-by-Run**（边执行边定义）。与 TensorFlow 的 Static Graph 不同，Inductor 的 IR 在 Python 解释器中**逐步构建**，每执行一个算子就立即生成对应的 IR 节点。这使得 IR 构建过程可以利用 Python 的全部控制流能力。

**核心示例**（`torch.log2(x)` 的 Inductor IR，论文 Figure 2）：

```python
def inner_fn_buf0(index):
    i0, i1 = index
    tmp0 = ops.load("arg0_1", i0 * s1 + i1)      # 从输入 buffer 加载
    tmp1 = ops.log(tmp0)                            # 计算 log
    tmp2 = ops.constant(1.4426950408889634, torch.float32)  # 1/ln(2)
    tmp3 = ops.mul(tmp1, tmp2)                      # log(x) * (1/ln(2)) = log2(x)
    return tmp3
```

> 完整的 IR 结构（TensorBox/StorageBox/ComputedBuffer 包装）和 V.ops 虚拟化机制的深入讲解，见 [阶段一 6.1-6.2 节](phase1_global_view.md)。

**关键观察**：
- `inner_fn_buf0` 是一个普通 Python 函数——这就是 "define-by-run" 的含义。IR 就是用可执行的 Python 代码来定义循环体
- `s0`, `s1` 是 SymPy 符号变量，代表动态的 tensor 大小
- `ops.load` / `ops.log` / `ops.mul` 等是 Inductor 的 **54 个原语操作**
- `TensorBox` → `StorageBox` → `ComputedBuffer` 是延迟求值的三层包装
- 这个函数最初不是作为单一平坦函数构建的，而是由多个小函数闭包组合而成——`ops.mul` 的函数调用 `ops.log` 的函数，后者又调用加载输入的函数

### 4.2 Inductor 的 54 个原语操作（ops.*）

论文 Section 4.3 列出了 Inductor loop-level IR 的所有原语操作：

| 类别 | 操作 | 说明 |
|------|------|------|
| 内存访问 | `ops.load`, `ops.store` | 从 buffer 名 + SymPy 索引读写 Tensor 内存 |
| 归约 | `ops.reduction` | 类似 `ops.store`，写入时隐式归约。支持：argmin, argmax, any, max, min, prod, sum, xor_sum, welford_combine |
| 索引转换 | `ops.index_expr` | SymPy 表达式 → 计算值（用于索引到计算的转换） |
| 间接索引 | `ops.indirect_indexing` | 计算值 → SymPy 表达式（引入动态绑定的符号变量，用于 gather/scatter） |
| 条件执行 | `ops.masked` | 接受条件和 Python 函数，映射到 Triton mask 或 C++ 条件语句 |
| 随机数 | `ops.load_seed`, `ops.rand`, `ops.randn`, `ops.randint64` | 随机数生成 |
| 逐元素数学 | ~40+ 个操作 | abs, sin, cos, exp, log, sqrt, div, mul, add, ... |

这些操作通过 `V.ops` 虚拟化机制分发，替换不同的 handler 即可实现不同语义（代码生成、分析、优化）。

### 4.3 延迟求值（Lazy Evaluation）

Inductor 的 IR 层使用 `TensorBox` → `StorageBox` → `Buffer` 的层次实现延迟求值：

1. **TensorBox**：包装一个数据引用。支持 view 操作（reshape, permute 等）而不实际生成代码
2. **StorageBox**：管理存储。跟踪是否已被 realize（物化）
3. **Buffer**：实际的命名存储单元。对应生成的代码中的一个变量

当调用 `realize()` 时，TensorBox 被物化为 Buffer，此时才真正进入调度和代码生成。

### 4.4 虚拟化操作系统（V.ops）

`virtualized.py` 中的 `V` 对象是 Inductor 的核心设计模式：

```python
from torch._inductor.virtualized import V, ops

# 编译时：安装代码生成 handler
with V.set_ops_handler(codegen_handler):
    loop_body()  # 生成代码

# 分析时：安装分析 handler
with V.set_ops_handler(analysis_handler):
    loop_body()  # 收集信息

# 传播时：安装传播 handler
with V.set_ops_handler(dtype_prop_handler):
    loop_body()  # 传播 dtype
```

同一个 `loop_body` 函数通过替换 `V.ops` 实现不同的语义。这就是 "define-by-run" IR。

### 4.5 后端注册机制

每个后端通过 `register_backend_for_device()` 注册：

```python
# codegen/triton.py 中
register_backend_for_device("cuda", TritonScheduling, PythonWrapperCodegen)
register_backend_for_device("xpu", XPUScheduling, PythonWrapperCodegen)

# codegen/cpp.py 中
register_backend_for_device("cpu", CppScheduling, PythonWrapperCodegen)
```

调度器根据张量的设备类型自动选择对应后端的 Scheduling 和 Kernel 类。

### 4.6 Autotuning 流程

1. `lowering.py` 中为 GEMM/Conv 等操作注册多个候选实现
2. `select_algorithm.py` 为每个候选创建 benchmark request
3. `autotune_process.py` 在子进程中运行 benchmark
4. 最快的实现被选中，结果缓存到 `autotune_cache`

### 4.7 Inlining vs Fusion：两个不同的优化阶段

论文 Table 4 的消融实验揭示了一个关键区分：kernel 组合发生在**两个不同阶段**。

**Inlining（内联）**——发生在 **Lowering 阶段**（`graph.py` / `lowering.py`）：
- 当阈值满足时，将 pointwise kernel 的函数体**复制**到所有消费者中
- 避免中间结果的物化（realize），让多个操作在同一个函数闭包内执行

**Fusion（融合）**——发生在 **Scheduling 阶段**（`scheduler.py`）：
- 将剩余的独立 kernel 组合为单个 kernel
- 通过 `can_fuse()` 和 `score_fusion()` 两个关键函数控制

**两者关系**（论文原话）：
> "The biggest speedups in TorchInductor come from combining pointwise, reduction, and scatter kernels together into a smaller number of fused kernels. Without both of those passes, TorchInductor generates slowdowns rather than speedups."

关键结论：**fusion + inlining 是加速的最大来源**（去掉后推理从 1.91x 变为 0.80x，反而比 eager 模式更慢）。没有它们，分解（decomposition）将大的优化算子拆成了很多小算子，必须靠融合重新组合才能恢复性能。

> 完整消融实验数据表和深入的因果分析，见 [阶段一 2.3 节](phase1_global_view.md) 和 [阶段四 1.2 节](phase4_scheduling_fusion.md)。

### 4.8 融合算法概览

论文 Section 4.4 描述了调度器的融合贪心算法。核心函数 `can_fuse()` 和 `score_fusion()` 控制融合的合法性和优先级，在最多 10 轮贪心循环中迭代融合直到收敛。

> 融合算法的详细源码讲解、UML 图和 Debug 路线，见 [阶段四：调度与融合](phase4_scheduling_fusion.md)。

### 4.9 生成的 Triton 代码示例

论文 Figure 3 展示了 `torch.log2` 最终生成的 Triton kernel 代码。核心要点是 `@pointwise` 装饰器编码了 block size 启发式、autotuning 和 AOT 编译的样板代码，而 body 中的 `tl.load / tl.log / tl.store` 则直接对应 inner_fn 中的 `ops.load / ops.log / ops.store`。

> 完整的 Triton 和 C++ 代码生成过程，见 [阶段五：代码生成](phase5_codegen.md)。
- **CSE（公共子表达式消除）** 在打印代码行时通过缓存实现，中间变量命名为 `tmp0`, `tmp1`, ...
- 对非整除 XBLOCK 的大小，末尾元素通过 mask 屏蔽

### 4.10 Reduction 代码生成的两种模式

论文 Section 4.5 描述了 reduction kernel 的两种代码生成策略：

1. **Persistent Reduction**（小归约）：整个 reduction 维度在单个 block 内加载，数据保持在寄存器/共享内存中。直接映射到 Triton reduction 操作符
2. **Loop-Based Reduction**（大归约）：使用整个 block 作为累加器，在循环中迭代，循环结束时调用一次 Triton reduction

### 4.11 C++ 后端的两种变体

论文 Section 4.6 描述了 CPU 后端的两种代码生成变体：

1. **向量化变体**：执行 tiling，将大多数操作映射到 `at::vec::Vectorized` 类（每次处理 16 个元素，与标准 PyTorch kernel 向量化方式相同，支持多种 SIMD 指令集）
2. **非向量化变体**：生成标准 C++ 代码，使用大量 STL 函数

两种变体都使用 `#pragma omp for` 进行 OpenMP 并行化，有启发式逻辑决定并行化多少层循环。

### 4.12 Wrapper 代码生成的两种变体

论文 Section 4.7 描述了 wrapper codegen 的两种实现：

1. **Python Wrapper**：更灵活，支持一些 C++ wrapper 不支持的边界情况
2. **C++ Wrapper**：更低的开销

当启用 `mode="reduce-overhead"` 时，Inductor 使用 **CUDA Graphs** 在 CUDA driver 层面录制和重放 kernel launch，开销比 C++ wrapper 更低。CUDA Graphs 仅在安全条件满足时自动启用（动态形状、非 CUDA tensor 等情况下自动禁用）。

### 4.13 算子分解（Decomposition）示例

论文 Section 4.2 给出了一个具体的分解示例：

```python
log2_scale = 1 / math.log(2)

@register_decomposition(torch.ops.aten.log2)
def log2(x):
    return torch.log(x) * log2_scale
```

`log2` 被分解为 `log` + `mul`，这个过程递归追踪并归一化，可能触发更多分解直到不动点。分解集合不能有环。

**规模**：论文写作时，Inductor 使用了 **191 个分解**（387 含重载），**433 个 lowering**（1605 含重载）。未知算子自动转为 **fallback kernel** 运行原始 PyTorch 代码。

### 4.14 动态形状（Dynamic Shapes）

论文 Section 5 专门讨论了动态形状支持——这是 Inductor 区别于许多其他编译器的核心能力：

| 机制 | 说明 |
|------|------|
| **SymPy 符号变量** | 张量大小用 SymPy 符号表示（如 `s0`, `s1`），而非具体数值。所有索引和形状计算都在符号域进行 |
| **0/1 特化** | 如果输入大小为 0 或 1，直接视为常数并添加 guard（不做符号推理）。因为 PyTorch 代码经常测试 "size == 0"（如空张量总是 contiguous），0/1 特化大幅简化了符号推理 |
| **Size Hint** | 每个符号变量有一个 "hint"（首次 JIT 编译时的具体值），用于确定控制流分支。只选择一个分支并 guard，不需要表示条件表达式 |
| **Hint-Free（Unbacked）符号整数** | 来自数据依赖操作（如 `.nonzero()`、`.item()`）的大小变量，实际值未知。不能对其做控制流，必须 graph break |
| **Meta Functions** | 为所有算子定义 meta 函数，在输入符号形状上传播输出形状。覆盖 2657/3028 个算子 |

`sizevars.py` 中的 `SimplifyIndexing` 是动态形状推理的核心实现——在编译过程中增量简化 SymPy 表达式。

---

## 五、推荐学习路径

以下路径与 [phase1_global_view.md](phase1_global_view.md) → [phase5_codegen.md](phase5_codegen.md) 系列教程一一对应。每份教程文档都包含设计哲学、调用栈、UML 图、数据流加工、Debug 路线、交叉校验等完整教学结构。

### 阶段一：建立全局观（1-2 天）→ [phase1_global_view.md](phase1_global_view.md)
1. `compile_fx.py` → `compile_fx_inner()` 走一遍主流程
2. 搭配 `config.py` 了解可配置项
3. **教程重点**：设计哲学（Define-by-Run IR）、延迟求值、V.ops 虚拟化、8 步 Debug 路线

### 阶段二：理解 FX 优化（2-3 天）→ [phase2_fx_optimization.md](phase2_fx_optimization.md)
1. `pattern_matcher.py` → 模式匹配引擎
2. `fx_passes/pre_grad.py` → 理解 Pass 注册机制
3. 选 1-2 个具体 Pass 深入（推荐 `fuse_attention.py` 或 `efficient_conv_bn_eval.py`）
4. **教程重点**：三阶段 Pass 体系、模式匹配 DAG 构建、Attention 融合、Conv-BN 折叠

### 阶段三：理解 Lowering（3-5 天）→ [phase3_lowering.md](phase3_lowering.md)
1. `graph.py` → `GraphLowering.run_node()` 和 `call_function()`
2. `lowering.py` → 从简单算子（add, mul）到复杂算子（mm, convolution）
3. `ir.py` → 重点理解 `TensorBox`、`Buffer`、`Layout` 类层次
4. `virtualized.py` → 理解 `V.ops` 机制
5. **教程重点**：make_pointwise 闭包构造、Inlining 机制、FlexibleLayout vs FixedLayout、Reduction 多层优化

### 阶段四：理解调度与融合（3-5 天）→ [phase4_scheduling_fusion.md](phase4_scheduling_fusion.md)
1. `scheduler.py` → `Scheduler.run()` 主流程
2. `dependencies.py` → 依赖分析
3. `memory.py` → 内存规划
4. **教程重点**：贪心融合算法（can_fuse + score_fusion）、三类依赖（MemoryDep/StarDep/WeakDep）、水平/垂直/Template 融合、内存优化排序

### 阶段五：理解代码生成（5-7 天）→ [phase5_codegen.md](phase5_codegen.md)
1. `codegen/common.py` → 后端注册框架
2. `codegen/triton.py` → Triton kernel 生成（GPU 路径）
3. `codegen/cpp.py` → C++ kernel 生成（CPU 路径）
4. `codegen/wrapper.py` → Python wrapper 生成
5. `output_code.py` → 编译产物封装
6. **教程重点**：CSE 消除、TritonKernel.load/store/indexing、CppVecKernel 向量化、OpenMP 并行、MemoryPlanner 池化、CompiledFxGraph 封装

### 阶段六：高级主题（按需）
- Autotuning: `select_algorithm.py` + `autotune_process.py`
- CUDA Graph: `cudagraph_trees.py`
- 分布式: `comms.py`
- oneDNN: `mkldnn_lowerings.py`
- CUTLASS: `codegen/cutlass/`
- 动态形状: `sizevars.py`

---

## 六、文件依赖关系图（简化）

```
compile_fx.py
    ├── fx_passes/* (图优化)
    ├── graph.py :: GraphLowering
    │       ├── lowering.py (算子翻译)
    │       ├── decomposition.py (算子分解)
    │       ├── ir.py (IR 节点)
    │       └── virtualized.py (V.ops)
    ├── scheduler.py
    │       ├── dependencies.py
    │       ├── memory.py
    │       └── select_algorithm.py
    ├── codegen/*
    │       ├── common.py (框架)
    │       ├── triton.py (GPU)
    │       ├── cpp.py (CPU)
    │       ├── cutlass/ (NVIDIA GEMM)
    │       ├── rocm/ (AMD GPU)
    │       └── wrapper.py (封装)
    ├── output_code.py
    ├── codecache.py
    ├── config.py
    └── cudagraph_trees.py
```

---

## 七、按文件大小排序的核心文件 Top 20

| 排名 | 文件 | 行数 | 层次 |
|------|------|------|------|
| 1 | `ir.py` | ~10600 | IR |
| 2 | `lowering.py` | ~8300 | Lowering |
| 3 | `scheduler.py` | ~9500 | 调度 |
| 4 | `compile_fx.py` | ~3800 | 入口 |
| 5 | `config.py` | ~3200 | 配置 |
| 6 | `comms.py` | ~3200 | 分布式 |
| 7 | `graph.py` | ~3300 | Lowering |
| 8 | `select_algorithm.py` | ~6800 | Autotuning |
| 9 | `cpp_builder.py` | ~2800 | 构建 |
| 10 | `codecache.py` | ~5500 | 缓存 |
| 11 | `utils.py` | ~4600 | 工具 |
| 12 | `cudagraph_trees.py` | ~3200 | CUDA Graph |
| 13 | `pattern_matcher.py` | ~2650 | 模式匹配 |
| 14 | `codegen/triton.py` | 很大 | 代码生成 |
| 15 | `codegen/cpp.py` | 很大 | 代码生成 |
| 16 | `codegen/common.py` | 很大 | 代码生成 |
| 17 | `autotune_process.py` | ~1500 | Autotuning |
| 18 | `mkldnn_lowerings.py` | ~1700 | oneDNN |
| 19 | `dependencies.py` | ~1000 | 依赖 |
| 20 | `virtualized.py` | ~500 | 虚拟化 |
