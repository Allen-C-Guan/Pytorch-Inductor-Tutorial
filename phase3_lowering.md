# 阶段三：理解 Lowering —— 从 FX Graph 到 Inductor IR 的翻译

> **定位**：本文档深入 Inductor 编译管线中最核心的翻译层——**Lowering**。读完本文档，你应当理解：FX Graph 如何通过 GraphLowering 逐节点翻译为 Inductor IR、算子 Lowering 注册表的架构、IR 节点的类层次与延迟求值机制、V.ops 虚拟化系统如何实现 define-by-run，以及算子分解与 Lowering 的协同关系。
>
> **权威参考**：
> - PyTorch 2 论文 (ASPLOS 2024): *"PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation"* — Section 4.3 (Loop-Level IR), Section 4.2 (Decomposition)
> - TorchInductor 设计帖: [dev-discuss.pytorch.org/t/torchinductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
> - PyTorch dev-discuss: [Inductor file structure explanation](https://dev-discuss.pytorch.org/t/inductor-file-structure-explanation/1860)
>
> **源码版本**：基于 `main` 分支截取（最近变更: commit `d63aab0`, 2026-04-07），行号可能随代码演进偏移，请以实际源码为准。
>
> **系列导航**：[全景总览](inductor_overview.md) | [← 阶段一：全局观](phase1_global_view.md) | [← 阶段二：FX 优化](phase2_fx_optimization.md) | **阶段三：Lowering** | [阶段四：调度与融合 →](phase4_scheduling_fusion.md) | [阶段五：代码生成 →](phase5_codegen.md)

---

## 一、设计思想 / 设计哲学

### 1.1 为什么需要 Lowering？

在 Inductor 编译管线中，Lowering 是连接两个世界的桥梁：

```
FX Graph（Python 计算图，ATen 算子）  ──Lowering──▶  Inductor IR（Loop-Level IR，54 个原语操作）
```

**为什么不能直接从 FX Graph 生成代码？**

1. **抽象层次不匹配**：FX Graph 操作的是 ATen 算子（如 `aten.convolution`），而代码生成器需要的是循环级操作（load/store/reduction）。两者之间有一个巨大的抽象鸿沟。
2. **优化机会缺失**：FX Graph 是 eager 语义的直接映射，没有延迟求值、布局优化、融合决策等信息。Lowering 过程中构建的 IR 为后续调度和融合提供了完整的优化空间。
3. **后端无关性**：Inductor 支持 Triton（GPU）、C++/OpenMP（CPU）、CUTLASS 等多个后端。IR 层提供了后端无关的中间表示，让同一个 IR 可以翻译为不同后端的代码。

### 1.2 Lowering 的三根支柱

Lowering 不仅仅是"翻译"——它由三个紧密协作的子系统组成：

| 支柱 | 核心文件 | 职责 |
|------|---------|------|
| **GraphLowering** | `graph.py` | FX Interpreter，逐节点调度翻译，管理全局状态 |
| **算子 Lowering 注册表** | `lowering.py` | 为每个 ATen 算子注册"如何翻译为 IR"的函数 |
| **IR 节点定义** | `ir.py` | 定义所有 IR 节点类型（TensorBox/Buffer/Pointwise/Reduction 等） |

加上两个辅助系统：

| 辅助系统 | 核心文件 | 职责 |
|----------|---------|------|
| **V.ops 虚拟化** | `virtualized.py` | 实现策略模式的动态切换，让同一个 IR 函数在不同阶段有不同语义 |
| **算子分解** | `decomposition.py` | 在 Lowering 之前/期间将复杂算子拆解为更基础的组合 |

### 1.3 核心设计理念回顾：Define-by-Run IR

Lowering 产出的 IR 是 **define-by-run** 的——IR 的循环体是一个可执行的 Python 函数：

```python
# torch.log2(x) 的 Inductor IR（论文 Figure 2）
def inner_fn_buf0(index):
    i0, i1 = index
    tmp0 = ops.load("arg0_1", i0 * s1 + i1)           # 从 buffer 加载
    tmp1 = ops.log(tmp0)                                # 计算 log
    tmp2 = ops.constant(1.4426950408889634, torch.float32)
    tmp3 = ops.mul(tmp1, tmp2)                          # log(x) * (1/ln(2)) = log2(x)
    return tmp3
```

这里的 `ops.log`、`ops.mul` 并不是真正计算——它们通过 `V.ops` 虚拟化接口分发。替换不同的 handler，同一个函数就能做不同的事：代码生成、类型传播、依赖分析、索引简化。**这个设计理念是 Lowering 的灵魂。**

### 1.4 Lowering 与 Inlining 的关系

论文 Table 4 的消融实验揭示了 Lowering 阶段一个关键行为——**Inlining**：

> Inlining 发生在 **Lowering 阶段**（`graph.py` / `lowering.py`）：将 pointwise kernel 的函数体**复制**到消费者中，避免中间结果物化。

具体来说，当 `make_pointwise()` 创建一个 `Pointwise` IR 节点时，它的 `inner_fn` 是一个闭包。后续的消费者可以通过 `make_loader()` 获取这个闭包并组合到自己的 `inner_fn` 中——这就是 Inlining。只有当某个条件触发 `realize()` 时，才会物化为 `ComputedBuffer`，结束 Inlining 链。

**没有 Inlining 的后果**：消融实验显示，去掉 Inlining + Fusion 后，推理从 1.91x 加速变为 0.80x **减速**。因为分解（Decomposition）将大算子拆成了很多小算子，如果不通过 Inlining 重新粘合，每个小算子都变成独立的 kernel launch。

---

## 二、主体核心调用栈

### 2.1 Lowering 在编译管线中的位置

```
compile_fx.py:1473  GraphLowering(gm, example_inputs, ...)
    │  ← FX 优化（Phase 2）已完成，FX Graph 已标准化
    │
    ▼
compile_fx.py:1503-1508
    with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
        graph.run(*example_inputs)
        │  ← 核心入口：逐节点执行 FX Graph，翻译为 IR
        │
        ├── graph.py:1049  GraphLowering.run(*args)
        │       └── super().run(*args)  ← torch.fx.Interpreter.run()
        │           │
        │           └── [遍历所有 FX Node]
        │               │
        │               ├── placeholder 节点 → graph.py:1209  placeholder()
        │               │       └── 创建 InputBuffer → 包装为 TensorBox
        │               │
        │               ├── get_attr 节点 → graph.py:1486  get_attr()
        │               │       └── 常量处理（小常量内联、大常量注册为 buffer）
        │               │
        │               ├── call_function 节点 → graph.py:1774  run_node()
        │               │       │
        │               │       └── graph.py:1319  call_function(target, args, kwargs)
        │               │           │
        │               │           ├── 检查 lowerings[target] 是否存在
        │               │           ├── 不存在 → make_fallback() 创建 fallback kernel
        │               │           ├── 应用布局约束（layout_constraints）
        │               │           ├── 优先尝试 user_lowerings[target]
        │               │           └── 调用 lowerings[target](*args, **kwargs)
        │               │               │
        │               │               ├── [简单算子] lowering.py:668  make_pointwise()
        │               │               │       └── ir.py:1105  Pointwise.create()
        │               │               │           └── 返回 TensorBox(StorageBox(Pointwise))
        │               │               │
        │               │               ├── [归约算子] lowering.py:6644  make_reduction()
        │               │               │       └── ir.py:1255  Reduction.create()
        │               │               │           └── 返回 TensorBox(StorageBox(Reduction))
        │               │               │
        │               │               └── [复杂算子] lowering.py:2506  make_fallback()
        │               │                       └── ir.py:6181  FallbackKernel.create()
        │               │                           └── 返回包装后的 ATen 调用
        │               │
        │               └── output 节点 → graph.py:1547  output()
        │                       └── 收集 graph_outputs，realize 所有输入
        │
        ▼ graph.run() 完成
        │  此时所有 FX Node 已翻译为 IR
        │  graph.operations 和 graph.buffers 已填充
        │
        ▼ compile_fx.py:1535
        compiled_fn = graph.compile_to_module()
            │
            ├── graph.py:2546  codegen()
            │       ├── _update_scheduler()  ← 创建 Scheduler
            │       ├── scheduler.codegen()   ← IR → Triton/C++ 代码
            │       └── wrapper_code.generate() ← 生成 wrapper
            │
            └── 返回 CompiledFxGraph
```

### 2.2 call_function 核心调度栈

```
graph.py:1319  call_function(target, args, kwargs)
    │
    ├── Step 1: getitem 特殊处理 (L1320-1321)
    │   if target is operator.getitem:
    │       return super().call_function(...)
    │
    ├── Step 2: 模式匹配 Lowering (L1323-1328)
    │   if hasattr(target, "_inductor_lowering_function"):
    │       return target(*args, **kwargs)
    │
    ├── Step 3: 缺失 Lowering 处理 (L1330-1391)
    │   if target not in lowerings:
    │       ├── FALLBACK_ALLOW_LIST → make_fallback(warn=False)
    │       ├── config.implicit_fallbacks → make_fallback()
    │       └── 否则 → raise MissingOperatorWith/WithoutDecomp
    │
    ├── Step 4: 应用布局约束 (L1393-1427)
    │   if layout_constraints:
    │       args, kwargs = constrain_to_fake_tensors(args, kwargs, ...)
    │
    ├── Step 5: 执行 Lowering (L1429-1462)
    │   ├── 优先: user_lowerings[target](*args, **kwargs)
    │   └── 兜底: lowerings[target](*args, **kwargs)
    │
    └── Step 6: 传播 mutation (L1458-1462)
        if layout_constraints:
            self.propagate_mutation(n, old_args, old_kwargs, args, kwargs)
```

---

## 三、主体流程梳理

### 3.1 数据流加工：FX Graph → Inductor IR

#### 原始材料
- **输入**：经过三阶段 FX 优化（Phase 2）后的 FX GraphModule
- **形态**：Python 计算图，所有算子已标准化为 ATen 操作，前后向图已分离
- **额外信息**：example_inputs（真实张量，用于推理形状）、shape_env（符号变量环境）

#### 加工过程（GraphLowering.run()）

**加工步骤 1：输入节点翻译（placeholder）**

```
graph.py:1209  placeholder(target, args, kwargs)
│  输入：FX placeholder 节点 + example_inputs 中的示例张量
│  加工：
│  ├── 提取示例张量的形状和 stride
│  ├── 符号化：static_sizes_strides() 或 symbolic_sizes_strides()
│  ├── 创建 InputBuffer(name=target, layout=FixedLayout(...))
│  └── 包装为 TensorBox(StorageBox(InputBuffer))
│  变化：
│  ├── 从 FX Node → IR InputBuffer（命名存储单元）
│  ├── 增加：精确的符号形状信息（SymPy 变量）
│  └── 增加：固定的内存布局（FixedLayout，stride 已确定）
│  价值：输入是 FX Graph → IR 的起点，后续所有计算以此为数据源
│
▼ 产品：TensorBox 输入变量，存入 graph_inputs 字典
```

**加工步骤 2：常量节点翻译（get_attr）**

```
graph.py:1486  get_attr(target, args, kwargs)
│  输入：FX get_attr 节点（模型参数、常量）
│  加工路径分叉：
│  ├── 小常量（shape == (1,) and size <= 8）→ 内联为 Constant
│  │   变化：直接在 IR 中表示为字面值
│  │   价值：避免为小常量分配 buffer
│  │
│  └── 大常量 → 注册为 ConstantBuffer 或 ShapeAsConstantBuffer
│      变化：创建命名的 buffer 存储
│      价值：大常量需要独立的内存分配
│
▼ 产品：Constant / TensorBox(ConstantBuffer)
```

**加工步骤 3：计算节点翻译（call_function）—— Lowering 的核心**

```
graph.py:1319  call_function(target, args, kwargs)
│  输入：FX call_function 节点（ATen 算子 + 已翻译为 IR 的参数）
│  加工路径根据算子类型分叉：
│
│  ├── [路径 A] 简单逐元素算子（add, mul, relu, ...）
│  │   lowerings[target] → make_pointwise(ops_wrapper(target_name))
│  │   │
│  │   ├── 为每个输入创建 loader 函数（从 buffer 加载数据）
│  │   ├── 构造 inner_fn = lambda idx: fn(load_a(idx), load_b(idx))
│  │   ├── 处理广播（broadcast）和类型提升（type promotion）
│  │   └── 返回 Pointwise.create(device, dtype, inner_fn, ranges)
│  │       └── TensorBox(StorageBox(Pointwise(...)))
│  │
│  │   变化：
│  │   ├── 从 ATen 算子 → define-by-run 的 inner_fn 闭包
│  │   ├── 增加：延迟求值能力（不立即执行）
│  │   └── 增加：可组合的函数体（支持 Inlining）
│  │   价值：pointwise 操作可以通过闭包组合实现 Inlining 优化
│  │
│  ├── [路径 B] 归约算子（sum, max, argmax, ...）
│  │   lowerings[target] → make_reduction(reduction_type)
│  │   │
│  │   ├── 确定归约维度和类型
│  │   ├── 构造归约 inner_fn
│  │   ├── Reduction.create() 处理多种优化路径：
│  │   │   ├── reduction_numel == 0 → Pointwise（常量）
│  │   │   ├── reduction_numel == 1 → Pointwise（无归约）
│  │   │   ├── 小归约 → Pointwise（展开归约）
│  │   │   └── 大归约 → Reduction（可能分多层）
│  │   └── 立即 realize()（归约必须物化）
│  │
│  │   变化：
│  │   ├── 从归约 ATen 算子 → Reduction IR 节点
│  │   ├── 增加：分块策略（多层归约优化）
│  │   └── 增加：归约提示（ReductionHint）
│  │   价值：归约是性能关键操作，需要专门的 IR 表示和优化
│  │
│  └── [路径 C] 复杂算子（mm, convolution, ...）
│      make_fallback(op) → FallbackKernel.create(op, *args)
│      │
│      ├── 确保 realize 所有输入（冻结布局）
│      ├── 创建 FallbackKernel IR 节点
│      └── 运行时调用原始 ATen 实现（cuBLAS/cuDNN 等）
│
│      变化：
│      ├── 从 ATen 算子 → 外部 kernel 调用封装
│      ├── 增加：布局约束（要求 contiguous 等）
│      └── 增加：autotuning 候选（如适用）
│      价值：复杂算子由高度优化的库实现，不需要重新生成 kernel
│
▼ 产品：TensorBox 包装的 IR 节点（Pointwise / Reduction / FallbackKernel）
```

**加工步骤 4：后处理与 Inlining 决策（run_node）**

```
graph.py:1774  run_node(n)
│  输入：FX Node + call_function 的返回值
│  加工：
│  ├── 布局优化（channels-last 决策）
│  │   对 4D 中间结果，如果消费者偏好 channels-last → 冻结布局
│  │
│  ├── 输出 stride 匹配
│  │   对用户可见输出，确保 stride 与 eager 一致
│  │
│  ├── Realize 提示
│  │   ├── 多用户 + 某些用户需要 realized → realize()
│  │   ├── 输出节点 → realize()
│  │   ├── 过多读取 → realize()
│  │   └── inner_fn 过大（>100 操作）→ realize()（防 RecursionError）
│  │
│  └── 流边界处理
│      跨 CUDA stream 时强制 realize（防跨 stream Inlining）
│
│  变化：
│  ├── 增加：布局优化决策（channels-last / contiguous）
│  ├── 增加：Inlining 边界决策（哪些节点需要物化）
│  └── 结构：从纯计算图 → 带优化信息的 IR DAG
│  价值：Inlining 决策直接影响后续融合效率
│
▼ 产品：带优化信息的 IR 节点
```

**加工步骤 5：输出收集与收尾（output + finalize）**

```
graph.py:1547  output() + graph.py:1655  finalize()
│  输入：所有节点翻译完成后的环境
│  加工：
│  ├── 收集 graph_outputs（所有输出值）
│  ├── realize 所有 extern kernel 输入
│  ├── stride 匹配（确保输出 stride 正确）
│  ├── 处理 mutated inputs（in-place 操作的副作用传播）
│  └── finalize()：对所有 buffer 调用 decide_layout()
│
│  变化：
│  ├── 确定：所有 buffer 的最终布局（FlexibleLayout → FixedLayout）
│  ├── 确定：完整的 IR DAG（所有依赖关系已建立）
│  └── 确定：操作列表和缓冲区列表已就绪
│  价值：准备送入调度与融合阶段
│
▼ 最终产品：完整的 Inductor IR 图
```

#### 输出产品
- **Inductor IR 图**：包含 TensorBox/Buffer/Pointwise/Reduction/ExternKernel 的计算 DAG
- **操作列表**：`graph.operations` — 所有 IR 操作（ComputedBuffer / ExternKernel）
- **缓冲区列表**：`graph.buffers` — 所有命名存储单元
- **新增功能**：
  - 可进行融合、内存规划的底层表示
  - 支持延迟求值和动态形状
  - 提供完整的依赖信息和布局决策

**数据流示意**：

```
优化后的 FX Graph（来自 Phase 2）
│  形态：Python 计算图，ATen 算子，已标准化
│
▼ placeholder 翻译
输入 TensorBox（InputBuffer 包装）
│  变化：FX placeholder → IR InputBuffer + 符号形状
│
▼ get_attr 翻译
常量 TensorBox（Constant / ConstantBuffer）
│  变化：模型参数 → IR 常量节点
│
▼ call_function 翻译（逐节点）
中间 TensorBox（Pointwise / Reduction / FallbackKernel）
│  变化：ATen 算子 → define-by-run IR 节点
│  增加：延迟求值、Inlinable 函数体
│
▼ run_node 后处理
带优化信息的 IR 节点
│  变化：增加布局决策、Inlining 边界
│
▼ output + finalize
完整 IR DAG
│  变化：所有 buffer 布局已确定，DAG 完整
│
▼ 送入下一道工序：调度与融合（Phase 4）
```

---

## 四、UML 图 / 架构设计

### 4.1 Lowering 系统总架构

```
┌─────────────────────────────────────────────────────────────────┐
│                FX Graph（经 Phase 2 优化后）                      │
│                call_function(aten.add, [a, b], {})               │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│             GraphLowering (graph.py:356)                          │
│             FX Interpreter + 全局状态管理器                        │
├──────────────────────────────────────────────────────────────────┤
│  placeholder()  → 创建 InputBuffer                              │
│  get_attr()     → 创建 Constant / ConstantBuffer                │
│  call_function() → 查表 + 调用 lowering 函数                     │
│  run_node()     → 后处理（布局优化、Inlining 决策、realize 提示） │
│  output()       → 收集输出 + stride 匹配                        │
│  finalize()     → 决定所有 buffer 布局                           │
├──────────────────────────────────────────────────────────────────┤
│  全局状态：                                                       │
│  ├── graph_inputs: dict[str, TensorBox]                         │
│  ├── graph_outputs: list[IRNode]                                │
│  ├── buffers: list[Buffer]                                      │
│  ├── operations: list[Operation]                                │
│  ├── constants: dict[str, Tensor]                               │
│  └── sizevars: SizeVarAllocator（符号变量分配器）                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │ call_function 查表
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│             Lowering 注册表 (lowering.py:115-117)                 │
├──────────────────────────────────────────────────────────────────┤
│  lowerings: dict[OpOverload, Callable]    ← 主注册表            │
│  user_lowerings: dict[OpOverload, Callable] ← 用户扩展点       │
│  fallbacks: set[OpOverload]               ← fallback 算子集合  │
├──────────────────────────────────────────────────────────────────┤
│  注册方式：                                                       │
│  ├── @register_lowering([aten.X]) def lower_X(...) → IR         │
│  ├── register_pointwise(aten.X) → make_pointwise(ops_wrapper()) │
│  ├── make_reduction("sum") → Reduction.create()                 │
│  └── make_fallback(aten.X) → FallbackKernel.create()            │
└──────────────────────┬──────────────────────────────────────────┘
                       │ 返回 IR 节点
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│             IR 节点层次 (ir.py)                                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  IRNode (L548) ── 抽象基类                                      │
│  ├── 延迟求值包装：                                              │
│  │   ├── TensorBox (L9412) ── 张量容器                          │
│  │   │   └── data = StorageBox                                   │
│  │   └── StorageBox (L9427) ── 存储容器                         │
│  │       └── data = Buffer / Loops / View                       │
│  │           └── realize() → ComputedBuffer                     │
│  │                                                               │
│  ├── Buffer (L4555) ── 命名存储单元                             │
│  │   ├── InputBuffer (L4708) ── 输入缓冲区                      │
│  │   ├── ComputedBuffer (L4781) ── 计算产生的缓冲区              │
│  │   └── TemplateBuffer (L5230) ── 模板 kernel 缓冲区           │
│  │                                                               │
│  ├── Loops (L960) ── 循环计算基类                                │
│  │   ├── Pointwise (L1105) ── 逐元素计算                        │
│  │   │   └── Scatter (L1146) ── scatter 操作                   │
│  │   ├── Reduction (L1255) ── 归约计算                          │
│  │   ├── Scan ── 扫描计算                                       │
│  │   └── Sort ── 排序计算                                       │
│  │                                                               │
│  ├── BaseView (L2878) ── 零拷贝视图                              │
│  │   ├── ExpandView (L2976) ── 广播扩展                         │
│  │   ├── PermuteView (L3071) ── 维度重排                        │
│  │   ├── SqueezeView (L3120) ── 维度压缩                        │
│  │   └── View/GenericView (L3189) ── reshape/view              │
│  │                                                               │
│  ├── ExternKernel (L6181) ── 外部库调用                         │
│  │   └── FallbackKernel ── ATen fallback                       │
│  │                                                               │
│  └── Layout (L3846) ── 内存布局                                  │
│      ├── FixedLayout (L4137) ── 固定布局（不可变）               │
│      └── FlexibleLayout (L4145) ── 可优化布局（冻结前可调）      │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 Lowering 注册与执行 UML

```
                    ┌────────────────────────────────┐
                    │     FX Graph Node               │
                    │  target = aten.add.Tensor       │
                    │  args = [TensorBox_a, TensorBox_b] │
                    └───────────┬────────────────────┘
                                │
                    GraphLowering.call_function()
                                │
                    ┌───────────▼────────────────────┐
                    │   查找 lowerings[target]         │
                    │                                  │
                    │   优先级：                        │
                    │   1. user_lowerings[target]      │
                    │   2. lowerings[target]           │
                    │   3. make_fallback(target)       │
                    └───────────┬────────────────────┘
                                │
                ┌───────────────┼───────────────────┐
                │               │                   │
     ┌──────────▼─────┐ ┌──────▼──────────┐ ┌─────▼───────────┐
     │ make_pointwise │ │ make_reduction  │ │ make_fallback   │
     │ (lowering.py   │ │ (lowering.py    │ │ (lowering.py    │
     │  :668)         │ │  :6644)         │ │  :2506)         │
     ├────────────────┤ ├─────────────────┤ ├─────────────────┤
     │ 创建 loader    │ │ 创建归约函数    │ │ 创建            │
     │ 构造 inner_fn  │ │ Reduction.create│ │ FallbackKernel  │
     │ Pointwise      │ │ + realize()     │ │ + 布局约束      │
     │ .create()      │ │                 │ │                 │
     └───────┬────────┘ └───────┬─────────┘ └───────┬─────────┘
             │                  │                   │
             ▼                  ▼                   ▼
     ┌──────────────────────────────────────────────────────┐
     │                   TensorBox                           │
     │         (延迟求值包装，统一返回类型)                    │
     │                                                       │
     │   TensorBox(StorageBox(Pointwise))                    │
     │   TensorBox(StorageBox(ComputedBuffer(Reduction)))    │
     │   TensorBox(StorageBox(TemplateBuffer))               │
     └──────────────────────────────────────────────────────┘
```

### 4.3 延迟求值包装层次

```
┌─────────────────────────────────────────────────────────────────┐
│  TensorBox (ir.py:9412)                                         │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  StorageBox (ir.py:9427)                                  │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │  Pointwise / Reduction / Buffer / View               │ │  │
│  │  │                                                      │ │  │
│  │  │  Pointwise:                                          │ │  │
│  │  │  ├── device: torch.device                            │ │  │
│  │  │  ├── dtype: torch.dtype                              │ │  │
│  │  │  ├── inner_fn: Callable → 计算 closure               │ │  │
│  │  │  └── ranges: Sequence[Expr] → 符号形状               │ │  │
│  │  │                                                      │ │  │
│  │  │  ComputedBuffer (realize() 产物):                     │ │  │
│  │  │  ├── name: str ("buf0", "buf1", ...)                 │ │  │
│  │  │  ├── layout: FlexibleLayout → FixedLayout            │ │  │
│  │  │  └── data: Loops (Pointwise / Reduction)             │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘

物化链：
TensorBox.data = StorageBox          ← 延迟求值容器
StorageBox.data = Pointwise          ← 未物化的计算描述
StorageBox.realize()                 ← 触发物化
  → StorageBox.data = ComputedBuffer ← 物化后的命名存储
  → ComputedBuffer 注册到 graph.buffers 和 graph.operations
```

### 4.4 V.ops 虚拟化系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│              V (virtualized.py:364)                               │
│              统一访问接口                                         │
├─────────────────────────────────────────────────────────────────┤
│  V.ops      → _ops._get_handler()   ← 当前操作处理器           │
│  V.graph    → _graph._get_handler()  ← 当前 GraphLowering      │
│  V.kernel   → _kernel._get_handler() ← 当前 Kernel 对象        │
│                                                                  │
│  V.set_ops_handler(handler)    → 替换 ops，返回 context manager │
│  V.set_graph_handler(graph)    → 替换 graph，返回 context mgr   │
└─────────────────────────────────────────────────────────────────┘

                    Virtualized<T> (L112)
                    基于 threading.local() 的动态作用域
                    ┌─────────────────────────────┐
                    │ _set_handler(value)          │
                    │   prior = _get_handler()     │
                    │   setattr(threadlocal, key,  │
                    │            value)             │
                    │   return context_manager(    │
                    │       restore=prior)          │
                    │                              │
                    │ _get_handler()                │
                    │   return getattr(threadlocal, │
                    │       key, default)           │
                    └─────────────────────────────┘

                    使用场景：

         编译时      V.set_graph_handler(graph)
         (graph.py)  → 所有内部代码通过 V.graph 访问当前图

         Lowering时  V.ops 默认为 MockHandler
         (lowering.py) → ops.add(a, b) 返回字符串 "ops.add(a, b)"
                         但 inner_fn 闭包被保存，在代码生成时才执行

         代码生成时  V.set_ops_handler(KernelFormatterHandler)
         (codegen/)  → ops.add(a, b) 生成 "tmp3 = tmp1 * tmp2" 代码

         分析时      V.set_ops_handler(RecordLoadStore)
         (dependencies.py) → ops.load("buf0", idx) 记录依赖关系

         类型传播时  V.set_ops_handler(DtypePropagationOpsHandler)
         (dtype_propagation.py) → ops.add(a, b) 推断结果 dtype
```

---

## 五、关键思想代码讲解

### 5.1 make_pointwise —— 最常用的 Lowering 模式

**文件**：[lowering.py:668-770](torch/_inductor/lowering.py#L668)

超过一半的 ATen 算子通过 `make_pointwise` 翻译为 IR。这是理解 Lowering 的最佳起点。

```python
def make_pointwise(fn, override_return_dtype=None, ...):
    """Wraps a pointwise fn and returns a function representing
    the pointwise in the define-by-run IR."""

    def inner(*inputs: TensorBox, alpha=None):
        # 1. 常量提升：将标量常量包装为 IR 节点
        inputs = promote_constants(inputs, override_return_dtype)

        # 2. 为每个输入创建 loader 函数
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()  # 输出形状 = 输入形状
        dtype = override_return_dtype or inputs[0].get_dtype()

        # 3. 构造 inner_fn —— define-by-run 的核心
        def inner_fn(index):
            assert len(index) == len(ranges)
            # 每个 loader 从对应的输入 buffer 加载数据
            # fn 是 ops_wrapper(name) → 调用 ops.add / ops.mul 等
            return fn(*[load(index) for load in loaders])

        # 4. 创建 Pointwise IR 节点
        return Pointwise.create(
            device=device,
            dtype=dtype,
            inner_fn=inner_fn,  # ← 闭包：延迟求值的核心
            ranges=ranges,       # ← 符号形状
        )

    return inner
```

**关键设计洞察**：

1. **`inner_fn` 是闭包，不是值**：它描述"在某个索引处计算什么"，而非"计算结果是什么"。这就是 define-by-run。
2. **`loaders` 支持 Inlining**：`make_loader()` 返回的是另一个 `inner_fn`。当消费者调用 `x.make_loader()` 时，它获取的是生产者的 `inner_fn`，可以直接组合到自己的 `inner_fn` 中——无需物化中间结果。
3. **`fn` 通过 V.ops 分发**：`ops_wrapper("add")` 返回 `lambda *args: ops.add(*args)`。`ops.add` 的行为取决于当前安装的 V.ops handler——代码生成时生成代码字符串，分析时记录依赖。

**具体示例——add 算子的 Lowering**：

```python
# lowering.py:7279
add = register_pointwise(
    aten.add,
    allow_alpha=True,           # 支持 alpha 参数：add(a, b, alpha=2)
    use_fma_for_alpha=True,     # CUDA 上使用 FMA 指令
    override_fn_when_input_bool="logical_or",  # bool + bool → logical_or
)
```

`register_pointwise`（lowering.py:997）内部调用链：

```
register_pointwise(aten.add)
    → ops_wrapper("add")       → lambda *args: ops.add(*args)
    → make_pointwise(fn)        → 创建 inner 函数
    → register_lowering(aten.add)(inner)  → 注册到 lowerings 字典
```

### 5.2 make_reduction —— 归约算子的 Lowering

**文件**：[lowering.py:6644-6671](torch/_inductor/lowering.py#L6644)

```python
def make_reduction(reduction_type: ReductionType, override_return_dtype=None):
    def inner(x, axis=None, keepdims=False, *, dtype=None):
        kwargs = _make_reduction_inner(
            x, axis=axis, keepdims=keepdims,
            dtype=dtype, override_return_dtype=override_return_dtype,
            reduction_type=reduction_type,
        )
        # Reduction.create() 有多种优化路径
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)

        # 归约操作必须立即 realize
        if isinstance(result.data.data, Reduction):
            result.realize()

        return result
    return inner
```

**Reduction.create() 的优化决策**（ir.py:1541-1740）：

```python
@classmethod
def create(cls, device, dst_dtype, src_dtype, inner_fn,
           ranges, reduction_ranges, reduction_type, ...):
    reduction_numel = product(reduction_ranges)

    if reduction_numel == 0:
        # 空归约 → 返回常量（如 sum([]) = 0）
        return Pointwise.create(device, dst_dtype, lambda idx: zero_constant, ranges)

    if reduction_numel == 1:
        # 单元素归约 → 退化 pointwise
        return Pointwise.create(device, dst_dtype, inner_fn_modified, ranges)

    if reduction_numel <= threshold:
        # 小归约 → 展开为 pointwise（更快，因为无归约 kernel 开销）
        return Pointwise.create(device, dst_dtype, unrolled_reduction_fn, ranges)

    # 大归约 → 标准 Reduction IR
    # 可能分多层：第一层归约到中间 buffer，第二层归约到最终结果
    return TensorBox.create(Reduction(device, dst_dtype, inner_fn,
                                       ranges, reduction_ranges, reduction_type))
```

**为什么归约必须 realize？**

归约改变了张量的形状（如 `[B, N, D] → [B, D]`），这意味着它的输出需要一个新的、更小的 buffer。与 pointwise 不同，归约不能简单地通过闭包组合——它需要独立的存储空间。因此 `result.realize()` 将其物化为 `ComputedBuffer`。

### 5.3 make_fallback —— 复杂算子的 Lowering

**文件**：[lowering.py:2506-2561](torch/_inductor/lowering.py#L2506)

```python
def make_fallback(op, layout_constraint=None, warn=True, override_decomp=False, ...):
    # 关键检查：不能同时有分解和 fallback
    check_decomps = get_decomp_fn() if get_decomp_fn is not None else decompositions
    assert op not in check_decomps or override_decomp, (
        f"both a fallback and a decomp for same op: {op}"
    )

    def register_fallback(op_overload):
        # 1. 确保 fallback 算子的输入已 realize（布局已确定）
        add_needs_realized_inputs(op_overload)

        # 2. 可选的布局约束（如 require_contiguous）
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)

        # 3. 注册为 lowering（使用 fallback_handler）
        return register_lowering(op_overload, type_promotion_kind=None)(
            fallback_handler(op_overload)
        )

    # 处理 OpOverloadPacket（多个重载）或单个 OpOverload
    if isinstance(op, torch._ops.OpOverloadPacket):
        for ol in op.overloads():
            register_fallback(getattr(op, ol))
    ...
```

**fallback_handler 的实现**：

```python
def fallback_handler(kernel, add_to_fallback_set=True):
    def handler(*args, **kwargs):
        def wrap_tensors(x):
            return x.wrap_for_lowering() if isinstance(x, ir.IRNode) else x

        # FallbackKernel.create() 创建外部 kernel 调用
        return pytree.tree_map(
            wrap_tensors, ir.FallbackKernel.create(kernel, *args, **kwargs)
        )

    handler._is_fallback_handler = True
    return handler
```

**FallbackKernel**（ir.py）封装了对原始 ATen 算子的调用。在运行时，它直接调用 cuBLAS、cuDNN 等高度优化的库实现。

**典型外部 kernel 算子**：
- `aten.mm` / `aten.addmm` — 矩阵乘法。在 GPU 上可能走 Triton mm template 或 cuBLAS，在 CPU 上走 oneDNN；当无专用 template 时才 fallback 到 `FallbackKernel`
- `aten.convolution` — 卷积（cuDNN / oneDNN）
- `aten.scaled_dot_product_attention` — SDPA（Flash Attention / Memory Efficient Attention）

### 5.4 ops_wrapper —— V.ops 的字符串桥接

**文件**：[ir.py](torch/_inductor/ir.py)（定义）

```python
def ops_wrapper(name: str) -> Callable[..., OpsValue]:
    assert isinstance(name, str), type(name)
    def fn(*args: object, **kwargs: object) -> OpsValue:
        return getattr(ops, name)(*args, **kwargs)
    return fn
```

**为什么需要字符串桥接？**

```python
# 在 lowering.py 中：
fn = ops_wrapper("add")       # 返回 lambda *args: ops.add(*args)
result = make_pointwise(fn)   # fn 被嵌入 inner_fn 闭包

# inner_fn 在不同阶段有不同行为：
# 编译时（默认 MockHandler）：ops.add(a, b) → "ops.add(a, b)" 字符串
# 代码生成时：ops.add(a, b) → "tmp3 = tmp1 + tmp2" Triton/C++ 代码
# 分析时：ops.add(a, b) → 记录 dtype、依赖等
```

`ops` 是从 `virtualized.py` 导入的全局变量——它实际指向 `_ops._get_handler()`，返回当前安装的 handler。

---

## 六、关键源码讲解

### 6.1 GraphLowering.call_function —— Lowering 的调度枢纽

**文件**：[graph.py:1319-1476](torch/_inductor/graph.py#L1319)

```python
def call_function(self, target, args, kwargs):
    # Step 1: getitem 特殊处理——直接返回 Python 原生操作
    if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
        return super().call_function(target, args, kwargs)

    # Step 2: 模式匹配注册的 passthrough lowering
    if hasattr(target, "_inductor_lowering_function"):
        return target(*args, **kwargs)

    # Step 3: 缺失 lowering → 创建 fallback 或报错
    if target not in lowerings:
        assert isinstance(target, torch._ops.OpOverload)
        base_name = target.name().split(".")[0]

        if base_name in FALLBACK_ALLOW_LIST:
            make_fallback(target, warn=False, get_decomp_fn=self.get_decomp_fn,
                          override_decomp=True)
        elif config.implicit_fallbacks:
            # 决定布局约束
            tag = get_layout_constraint_tag(target, with_default=False)
            decided_constraint = tag_to_layout_constraint(default_tag)
            make_fallback(target, layout_constraint=decided_constraint,
                          get_decomp_fn=self.get_decomp_fn)
        elif get_decompositions([target]):
            raise MissingOperatorWithDecomp(target, args, kwargs)
        else:
            raise MissingOperatorWithoutDecomp(target, args, kwargs)

    # Step 4: 应用布局约束（如 require_contiguous）
    layout_constraints = maybe_layout_constraints(target)
    if layout_constraints:
        args, kwargs = constrain_to_fake_tensors(args, kwargs, fake_args, fake_kwargs)

    # Step 5: 执行 lowering —— 优先级：user_lowerings > lowerings
    try:
        out = None
        if target in user_lowerings and target not in V.active_user_lowering_ops:
            out = user_lowerings[target](*args, **kwargs)

        if out is None:
            if target in lowerings:
                out = lowerings[target](*args, **kwargs)
            else:
                out = fallback_handler(target, add_to_fallback_set=False)(*args, **kwargs)
    except Exception as e:
        raise LoweringException(e, target, args, kwargs) from None

    # Step 6: 传播 mutation（如果布局约束创建了副本）
    if layout_constraints:
        self.propagate_mutation(n, old_args, old_kwargs, args, kwargs)

    return out
```

**设计要点**：

1. **三层优先级**：`user_lowerings` → `lowerings` → `fallback_handler`。用户可以通过 `register_lowering` 扩展或覆盖默认行为。
2. **缺失算子策略**：不是所有 ATen 算子都有显式 lowering。`FALLBACK_ALLOW_LIST` 中的算子自动转为 fallback；`config.implicit_fallbacks` 允许更宽松的策略。
3. **布局约束**：某些算子要求输入是 contiguous 或特定布局。`constrain_to_fake_tensors()` 通过比较 FakeTensor 的形状/stride 来确保 IR 节点满足约束。

### 6.2 GraphLowering.run_node —— Inlining 决策中心

**文件**：[graph.py:1774-2186](torch/_inductor/graph.py#L1774)

`run_node()` 在 `call_function()` 返回后执行后处理，是 Inlining 决策的核心。

```python
def run_node(self, n):
    # 前置处理...
    result = super().run_node(n)  # 调用 call_function / placeholder 等

    # ── 关键后处理 ──

    # 1. Channels-last 布局优化
    if isinstance(result, TensorBox) and len(result.get_size()) == 4:
        # 检查下游消费者是否偏好 channels-last
        if downstream_prefers_channels_last(n):
            result.freeze_layout_with_fill_order(
                order=(0, 3, 1, 2)  # NCHW → channels-last 填充顺序
            )

    # 2. 输出 stride 匹配
    if is_output_or_as_strided_input(n):
        strides = n.meta["val"].stride()
        result = ir.ExternKernel.require_exact_strides(result, strides)

    # 3. Realize 提示——决定 Inlining 边界
    if isinstance(result, TensorBox):
        _data = result.data

        # 3a. 多用户场景：某些用户需要 realized 输入
        if _data.has_exceeded_max_reads():
            result.realize()

        # 3b. 输出节点：必须 realize
        if is_output_node(n):
            result.realize()

        # 3c. Buffer 重用提示
        if isinstance(_data, StorageBox) and _data.should_realize_on_reuse(len(n.users)):
            result.realize()

        # 3d. inner_fn 过大 → 强制 realize（防 RecursionError）
        if isinstance(_data.data, Pointwise) and _data.data.has_large_inner_fn(threshold=100):
            result.realize()

    # 4. 标记 origin node（用于调试和错误追踪）
    if isinstance(result, ir.IRNode):
        result.origin_node = n

    return result
```

**Inlining 决策总结**：

| 条件 | 行为 | 原因 |
|------|------|------|
| 多用户 + 超过读取阈值 | realize | 避免重复计算 |
| 输出节点 | realize | 输出需要固定布局 |
| 用户需要 realized 输入 | realize | ExternKernel 需要连续布局 |
| inner_fn 过大（>100 ops） | realize | 防止 RecursionError |
| 流边界不同 | realize | 防止跨 stream Inlining |
| **其他情况** | **不 realize** | **保持 Inlining，允许后续融合** |

### 6.3 StorageBox.realize() —— 延迟求值的物化

**文件**：[ir.py:9443-9470](torch/_inductor/ir.py#L9443)

> 以下为简化展示，实际实现中注册逻辑在 `ComputedBuffer` 构造过程中完成，行号仅作参考。

```python
def realize(self) -> str | None:
    # 已物化的节点直接返回名称
    if IRNode.is_realized_node(self.data):
        return self.data.get_name()

    # 只能物化 Loops 子类（Pointwise / Reduction / Scan / Sort）
    assert isinstance(self.data, (Pointwise, Reduction, Scan, Sort))

    # 包装为 ComputedBuffer——这是物化的核心步骤
    self.data = ComputedBuffer(
        name=None,
        layout=FlexibleLayout(
            device=self.data.get_device(),
            dtype=self.data.get_dtype(),
            size=self.data.get_size(),
        ),
        data=self.data,  # 原始的 Pointwise / Reduction 保留在 data 字段中
    )

    # 注册到 GraphLowering 的全局状态
    self.data.name = V.graph.register_buffer(self.data)
    V.graph.register_operation(self.data)
    return self.data.name
```

**物化的含义**：

1. **命名**：`ComputedBuffer` 获得一个唯一名称（如 `"buf5"`），对应最终生成代码中的一个变量。
2. **布局决策**：创建 `FlexibleLayout`，后续可以被调度器优化（选择 contiguous / channels-last 等）。
3. **注册**：buffer 和 operation 分别注册到 `graph.buffers` 和 `graph.operations`，进入全局调度。

**未物化 vs 已物化的区别**：

```
未物化（可 Inlining）：
  TensorBox(StorageBox(Pointwise(inner_fn, ranges)))
  → inner_fn 是闭包，可以被消费者组合
  → 没有命名 buffer，没有独立的存储

已物化（结束 Inlining）：
  TensorBox(StorageBox(ComputedBuffer(name="buf5", layout=FlexibleLayout, data=Pointwise(...))))
  → 有命名 buffer，有独立存储
  → Pointwise 保留在 data 字段，用于代码生成
```

### 6.4 GraphLowering.placeholder —— 输入节点翻译

**文件**：[graph.py:1209-1316](torch/_inductor/graph.py#L1209)

```python
def placeholder(self, target, args, kwargs):
    self.placeholder_idx += 1
    example = super().placeholder(target, args, kwargs)

    # SymType（符号整数/浮点数）→ 直接转为 SymPy 表达式
    if isinstance(example, SymTypes):
        expr = example.node.expr  # SymPy 表达式
        self.graph_inputs[target] = expr
        self.graph_input_names.append(target)
        return expr

    # 标量（int/bool/float）→ SymPy 常量
    elif isinstance(example, (int, bool, float)):
        self.graph_inputs[target] = sympy.sympify(example)
        self.graph_input_names.append(target)
        return sympy.sympify(example)

    # 张量 → 创建 InputBuffer + TensorBox 包装
    assert isinstance(example, torch.Tensor)
    if not example._has_symbolic_sizes_strides:
        sizes, strides = self.static_sizes_strides(example)  # 固定形状
    else:
        sizes, strides = self.symbolic_sizes_strides(example)  # 符号形状

    # 区分普通输入和捐赠的 backward 输入
    if self.is_backward and self.placeholder_idx in self.bw_donated_idxs:
        tensor = TensorBox.create(DonatedBuffer(
            name=target,
            layout=FixedLayout(example.device, example.dtype, sizes, strides),
        ))
    else:
        tensor = TensorBox.create(InputBuffer(
            name=target,
            layout=FixedLayout(example.device, example.dtype, sizes, strides),
        ))

    self.graph_inputs[target] = tensor
    self.graph_input_names.append(target)
    self.graph_inputs_original[target] = tensor.data.data
    return tensor
```

**设计要点**：
- 输入使用 `FixedLayout`（布局不可变，因为是外部提供的张量）
- 符号形状通过 `symbolic_sizes_strides()` 提取 SymPy 变量（如 `s0`, `s1`）
- `DonatedBuffer` 用于 backward 图中可复用的输入（in-place 优化）

### 6.5 分解表的选择与传递

**文件**：[decomposition.py:964-973](torch/_inductor/decomposition.py#L964)

```python
@functools.cache
def fast_random_decomps():
    return {**decompositions, **extra_random_decomps}

def select_decomp_table():
    """decomps can change based on config"""
    if config.fallback_random:
        return decompositions          # 不包含随机分解
    if config.fallback_embedding_bag_byte_unpack:
        decompositions.pop(torch.ops.quantized.embedding_bag_byte_unpack.default, None)
        return decompositions
    result = fast_random_decomps()     # 默认：基础 + 随机分解
    return result
```

**分解与 Lowering 的互斥关系**（lowering.py:2513-2542）：

```python
def make_fallback(op, ...):
    check_decomps = get_decomp_fn() if get_decomp_fn is not None else decompositions
    # 关键断言：同一个算子不能同时有分解和 fallback
    assert op not in check_decomps or override_decomp, (
        f"both a fallback and a decomp for same op: {op}"
    )
```

**设计哲学**：分解 > 自定义 Lowering > Fallback。如果算子有分解规则，优先使用分解（让分解后的基础算子走标准 lowering），除非显式 `override_decomp=True`。

### 6.6 具体算子 Lowering 示例

**简单算子——mul**（lowering.py:7001-7008）：

```python
@register_lowering([aten.mul], broadcast=True)
def mul(a, b):
    both_bool = is_boolean_type(a) and is_boolean_type(b)
    if both_bool:
        return logical_and(a, b)
    else:
        fn = ops_wrapper(aten.mul.__name__)  # → ops.mul
        return make_pointwise(fn)(a, b)
```

**归约算子——sum**（lowering.py:7096-7104）：

```python
@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    if (is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())) and dtype is None:
        dtype = torch.int64
    fn = make_reduction("sum", override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)
```

**复杂算子——avg_pool2d**（lowering.py:6688-6703, 6036-6135）：

```python
@register_lowering(aten.avg_pool2d, type_promotion_kind=None)
def avg_pool2d(x, kernel_size, stride=(), padding=0, ceil_mode=False, ...):
    return _avg_poolnd(x, kernel_size, stride, padding, ceil_mode, ...)

def _avg_poolnd(x, kernel_size, stride, padding, ..., dim):
    # ... 参数归一化 ...
    def fn_inner(idx, reduction_idx):
        # 将输出索引映射回输入索引（考虑 stride 和 padding）
        prefix = idx[:-dim]
        bh = idx[-dim:]
        ih = reduction_idx
        ih = [bh[i] * stride[i] + ih[i] - padding[i] for i in range(dim)]
        return x_loader([*prefix, *ih])

    # 创建 Reduction IR 节点
    rv = Reduction.create(
        reduction_type="sum",
        input_node=x,
        device=device,
        dst_dtype=output_dtype,
        src_dtype=dtype,
        inner_fn=fn_inner,          # ← 自定义的索引映射
        ranges=new_size,             # ← 输出形状
        reduction_ranges=kernel_size, # ← 归约维度
    )
    # ... 缩放为平均值 ...
    return rv
```

**条件分解——addmm**（decomposition.py:378-415）：

```python
@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(self, mat1, mat2, out_dtype=None, beta=1, alpha=1):
    if mat1.device.type not in ["cpu", "mps"]:
        if (statically_known_true(mat1.size(-1) == 1)
            and statically_known_true(mat1.size(0) != 1)
            and statically_known_true(mat2.size(1) != 1)):
            # 小矩阵：分解为逐元素操作更快
            counters["inductor"]["decompose_addmm"] += 1
            out = mat1 * mat2
            return alpha * out + beta * self
    # 条件不满足 → 不分解，保留原始算子走 fallback
    return NotImplemented
```

---

## 七、核心技术

### 7.1 延迟求值与 Inlining 的实现机制

延迟求值是 Inductor 性能优化的基石。具体实现通过三层包装：

```
TensorBox ──wraps──▶ StorageBox ──wraps──▶ Pointwise / Reduction / Buffer
```

**Inlining 的实现原理**：

```python
# 假设用户代码：result = relu(x + y)
# Lowering 过程：

# Step 1: x + y 的 lowering
def add_inner_fn(index):
    return ops.add(x_loader(index), y_loader(index))
add_result = TensorBox(StorageBox(Pointwise(inner_fn=add_inner_fn, ranges=[N])))

# Step 2: relu(result) 的 lowering
# relu 的 lowering 调用 add_result.make_loader()，获取 add_inner_fn
relu_loader = add_result.make_loader()  # → add_inner_fn 本身！

def relu_inner_fn(index):
    val = relu_loader(index)  # → ops.add(x_loader(index), y_loader(index))
    return ops.relu(val)       # → ops.relu(ops.add(x_load, y_load))

relu_result = TensorBox(StorageBox(Pointwise(inner_fn=relu_inner_fn, ranges=[N])))

# 最终 inner_fn 闭包链：
# relu_inner_fn → relu(add_inner_fn) → relu(add(x, y))
# 这就是 Inlining！x + y 的计算没有物化，而是被嵌入 relu 的 inner_fn 中
```

**make_loader 如何实现 Inlining**：

`Pointwise.make_loader()`（ir.py:1106）直接返回 `self.inner_fn`——不创建任何新的 buffer。消费者获取到的是生产者的计算函数，可以组合到自己的 inner_fn 中。

### 7.2 FlexibleLayout vs FixedLayout 的设计

**文件**：[ir.py:4137-4554](torch/_inductor/ir.py#L4137)

```
FixedLayout（不可变）：
├── 用于：输入张量、已物化的 buffer
├── stride 已确定，不能修改
└── make_indexer() 返回固定的索引计算公式

FlexibleLayout（可优化）：
├── 用于：ComputedBuffer（物化后的中间结果）
├── stride 待定，可被调度器优化
├── 支持冻结为多种布局：
│   ├── as_stride_order(order) → 按指定 stride 顺序
│   ├── as_fill_order(order) → 按指定填充顺序
│   ├── as_channels_last() → channels-last 布局
│   └── as_exact_strides(strides) → 精确指定 strides
└── 冻结后变为 FixedLayout
```

**设计意图**：FlexibleLayout 允许调度器在看到完整的 IR 图后，再决定每个中间 buffer 的最优布局。例如，如果消费者偏好 channels-last，生产者可以被冻结为 channels-last，从而避免额外的 transpose。

### 7.3 View 操作的零拷贝实现

**文件**：[ir.py:2878-3600](torch/_inductor/ir.py#L2878)

View 操作（expand, permute, squeeze, reshape 等）在 IR 层面是零拷贝的——它们只修改索引映射，不复制数据。

```python
# ExpandView：广播维度
class ExpandView(BaseView):
    def make_reindexer(self):
        # 广播维度：将大小为1的维度映射为 0（每次都读同一个位置）
        def reindex(index):
            return [0 if self.old_size[i] == 1 else idx
                    for i, idx in enumerate(index)]
        return reindex

# PermuteView：维度重排
class PermuteView(BaseView):
    def make_reindexer(self):
        def reindex(index):
            # 逆排列：将输出索引映射回输入索引
            inv_perm = [0] * len(self.dims)
            for i, d in enumerate(self.dims):
                inv_perm[d] = i
            return [index[inv_perm[i]] for i in range(len(index))]
        return reindex

# View（reshape）：通用形状变换
# 通过计算索引变换函数实现
```

**优化路径**：如果底层有已知的 storage 和 layout，View 操作可以直接创建 `ReinterpretView`——修改 layout 中的 stride 信息，完全避免创建 View IR 节点。

```python
# ExpandView.create() 的优化路径
@classmethod
def create(cls, x, new_size):
    if is_storage_and_layout(x):
        storage, old_layout = as_storage_and_layout(x)
        # 直接计算新的 strides（广播维度 stride 设为 0）
        new_layout = FixedLayout(old_layout.device, old_layout.dtype,
                                  new_size, new_strides)
        return ReinterpretView(data=storage, layout=new_layout)
    return ExpandView(data=x, size=new_size)
```

### 7.4 Reduction 的多层优化

**文件**：[ir.py:1541-1740](torch/_inductor/ir.py#L1541)

`Reduction.create()` 包含多种优化路径：

| 归约大小 | 策略 | 原因 |
|----------|------|------|
| 0 | 返回常量 Pointwise | 空归约的结果是确定的（如 sum([])=0） |
| 1 | 退化为 Pointwise | 单元素无需归约循环 |
| 小（≤ 阈值）| 展开为 Pointwise | 避免 kernel launch 开销 |
| 大 | 标准 Reduction | 需要归约循环 |
| 非常大 | 多层 Reduction | 并行归约：第一层部分归约，第二层最终归约 |

### 7.5 broadcast 的符号化实现

**文件**：[lowering.py:549-572](torch/_inductor/lowering.py#L549)

```python
def broadcast_symbolic_shapes(a, b):
    """广播逻辑，基于符号形状"""
    b = tuple(b)
    if not a or a == b:
        return b

    output = []
    for x, y in itertools.zip_longest(reversed(a), reversed(b),
                                       fillvalue=sympy.S.One):
        if V.graph.sizevars.is_size_one_or_false(y):  # y == 1
            output.append(x)
        elif V.graph.sizevars.is_size_one_or_false(x):  # x == 1
            output.append(y)
        else:
            V.graph.sizevars.check_equals(x, y)  # 必须相等
            # 选择符号更少的表达式（简化后续计算）
            if len(sympy.expand(y).free_symbols) < len(sympy.expand(x).free_symbols):
                output.append(y)
            else:
                output.append(x)
    return tuple(reversed(output))
```

**关键洞察**：广播操作在 SymPy 符号域进行，支持动态形状。`is_size_one_or_false()` 不仅检查 `== 1`，还利用符号推理判断是否可能为 1。

---

## 八、自主学习 Debug 路线

### 路线总览

```
Step 1: 观察完整 Lowering 过程
    │
    Step 2: 断点走读 call_function
    │
    Step 3: 观察不同类型算子的 Lowering
    │
    Step 4: 观察延迟求值与 Inlining
    │
    Step 5: 观察 Reduction 的优化路径
    │
    Step 6: 观察 V.ops 的多态行为
    │
    Step 7: 观察算子分解与 Lowering 的交互
```

### Step 1: 观察完整 Lowering 过程

**目标**：确认 FX Graph 中的每个节点都被正确翻译为 IR。

**操作**：创建脚本 `agent_space/debug_lowering_step1.py`：

```python
import torch
import torch._inductor.config as config
import logging

# 开启 Lowering 相关日志
torch._logging.set_logs(
    inductor=logging.DEBUG,
    output_code=logging.DEBUG,
)

@torch.compile(mode="default")
def simple_model(x, y):
    return torch.relu(x + y) * 2.0

x = torch.randn(4, 4, device="cpu")
y = torch.randn(4, 4, device="cpu")
result = simple_model(x, y)
print(f"Match: {torch.allclose(result, torch.relu(x + y) * 2.0)}")
```

**关注点**：
- 终端输出中 `GraphLowering.run` 的日志
- 每个 FX Node 被翻译为哪种 IR 节点？
- 最终有多少个 operation 和 buffer？

**输入**：编译环境 + 日志配置
**输出**：理解 FX Node → IR Node 的完整映射

### Step 2: 断点走读 call_function

**目标**：用 pdb 走通一次完整的 call_function 调度。

**操作**：设置断点：

```
断点 1: graph.py:1319  call_function()
    → 观察 target 是什么算子
    → 观察 args 的类型（TensorBox? StorageBox? scalar?）
    → 单步进入 lowerings[target] 观察返回值

断点 2: lowering.py:7001  mul() 函数
    → 观察 make_pointwise 如何构造 inner_fn
    → 观察 Pointwise.create() 的返回值

断点 3: ir.py:9420  TensorBox.create()
    → 观察 StorageBox 如何包装 Pointwise
    → 确认返回值的完整结构
```

**关注点**：
- `lowerings` 字典中 `target` 对应的函数是什么？
- 返回值的类型层次：`TensorBox(StorageBox(Pointwise(...)))`
- `inner_fn` 闭包捕获了哪些变量？

**输入**：pdb 或 IDE debugger
**输出**：理解 call_function 的完整调度逻辑

### Step 3: 观察不同类型算子的 Lowering

**目标**：理解 pointwise、reduction、fallback 三种 Lowering 路径。

**操作**：创建脚本：

```python
import torch

@torch.compile(mode="default")
def mixed_ops(x, w):
    # Pointwise: add + relu
    a = x + 1.0
    b = torch.relu(a)
    # Reduction: sum
    c = b.sum(dim=-1)
    # Fallback: mm
    d = x @ w
    return c, d

x = torch.randn(4, 8)
w = torch.randn(8, 4)
result = mixed_ops(x, w)
```

**关注点**（在 graph.py:1319 设断点）：
- `x + 1.0`：target = `aten.add`，走 make_pointwise 路径
- `b.sum(dim=-1)`：target = `aten.sum`，走 make_reduction 路径
- `x @ w`：target = `aten.mm`，GPU 上可能走 Triton mm template，无专用 template 时走 fallback 路径
- 每种路径返回的 IR 节点类型有何不同？

**输入**：包含三种类型算子的模型
**输出**：理解三种 Lowering 路径的区别

### Step 4: 观察延迟求值与 Inlining

**目标**：验证 pointwise 操作确实通过闭包组合实现 Inlining。

**操作**：在 `graph.run()` 执行完毕后观察 IR 结构：

```python
# 在 graph.py:2546 codegen() 入口处设断点，观察：
import torch._inductor.virtualized as V

graph = V.graph
print("=== Operations ===")
for op in graph.operations:
    print(f"  {type(op).__name__}: {op.get_name()}")
    if hasattr(op, 'data') and hasattr(op.data, 'inner_fn'):
        print(f"    inner_fn: {op.data.inner_fn}")

print("=== Buffers ===")
for buf in graph.buffers:
    print(f"  {buf.get_name()}: dtype={buf.get_dtype()}, "
          f"size={buf.get_layout().size}")
```

**关注点**：
- 有多少个 Pointwise operation？如果 `relu(x + y)` 只产生一个 operation，说明 Inlining 成功
- 有多少个 Buffer？Inlining 后中间 buffer 不应该存在
- `inner_fn` 闭包的内容——能否看到嵌套的 ops.add + ops.relu？

**输入**：graph.run() 后的 GraphLowering 对象
**输出**：验证 Inlining 的实际效果

### Step 5: 观察 Reduction 的优化路径

**目标**：理解 Reduction.create() 的多种优化路径。

**操作**：在 ir.py:1541 Reduction.create() 设断点：

```python
@torch.compile(mode="default")
def test_reductions(x):
    # 大归约
    a = x.sum()              # reduction_numel 很大
    # 小归约
    b = x.sum(dim=-1)        # reduction_numel 可能较小
    return a, b

x = torch.randn(32, 64)
result = test_reductions(x)
```

**关注点**：
- `reduction_numel` 的值——大归约和小归约分别走了哪条路径？
- 大归约是否被分成了多层？
- 小归约是否被展开为 pointwise？

**输入**：包含不同规模归约的模型
**输出**：理解 Reduction 的优化决策

### Step 6: 观察 V.ops 的多态行为

**目标**：理解同一个 inner_fn 在不同阶段的不同语义。

**操作**：设置断点：

```
断点 1: lowering.py:7007  mul() 中调用 make_pointwise
    → 此时 V.ops 是什么？（应该是 MockHandler 或 lowering 时的 handler）
    → ops.add(a, b) 返回什么？

断点 2: dependencies.py:680  RecordLoadStore 使用
    → V.set_ops_handler(rw) 后，ops.load 变成了什么？

断点 3: codegen/triton.py  TritonKernel 代码生成
    → V.set_ops_handler 后，ops.add 变成了什么？
    → 生成的是什么代码？
```

**关注点**：
- 同一个 `ops.add(a, b)` 在不同阶段的行为差异
- V.ops 的 handler 是如何被安装和替换的？

**输入**：多阶段断点
**输出**：理解 V.ops 的策略模式

### Step 7: 观察算子分解与 Lowering 的交互

**目标**：理解分解如何改变进入 Lowering 的算子集合。

**操作**：

```python
import torch

@torch.compile(mode="default")
def test_decomp_interact(x):
    return torch.log2(x)  # 应被分解为 log(x) * constant

x = torch.randn(4, 4)
result = test_decomp_interact(x)

# 在 graph.py:1319 设断点，观察：
# - 是否还有 aten.log2 节点？还是已经被分解为 aten.log + aten.mul？
# - 分解后的 log 和 mul 分别走什么 lowering 路径？
# - 两个操作是否被 Inlining 为一个 Pointwise？
```

**关注点**：
- 分解在何时发生？是在 Lowering 之前（AOTAutograd 阶段）还是期间？
- 分解后的碎片是否被 Inlining 重新粘合？
- 最终生成了几个 kernel？

**输入**：包含可分解算子的模型
**输出**：理解分解与 Lowering 的协同

---

## 九、数据流加工过程重点

### 9.1 FX Graph 在 Lowering 过程中的形态演变

```
阶段 0: 优化后的 FX Graph（来自 Phase 2）
│  形态：
│  ├── 所有算子已标准化为 ATen 操作
│  ├── 前后向图已分离
│  ├── SDPA、Conv-BN 等优化已应用
│  └── 节点类型：placeholder / get_attr / call_function / output
│
▼ placeholder 翻译
输入 IR（InputBuffer + TensorBox）
│  形态：
│  ├── 每个 placeholder → InputBuffer(name, FixedLayout)
│  ├── 包装为 TensorBox(StorageBox(InputBuffer))
│  ├── 形状：FixedLayout 包含精确的 SymPy 表达式
│  └── 新增：符号形状（SymPy 变量 s0, s1 等）
│
▼ get_attr 翻译
常量 IR
│  形态：
│  ├── 小常量（≤8 元素）→ Constant（内联值）
│  ├── 大常量 → ConstantBuffer（命名存储）
│  └── 新增：常量值直接嵌入 IR 或分配 buffer
│
▼ call_function 翻译（逐节点）
计算 IR
│  形态：
│  ├── Pointwise IR（逐元素计算）：
│  │   └── inner_fn = 闭包，包含 ops.add / ops.mul 等调用
│  │   └── ranges = SymPy 符号形状
│  │   └── 包装为 TensorBox(StorageBox(Pointwise))
│  │
│  ├── Reduction IR（归约计算）：
│  │   └── inner_fn + reduction_ranges
│  │   └── reduction_type = "sum" / "max" / ...
│  │   └── 通常已 realize → ComputedBuffer
│  │
│  └── ExternKernel / FallbackKernel（外部调用）：
│      └── 封装原始 ATen 算子调用
│      └── 布局约束（require_contiguous 等）
│
│  关键变化：
│  ├── [新增] 延迟求值能力（未物化的闭包）
│  ├── [新增] Inlining 能力（通过 make_loader 组合闭包）
│  ├── [新增] 符号形状信息（SymPy 变量在所有 IR 节点中传播）
│  └── [结构] 从 ATen 算子 → define-by-run 的 54 个原语操作
│
▼ run_node 后处理
带优化信息的 IR
│  变化：
│  ├── [新增] channels-last 布局决策（部分中间 buffer）
│  ├── [新增] Inlining 边界标记（realize 决策）
│  ├── [新增] stride 匹配（输出和 as_strided 输入）
│  └── [优化] 过大的 inner_fn 被强制物化（防 RecursionError）
│
▼ output + finalize
完整 IR DAG
│  形态：
│  ├── graph.operations: 所有 IR 操作列表
│  ├── graph.buffers: 所有命名存储单元列表
│  ├── graph.graph_outputs: 输出 IR 节点列表
│  ├── 所有 buffer 布局已决策（FlexibleLayout → FixedLayout）
│  └── 所有 mutated inputs 已处理
│
▼ 最终产品：完整的 Inductor IR 图
   特性：
   ├── 延迟求值：部分计算仍以闭包形式存在（可被后续融合）
   ├── 符号形状：所有形状用 SymPy 变量表示（支持动态形状）
   ├── 优化决策：布局、Inlining 边界已确定
   └── 完整依赖：所有 buffer 的读写依赖已建立
   └── 准备送入调度与融合阶段（Phase 4）
```

### 9.2 关键转变点分析

**转变 1：ATen 算子 → define-by-run 闭包**

```
原始：                          翻译后：
FX Node: aten.add(a, b)        Pointwise IR:
                                  inner_fn = lambda idx: ops.add(a_loader(idx), b_loader(idx))
                                  ranges = [s0, s1]

变化：
├── 从"做什么"（ATen 语义）→ "怎么做"（循环体函数）
├── 增加：延迟求值能力（闭包不立即执行）
├── 增加：Inlining 能力（make_loader 返回闭包本身）
└── 增加：V.ops 多态性（同一函数在不同阶段有不同语义）
```

**转变 2：未物化 → 物化（realize）**

```
物化前（可 Inlining）：                   物化后（不可 Inlining）：
TensorBox(StorageBox(Pointwise))         TensorBox(StorageBox(ComputedBuffer(
  ├── 无命名 buffer                          name="buf5",
  ├── 无固定布局                              layout=FlexibleLayout,
  ├── inner_fn 可被消费者组合                 data=Pointwise(...)))
  └── 不在 graph.buffers 中                 ├── 有命名 buffer
                                            ├── 布局待优化（FlexibleLayout）
                                            ├── inner_fn 保留（用于代码生成）
                                            └── 已注册到 graph.buffers 和 graph.operations

触发条件：
├── 多用户超过读取阈值
├── 输出节点
├── ExternKernel 输入（需要固定布局）
├── inner_fn 过大
└── 跨 CUDA stream 边界
```

**转变 3：FlexibleLayout → FixedLayout（布局冻结）**

```
FlexibleLayout（可优化）：            FixedLayout（不可变）：
├── stride 未确定                     ├── stride 已确定
├── 可被调度器选择最优布局             ├── 用于输入张量和最终输出
├── as_stride_order() → contiguous    └── make_indexer() 返回固定公式
├── as_channels_last()
└── 最终在 finalize() 或调度器中冻结
```

### 9.3 每种 Lowering 路径的数据流对比

```
                    输入                       加工                         输出
                    ────                       ────                         ────
Pointwise     TensorBox args             make_pointwise(fn)           TensorBox(StorageBox(Pointwise))
              (可能含 View)              ├── promote_constants          ├── inner_fn 闭包
                                        ├── create loaders             ├── ranges = 符号形状
                                        ├── 构造 inner_fn              └── 未物化（可 Inlining）
                                        └── Pointwise.create()

Reduction     TensorBox args             make_reduction(type)          TensorBox(StorageBox(ComputedBuffer(Reduction)))
              (可能含 View)              ├── _make_reduction_inner      ├── inner_fn 闭包
                                        ├── Reduction.create()         ├── ranges + reduction_ranges
                                        │   ├── 空/小 → Pointwise     └── 已 realize（必须物化）
                                        │   └── 大 → Reduction
                                        └── realize()

Fallback      TensorBox args             make_fallback(op)             TensorBox(StorageBox(TemplateBuffer/FallbackKernel))
              (必须 realize)             ├── add_needs_realized_inputs  ├── 布局约束
                                        ├── add_layout_constraint      ├── 封装 ATen 调用
                                        └── fallback_handler()         └── 外部库实现
```

---

## 十、交叉校验报告

> 校验时间：2026-04-15
> 校验方法：对比 PyTorch 2 论文 (ASPLOS 2024)、TorchInductor 设计帖 (dev-discuss #747)、PyTorch 源码 (main 分支)

### 校验结果汇总

| 校验项 | 来源 | 结果 |
|--------|------|------|
| GraphLowering 继承 torch.fx.Interpreter | 源码 graph.py:356 | **通过** |
| call_function 查找优先级：user_lowerings → lowerings → fallback | 源码 graph.py:1429-1462 | **通过** |
| make_pointwise 创建 Pointwise IR 节点 | 源码 lowering.py:668 | **通过** |
| make_reduction 创建 Reduction IR 节点并 realize | 源码 lowering.py:6644 | **通过** |
| make_fallback 创建 FallbackKernel | 源码 lowering.py:2506 | **通过** |
| 分解与 fallback 互斥（assert 检查） | 源码 lowering.py:2513-2514 | **通过** |
| TensorBox/StorageBox/Buffer 延迟求值链 | 源码 ir.py:9412, 9427, 4555 | **通过** |
| Pointwise.make_loader() 返回 inner_fn（Inlining 机制） | 源码 ir.py:1106 | **通过** |
| StorageBox.realize() 包装为 ComputedBuffer | 源码 ir.py:9443-9470 | **通过** |
| FlexibleLayout 可冻结为 FixedLayout | 源码 ir.py:4145-4554 | **通过** |
| V.ops 基于 threading.local() 动态作用域 | 源码 virtualized.py:112-159 | **通过** |
| select_decomp_table() 基于 config 选择分解表 | 源码 decomposition.py:964 | **通过** |
| Reduction.create() 多种优化路径 | 源码 ir.py:1541-1740 | **通过** |
| 54 个原语操作（ops.*） | 论文 Section 4.3 | **通过** |
| Inlining 消融实验数据（去 inlining 推理 -0.33） | 论文 Table 4 | **通过** |
| View 操作零拷贝（ExpandView/PermuteView） | 源码 ir.py:2976, 3071 | **通过** |

### 修正记录

| 修正内容 | 修正原因 |
|----------|----------|
| mm/convolution 使用外部 kernel 机制（GPU 上可能走专用 template，无 template 时 fallback） | 源码确认 make_fallback(aten.addbmm) 等，mm 在 GPU 上可通过 Triton mm template 或 cuBLAS 实现 |
| Reduction.create() 包含多层优化（空/小/大归约路径） | 源码 ir.py:1541-1740 确认有四种优化路径 |
| 分解表总量为 ~1145 个（core 1107 + random 38） | 源码分解：1001 core + 106 inductor-specific + 38 random ≈ 1145 |

### 权威出处

- PyTorch 2 论文 (ASPLOS 2024): [ACM DL](https://dl.acm.org/doi/10.1145/3620665.3640366) | [PDF](https://docs.pytorch.org/assets/pytorch2-2.pdf)
- TorchInductor 设计帖: [dev-discuss #747](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747)
- Inductor 文件结构讨论: [dev-discuss #1860](https://dev-discuss.pytorch.org/t/inductor-file-structure-explanation/1860)
- PyTorch 源码: `torch/_inductor/` 目录，main 分支 (2026-04)

---

## 附录 A：关键源码文件索引

| 文件 | 核心行号 | 核心内容 |
|------|---------|---------|
| `graph.py` | L356 | `GraphLowering` 类定义 |
| `graph.py` | L1049 | `run()` 入口 |
| `graph.py` | L1209 | `placeholder()` 输入翻译 |
| `graph.py` | L1319 | `call_function()` 算子调度 |
| `graph.py` | L1486 | `get_attr()` 常量处理 |
| `graph.py` | L1547 | `output()` 输出收集 |
| `graph.py` | L1655 | `finalize()` 布局决策 |
| `graph.py` | L1774 | `run_node()` Inlining 决策 |
| `graph.py` | L2546 | `codegen()` IR → 代码 |
| `lowering.py` | L115-117 | `lowerings` / `user_lowerings` 注册表 |
| `lowering.py` | L528 | `register_lowering()` 装饰器 |
| `lowering.py` | L668 | `make_pointwise()` 逐元素 Lowering |
| `lowering.py` | L6644 | `make_reduction()` 归约 Lowering |
| `lowering.py` | L2506 | `make_fallback()` fallback 注册 |
| `lowering.py` | L7001 | `mul()` 典型 pointwise Lowering |
| `lowering.py` | L7096 | `sum_()` 典型 reduction Lowering |
| `ir.py` | L548 | `IRNode` 基类 |
| `ir.py` | L960 | `Loops` 基类 |
| `ir.py` | L1105 | `Pointwise` 逐元素 IR |
| `ir.py` | L1255 | `Reduction` 归约 IR |
| `ir.py` | L2878 | `BaseView` 视图基类 |
| `ir.py` | L3846 | `Layout` 基类 |
| `ir.py` | L4137 | `FixedLayout` 固定布局 |
| `ir.py` | L4145 | `FlexibleLayout` 可优化布局 |
| `ir.py` | L4555 | `Buffer` 缓冲区基类 |
| `ir.py` | L4781 | `ComputedBuffer` 计算缓冲区 |
| `ir.py` | L5230 | `TemplateBuffer` 模板缓冲区 |
| `ir.py` | L6181 | `ExternKernel` 外部调用 |
| `ir.py` | L9261 | `MutableBox` 延迟求值基类 |
| `ir.py` | L9412 | `TensorBox` 张量容器 |
| `ir.py` | L9427 | `StorageBox` 存储容器 |
| `virtualized.py` | L112 | `Virtualized<T>` 类 |
| `virtualized.py` | L187-196 | V 实例定义（_ops, _graph, _kernel） |
| `virtualized.py` | L364 | `_V` 统一访问类 |
| `decomposition.py` | L964 | `select_decomp_table()` |
| `decomposition.py` | L250 | `silu` 分解 |
| `decomposition.py` | L378 | `addmm` 条件分解 |

---

## 附录 B：Phase 3 检验清单

完成以上 7 步 debug 路线后，你应当能够回答以下问题：

- [ ] GraphLowering 的 `call_function()` 如何查找并调用 lowering 函数？优先级是什么？
- [ ] `make_pointwise()` 如何构造 `inner_fn` 闭包？这个闭包为什么支持 Inlining？
- [ ] `TensorBox` → `StorageBox` → `Buffer` 的延迟求值链是什么？`realize()` 触发了什么？
- [ ] `Pointwise.make_loader()` 如何实现 Inlining？返回的是什么？
- [ ] `make_reduction()` 和 `make_pointwise()` 的关键区别是什么？为什么归约必须 realize？
- [ ] `make_fallback()` 解决什么问题？为什么有些算子使用 fallback，而 mm 在 GPU 上可走专用 template？
- [ ] `FlexibleLayout` 和 `FixedLayout` 的区别是什么？布局在什么时候冻结？
- [ ] `V.ops` 的虚拟化机制如何让同一个 inner_fn 在不同阶段有不同语义？
- [ ] View 操作（expand/permute/reshape）如何在 IR 层面实现零拷贝？
- [ ] 你能否在不查看本文档的情况下，画出 Lowering 的完整数据流和 IR 类层次？
