# 第七章：内核代码生成 —— 从执行计划到 Kernel 代码

## 引言

在前面的章节中，我们追踪了计算图从 Python 前端到 IR 中间表示，再到 Scheduler 调度计划的全过程。调度阶段将 IR 节点组织成可融合的组，确定了执行顺序和内存布局。然而，调度计划本身只是一份"蓝图"——它告诉系统"做什么"，但没有回答"怎么做"。

**内核代码生成（Kernel Codegen）** 正是回答"怎么做"的环节。它将调度器输出的 `SchedulerNode`（或融合后的 `FusedSchedulerNode`）转化为可在真实硬件上执行的代码：在 GPU 上是 Triton Python kernel，在 CPU 上是 C++ 循环嵌套。

用编译器的术语来说，如果调度阶段对应于指令调度（instruction scheduling），那么代码生成阶段对应于**指令选择（instruction selection）和代码发射（code emission）**。这是编译器后端的最后一步，也是与目标硬件最紧密相关的一步。

本章将深入剖析 Inductor 的内核代码生成架构，从基类 `Kernel` 的三段式模型出发，逐步展开到 `TritonKernel`（GPU）和 `CppKernel`（CPU）的具体实现，最后讨论模板 kernel 和辅助类。读完本章后，你将理解一条计算指令从 IR 闭包到最终 Triton/C++ 代码的完整转化路径。

---

## 7.1 Kernel 基类 —— 三段式代码组织

### 7.1.1 设计动机

编译器后端的一个核心挑战是：如何将一组抽象的 IR 操作有序地组织成目标代码。对于一个典型的计算 kernel，例如 `c = relu(a + b)`，我们需要生成以下几类代码：

1. 从全局内存读取 `a` 和 `b`（load）
2. 执行 `a + b` 和 `relu`（compute）
3. 将结果写回全局内存（store）

这三类操作有不同的特性和约束：load 和 store 涉及内存访问，需要处理边界检查（boundary masking）；compute 是纯计算，可以自由重排。将它们分离到不同的代码段中，既方便后续优化（如合并连续 load、消除冗余 store），也使代码结构更清晰。

### 7.1.2 核心数据结构

源码位置：`torch/_inductor/codegen/common.py:1953`

```python
class Kernel(CodeGen, Generic[CSEVariableType]):
    newvar_prefix: str = ""
    suffix: str = ""
    overrides: Optional[Callable[[], OpsHandler[Any]]] = None

    def __init__(self, args=None, increase_kernel_count=True):
        super().__init__()
        if increase_kernel_count:
            metrics.generated_kernel_count += 1
        self.args = args or KernelArgs()
        self.loads = IndentedBuffer()      # 输入读取段
        self.compute = IndentedBuffer()    # 计算段
        self.stores = IndentedBuffer()     # 输出写入段

        self.cse: CSE[CSEVariableType, Any] = CSE(self.newvar_prefix, self.suffix)
        self.must_keep_buffers: OrderedSet[str] = OrderedSet()
        self.store_buffer_names: OrderedSet[str] = OrderedSet()
        self._load_mask: Optional[str] = None
        self._load_other: Union[None, int, float] = None
        self.current_node: Optional[SchedulerNode] = None
        self.removed_buffers: OrderedSet[str] = OrderedSet()
        self.inplace_update_buffers: dict[str, str] = {}
        self.min_elem_per_thread = 1
        self.kernel_name: Optional[str] = None
```

`Kernel` 是所有 kernel 代码生成的基类，使用泛型参数 `CSEVariableType` 允许不同后端定义自己的变量类型（如 Triton 的 `TritonCSEVariable`）。

**三段式模型**的三个 `IndentedBuffer` 是 `Kernel` 最核心的数据结构：

```
┌─────────────────────────────────────────────┐
│  self.loads (IndentedBuffer)                │  ← 输入读取
│    tmp0 = tl.load(x_ptr + offsets, mask=mask)│
│    tmp1 = tl.load(y_ptr + offsets, mask=mask)│
├─────────────────────────────────────────────┤
│  self.compute (IndentedBuffer)              │  ← 计算
│    tmp2 = tmp0 + tmp1                        │
│    tmp3 = tl.where(tmp2 > 0, tmp2, 0)       │
├─────────────────────────────────────────────┤
│  self.stores (IndentedBuffer)               │  ← 输出写入
│    tl.store(out_ptr + offsets, tmp3, mask=...)│
└─────────────────────────────────────────────┘
```

不同的后端会以不同方式消费这三个段：

| 后端     | loads            | compute          | stores              |
| -------- | ---------------- | ---------------- | ------------------- |
| Triton   | `tl.load(...)`   | 算术表达式        | `tl.store(...)`    |
| C++ 标量 | `auto tmp0 = buf[i]` | `auto tmp1 = ...` | `out[i] = tmp1;`  |
| C++ 向量 | `vec.load(buf+i)`    | `vec_op(...)`     | `vec.store(out+i)` |

### 7.1.3 上下文管理器模式

`Kernel` 使用 Python 的上下文管理器（context manager）模式来管理局部环境：

```python
def __enter__(self) -> Self:
    super().__enter__()
    assert self.overrides
    # 1. 安装 CSEProxy + 后端 overrides 作为当前 ops handler
    self.exit_stack.enter_context(
        V.set_ops_handler(CSEProxy(self, self.overrides()))
    )
    # 2. 将自身设置为当前 kernel
    self.exit_stack.enter_context(V.set_kernel_handler(self))
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    # 清理 kernel 局部 buffer
    self.remove_kernel_local_buffers()
    super().__exit__(exc_type, exc_val, exc_tb)
```

这段代码的含义非常深刻。在 `__enter__` 中：

1. **安装 CSEProxy**：将 `CSEProxy(kernel, backend_overrides)` 推入 `V.ops` 的 handler 栈。此后所有 `V.ops.load()`、`V.ops.add()` 调用都会经过 CSEProxy 的拦截，CSEProxy 在命中缓存时直接返回已有变量，否则委托给后端 overrides 生成实际代码。

2. **设置当前 kernel**：将自身推入 `V.kernel`，使得在任何嵌套调用中都能访问当前 kernel 的状态（如 `cse`、`args`、`range_trees` 等）。

在 `__exit__` 中，执行 `remove_kernel_local_buffers()`——如果一个 buffer 在当前 kernel 内被创建且最后使用也在当前 kernel 内，那么它可以被安全移除，不需要作为 kernel 的输出参数传递。

用编译器的类比：这就像进入一个基本块（basic block）时设置活跃变量集合，离开时清理死代码。`__enter__`/`__exit__` 确保 handler 栈的正确恢复，即使内部代码抛出异常。

典型的使用模式如下：

```python
with TritonKernel(tiling, features=features) as kernel:
    # 此时 V.ops = CSEProxy(TritonKernelOverrides())
    # V.kernel = kernel
    # IR 闭包执行，所有 ops 调用经过 CSEProxy → TritonKernelOverrides
    for node in fused_nodes:
        node.codegen()
# 离开时 V.ops 和 V.kernel 自动恢复
```

### 7.1.4 虚方法接口

`Kernel` 定义了一组必须由子类实现的虚方法：

```python
def load(self, name: str, index: sympy.Expr) -> CSEVariable:
    raise NotImplementedError

def store(self, name: str, index: sympy.Expr, value: CSEVariable, mode=None) -> None:
    raise NotImplementedError

def store_reduction(self, name: str, index: sympy.Expr, value: CSEVariable) -> None:
    raise NotImplementedError

def reduction(self, dtype, src_dtype, reduction_type, value) -> CSEVariable:
    raise NotImplementedError

def scan(self, dtypes, combine_fn, values) -> tuple[CSEVariable, ...]:
    raise NotImplementedError

def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
    raise NotImplementedError

def index_to_str(self, index: sympy.Expr) -> str:
    raise NotImplementedError
```

这些方法构成了 kernel 代码生成的核心接口。`load` 和 `store` 负责内存访问，`reduction` 和 `scan` 处理特殊归约/前缀扫描模式，`var_ranges` 返回循环变量的范围映射。不同后端根据自身硬件特性提供不同的实现。

### 7.1.5 swap_buffers：临时切换代码缓冲区

```python
@contextlib.contextmanager
def swap_buffers(self, lb, cb=None, sb=None):
    if cb is None:
        cb = lb
    if disallow_stores := sb is None:
        sb = IndentedBuffer()
    # 保存当前缓冲区
    loads, compute, stores, cse = self.loads, self.compute, self.stores, self.cse
    # 切换到临时缓冲区
    self.loads, self.compute, self.stores = lb, cb, sb
    self.cse = cse.scoped_copy()  # CSE 使用作用域副本
    try:
        yield
    finally:
        # 恢复原始缓冲区
        self.loads, self.compute, self.stores = loads, compute, stores
        self.cse = cse
        if disallow_stores:
            assert not sb, "unexpected store inside swap_buffers"
```

`swap_buffers` 是一个精巧的机制，用于在特定场景下临时重定向代码输出。例如，在处理归约操作时，归约循环体内的 load/compute 需要写入临时缓冲区，而不是直接写入主缓冲区。`cse.scoped_copy()` 确保临时作用域内的 CSE 缓存不会污染主作用域。

编译器类比：这类似于编译器中的"临时代码缓冲区"或"指令暂存区"——在复杂控制流（如循环展开、条件分支）中，先收集局部代码片段，再在适当时机拼接到主输出。

### 7.1.6 输入与输出

- **输入**：`SchedulerNode`（或 `FusedSchedulerNode`），包含 IR 闭包（`LoopBody`），描述需要执行的计算。
- **输出**：完整的 kernel 源代码字符串（Triton Python 或 C++），包含函数签名、参数声明和计算逻辑。

---

## 7.2 SIMDKernel —— GPU/SIMD 基类

### 7.2.1 设计动机

`Kernel` 定义了三段式模型，但没有涉及任何关于"如何遍历数据"的逻辑。对于 GPU 和 SIMD 后端，数据遍历方式是相似的：将多维张量展平为一维索引空间，每个线程（或 SIMD lane）处理索引空间中的一个连续块。这种共性被抽象为 `SIMDKernel`。

源码位置：`torch/_inductor/codegen/simd.py:371`

```python
class SIMDKernel(Kernel[CSEVariableType], Generic[CSEVariableType]):
    """
    Common base class for Triton/Halide codegen which both use
    flattened indexing rather than loop nests.
    """
```

`SIMDKernel` 是 `TritonKernel` 的直接父类，也是所有使用展平索引（而非循环嵌套）的 kernel 的共同基类。

### 7.2.2 展平索引（Flattened Indexing）

CPU 后端（`CppKernel`）使用嵌套循环来遍历多维张量：

```cpp
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        out[i * N + j] = x[i * N + j] + y[i * N + j];
```

GPU 后端不使用循环嵌套。取而代之的是将所有维度展平为一个一维索引空间，每个 GPU 线程块（thread block）处理索引空间中的一段连续区域：

```python
# Triton kernel 中
pid = tl.program_id(0)             # 当前线程块的 ID
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # 该线程块处理的索引
mask = offsets < N                   # 边界检查
x = tl.load(x_ptr + offsets, mask=mask)   # 读取
```

对于一个 `[M, N]` 的二维张量，逻辑上的 `(i, j)` 坐标被映射为扁平索引 `i * N + j`，而 GPU 直接在扁平索引空间上并行。

这种展平不是简单的 reshape——它需要正确处理 stride、padding 和非连续张量。`SIMDKernel` 的索引简化逻辑负责将这些复杂的索引表达式化简为高效的扁平地址。

### 7.2.3 Range Trees —— 迭代域的层次表示

Range tree 是 `SIMDKernel` 的核心数据结构，表示 kernel 的迭代域（iteration domain）。

```python
def __init__(self, tiling, features, pid_cache=None, ...):
    ...
    self.numels = {
        prefix: V.graph.sizevars.simplify(val) for prefix, val in tiling.items()
    }
    self.range_trees: list[IterationRangesRoot] = []
    self.range_tree_nodes: dict[sympy.Symbol, IterationRangesEntry] = {}
    ...
    self.initialize_range_tree(pid_cache)
```

**numels**：每个前缀维度对应的元素数量。对于 pointwise kernel，通常只有一个 `x` 维度；对于 reduction kernel，会有 `x`（非归约维度）和 `r0_`（归约维度）。

**range_tree 的层次结构**：

```
IterationRangesRoot (顶层)
  ├── prefix: "x"         (维度标识)
  ├── numel: M * N        (该维度的元素数)
  ├── index: 0            (对应 tl.program_id 的哪个维度)
  ├── is_loop: False      (pointwise) / True (reduction 中的归约维度)
  └── children: IterationRangesEntry (子节点，更细粒度的索引)
```

每个 range tree 对应一个 GPU 维度：

| Range Tree 前缀 | GPU 网格维度    | 含义              |
| ---------------- | --------------- | ----------------- |
| `x`              | `tl.program_id(0)` | 第一个空间维度   |
| `y`              | `tl.program_id(1)` | 第二个空间维度   |
| `z`              | `tl.program_id(2)` | 第三个空间维度   |
| `r0_`            | 归约循环         | 归约维度         |

### 7.2.4 SIMDKernel 到 Triton 编程模型的映射

`SIMDKernel` 的抽象概念与 Triton 的编程模型有直接的对应关系：

```
SIMDKernel 概念          →    Triton 概念
───────────────────           ────────────────
range_tree["x"]          →    tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
range_tree["r0_"]        →    for roffset in range(0, r0_numel, R0BLOCK):
numel["x"]               →    grid 大小（grid[0] = ceil(xnumel / XBLOCK)）
load(name, flat_index)   →    tl.load(ptr + index, mask=mask)
store(name, flat_index)  →    tl.store(ptr + index, value, mask=mask)
```

这种映射是在 `SIMDKernel` 的子类中具体实现的。`SIMDKernel` 提供了索引简化和 range tree 管理的基础设施，而具体的 `tl.load()`/`tl.store()` 调用由 `TritonKernel` 的 overrides 生成。

### 7.2.5 var_ranges：循环变量的值域

```python
def var_ranges(self) -> dict[sympy.Symbol, sympy.Expr]:
    return dict(
        itertools.chain.from_iterable(
            tree.var_ranges.items() for tree in self.range_trees
        )
    )
```

`var_ranges` 返回一个从循环变量符号到其值域的映射。这个映射被用于两个地方：

1. **索引简化**：将多维索引表达式化简为扁平索引。
2. **值域分析**：为 CSE 和边界检查提供符号范围信息。

例如，对于一个 `[M, N]` 的 kernel，`var_ranges` 可能返回 `{x0: M*N}`（展平后），或者 `{i: M, j: N}`（如果保留了多维度结构）。

---

## 7.3 TritonKernel —— Triton 后端核心

### 7.3.1 架构总览

源码位置：`torch/_inductor/codegen/triton.py:1620`

`TritonKernel` 是 Inductor 中最重要的 kernel 类，负责生成 GPU 上运行的 Triton kernel 代码。它的继承关系如下：

```
Kernel[CSEVariable]
  └── SIMDKernel[CSEVariable]
        └── TritonKernel    ← 本节主角
              ├── TritonSplitScanKernel  (前缀扫描)
              └── TritonTemplateKernel   (模板 kernel)
```

`TritonKernel` 扩展了 `SIMDKernel`，添加了：

- Triton 特定的代码生成模式（`tl.load`/`tl.store`/`tl.where`）
- 指针管理与算术
- 边界 mask 生成
- Autotuning 配置支持
- Reduction 循环模式
- Block pointer 优化

```python
class TritonKernel(SIMDKernel[TritonCSEVariable]):
    overrides = TritonKernelOverrides  # 后端 overrides 绑定
    kexpr: Callable[[sympy.Expr], str] = texpr  # 表达式序列化
    allow_block_ptr = True  # 允许使用 Triton block pointer 优化
```

### 7.3.2 初始化过程

```python
def __init__(self, tiling, min_elem_per_thread=0, optimize_mask=True,
             fixed_config=None, **kwargs):
    self.optimize_mask = optimize_mask
    self.fixed_config = fixed_config
    super().__init__(tiling, **kwargs)       # 调用 SIMDKernel.__init__

    # Triton 特有的数据结构
    self.cse = TritonCSE(self.newvar_prefix, self.suffix)  # Triton 专用 CSE
    self.post_loop_combine = IndentedBuffer()   # 归约后的 combine 代码
    self.post_loop_store = IndentedBuffer()     # 归约后的 store 代码
    self.outside_loop_vars = OrderedSet()       # 归约循环外可见的变量
    self.min_elem_per_thread = min_elem_per_thread
    self.block_ptr_id = itertools.count()       # block pointer ID 生成器
    self.helper_functions = HelperFunctions()   # 辅助函数收集器
    self.autotune_hints = OrderedSet()          # autotuning 提示
    self.triton_meta = None                     # Triton 元数据

    # 根据 kernel 类型生成初始化代码
    if self.inside_reduction:
        self.codegen_reduction_numels(self.body)  # 归约维度初始化
    if self.cooperative_reduction:
        self.init_cooperative_reduction()          # 协作归约初始化
    self.codegen_range_tree()                      # Range tree 代码生成
    if self.cooperative_reduction:
        self.init_cooperative_reduction_mask()     # 归约 mask 初始化
```

初始化过程清晰展示了 Triton kernel 的代码组织：先处理归约维度，再建立 range tree 索引，最后设置 mask。这些初始化代码被写入 `self.body` 缓冲区，构成 kernel 的"序言"（prologue）。

### 7.3.3 TritonKernelOverrides —— 核心代码生成器

源码位置：`torch/_inductor/codegen/triton.py:1321`

`TritonKernelOverrides` 继承自 `TritonOverrides`，在 kernel 作用域内提供具体的代码生成方法：

```python
class TritonKernelOverrides(TritonOverrides):
    """Map element-wise ops to Triton within a TritonKernel.

    Unlike TritonOverrides, these assume the code is going to be inserted into
    the body of the main triton kernel and so it may use indexing and mask
    variables which are assumed to already be defined in the current scope.
    """
```

关键的区别：`TritonOverrides` 是独立的（不依赖任何上下文），而 `TritonKernelOverrides` 假设当前正处于一个 Triton kernel 内部，可以使用 `xoffset`、`xmask`、`XBLOCK` 等由 `TritonKernel` 序言定义的变量。

一些核心方法：

**index_expr**：将 sympy 索引表达式转化为 Triton 变量

```python
@classmethod
def index_expr(cls, expr, dtype):
    indexing = V.kernel.indexing(expr, block_ptr=False)
    assert isinstance(indexing, IndexingOptions)
    # 将索引表达式生成为 compute 段中的 CSE 变量
    var = V.kernel.cse.generate(
        V.kernel.compute,
        indexing.index_str,
        bounds=get_bounds_index_expr(expr),
        dtype=dtype,
    )
    var.mask_vars = indexing.mask_vars
    return var
```

**masked**：条件执行（处理 mask）

```python
@staticmethod
def masked(mask, body, other):
    # 如果是 tl.load + mask + other 值，可以合并到 tl.load 中
    # 否则使用 tl.where 进行条件选择
    ...
```

### 7.3.4 代码生成流程

当 `TritonScheduling` 调用 `TritonKernel.codegen()` 时，完整的代码生成流程如下：

**第一步：初始化 kernel 并进入上下文**

```python
kernel = TritonKernel(tiling, features=features)
with kernel:  # 安装 CSEProxy + TritonKernelOverrides
    ...
```

**第二步：生成 kernel 序言**

Triton kernel 的序言包含：
- 线程块索引计算：`pid = tl.program_id(0)`
- 偏移量计算：`xoffset = pid * XBLOCK`
- 元素索引：`xindex = xoffset + tl.arange(0, XBLOCK)`
- 边界 mask：`xmask = xindex < xnumel`

```python
def codegen_range_tree(self):
    for tree in self.range_trees:
        if not tree.is_loop:
            self.iteration_ranges_codegen_header(tree, self.body)
        elif self.inside_reduction:
            self.body.writeline(
                f"{tree.prefix}base = {self.iteration_ranges_ranges_code(tree)}"
            )
```

**第三步：执行 IR 闭包**

IR 闭包的执行是代码生成的核心。调度器依次调用每个 `SchedulerNode` 的 `codegen()` 方法，后者执行其 `LoopBody` 闭包。闭包内部的 `ops.load()`、`ops.add()`、`ops.store()` 调用通过 CSEProxy 分发到 `TritonKernelOverrides`，生成 Triton 代码。

**第四步：组装最终 kernel**

```python
def codegen_kernel(self, name=None):
    code = IndentedBuffer()
    # 生成 imports
    code.splice(gen_common_triton_imports())

    # 生成函数签名
    argdefs, _, signature, _ = self.args.python_argdefs()
    # 添加 numel 参数和 BLOCK_SIZE constexpr 参数
    ...

    # 生成 @triton_heuristics 装饰器和 @triton.jit
    code.splice(heuristics_line)

    # 生成函数定义
    code.writeline(f"def {name}({', '.join(x.full_name() for x in argdefs)}):")
    with code.indent():
        self.codegen_static_numels(code)  # 静态 numel 常量
        for old, new in self.args.aliases():  # inplace 别名
            code.writeline(f"{old} = {new}")
        code.splice(self.body)  # kernel 主体

    return code.getvalue()
```

最终输出的完整 Triton kernel 具有如下结构：

```python
import triton
import triton.language as tl
...

@triton_heurics.pointwise(
    size_hints=[4096],
    filename=__file__,
    triton_meta={...},
    inductor_meta={...}
)
@triton.jit
def kernel_name(in_ptr0, in_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096  # 静态 numel（如果已知）
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    # --- loads 段 ---
    tmp0 = tl.load(in_ptr0 + xindex, mask=xmask)
    tmp1 = tl.load(in_ptr1 + xindex, mask=xmask)
    # --- compute 段 ---
    tmp2 = tmp0 + tmp1
    tmp3 = tl.where(tmp2 > 0, tmp2, 0)  # relu
    # --- stores 段 ---
    tl.store(out_ptr2 + xindex, tmp3, mask=xmask)
```

### 7.3.5 indexing 方法：索引计算的核心

源码位置：`torch/_inductor/codegen/triton.py:1784`

`indexing` 是 `TritonKernel` 中最复杂的方法之一，负责将 sympy 索引表达式转化为 Triton 可以直接使用的索引和 mask：

```python
def indexing(self, index, *, copy_shape=None, dense_indexing=False,
             override_mask=None, block_ptr=False):
    """
    Compute the index and mask to pass to tl.load() or tl.store()
    """
    index = self.prepare_indexing(index)
    index_vars = index.free_symbols

    # 1. 收集需要参与的 mask 变量
    mask_vars = OrderedSet()
    for var in sorted(index_vars, key=operator.attrgetter("name")):
        if symbol_is_type(var, SymT.TMP):
            # 间接索引（indirect indexing），继承源变量的 mask
            cse_var = self.cse.varname_map[var.name]
            mask_vars.update(cse_var.mask_vars)
        else:
            # 直接索引，使用对应维度的 mask
            prefix_matches = [
                prefix_str[symt]
                for symt in TritonSymbols.block_types
                if symbol_is_type(var, symt)
            ]
            mask_vars.add(f"{prefix_matches[0]}mask")

    # 2. 尝试匹配 block pointer 模式（性能优化）
    if block_ptr and self.allow_block_ptr and config.triton.use_block_ptr:
        ...

    # 3. 返回 IndexingOptions（包含 index_str 和 mask_vars）
    return IndexingOptions(index_str=..., mask_vars=mask_vars)
```

这个方法的输出是一个 `IndexingOptions` 对象，包含序列化后的索引字符串（如 `"xindex"` 或 `"xindex * stride"`）和需要应用的 mask 变量集合。

### 7.3.6 codegen_body：三段代码的拼接

```python
def codegen_body(self):
    """
    Concat output code from index_code, loads, compute, stores, suffix
    into self.body.
    """
    if not (self.indexing_code or self.loads or self.stores or self.compute
            or self.post_loop_combine or self.post_loop_store):
        return

    if self.inside_reduction and len(loop_trees) > 0:
        # 归约模式：需要生成循环结构
        for level, tree in enumerate(loop_trees):
            with self.body.indent(offset=level):
                self.body.writeline(
                    f"for {prefix}offset in range(0, {prefix}numel, {prefix.upper()}BLOCK):"
                )
            with self.body.indent(offset=level + 1):
                self.iteration_ranges_codegen_header(tree, self.body)

        # 循环体内：indexing + loads + compute + stores
        with self.body.indent(offset=len(loop_trees)):
            self.codegen_reduction_indices(self.body)
            self.body.splice(self.indexing_code)
            self.body.splice(self.loads)
            self.body.splice(self.compute)
            self.body.splice(self.stores)

        # 循环外：post_loop_combine + post_loop_store
        self.body.splice(self.post_loop_combine)
        self.body.splice(self.post_loop_store)
    else:
        # Pointwise 模式：直接拼接
        self.body.splice(self.indexing_code)
        self.body.splice(self.loads)
        self.body.splice(self.compute)
        self.body.splice(self.stores)
```

对于 pointwise kernel，三段代码按顺序拼接。对于 reduction kernel，loads/compute/stores 被放入一个 Python `for` 循环体内，循环外的 `post_loop_combine` 和 `post_loop_store` 负责归约的最终合并和结果写出。

### 7.3.7 Autotuning 集成

Triton kernel 的性能高度依赖于配置参数：`BLOCK_SIZE`（每个线程块处理的元素数）、`num_warps`（每个线程块的 warp 数）、`num_stages`（pipeline 阶段数）。不同的输入大小和硬件特性需要不同的最优配置。

Inductor 通过 autotuning 机制自动选择最优配置：

```
TritonScheduling 的 autotuning 流程：

1. 创建 kernel，使用默认配置（如 BLOCK_SIZE=128）
2. 生成 kernel 代码
3. 如果启用 benchmark：
   a. 在 GPU 上运行 benchmark
   b. 尝试多种配置（BLOCK_SIZE, num_warps 的组合）
   c. 选择最快的配置
4. 使用最优配置重新生成 kernel
```

在生成的代码中，autotuning 体现为 `@triton_heuristics` 装饰器：

```python
@triton_heuristics.pointwise(
    size_hints=[4096],              # 输入大小的提示
    filename=__file__,
    triton_meta={...},              # Triton 编译器元数据
    inductor_meta={...},            # Inductor 元数据
    min_elem_per_thread=1           # 每线程最少处理的元素数
)
@triton.jit
def kernel_name(...):
    ...
```

### 7.3.8 完整追踪示例：融合 relu + add

让我们追踪一个完整的代码生成过程。假设我们有以下 PyTorch 代码：

```python
import torch

@torch.compile
def f(x, y):
    return torch.relu(x + y)
```

经过前端、Dynamo、AOTAutograd、Inductor lowering 后，调度器生成了一个融合的 `FusedSchedulerNode`，包含两个 IR 操作：`add` 和 `relu`。

**Step 1：TritonScheduling 创建 TritonKernel**

```python
# TritonScheduling.codegen_nodes() 中
tiling = {"x": N}  # N = x.numel()
features = SIMDKernelFeatures([], N)
kernel = TritonKernel(tiling, features=features)
```

**Step 2：进入 kernel 上下文**

```python
with kernel:
    # V.ops = CSEProxy(TritonKernelOverrides())
    # V.kernel = kernel
```

此时 kernel 的 body 已包含序言代码：
```python
xoffset = tl.program_id(0) * XBLOCK
xindex = xoffset + tl.arange(0, XBLOCK)
xmask = xindex < xnumel
```

**Step 3：执行 add IR 闭包**

IR 闭包内的 FX graph 包含：
```
load(x, index) → load(y, index) → add(loaded_x, loaded_y) → store("buf_add", index, result)
```

3a. `V.ops.load("x", xindex)`：
- CSEProxy 检查 CSE 缓存：未命中
- 委托给 `TritonKernelOverrides`（无 `load` 方法，使用 kernel 的 `load`）
- `TritonKernel.load("x", xindex)` 生成：
  ```python
  tmp0 = tl.load(in_ptr0 + xindex, mask=xmask)
  ```
  写入 `kernel.loads`，返回 `TritonCSEVariable("tmp0")`

3b. `V.ops.load("y", xindex)`：
- CSE 缓存未命中
- 生成：
  ```python
  tmp1 = tl.load(in_ptr1 + xindex, mask=xmask)
  ```
  写入 `kernel.loads`，返回 `TritonCSEVariable("tmp1")`

3c. `V.ops.add(tmp0, tmp1)`：
- CSEProxy 检查 CSE 缓存：键 `("add", "tmp0", "tmp1")` 未命中
- 委托给 `TritonKernelOverrides.add(tmp0, tmp1)`
- `TritonOverrides.add` 返回字符串 `"tmp0 + tmp1"`
- CSEProxy 调用 `cse.generate(compute, "tmp0 + tmp1", ...)` 生成：
  ```python
  tmp2 = tmp0 + tmp1
  ```
  写入 `kernel.compute`，返回 `TritonCSEVariable("tmp2")`

3d. `V.ops.store("buf_add", xindex, tmp2)`：
- 更新 store cache：`buf_add → tmp2`
- kernel 的 `store` 方法生成：
  ```python
  tl.store(buf_add_ptr + xindex, tmp2, mask=xmask)
  ```
  写入 `kernel.stores`

**Step 4：执行 relu IR 闭包**

IR 闭包内的 FX graph：
```
load("buf_add", index) → relu(loaded) → store("output", index, result)
```

4a. `V.ops.load("buf_add", xindex)`：
- **CSE store cache 命中**！`buf_add` 的最近一次 store 值是 `tmp2`
- 直接返回 `TritonCSEVariable("tmp2")`，不生成任何 load 代码

4b. `V.ops.relu(tmp2)`：
- CSE 缓存未命中
- `TritonOverrides.relu(tmp2)` 返回 `"tl.where(tmp2 > 0, tmp2, 0)"`
- CSEProxy 生成：
  ```python
  tmp3 = tl.where(tmp2 > 0, tmp2, 0)
  ```
  写入 `kernel.compute`，返回 `TritonCSEVariable("tmp3")`

4c. `V.ops.store("output", xindex, tmp3)`：
- 更新 store cache：`output → tmp3`
- 生成：
  ```python
  tl.store(out_ptr2 + xindex, tmp3, mask=xmask)
  ```
  写入 `kernel.stores`

**Step 5：退出 kernel 上下文**

```python
# kernel.__exit__()
# remove_kernel_local_buffers() 发现 "buf_add" 在 kernel 内创建且最后使用也在 kernel 内
# → 标记为 removed，不作为输出参数
```

**Step 6：调用 codegen_kernel() 组装最终代码**

```python
code = kernel.codegen_kernel(name="fused_relu_add")
```

最终生成的完整 Triton kernel：

```python
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import pointwise
...

@pointwise(
    size_hints=[4096],
    filename=__file__,
    triton_meta={"signature": {...}, "device": ...},
    inductor_meta={"kernel_name": "fused_relu_add", ...}
)
@triton.jit
def fused_relu_add(in_ptr0, in_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    # --- loads ---
    tmp0 = tl.load(in_ptr0 + xindex, mask=xmask)
    tmp1 = tl.load(in_ptr1 + xindex, mask=xmask)
    # --- compute ---
    tmp2 = tmp0 + tmp1
    tmp3 = tl.where(tmp2 > 0, tmp2, 0)
    # --- stores ---
    tl.store(out_ptr2 + xindex, tmp3, mask=xmask)
```

注意几个关键点：

1. **CSE store cache 的效果**：`buf_add` 的 load 被 store cache 拦截，避免了冗余的 `tl.load`。
2. **中间 buffer 消除**：`buf_add` 被识别为 kernel-local buffer，从输出参数中移除。
3. **三段式组织**：load、compute、store 代码被分别收集，最终按顺序拼接。

### 7.3.9 输入与输出

- **输入**：`FusedSchedulerNode`（包含多个 `SchedulerNode` 的融合组），以及 tiling 配置和 kernel 特征信息。
- **输出**：完整的 `@triton.jit` kernel Python 源代码 + autotuning 元数据 + launch wrapper 代码。

---

## 7.4 TritonSplitScanKernel —— 分裂扫描模式

### 7.4.1 问题背景

前缀扫描（prefix scan）操作，如 `torch.cumsum`、`torch.cumprod`、`torch.cummax`，在并行计算中是一个经典难题。与 pointwise 或 reduction 操作不同，scan 操作具有**数据依赖性**：每个输出元素依赖于所有前驱元素的组合结果。

```
输入:   [1, 3, 5, 2, 4]
cumsum: [1, 4, 9, 11, 15]
         ↑  ↑  ↑   ↑   ↑
         1  1+3 1+3+5 ...  逐元素依赖
```

简单的并行-for 模式无法处理这种依赖。`TritonKernel` 的 pointwise 模式假设每个元素的计算是独立的，reduction 模式假设所有元素归约为一个值。两者都不适用于 scan。

### 7.4.2 Split-Scan 算法

源码位置：`torch/_inductor/codegen/triton_split_scan.py:21`

`TritonSplitScanKernel` 使用**分裂扫描（split-scan）**算法，将 scan 操作分解为三个阶段：

```
阶段 1：局部扫描（Local Scan）
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Block 0  │ │ Block 1  │ │ Block 2  │
│ [1,3,5,2]│ │ [4,7,1,3]│ │ [2,8,6,5]│
│ →[1,4,9,11]│→[4,11,12,15]│→[2,10,16,21]│
│ 携带=11  │ │ 携带=15  │ │ 携带=21  │
└──────────┘ └──────────┘ └──────────┘

阶段 2：携带值扫描（Carry Scan）
  携带值 = [11, 15, 21]
  cumsum = [11, 26, 47]

阶段 3：修正（Fixup）
  Block 0: [1, 4, 9, 11]     (不变)
  Block 1: [4+11, 11+11, 12+11, 15+11] = [15, 22, 23, 26]
  Block 2: [2+26, 10+26, 16+26, 21+26] = [28, 36, 42, 47]
```

这种算法允许不同 block 之间的并行执行（阶段 1 和阶段 3），同时通过全局 workspace buffer 进行跨 block 通信（阶段 2）。

### 7.4.3 实现要点

```python
class TritonSplitScanKernel(TritonKernel):
    """Generates a triton kernel that supports ops.scan calls while also
    splitting the reduction dimension over multiple triton programs.

    For this kernel, loop numels will always take the form (xdim, rdim)
    and the grid has the shape (CeilDiv(rdim, RBLOCK), xdim).
    Communication between blocks occurs within a global memory workspace
    buffer, which must be zero-filled before launching the kernel.

    Note that generation for ops.reduction is not supported.
    """

    def __init__(self, tiling, pid_cache=None, fixed_config=None, **kwargs):
        super().__init__(tiling, **kwargs)
        self.no_x_dim = True  # 使用特殊的 x 维度处理

    def should_use_persistent_reduction(self) -> bool:
        return False  # 禁用持久归约

    def should_use_cooperative_reduction(self) -> bool:
        return False  # 禁用协作归约

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError("NYI TritonSplitDimKernel reductions")
```

关键设计决策：

1. **Grid 布局**：`(CeilDiv(rdim, RBLOCK), xdim)`——归约维度在 grid 的第一个维度，允许足够多的 block 并行处理。
2. **Workspace buffer**：使用全局内存作为跨 block 通信的中间缓冲区，在 kernel 启动前需要被零初始化。
3. **不支持 reduction**：scan kernel 只处理 scan 操作，不混合 reduction 操作。

### 7.4.4 通信策略

`TritonSplitScanKernel` 采用 **Decoupled Look-Back** 算法（参考 NVIDIA 2016 论文 "Single-pass Parallel Prefix Scan with Decoupled Look-back"），将 workspace 分为三个区域：

```
Workspace 布局（每个 block 对应一个槽位）：
┌──────────────┬──────────────┬──────────────┐
│ prefix 值     │ 携带值        │ agg 值        │
│ (本 block 及  │ (本 block 的  │ (本 block 的  │
│  之前所有     │ 局部 scan    │ 聚合值，      │
│  block 的     │ 最终值)       │ 不含之前)     │
│  scan 结果)   │              │              │
└──────────────┴──────────────┴──────────────┘
```

每个 block 依次执行：
1. 计算本 block 的局部 scan 和 agg 值
2. 将 agg 值写入 workspace
3. 从后往前扫描（look-back），累加前面所有 block 的 prefix
4. 用累加得到的 carry 值修正本 block 的输出

---

## 7.5 CppKernel 继承树

### 7.5.1 架构总览

CPU 端的 kernel 代码生成由一棵继承树组成，每个类针对不同的优化场景：

```
Kernel
  └── CppKernel              (标量 C++ 循环)
        └── CppVecKernel     (SIMD 向量化)
              └── CppTile2DKernel  (2D 分块向量化)
        └── CppKernelProxy   (代理/协调器，非独立 kernel)
        └── OuterLoopFusedKernel  (外层循环融合)
```

与 Triton 后端的"展平索引"哲学不同，CPU 后端使用**嵌套循环**来遍历数据，这与 CPU 的顺序执行模型一致。每个子类代表一种不同的优化策略。

### 7.5.2 CppKernel —— 标量 C++ 循环

源码位置：`torch/_inductor/codegen/cpp.py:1876`

```python
class CppKernel(Kernel):
    overrides = CppOverrides
    sexpr = cexpr             # C++ 表达式序列化器
    newvar_prefix = "auto "   # C++ auto 类型声明
    suffix = ";"              # C++ 语句分号
```

`CppKernel` 生成最基础的 C++ 嵌套循环代码。`newvar_prefix = "auto "` 和 `suffix = ";"` 是 C++ 语法的要求——每个新生成的变量声明为 `auto tmp0 = ...;`。

生成的代码示例：

```cpp
auto tmp0 = in_ptr0[i * N + j];     // load
auto tmp1 = in_ptr1[i * N + j];     // load
auto tmp2 = tmp0 + tmp1;            // compute
auto tmp3 = std::max(tmp2, (scalar_t)0);  // compute (relu)
out_ptr0[i * N + j] = tmp3;         // store
```

完整的嵌套循环：

```cpp
extern "C" void kernel(const float* in_ptr0, const float* in_ptr1,
                       float* out_ptr0, int64_t M, int64_t N) {
    #pragma omp parallel for
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            auto tmp0 = in_ptr0[i * N + j];
            auto tmp1 = in_ptr1[i * N + j];
            auto tmp2 = tmp0 + tmp1;
            auto tmp3 = std::max(tmp2, (float)0);
            out_ptr0[i * N + j] = tmp3;
        }
    }
}
```

`CppKernel` 的初始化包含大量与 reduction 相关的缓冲区：

```python
def __init__(self, args, num_threads):
    super().__init__(args)
    self.active_ranges = {}
    self.ranges = []
    self.itervars = []
    self.reduction_depth = None
    self.reduction_prefix = IndentedBuffer()
    self.reduction_suffix = IndentedBuffer()
    self.parallel_reduction_prefix = IndentedBuffer()
    self.parallel_reduction_suffix = IndentedBuffer()
    self.local_reduction_init = IndentedBuffer()
    self.local_reduction_stores = IndentedBuffer()
    self.reduction_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="tmp_acc")
    self.welford_helper_cse = CSE(self.newvar_prefix, self.suffix, name_prefix="welford_helper")
    self.preloads = IndentedBuffer()
    self.poststores = IndentedBuffer()
    self.num_threads = num_threads
```

CPU 端的归约比 GPU 端更复杂，因为需要处理 OpenMP 并行归约（多线程）和串行归约（单线程）两种模式。`parallel_reduction_prefix` 和 `non_parallel_reduction_prefix` 分别对应这两种模式的初始化代码。

### 7.5.3 CppVecKernel —— SIMD 向量化

源码位置：`torch/_inductor/codegen/cpp.py:2449`

```python
class CppVecKernel(CppKernel):
    overrides = CppVecOverrides

    def __init__(self, args, num_threads, tiling_factor, tiling_idx, tail_size=None):
        super().__init__(args, num_threads)
        self.vec_isa = cpu_vec_isa.pick_vec_isa()   # 选择 SIMD ISA（AVX2/AVX512等）
        self.tiling_factor = tiling_factor            # 每次迭代处理的元素数
        self.tiling_idx = tiling_idx                  # 被向量化的循环变量索引
        self.tail_size = tail_size                    # 尾部剩余元素数
```

`CppVecKernel` 使用 CPU 的 SIMD 指令（如 AVX2、AVX-512、NEON）来加速计算。`tiling_factor` 表示每次向量操作处理的元素数，例如 AVX2 的 `float32` 向量宽度为 8（256 bit / 32 bit）。

生成的代码示例（AVX2 向量化的 relu + add）：

```cpp
extern "C" void kernel(const float* in_ptr0, const float* in_ptr1,
                       float* out_ptr0, int64_t N) {
    int64_t i = 0;
    // 向量化主循环：每次处理 8 个 float
    for (; i < N - (N % 8); i += 8) {
        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + i);
        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + i);
        auto tmp2 = tmp0 + tmp1;
        auto tmp3 = at::vec::clamp_min(tmp2, (float)0);  // relu
        tmp3.store(out_ptr0 + i);
    }
    // 尾部标量处理
    for (; i < N; i++) {
        auto tmp0 = in_ptr0[i];
        auto tmp1 = in_ptr1[i];
        auto tmp2 = tmp0 + tmp1;
        out_ptr0[i] = std::max(tmp2, (float)0);
    }
}
```

`CppVecKernel` 的关键方法：

- `_get_vec_type(dtype)`：返回向量类型，如 `at::vec::Vectorized<float>` 或 `at::vec::VectorizedN<float, 2>`（当 tiling_factor 超过一个向量宽度时）
- `_get_vec_load_line(var, index, dtype, load_mask)`：生成向量化 load 代码
- `_try_get_const_stride(index, itervar)`：检查索引是否具有常量步长，以优化地址计算

### 7.5.4 CppTile2DKernel —— 2D 分块向量化

源码位置：`torch/_inductor/codegen/cpp.py:3307`

```python
class CppTile2DKernel(CppVecKernel):
    """
    A vector kernel that handles the 2d tiles with the tile size defined
    in `tiling_factor` on the inner-most loop level and one of the outer
    loop level (`outer_tiling_idx`). When the data tile is accessed in a
    contiguous way from the outer loop axis, a transposition is applied
    on the tile to make the access contiguous from the inner-most loop axis.
    """
```

`CppTile2DKernel` 针对 2D 访问模式（如矩阵乘法中的分块访问）进行优化。它的核心思想是：当数据在某个外层循环维度上是连续访问的，先将数据 tile 加载到临时数组中，执行转置（transpose），然后在内层循环中以连续方式使用向量化操作。

循环结构示意：

```cpp
for (...) {                           // 外层循环
    for (int i_outer = 0; ...) {      // 分块的外层维度
        for (...) {                   // 中间循环
            for (int inner = 0; ...) {// 最内层循环
                // --- preloads (生成到 kernel.preloads) ---
                float tmp0[16*16];
                at::vec::transpose_mxn<...>(tmp0, in_ptr0 + ..., ...);
                float tmp1[16*16];
                // --- 向量化主循环 ---
                for (int i_inner = 0; ...) {  // kernel 内部循环
                    // vectorized loads/compute/stores
                }
                // --- poststores (生成到 kernel.poststores) ---
                at::vec::transpose_mxn(out_ptr0 + ..., tmp1, ...);
            }
            // 尾部处理（tail）
        }
        // 尾部处理（outer tail）
    }
}
```

### 7.5.5 OuterLoopFusedKernel —— 外层循环融合

源码位置：`torch/_inductor/codegen/cpp.py:4412`

```python
class OuterLoopFusedKernel(CppKernel):
    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.inner: list[LoopNest] = []
```

当多个独立的操作共享相同的外层循环边界时，`OuterLoopFusedKernel` 将它们融合到同一个外层循环中：

```cpp
for (int i = 0; i < M; i++) {       // 共享的外层循环
    // 操作 1：add
    for (int j = 0; j < N; j++) {
        out1[i*N+j] = a[i*N+j] + b[i*N+j];
    }
    // 操作 2：mul（共享同一个 i 循环，但内部 j 循环独立）
    for (int j = 0; j < N; j++) {
        out2[i*N+j] = c[i*N+j] * d[i*N+j];
    }
}
```

`decide_parallel_depth` 方法决定在哪个循环层级启用 OpenMP 并行，确保所有融合的子 kernel 在并行深度上达成一致。

### 7.5.6 CppKernelProxy —— 代码生成协调器

源码位置：`torch/_inductor/codegen/cpp.py:3800`

```python
class CppKernelProxy(CppKernel):
    kernel_cls: type[CppKernel] = CppKernel
    vec_kernel_cls: type[CppVecKernel] = CppVecKernel
    tile2d_kernel_cls: type[CppTile2DKernel] = CppTile2DKernel

    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.kernel_group = kernel_group
        self.loop_nest = None
        self.call_ranges = None
        self.picked_vec_isa = cpu_vec_isa.pick_vec_isa()
        self.kernels: list[CppKernel] = []
```

`CppKernelProxy` **不是**一个独立的 kernel——它是一个**代理/协调器**，管理多个子 kernel 的创建和代码拼接。

它的核心职责是**决策使用哪种 kernel 类型**：

```
CppKernelProxy 的决策流程：

1. 分析节点的循环结构
2. 如果可以向量化：
   a. 检查是否是 2D tile 模式 → 使用 CppTile2DKernel
   b. 否则 → 使用 CppVecKernel
3. 如果不能向量化：
   → 使用 CppKernel（标量）
```

编译器类比：`CppKernelProxy` 类似于编译器中的**代码生成协调器（code generation coordinator）**——它不直接生成代码，而是根据目标特征选择最合适的代码生成策略，并协调多个代码生成器的输出。

---

## 7.6 辅助类

### 7.6.1 KernelArgs —— kernel 参数管理器

源码位置：`torch/_inductor/codegen/common.py:1392`

```python
class KernelArgs:
    def __init__(self):
        self.input_buffers: dict[str, str] = {}       # 输入缓冲区映射
        self.output_buffers: dict[str, Union[str, RemovedArg]] = {}  # 输出缓冲区映射
        self.inplace_buffers: dict[str, Union[InplacedBuffer, RemovedArg]] = {}  # 原地缓冲区
        self.sizevars: dict[sympy.Expr, str] = {}     # 大小变量映射
        self.workspace_args: list[WorkspaceArg] = []   # workspace 参数
```

`KernelArgs` 管理一个 kernel 的所有参数。它的核心功能是将逻辑 buffer 名称映射到物理参数名称：

```python
def input(self, name: str) -> str:
    # 查找或分配输入参数名称
    # 第一次遇到 "x" → "in_ptr0"
    # 第一次遇到 "y" → "in_ptr1"
    if name in self.output_buffers:
        return self.output_buffers[name]     # 输出也是输入（inplace）
    if name in self.inplace_buffers:
        return self.inplace_buffers[name].inner_name  # 原地操作
    return self._lookup("in_ptr", self.input_buffers, name)  # 新输入
```

参数映射的示例：

```
逻辑 buffer 名     →    参数名       →    说明
"buf_add"          →    "in_ptr0"    →    add 的输入 x
"buf_bias"         →    "in_ptr1"    →    add 的输入 y
"output"           →    "out_ptr2"   →    最终输出
"buf_add" (removed)→    REMOVED      →    kernel-local，不传参
"workspace"        →    (workspace)  →    split-scan workspace
```

`python_argdefs()` 方法生成完整的函数参数声明：

```python
argdefs, call_args, signature, _ = self.args.python_argdefs()
# argdefs: ["in_ptr0", "in_ptr1", "out_ptr2", "xnumel"]
# call_args: 对应的实际参数
# signature: 类型签名信息
```

### 7.6.2 CSE —— 公共子表达式消除

源码位置：`torch/_inductor/codegen/common.py:1772`

CSE（Common Subexpression Elimination）是代码生成阶段最重要的优化之一。它避免对相同的计算生成重复代码。

```python
class CSE(Generic[CSEVariableType, AugmentedKeyT]):
    def __init__(self, prefix="", suffix="", name_prefix="tmp", ...):
        self._cache: MutableMapping[AugmentedKeyT, CSEVariableType] = {}
        self.store_cache: MutableMapping[str, CSEVariableType] = {}
        self.reduction_cache: MutableMapping[ReductionCacheKey, CSEVariableType] = {}
        self.iter_buffer_ids: itertools.count[int] = itertools.count()
```

CSE 维护三个缓存：

**1. _cache：表达式缓存**

检测重复的计算表达式。键是 `(op_name, arg1, arg2, ...)` 元组，值是已生成的 CSEVariable。

```
第一次：ops.add(a, b)
  → 缓存未命中
  → 生成 "tmp0 = a + b"
  → 缓存：("add", "tmp_in0", "tmp_in1") → CSEVariable("tmp0")

第二次：ops.add(a, b)  （相同参数）
  → 缓存命中！直接返回 CSEVariable("tmp0")
  → 不生成任何代码
```

**2. store_cache：存储缓存**

跟踪最近一次对每个 buffer 的 store 操作，消除紧随其后的冗余 load。

```
ops.store("buf", idx, val) → store_cache["buf"] = val

ops.load("buf", idx) → 检查 store_cache["buf"]
  → 如果命中：直接返回 val，不生成 tl.load
  → 如果未命中：正常生成 tl.load
```

这个优化在融合 kernel 中特别有效：一个节点的 store 结果直接作为下一个节点的 load 输入，无需通过全局内存中转。

**3. reduction_cache：归约缓存**

避免对相同的归约操作生成重复代码。

**generate 方法**：CSE 的核心入口

```python
def generate(self, buffer, value, bounds=None, dtype=None):
    # 将 value 包装为 CSEVariable
    var_name = f"{self.name_prefix}{next(self.iter_buffer_ids)}"
    var = V.kernel.create_cse_var(var_name, bounds, dtype)

    # 写入目标缓冲区
    buffer.writeline(
        DeferredLine(name, f"{self.prefix}{var_name} = {value}{self.suffix}")
    )

    # 增加使用计数
    var.use_count += 1
    return var
```

**invalidate 方法**：在归约循环边界调用

```python
def invalidate(self, keep_vars):
    # 清除不包含在 keep_vars 中的 store_cache 条目
    for name, tmp in [*self.store_cache.items()]:
        if tmp not in keep_vars:
            del self.store_cache[name]
    # 清除表达式缓存
    if keep_vars:
        self._cache = {k: v for k, v in self._cache.items() if v in keep_vars}
    else:
        self._cache = {}
```

在归约循环中，每次循环迭代都会改变归约变量的值。因此，在循环边界处必须使 CSE 缓存失效（除了跨越循环迭代的归约累加器变量）。`invalidate` 方法正是处理这种情况。

### 7.6.3 CSEProxy —— CSE 与后端 Overrides 的桥梁

源码位置：`torch/_inductor/codegen/common.py:2375`

`CSEProxy` 是一个 `OpsHandler`，它拦截所有 `V.ops` 调用，首先尝试 CSE，如果未命中则委托给后端 overrides：

```python
class CSEProxy(DefaultHandler):
    def __init__(self, kernel, parent_handler):
        self.vr_analysis = ValueRangeAnalysis()
        self.kernel = kernel
        self.parent_handler = parent_handler  # 后端 overrides（如 TritonKernelOverrides）

    def _default(self, name, args, kwargs):
        # 1. 计算值域
        bounds = self._bound_variable(name, *args, **kwargs)
        # 2. 调用后端 overrides 生成代码字符串
        value = getattr(self.parent_handler, name)(*args, **kwargs)
        # 3. 传播 dtype
        ...
        # 4. 通过 CSE 生成或缓存变量
        csevar = V.kernel.cse.generate(V.kernel.compute, v, bounds=bounds, dtype=...)
        return csevar
```

调用链路：

```
IR 闭包调用 V.ops.add(a, b)
  → CSEProxy._default("add", (a, b), {})
    → 检查 CSE 缓存
      → 命中：直接返回缓存的 CSEVariable
      → 未命中：
        → TritonKernelOverrides.add(a, b) → "a + b"（字符串）
        → CSE.generate(compute, "a + b") → CSEVariable("tmp2")
        → compute.writeline("tmp2 = a + b")
        → 缓存 ("add", a, b) → tmp2
    → 返回 CSEVariable("tmp2")
```

### 7.6.4 CSEVariable —— 代码中的命名变量

源码位置：`torch/_inductor/codegen/common.py:1725`

```python
class CSEVariable:
    def __init__(self, name, bounds, dtype=None):
        self.name = name              # 变量名（如 "tmp0"）
        self.bounds = bounds          # 值域范围（ValueRanges）
        self.use_count = 1            # 被引用次数
        self.dtype = dtype            # 数据类型（torch.float32 等）
```

`CSEVariable` 是代码生成阶段的基本流通货币。IR 闭包的每次 `ops.xxx()` 调用都返回一个 `CSEVariable`，后续的操作使用它作为输入。`use_count` 跟踪变量的使用次数，可用于死代码消除。

不同后端可以扩展 `CSEVariable`：

- `TritonCSEVariable`：添加 `mask_vars` 属性，记录该变量依赖的 mask 变量集合
- `CppCSEVariable`：添加 `is_vec` 属性，标记变量是否是向量类型

### 7.6.5 IndentedBuffer —— 代码缓冲区

源码位置：`torch/_inductor/utils.py:1198`

```python
class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent=0):
        self._lines = []
        self._indent = initial_indent
```

`IndentedBuffer` 是所有代码生成的基础工具类，提供自动缩进管理：

```python
buf = IndentedBuffer()
buf.writeline("def kernel():")       # "def kernel():"
with buf.indent():
    buf.writeline("x = 1")           # "    x = 1"
    buf.writeline("y = 2")           # "    y = 2"
    with buf.indent():
        buf.writeline("return x + y") # "        return x + y"
```

核心方法：

- `writeline(line)`：写入一行代码（自动添加前导空格）
- `splice(code)`：批量插入多行代码
- `indent()` / `dedent()`：增加/减少缩进级别（上下文管理器）
- `getvalue()`：返回完整的代码字符串

### 7.6.6 BracesBuffer —— C++ 花括号缓冲区

源码位置：`torch/_inductor/codegen/common.py:1348`

```python
class BracesBuffer(IndentedBuffer):
    def indent(self, offset=1):
        @contextmanager
        def ctx():
            for _ in range(offset):
                self.writeline("{")
                self._indent += 1
            yield
            for _ in range(offset):
                self._indent -= 1
                self.writeline("}")
        return ctx()
```

`BracesBuffer` 扩展了 `IndentedBuffer`，在每次增加缩进时自动写入 `{`，减少缩进时自动写入 `}`。这是 C/C++ 代码生成的专用工具：

```cpp
for (int i = 0; i < N; i++) {    ← indent() 自动添加
    auto tmp0 = buf[i];           ← 自动缩进
}                                 ← dedent() 自动添加
```

---

## 7.7 模板 Kernel

### 7.7.1 设计动机

并非所有操作都适合自动生成代码。对于某些计算密集型操作，手工优化的 kernel 模板性能远优于自动生成的代码。典型的例子包括：

- **矩阵乘法（GEMM）**：涉及复杂的分块策略、共享内存利用和流水线优化
- **Flash Attention**：特定的 IO 感知算法，分块加载 Q/K/V
- **Split-K GEMM**：将 GEMM 的归约维度拆分到多个线程块

这些操作的 kernel 由专家手工编写，作为"模板"使用。Inductor 在调度阶段识别匹配的操作，将它们路由到模板 kernel 而非自动生成。

### 7.7.2 TritonTemplateKernel

源码位置：`torch/_inductor/select_algorithm.py:294`

```python
class TritonTemplateKernel(TritonKernel):
    def __init__(self, kernel_name, input_nodes, output_node, defines,
                 num_stages, num_warps, grid_fn, meta, call_sizes, ...):
        numel = sympy_product(output_node.get_size())
        super().__init__(
            {"x": numel, "r0_": sympy.S.One},
            features=SIMDKernelFeatures([], numel),
        )
        self.kernel_name = kernel_name    # 模板 kernel 的名称
        self.input_nodes = input_nodes    # 输入节点
        self.output_node = output_node    # 输出节点
        self.defines = defines            # 模板定义代码
        self.num_stages = num_stages      # pipeline 阶段数
        self.num_warps = num_warps        # warp 数
        self.grid_fn = grid_fn            # grid 大小函数
        self.meta = meta                  # Triton 元数据
        ...
```

`TritonTemplateKernel` 继承自 `TritonKernel`，但与常规 `TritonKernel` 的关键区别是：**kernel 主体不是通过 IR 闭包自动生成的，而是使用预定义的模板代码**。

模板 kernel 的典型使用流程：

```
1. Scheduler 识别到一个 GEMM 操作
2. 创建 TemplateSchedulerNode（而非普通 SchedulerNode）
3. TritonScheduling 将 TemplateSchedulerNode 路由到 TritonTemplateKernel
4. 模板 kernel 使用预定义的 Triton 代码，填入参数（BLOCK_M, BLOCK_N, 等）
5. Autotuning 调优模板参数
```

模板 kernel 通常有以下可调参数：

| 参数          | 含义                      | 典型值              |
| ------------- | ------------------------- | ------------------- |
| `BLOCK_M`     | M 维度的分块大小           | 64, 128, 256        |
| `BLOCK_N`     | N 维度的分块大小           | 64, 128, 256        |
| `BLOCK_K`     | K 维度的分块大小           | 32, 64              |
| `num_warps`   | 每 block 的 warp 数        | 4, 8                |
| `num_stages`  | pipeline 阶段数            | 2, 3, 4             |

### 7.7.3 CUDATemplateKernel

源码位置：`torch/_inductor/codegen/cuda/cuda_kernel.py:190`

```python
class CUDATemplateKernel(CUDAKernel):
    """
    Template kernels defined by CUDA / Cutlass in C++.
    """

    def __init__(self, kernel_name, runtime_arg_info, runtime_arg_values):
        super().__init__()
        self.kernel_name = kernel_name
        self.runtime_arg_info = runtime_arg_info
        self.runtime_arg_values = runtime_arg_values
```

`CUDATemplateKernel` 包装 CUTLASS 模板 kernel。CUTLASS 是 NVIDIA 提供的高性能矩阵操作库，使用 C++ 模板元编程实现。与 Triton 模板不同，CUTLASS 模板生成的是 C++/CUDA 代码而非 Triton Python。

`CUDATemplateKernel` 的 `def_kernel` 方法生成 CUDA kernel 的函数定义和参数绑定：

```python
def def_kernel(self, inputs, outputs, names_str="", input_reorder=None):
    """Hook called from template code to generate function definition."""
    ...
```

### 7.7.4 模板 vs 自动生成：决策机制

调度阶段负责决定使用模板还是自动生成。决策逻辑在 `TritonScheduling`（或 `CUDACombinedScheduling`）中：

```
调度决策流程：

SchedulerNode
  ├── 是否匹配某个模板的 pattern？
  │   ├── 是 → TemplateSchedulerNode → TritonTemplateKernel / CUDATemplateKernel
  │   │        （如：GEMM → flash_attention_template, split_k_gemm）
  │   └── 否 → 普通 SchedulerNode → TritonKernel / CppKernel
  │            （如：pointwise add, reduction sum）
  └── 特殊操作（scan） → TritonSplitScanKernel
```

模板的优势：

1. **极致性能**：手工优化的分块、共享内存利用、寄存器分配
2. **算法级优化**：如 Flash Attention 的 IO 感知算法，自动生成难以复制
3. **领域专家知识**：利用硬件特性的深度知识

自动生成的优势：

1. **通用性**：处理任意计算模式
2. **融合能力**：自动将多个操作融合到同一个 kernel
3. **维护成本低**：无需为每种操作手写 kernel

理想情况下，Inductor 会优先尝试模板匹配，未匹配的操作走自动生成路径。两种路径的代码最终通过统一的 wrapper 层被调用。

---

## 7.8 总结：从 IR 到机器码的完整路径

让我们用一个端到端的例子回顾整个代码生成过程：

```
Python 用户代码：
    c = torch.relu(a + b)

    ↓ Dynamo (符号执行)
    ↓ AOTAutograd (联合图)
    ↓ Inductor Lowering (生成 IR)

IR 表示：
    buf_add = Buffer("add", shape=[N])
    buf_relu = Buffer("relu", shape=[N])

    ↓ Scheduler (调度)
    ↓ 融合决策：add + relu → FusedSchedulerNode

    ↓ TritonScheduling (GPU 路径)

TritonKernel 代码生成：
    1. 创建 TritonKernel(tiling={"x": N}, features=...)
    2. kernel.__enter__() → 安装 CSEProxy + TritonKernelOverrides
    3. 执行 add IR 闭包 → loads/compute 段
    4. 执行 relu IR 闭包 → compute 段（CSE 命中 buf_add 的 load）
    5. kernel.__exit__() → 清理 kernel-local buffers
    6. codegen_kernel() → 组装完整 Triton kernel

最终输出（GPU）：
    @triton.jit
    def fused_relu_add(in_ptr0, in_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + xindex, mask=xmask)
        tmp1 = tl.load(in_ptr1 + xindex, mask=xmask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.where(tmp2 > 0, tmp2, 0)
        tl.store(out_ptr2 + xindex, tmp3, mask=xmask)

    ↓ CppScheduling (CPU 路径)

最终输出（CPU）：
    extern "C" void kernel(const float* in_ptr0, const float* in_ptr1,
                           float* out_ptr0, int64_t xnumel) {
        #pragma omp parallel for
        for (int64_t i = 0; i < xnumel; i++) {
            auto tmp0 = in_ptr0[i];
            auto tmp1 = in_ptr1[i];
            auto tmp2 = tmp0 + tmp1;
            auto tmp3 = std::max(tmp2, (float)0);
            out_ptr0[i] = tmp3;
        }
    }
```

**本章核心要点**：

1. `Kernel` 的三段式模型（loads / compute / stores）是所有代码生成的基础抽象，分离了内存访问和计算逻辑。
2. `SIMDKernel` 引入了展平索引和 range tree 的概念，是 GPU/SIMD 后端的共同基础设施。
3. `TritonKernel` 通过 `TritonKernelOverrides` + `CSEProxy` 的组合，将 IR 闭包执行转化为 Triton Python 代码。
4. CSE 的 store cache 是融合 kernel 避免冗余内存访问的关键机制。
5. CPU 后端使用嵌套循环模型，通过 `CppKernelProxy` 在标量、向量化和 2D 分块之间选择。
6. 模板 kernel 为特定操作提供了手工优化的高性能路径，与自动生成路径互补。

在下一章中，我们将讨论生成后的 kernel 如何被打包、序列化并最终执行——即 wrapper 层和运行时系统的设计。
