# 第二章：全局基础设施 —— Virtualized 动态作用域引擎

## 2.1 Virtualized<T> — 线程局部动态作用域容器

### 2.1.1 编译器中的全局状态问题

任何多遍（multi-pass）编译器都需要在各个阶段之间共享全局上下文。在传统的编译器实现中，这些状态通常通过以下三种方式管理：

1. **显式参数传递**：每个函数签名都携带一个 `context` 或 `pass_data` 参数。缺点是调用链越深，参数穿透越痛苦。
2. **全局变量**：直接定义模块级变量。缺点是多线程不安全，嵌套编译场景容易出错。
3. **单例对象**：用一个全局单例（如 LLVM 的 `LLVMContext`）封装所有状态。缺点是切换上下文需要手动保存/恢复，遗漏即出 bug。

GCC 是典型的"全局变量"派——源码中到处都是 `current_function_decl`、`cfun`、`input_location` 等全局变量。LLVM 则走"显式传递"路线——几乎所有 API 都要求传入 `LLVMContext&`。两条路都能工作，但各有痛点。

PyTorch Inductor 选择了一条不同的路：**动态作用域（Dynamic Scoping）**。

### 2.1.2 动态作用域：从 Lisp 到 Inductor

动态作用域（Dynamic Scoping）是一个古老的概念，最早可追溯到 Lisp 语言的 `fluid-let` 和 Scheme 的 `parameterize`。它的核心思想是：**变量的值不取决于它在源码中的词法位置（lexical scope），而取决于运行时的调用栈（call stack）。**

用一个简化的例子说明。在静态作用域（Python、C、Java）中：

```python
x = 10
def foo():
    print(x)  # 打印 10，取决于定义时的词法环境
```

在动态作用域（Lisp 的 `fluid-let`，或 Inductor 的 `V.ops`）中：

```python
# 伪代码：动态作用域
handler = DefaultHandler

def inner_fn():       # inner_fn 不知道当前 handler 是什么
    handler.add(a, b) # 取决于谁在调用它、运行时绑定了什么

with set_handler(RecordLoadStore()):  # 动态绑定 handler
    inner_fn()   # 此处 handler = RecordLoadStore

with set_handler(CSEProxy(...)):      # 动态绑定另一个 handler
    inner_fn()   # 此处 handler = CSEProxy
```

同一个 `inner_fn` 被调用了两次，每次 "看到" 的 `handler` 不同——这就是动态作用域的威力。

Inductor 的核心洞察是：**IR 循环体（loop body）本质上是一个 Python callable（闭包），它内部调用 `ops.add()`、`ops.load()`、`ops.store()` 等方法。如果 `ops` 的实际行为可以在运行时动态切换，那么同一个闭包就能被多次执行，每次产生完全不同的效果——依赖分析、值域推断、代码生成。** 这就是 Inductor 的 "define-by-run" 哲学：IR 不是一个数据结构，而是一个 Python callable，它的语义由当前活跃的 handler 决定。

### 2.1.3 核心实现

源码位置：`torch/_inductor/virtualized.py:107-153`

```python
class Virtualized(Generic[T]):
    """
    Implements a global variable that redirects via thread local variable
    (NB: construct this class to create the global variable; this is not
    a singleton class!)
    """

    def __init__(self, vname: str, default: Union[Callable[[], T], type[NullHandler]]):
        self._vname = vname
        self._key: str = f"__torchinductor_{vname}"
        self._default = default

    def _set_handler(self, value: T) -> AbstractContextManager[None]:
        prior = self._get_handler(False)
        setattr(threadlocal, self._key, value)

        @contextmanager
        def ctx():
            try:
                yield
            finally:
                self._set_handler(prior)

        return ctx()

    def _get_handler(self, check_poisoned: bool = True) -> T:
        try:
            value = getattr(threadlocal, self._key)
            if check_poisoned and value is _PoisonedVirtual:
                raise RuntimeError(
                    f"Attempt to use poisoned virtualized value '{self._vname}'."
                )
            return value
        except AttributeError:
            return self._default()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_handler(), name)
```

这段代码虽然只有不到 50 行，却承载了 Inductor 编译器最核心的基础设施。逐层拆解：

**存储层：`threading.local()`**

```python
threadlocal = local()  # virtualized.py:86
```

所有 handler 的当前值都存储在这个线程局部对象上。`threading.local()` 保证每个线程有独立的命名空间，多线程编译时互不干扰。这也是 Inductor 与 GCC 的关键区别——GCC 的 `current_function_decl` 是真正的全局变量，而 Inductor 的 `V.graph` 是线程局部的。

**设值层：`_set_handler()` —— 栈式 push/pop**

```python
def _set_handler(self, value: T) -> AbstractContextManager[None]:
    prior = self._get_handler(False)   # 保存旧值
    setattr(threadlocal, self._key, value)  # 设新值

    @contextmanager
    def ctx():
        try:
            yield
        finally:
            self._set_handler(prior)   # 恢复旧值

    return ctx()
```

`_set_handler` 实现了经典的栈式 push/pop 语义：

1. 保存当前值到 `prior`
2. 设置新值
3. 返回一个 context manager，其 `__exit__` 会恢复 `prior`

这意味着嵌套使用完全安全：

```python
with V.set_ops_handler(handler_A):        # push A
    # V.ops = handler_A
    with V.set_ops_handler(handler_B):    # push B (prior = A)
        # V.ops = handler_B
        pass
    # V.ops = handler_A (自动恢复)
# V.ops = 之前的 handler (自动恢复)
```

**取值层：`_get_handler()` —— 带哨兵检查的读取**

```python
def _get_handler(self, check_poisoned: bool = True) -> T:
    try:
        value = getattr(threadlocal, self._key)
        if check_poisoned and value is _PoisonedVirtual:
            raise RuntimeError(...)
        return value
    except AttributeError:
        return self._default()
```

如果线程局部存储中没有对应键，则调用构造时传入的 `default` 工厂函数。对于大多数上下文，默认工厂是 `NullHandler`——一个任何属性访问都会抛出 `AttributeError` 的哨兵对象。对于 `V.choices`，默认工厂是 `_choices_default()`——一个延迟初始化函数。

有一个特殊的防护机制：`_PoisonedVirtual`。当编译发生在子进程中时，父进程的 `V` 状态不会（也不应该）被子进程继承。Inductor 通过将子进程中的所有虚拟化值设为 `_PoisonedVirtual` 来阻止子进程误读父进程的上下文。

**代理层：`__getattr__` —— 透明转发**

```python
def __getattr__(self, name: str) -> Any:
    return getattr(self._get_handler(), name)
```

这行代码是点睛之笔。它让 `V.graph.sizevars` 这样的链式属性访问变成 `getattr(V._get_handler(), "sizevars")`——先从 threadlocal 取出当前 handler（比如一个 `GraphLowering` 实例），然后在那个实例上查找 `sizevars`。对调用者来说，`V.graph` 和直接持有 `GraphLowering` 引用完全一样，但值可以在运行时随时切换。

### 2.1.4 输入/输出分析

`Virtualized<T>` 本身没有显式的数据流。它不接收函数参数，也不返回计算结果。它扮演的是一个**隐式上下文（Implicit Context）**的角色：

```
写入（设值）:  V.set_xxx(value)   → 值存入 threadlocal
读取（取值）:  V.xxx              → 从 threadlocal 读取当前值
代理访问:    V.xxx.some_method() → 从 threadlocal 取值，调用其方法
```

编译器类比：这相当于编译器中的 **符号表栈（Symbol Table Stack）**。进入一个新作用域时 push 新的符号表，离开时 pop。只不过 Inductor 的符号表不是存储变量名到类型的映射，而是存储任意类型的"当前上下文"。

### 2.2 _V 门面 — 12 个全局上下文

### 2.2.1 _V 的设计

源码位置：`torch/_inductor/virtualized.py:336-411`

`_V` 是 Inductor 暴露给外部的统一入口，通过 `from torch._inductor.virtualized import V` 导入。它将所有 `Virtualized<T>` 实例的设值/取值操作封装为属性和方法：

```python
class _V:
    # re-export 常用类型
    MockHandler = MockHandler
    KernelFormatterHandler = KernelFormatterHandler
    WrapperHandler = WrapperHandler

    # 设值方法 — 直接委托到对应 Virtualized 实例的 _set_handler
    set_ops_handler = _ops._set_handler
    set_graph_handler = _graph._set_handler
    set_kernel_handler = _kernel._set_handler
    set_fake_mode = _fake_mode._set_handler
    # ... 更多

    # 取值方法 — 通过 property 实现
    @property
    def ops(self) -> OpsHandler[Any]:
        return _ops._get_handler()

    @property
    def graph(self) -> GraphLowering:
        return _graph._get_handler()

    @property
    def kernel(self):
        return _kernel._get_handler()
    # ... 更多

V = _V()
```

`_V` 不是 `Virtualized` 的子类，而是一个**门面（Facade）**——它聚合了所有 `Virtualized` 实例，对外提供统一接口。这种设计的好处是：调用者只需 `import V` 就能访问所有全局上下文，不需要知道内部有 12 个独立的 `Virtualized` 实例。

### 2.2.2 12 个全局上下文详表

下面逐个介绍 `V` 管理的 12 个全局上下文，包括类型、默认值、创建/销毁时机，以及在编译中的角色。

#### 完整上下文清单

```
Virtualized 实例的创建（virtualized.py:182-213）

 _ops          = Virtualized("ops",             MockHandler)          # L182
 _graph        = Virtualized("graph",            NullHandler)          # L185
 _real_inputs  = Virtualized("real_inputs",      NullHandler)          # L186
 _fake_mode    = Virtualized("fake_mode",        NullHandler)          # L187
 _kernel       = Virtualized("kernel",           NullKernelHandler)    # L188
 _debug        = Virtualized("debug",            NullHandler)          # L191
 _interpreter  = Virtualized("interpreter",      NullHandler)          # L192
 _aot_compilation = Virtualized("aot_compilation", NullHandler)        # L193
 _current_node = Virtualized("current_node",     NullHandler)          # L194
 _local_buffer_context = Virtualized("local_buffer_context", NullHandler) # L195
 _choices      = Virtualized("choices",          _choices_default)     # L213
```

#### 详细分析

| # | 访问路径 | 类型参数 | 默认值 | 语义 |
|---|---------|---------|--------|------|
| 1 | `V.graph` | `GraphLowering` | `NullHandler` | 当前正在编译的计算图 |
| 2 | `V.kernel` | `NullKernelHandler` | `NullKernelHandler()` | 当前正在 codegen 的 kernel |
| 3 | `V.ops` | `OpsHandler[Any]` | `MockHandler` | 当前 IR 操作处理器 |
| 4 | `V.choices` | `InductorChoices` | 懒初始化 `_choices_default()` | 编译启发式/调优策略 |
| 5 | `V.real_inputs` | `list[Tensor]` | `NullHandler` | 非 fake 的真实输入 |
| 6 | `V.fake_mode` | `FakeTensorMode` | `NullHandler` | Fake tensor 模式 |
| 7 | `V.debug` | `DebugContext` | `NullHandler` | 调试上下文 |
| 8 | `V.interpreter` | `InterpreterShim` | `NullHandler` | FX 解释器 shim |
| 9 | `V.aot_compilation` | `bool` | `NullHandler` | 是否 AOT 编译模式 |
| 10 | `V.current_node` | `fx.Node` | `NullHandler` | 当前处理的 FX 节点 |
| 11 | `V.local_buffer_context` | `LocalBufferContext` | `NullHandler` | 局部 buffer 管理上下文 |
| 12 | `ops`（模块级） | `OpsWrapper` | `OpsWrapper()` | ops 的算术重载入口 |

第 12 项 `ops` 是模块级变量（`virtualized.py:333`），不是 `V` 的属性，但它是 `V.ops` 机制的延伸——它在 `V.ops` 的基础上增加了 `OpsValue` 运算符重载。IR 代码中 `a * b + c` 这种写法之所以能工作，就是因为 `ops` 模块级变量在背后做了 wrap/unwrap。

#### 各上下文的生命周期

| V 上下文 | 创建（set）时机 | 销毁（restore）时机 | 存活范围 | 典型实例数 |
|---------|---------------|-------------------|---------|-----------|
| `V.graph` | `compile_fx.py` 编译开始 | 编译结束 | 整个编译过程（可嵌套子图） | 1 主图 + N 子图 |
| `V.kernel` | `Kernel.__enter__` | `Kernel.__exit__` | 单个 kernel 的 codegen 过程 | 0 或 1 |
| `V.ops` | `Kernel.__enter__` 或分析阶段 | 对应 `with` 块结束 | kernel codegen 或 IR 分析期间 | 0 或 1 |
| `V.choices` | 首次访问时懒初始化 | 编译进程结束 | 整个进程 | 1（全局共享） |
| `V.fake_mode` | `compile_fx.py` 编译开始 | 编译结束 | 整个编译过程 | 0 或 1 |
| `V.current_node` | `GraphLowering.run_node` | 下一个节点 | 单个 FX 节点的处理过程 | 0 或 1 |
| `V.local_buffer_context` | `LocalBufferContext.__enter__` | `__exit__` | kernel codegen 中局部 buffer 使用期间 | 0 或 1 |
| `V.real_inputs` | 仅 minifier/repro | repro 结束 | 很少使用 | 0 或 1 |
| `V.debug` | `DebugContext.__enter__` | `__exit__` | 调试期间 | 0 或 1 |
| `V.aot_compilation` | `compile_fx.py` | 编译结束 | 整个编译过程 | 0 或 1 |
| `V.interpreter` | `InterpreterShim` 使用时 | shim 结束 | loop body 解释执行期间 | 0 或 1 |

#### 上下文分类

按源码注释（virtualized.py:17-55）的三大使用模式分类：

**模式 1：隐式参数传递**

`V.current_node`、`V.aot_compilation` 是典型代表。它们的值在整个编译过程中可能频繁变化，但深层代码需要读取它们。如果通过参数传递，每个函数签名都要加一个 `current_node` 参数，极其冗余。通过 `V` 的动态作用域，深层代码直接 `V.current_node` 即可，不需要知道这个值从哪来。

**模式 2：编译全局状态**

`V.graph`、`V.fake_mode` 是典型代表。它们在编译开始时设置一次，编译结束时恢复。整个编译过程中基本不变（`V.graph` 在子图编译时会临时切换）。它们关联着复杂的内部状态（如 `V.graph` 持有所有 IR 节点、缓冲区映射、调度器等），不适合用全局函数替代。

**模式 3：define-by-run 的多态解释**

`V.ops`、`V.kernel` 是核心代表。这是 Inductor 最独特的使用模式：同一个 IR 闭包（loop body callable）被多次执行，每次安装不同的 handler，产生不同的效果。在依赖分析阶段安装 `RecordLoadStore`，在代码生成阶段安装 `CSEProxy(TritonOverrides)`——IR 闭包本身不变，改变的只是解释方式。

### 2.2.3 编译器类比

| 编译器 | 全局上下文机制 | 典型变量 | 线程安全 | 类型安全 |
|--------|--------------|---------|---------|---------|
| GCC | 全局变量 | `current_function_decl`, `cfun`, `input_location` | 不安全 | 不安全（C 语言） |
| LLVM | 显式传递 `LLVMContext&` | `LLVMContext`, `IRBuilder` | 安全 | 安全（C++） |
| Inductor | `Virtualized<T>` 动态作用域 | `V.graph`, `V.ops`, `V.kernel` | 安全（`threadlocal`） | 安全（泛型 `T`） |

Inductor 的方案兼具三重优势：
- **GCC 的便捷性**：深层代码直接访问 `V.xxx`，不需要穿透参数
- **LLVM 的安全性**：线程局部存储 + 泛型类型参数保证类型正确
- **独特的动态切换**：`with V.set_xxx()` 自动恢复，嵌套安全

## 2.3 NullHandler / NullKernelHandler — 哨兵模式

### 2.3.1 NullHandler：快速失败的哨兵

源码位置：`torch/_inductor/virtualized.py:91-97`

```python
class NullHandler:
    """
    Sentinel indicating that a global variable is unset ala None.  Typically,
    attempting to access the global variable before it's set is an error, but with
    NullHandler it won't fail until you try to access an attribute on it.
    """
```

`NullHandler` 是一个完全空的类——没有方法，没有属性，没有任何特殊行为。它的全部意义在于：**当你在 handler 未安装时访问 `V.graph.xxx`，你不会得到一个晦涩的 `AttributeError: 'NoneType' object has no attribute 'xxx'`，而是得到 `AttributeError: 'NullHandler' object has no attribute 'xxx'`。**

后者的错误信息要清晰得多——它明确告诉你"你在访问一个未设置的虚拟化全局变量"。

这是 **Null Object 模式** 的变体。经典的 Null Object 模式提供一个"什么都不做"的默认行为，而 `NullHandler` 则是"什么都不做，并且让你知道"。它是 **哨兵（Sentinel）** ——存在的目的不是为了提供功能，而是为了在错误发生时提供更清晰的诊断信息。

编译器类比：这相当于 C/C++ 中的 `assert(ptr != NULL)` 检查，只不过 `NullHandler` 用 Python 的属性查找机制自动完成了这个检查——任何对未初始化上下文的属性访问都会立即失败。

### 2.3.2 NullKernelHandler：安全的默认值

源码位置：`torch/_inductor/virtualized.py:156-179`

```python
class NullKernelHandler(NullHandler):
    """
    We need access `V.kernel.removed_buffers` in DeferredLine class when there
    is no kernel in the context. This happens when codegening the wrapper.
    Initialize `removed_buffers` and `inplaced_to_remove` explicitly so we don't
    need call 'getattr' with default value which is error prone to typo in
    attribute name.
    """

    def __init__(self):
        super().__init__()
        self.removed_buffers = OrderedSet[Any]()
        self.inplaced_to_remove = OrderedSet[Any]()
        self.index_dtype = "tl.int64"

    def get_index_dtype_as_torch_dtype(self):
        import torch
        if self.index_dtype == "tl.int64":
            return torch.int64
        elif self.index_dtype == "tl.int32":
            return torch.int32
        else:
            raise ValueError(f"Unknown dtype: {self.index_dtype}")
```

`NullKernelHandler` 与 `NullHandler` 的设计哲学完全不同。它不是用来"快速失败"的，而是用来**安全降级**的。

问题背景：在 wrapper codegen 阶段，编译器已经完成了所有 kernel 的代码生成，此时 `V.kernel` 已经恢复为默认值（没有活跃的 kernel）。但 wrapper 代码生成过程中需要读取 `V.kernel.removed_buffers`——一个记录了哪些 buffer 已经被优化掉的集合。如果默认值是 `NullHandler`，读取 `removed_buffers` 会直接崩溃。

解决方案是 `NullKernelHandler`：它提供了空的 `removed_buffers`（`OrderedSet()`）和空的 `inplaced_to_remove`。当代码在没有活跃 kernel 时查询这些属性，得到的是"没有任何 buffer 被移除"——语义上完全正确，不会导致错误行为。

此外，`NullKernelHandler` 还提供了 `index_dtype = "tl.int64"` 和 `get_index_dtype_as_torch_dtype()` 方法。这是因为索引 dtype 是许多代码路径都需要查询的属性，即使不在 kernel 内部也需要一个合理的默认值。

设计模式总结：

```
NullHandler          → "你没设值就访问，这是 bug"（快速失败）
NullKernelHandler    → "没设值没关系，给你安全的默认值"（安全降级）
```

两种哨兵服务于不同的场景，共同保证了 `V` 上下文访问的安全性。

### 2.3.3 _PoisonedVirtual：子进程隔离

源码位置：`torch/_inductor/virtualized.py:99-104`

```python
# If a virtualized value is set to _PoisonedVirtual then any attempt to get the
# value will result an an exception being raised. This is useful if we want to
# trap uninitialized reads of virtualized globals - for example when compiling
# in a subprocess we don't want the child reading globals that weren't copied
# from the parent.
_PoisonedVirtual = object()
```

`_PoisonedVirtual` 是第三种哨兵，用于**子进程隔离**。当 Inductor 使用子进程编译时（如 autotuning），父进程的 `threadlocal` 状态不应该被子进程看到——子进程可能有自己完全不同的编译上下文。如果子进程意外读取了父进程残留的 handler，可能产生难以追踪的 bug。

`_PoisonedVirtual` 被设置为所有虚拟化变量的值后，子进程中任何 `V.xxx` 的访问都会触发 `_get_handler()` 中的 `check_poisoned` 检查，抛出清晰的 `RuntimeError`：

```
RuntimeError: Attempt to use poisoned virtualized value 'graph'.
```

这比 `NullHandler` 更严格——`NullHandler` 允许你"知道自己没设值"，`_PoisonedVirtual` 则是"你不应该在这里访问这个值"。

三种哨兵的对比：

```
_PoisonedVirtual  → "你根本不该在这里"（进程隔离）
NullHandler       → "你没设值就用了"（编程错误）
NullKernelHandler → "没设值也安全"（合理降级）
```

## 2.4 V 的数据流追踪

### 2.4.1 完整编译时间线

下面追踪一次 `torch.compile()` 调用中 `V` 的所有上下文切换。以 Triton GPU 后端为例：

```
compile_fx_inner(graph):
  │
  │  ┌─ with V.set_fake_mode(fake_mode):                # 上下文 6: FakeTensorMode
  │  │  │
  │  │  │  ┌─ with V.set_aot_compilation(True/False):   # 上下文 9: AOT 标志
  │  │  │  │  │
  │  │  │  │  │  ─── Phase 0: 常量折叠（可选） ───
  │  │  │  │  │
  │  │  │  │  │  const_graph = GraphLowering(const_gm, ...)
  │  │  │  │  │  with V.set_graph_handler(const_graph):  # 上下文 1: 临时切换到常量子图
  │  │  │  │  │    const_graph.run()                     #   Lowering 常量子图
  │  │  │  │  │    const_graph.codegen_with_cpp_wrapper()
  │  │  │  │  │  └─ V.graph 恢复为 NullHandler           #   常子图编译结束
  │  │  │  │  │
  │  │  │  │  │  ─── Phase 1: 主图 Lowering ───
  │  │  │  │  │
  │  │  │  │  │  graph = GraphLowering(gm, ...)
  │  │  │  │  │  with V.set_graph_handler(graph):        # 上下文 1: 设置主图
  │  │  │  │  │    │
  │  │  │  │  │    graph.run(*example_inputs)
  │  │  │  │  │    │  │
  │  │  │  │  │    │  │  GraphLowering 继承自 torch.fx.Interpreter，
  │  │  │  │  │    │  │  逐节点遍历 FX Graph：
  │  │  │  │  │    │  │
  │  │  │  │  │    │  for node in fx_graph.nodes:
  │  │  │  │  │    │    with V.set_current_node(node):   # 上下文 10: 当前节点
  │  │  │  │  │    │      result = lowerings[target](*args)
  │  │  │  │  │    │      # 创建 IR 节点（Pointwise, Reduction 等）
  │  │  │  │  │    │      # IR 节点持有 inner_fn（Python 闭包）
  │  │  │  │  │    │      # inner_fn 内部调用 ops.load/add/store
  │  │  │  │  │    │      # 此时 V.ops = MockHandler（默认状态）
  │  │  │  │  │    │
  │  │  │  │  │    │  ─── Phase 2: Scheduling ───
  │  │  │  │  │    │
  │  │  │  │  │    graph.compile_to_fn()
  │  │  │  │  │    │  │
  │  │  │  │  │    │  scheduler = Scheduler(graph)
  │  │  │  │  │    │  │  │
  │  │  │  │  │    │  │  │  对每个 IR 节点进行多遍分析：
  │  │  │  │  │    │  │  │
  │  │  │  │  │    │  │  │  ─ 分析 1: 依赖分析 ─
  │  │  │  │  │    │  │  │  with V.set_ops_handler(
  │  │  │  │  │    │  │  │    RecordLoadStore(...)
  │  │  │  │  │    │  │  │  ):                             # 上下文 3: 依赖分析 handler
  │  │  │  │  │    │  │  │    inner_fn(*index_vars)        # ← inner_fn 第 1 次执行
  │  │  │  │  │    │  │  │    # RecordLoadStore 记录读/写依赖
  │  │  │  │  │    │  │  │    # 产出: ReadWrites{reads, writes}
  │  │  │  │  │    │  │  │  V.ops 恢复
  │  │  │  │  │    │  │  │
  │  │  │  │  │    │  │  │  ─ 分析 2: op 计数 ─
  │  │  │  │  │    │  │  │  with V.set_ops_handler(
  │  │  │  │  │    │  │  │    OpCounterCSE(V.MockHandler())
  │  │  │  │  │    │  │  │  ):                             # 上下文 3: 计数 handler
  │  │  │  │  │    │  │  │    inner_fn(*index_vars)        # ← inner_fn 第 2 次执行
  │  │  │  │  │    │  │  │    # 产出: OpCountResult{num_ops}
  │  │  │  │  │    │  │  │  V.ops 恢复
  │  │  │  │  │    │  │  │
  │  │  │  │  │    │  │  │  ─ 分析 3: 值域分析 ─
  │  │  │  │  │    │  │  │  with V.set_ops_handler(
  │  │  │  │  │    │  │  │    ValueRangeAnalysis()
  │  │  │  │  │    │  │  │  ):                             # 上下文 3: 值域 handler
  │  │  │  │  │    │  │  │    inner_fn(*index_vars)        # ← inner_fn 第 3 次执行
  │  │  │  │  │    │  │  │    # 产出: ValueRanges{lower, upper}
  │  │  │  │  │    │  │  │  V.ops 恢复
  │  │  │  │  │    │  │  │
  │  │  │  │  │    │  │  │  ─ 融合决策、拓扑排序 ─
  │  │  │  │  │    │  │  │  根据 ReadWrites 做融合
  │  │  │  │  │    │  │  │  根据 OpCountResult 做是否 realize 的决策
  │  │  │  │  │    │  │  │
  │  │  │  │  │    │  │  │  scheduler.compile()
  │  │  │  │  │    │  │  │  │
  │  │  │  │  │    │  │  │  │  ─── Phase 3: Codegen ───
  │  │  │  │  │    │  │  │  │
  │  │  │  │  │    │  │  │  │  对每个融合后的节点组：
  │  │  │  │  │    │  │  │  │  TritonScheduling.codegen_nodes(nodes)
  │  │  │  │  │    │  │  │  │  │
  │  │  │  │  │    │  │  │  │  │  with TritonKernel(...) as kernel:
  │  │  │  │  │    │  │  │  │  │    │
  │  │  │  │  │    │  │  │  │  │    │  Kernel.__enter__():
  │  │  │  │  │    │  │  │  │  │    │    V.ops   ← CSEProxy(kernel, TritonKernelOverrides())
  │  │  │  │  │    │  │  │  │  │    │    V.kernel ← TritonKernel 实例
  │  │  │  │  │    │  │  │  │  │    │
  │  │  │  │  │    │  │  │  │  │    │  对融合组内每个节点:
  │  │  │  │  │    │  │  │  │  │    │    node.codegen(index_vars)
  │  │  │  │  │    │  │  │  │  │    │    │
  │  │  │  │  │    │  │  │  │  │    │    │  with V.set_ops_handler(
  │  │  │  │  │    │  │  │  │  │    │    │    SimplifyIndexing(
  │  │  │  │  │    │  │  │  │  │    │    │      CSEProxy(...),
  │  │  │  │  │    │  │  │  │  │    │    │      var_ranges
  │  │  │  │  │    │  │  │  │  │    │    │    )
  │  │  │  │  │    │  │  │  │  │    │    │  ):  ← 包装 SimplifyIndexing
  │  │  │  │  │    │  │  │  │  │    │    │    self._body(*index_vars)
  │  │  │  │  │    │  │  │  │  │    │    │                            # ← inner_fn 第 4 次执行
  │  │  │  │  │    │  │  │  │  │    │    │    # 生成 Triton kernel 代码
  │  │  │  │  │    │  │  │  │  │    │    │    # 写入 V.kernel.compute 缓冲区
  │  │  │  │  │    │  │  │  │  │    │    │  V.ops 恢复（去掉 SimplifyIndexing）
  │  │  │  │  │    │  │  │  │  │    │
  │  │  │  │  │    │  │  │  │  │    │  Kernel.__exit__():
  │  │  │  │  │    │  │  │  │  │    │    V.ops   ← 恢复为之前的 handler
  │  │  │  │  │    │  │  │  │  │    │    V.kernel ← 恢复为 NullKernelHandler
  │  │  │  │  │    │  │  │  │  │
  │  │  │  │  │    │  │  │  │  ─── Phase 4: Wrapper Codegen ───
  │  │  │  │  │    │  │  │  │  │
  │  │  │  │  │    │  │  │  │  │  V.kernel = NullKernelHandler（默认值）
  │  │  │  │  │    │  │  │  │  │  V.ops = MockHandler（默认值）
  │  │  │  │  │    │  │  │  │  │
  │  │  │  │  │    │  │  │  │  │  wrapper_code.generate()
  │  │  │  │  │    │  │  │  │  │  → 组装所有 kernel 调用为 Python 函数
  │  │  │  │  │    │  │  │  │  │  → 读取 V.kernel.removed_buffers（安全，返回空集合）
  │  │  │  │  │    │  │  │  │
  │  │  │  │  │    │  │  │  └─── 编译完成
  │  │  │  │  │    │  │  │
  │  │  │  │  │    │  └─── graph.compile_to_fn() 返回
  │  │  │  │  │    │
  │  │  │  │  │    └─── V.graph 恢复
  │  │  │  │  │
  │  │  │  └─── V.aot_compilation 恢复
  │  │  │
  │  └─── V.fake_mode 恢复
  │
  └─── compile_fx_inner() 返回 compiled_fn
```

这个时间线揭示了 `V` 的核心设计原则：**长生命周期的上下文在外层设置，短生命周期的上下文在内层动态切换。**

```
外层（长生命周期）:
  V.fake_mode       ─── 整个编译过程
  V.aot_compilation ─── 整个编译过程
  V.graph           ─── 整个编译过程（子图编译时临时切换）
  V.choices         ─── 整个进程（懒初始化，永不销毁）

内层（短生命周期，频繁切换）:
  V.current_node    ─── 单个 FX 节点的处理
  V.ops             ─── 单次分析或单个 kernel 的 codegen
  V.kernel          ─── 单个 kernel 的 codegen
  V.local_buffer_context ─── kernel 内局部 buffer 使用期间
```

### 2.4.2 IR 闭包的四次生命

上面的时间线中最值得关注的是 **IR 闭包（inner_fn）的四次执行**。同一个 Python callable，在编译过程中被调用至少四次，每次安装不同的 `V.ops` handler，产生完全不同的效果。

假设有一个简单的 IR 闭包：

```python
def inner_fn(index):
    a = ops.load("buf_a", index)
    b = ops.load("buf_b", index)
    result = a * b + ops.constant(1.0, torch.float32)
    ops.store("buf_out", index, result)
```

下面追踪它在四次执行中的行为。

#### 第一次执行：依赖分析（RecordLoadStore）

```
Handler: RecordLoadStore (继承自 MockHandler)

执行过程:
  ops.load("buf_a", index)
    → RecordLoadStore.load() → 记录 MemoryDep("buf_a", index) → 返回 "buf_a" (字符串)
  ops.load("buf_b", index)
    → RecordLoadStore.load() → 记录 MemoryDep("buf_b", index) → 返回 "buf_b"
  a * b
    → OpsValue.__mul__ → ops.mul("buf_a", "buf_b")
    → MockHandler._default → 返回 "buf_a * buf_b" (仅传递值)
  ... + ops.constant(...)
    → MockHandler._default → 返回 "buf_a * buf_b + ops.constant(...)"
  ops.store("buf_out", index, ...)
    → RecordLoadStore.store() → 记录 MemoryDep("buf_out", index)

产出:
  ReadWrites = {
    reads:  { MemoryDep("buf_a", ...), MemoryDep("buf_b", ...) },
    writes: { MemoryDep("buf_out", ...) }
  }
```

`RecordLoadStore` 只关心 `load` 和 `store`——它们是内存访问点，决定了数据依赖。`mul`、`add` 等纯计算操作走 `MockHandler` 的字符串路径，只作为值的管道传递，不产生任何副作用。

#### 第二次执行：Op 计数（OpCounterCSE）

```
Handler: OpCounterCSE (带 CSE 的计数器)

执行过程:
  ops.load("buf_a", index)  → 计数器["load"] += 1
  ops.load("buf_b", index)  → 计数器["load"] += 1
  a * b                     → 计数器["mul"] += 1
  ... + constant            → 计数器["add"] += 1
  ops.store("buf_out", ...) → 计数器["store"] += 1

  注: 如果同一个子表达式出现多次，CSE 会去重，只计一次

产出:
  OpCountResult = { num_ops: 5, ... }
```

Scheduler 用这个计数决定：如果一个 loop body 的计算量太大（op 太多），就先把它"实体化"（realize）为一个独立 buffer，避免在融合后的 kernel 中重复计算。

#### 第三次执行：值域分析（ValueRangeAnalysis）

```
Handler: ValueRangeAnalysis

执行过程:
  ops.load("buf_a", index)
    → ValueRangeAnalysis.load() → 返回 ValueRanges.unknown()  # 不知 buffer 内容范围
  ops.load("buf_b", index)
    → ValueRangeAnalysis.load() → 返回 ValueRanges.unknown()
  a * b
    → ValueRangeAnalysis.mul() → ValueRanges(-inf, +inf)  # 两个 unknown 相乘仍 unknown
  ... + constant(1.0, float32)
    → ValueRangeAnalysis.add() → ValueRanges(-inf, +inf)

产出:
  ValueRanges = { lower: -inf, upper: +inf }  # 对于未知输入，结果也是未知
```

如果输入有已知的范围约束（比如 `buf_a` 是 sigmoid 输出，范围 `[0, 1]`），值域分析可以推导出更精确的输出范围。这些信息被用于：
- 决定 Triton kernel 的数据类型（是否可以用更低精度）
- SimplifyIndexing 中简化索引表达式（利用范围信息消除不可能的分支）

#### 第四次执行：代码生成（CSEProxy + TritonKernelOverrides）

```
Handler 链: SimplifyIndexing → CSEProxy → TritonKernelOverrides

执行过程:
  ops.load("buf_a", index)
    → SimplifyIndexing: 简化 index（如果有可化简的 sympy 表达式）
    → CSEProxy.load(): 检查 store cache → 没命中
    → TritonKernel.load(): 生成 "tl.load(buf_a_ptr + index, mask=...)"
    → CSE 分配: CSEVariable("tmp0", dtype=float32)
    → 写入 V.kernel.loads: "tmp0 = tl.load(buf_a_ptr + index, ...);"

  ops.load("buf_b", index)
    → 同上 → CSEVariable("tmp1")

  a * b
    → OpsValue.__mul__ → ops.mul(OpsValue("tmp0"), OpsValue("tmp1"))
    → OpsWrapper: unwrap → _ops.mul("tmp0", "tmp1")
    → CSEProxy._default("mul", ...):
        ① 值域追踪: ValueRangeAnalysis.mul(bounds_tmp0, bounds_tmp1)
        ② 代码生成: TritonKernelOverrides.mul("tmp0", "tmp1") → "tmp0 * tmp1"
        ③ dtype 推导: DtypePropagationOpsHandler.mul(float32, float32) → float32
        ④ CSE 去重: 检查 "tmp0 * tmp1" 是否已存在 → 否
           → V.kernel.compute 追加: "tmp2 = tmp0 * tmp1"
        → 返回 CSEVariable("tmp2")

  ... + constant(1.0, float32)
    → CSEProxy._default("add", ...):
        ① 值域追踪: bounds
        ② 代码生成: "tmp2 + 1.0"
        ③ dtype 推导: float32
        ④ CSE: "tmp3 = tmp2 + 1.0"
        → 返回 CSEVariable("tmp3")

  ops.store("buf_out", index, OpsValue("tmp3"))
    → OpsWrapper: unwrap → CSEProxy.store("buf_out", index, "tmp3")
    → 更新 store cache: {"buf_out": CSEVariable("tmp3")}
    → TritonKernel.store(): 生成 "tl.store(buf_out_ptr + index, tmp3, ...)"
    → 写入 V.kernel.stores: "tl.store(buf_out_ptr + index, tmp3, ...);"

产出:
  V.kernel.loads   ← "tmp0 = tl.load(...); tmp1 = tl.load(...);"
  V.kernel.compute ← "tmp2 = tmp0 * tmp1; tmp3 = tmp2 + 1.0;"
  V.kernel.stores  ← "tl.store(buf_out_ptr + index, tmp3, ...);"
```

这四行代码组成了一个完整的 Triton kernel 的计算部分。加上 kernel 的框架代码（函数签名、grid 配置等），就是一个可执行的 `@triton.jit` kernel。

#### 四次执行的对比总结

```
┌─────────────────┬─────────────────────┬─────────────────────────────────────┐
│ 执行次数        │ Handler             │ 产出                               │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ 第 1 次         │ RecordLoadStore     │ ReadWrites{reads, writes}           │
│ (依赖分析)      │                     │ 用于融合决策和拓扑排序              │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ 第 2 次         │ OpCounterCSE        │ OpCountResult{num_ops}              │
│ (计算量评估)    │                     │ 用于决定是否 realize               │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ 第 3 次         │ ValueRangeAnalysis  │ ValueRanges{lower, upper}           │
│ (值域推断)      │                     │ 用于 CSE 优化和索引简化            │
├─────────────────┼─────────────────────┼─────────────────────────────────────┤
│ 第 4 次         │ SimplifyIndexing    │ 目标语言代码字符串                  │
│ (代码生成)      │  → CSEProxy         │ 写入 V.kernel.compute              │
│                 │  → TritonOverrides  │ 最终成为 @triton.jit kernel        │
└─────────────────┴─────────────────────┴─────────────────────────────────────┘
```

**核心洞察**：IR 只定义了一次（在 Lowering 阶段创建 `inner_fn` 闭包时），但被解释了四次。每次解释使用不同的 handler，产生不同类型的结果。这正是 **define-by-run** 哲学的精髓：**IR 不是数据结构，而是 Python callable；handler 不是 visitor，而是解释器。**

### 2.4.3 V.graph 的嵌套切换：子图编译

`V.graph` 在编译过程中通常是稳定的，但在子图编译场景中会发生嵌套切换。一个典型的例子是 `Conditional` IR 节点（对应 Python 的 `if/else` 控制流）：

```
主图 codegen 中遇到 Conditional IR:
  │
  │  V.graph = 主图 GraphLowering
  │
  ├─ Conditional.codegen(wrapper_code)
  │    │
  │    ├─ 编译 true 分支:
  │    │    with V.set_graph_handler(true_subgraph.graph):  ← 临时切换
  │    │      │  V.graph = true_subgraph
  │    │      │  true_subgraph 内部的 lowering + codegen
  │    │      │  （V.graph.wrapper_code 仍指向主图的 wrapper，因为是共享的）
  │    │    ← 退出 with，V.graph 恢复为主图
  │    │
  │    ├─ 编译 false 分支:
  │    │    with V.set_graph_handler(false_subgraph.graph): ← 临时切换
  │    │      │  V.graph = false_subgraph
  │    │    ← 退出 with，V.graph 恢复为主图
  │    │
  │    └─ 生成 if/else wrapper 代码
  │
  └─ V.graph = 主图 GraphLowering（不变）
```

这种嵌套切换与 GCC 中编译内联函数时临时切换 `current_function_decl` 类似，但 Inductor 通过 context manager 的自动恢复机制，避免了 GCC 中"忘记恢复全局变量"的经典 bug。

### 2.4.4 V.kernel 的生命周期：单次 kernel codegen

`V.kernel` 的生命周期是所有上下文中最短的——仅在单个 kernel 的代码生成期间存在。以 CPU C++ kernel 为例：

```
时间 ──────────────────────────────────────────────────────────────►

  V.kernel = NullKernelHandler (默认空状态)
     │
     │  CppScheduling.codegen_node() 开始
     │
     ├─ CppKernelProxy(kernel_group) 创建（此时还不是 V.kernel）
     │
     ├─ proxy.codegen_nodes(nodes)
     │    └─ codegen_functions()
     │         └─ with kernel_group.new_kernel(CppKernel) as kernel:
     │              │
     │              ├─ CppKernel.__enter__():
     │              │    ├─ V.set_ops_handler(CSEProxy(kernel, CppOverrides()))
     │              │    │       ← V.ops = CSEProxy(CppOverrides)
     │              │    └─ V.set_kernel_handler(kernel)
     │              │           ← V.kernel = CppKernel 实例
     │              │
     │              │  ┌─ V.kernel = CppKernel ───────────────────────────┐
     │              │  │  V.ops = CSEProxy(CppOverrides)                  │
     │              │  │                                                  │
     │              │  │  遍历融合组内所有节点的 inner_fn:                │
     │              │  │    for node in nodes:                            │
     │              │  │      node.codegen(index_vars)                    │
     │              │  │        → inner_fn(*index_vars)                   │
     │              │  │          → ops.load(...) → V.kernel.loads        │
     │              │  │          → ops.add(...) → V.kernel.compute       │
     │              │  │          → ops.store(...) → V.kernel.stores      │
     │              │  │                                                  │
     │              │  │  V.kernel.removed_buffers 被填充                  │
     │              │  │  V.kernel.inplaced_to_remove 被填充              │
     │              │  └──────────────────────────────────────────────────┘
     │              │
     │              ├─ CppKernel.__exit__():
     │              │    ├─ remove_kernel_local_buffers()
     │              │    ├─ V.ops 恢复为 MockHandler
     │              │    └─ V.kernel 恢复为 NullKernelHandler
     │              │
     │              └─ 返回 kernel 实例
     │
     └─ kernel_group.finalize_kernel(proxy, nodes)
        → 组装 kernel 代码（循环嵌套 + loads + compute + stores）

  V.kernel = NullKernelHandler (恢复空状态)
```

注意 `Kernel.__enter__` 的精确行为（`codegen/common.py:2140-2147`）：

```python
def __enter__(self) -> Self:
    super().__enter__()
    assert self.overrides
    self.exit_stack.enter_context(
        V.set_ops_handler(CSEProxy(self, self.overrides()))
    )
    self.exit_stack.enter_context(V.set_kernel_handler(self))
    return self
```

它使用了 `contextlib.ExitStack` 来管理两个 context manager——`V.set_ops_handler` 和 `V.set_kernel_handler`。`ExitStack` 保证退出时按相反的顺序恢复，即使中途发生异常也不会泄漏状态。

### 2.4.5 V.ops 的切换序列：一次编译中的 handler 演替

一次完整的编译过程中，`V.ops` 会经历多次切换。以下是一个简化的切换序列：

```
初始状态:
  V.ops = MockHandler (virtualized.py 中 MockHandler 作为 _ops 的默认工厂)

Scheduler 构造阶段:
  │
  ├─ 依赖分析: with V.set_ops_handler(RecordLoadStore(...)):
  │    V.ops = RecordLoadStore
  │    执行 inner_fn → 收集 ReadWrites
  │  V.ops = MockHandler (恢复)
  │
  ├─ Op 计数: with V.set_ops_handler(OpCounterCSE(MockHandler())):
  │    V.ops = OpCounterCSE
  │    执行 inner_fn → 统计 op 数量
  │  V.ops = MockHandler (恢复)
  │
  ├─ 值域分析: with V.set_ops_handler(ValueRangeAnalysis()):
  │    V.ops = ValueRangeAnalysis
  │    执行 inner_fn → 收集值域范围
  │  V.ops = MockHandler (恢复)
  │
  ├─ 自由符号收集: with V.set_ops_handler(FreeSymbolsOpsHandler()):
  │    V.ops = FreeSymbolsOpsHandler
  │    执行 inner_fn → 收集 sympy 符号
  │  V.ops = MockHandler (恢复)

Scheduler codegen 阶段:
  │
  ├─ Kernel.__enter__():
  │    V.ops = CSEProxy(kernel, TritonKernelOverrides())
  │
  │  ├─ node.codegen():
  │  │    with V.set_ops_handler(
  │  │      SimplifyIndexing(V.get_ops_handler(), var_ranges)
  │  │    ):
  │  │      V.ops = SimplifyIndexing → CSEProxy → TritonKernelOverrides
  │  │      执行 inner_fn → 生成 Triton 代码
  │  │    V.ops = CSEProxy → TritonKernelOverrides (去掉 SimplifyIndexing)
  │
  ├─ Kernel.__exit__():
  │    V.ops = MockHandler (恢复)

Wrapper codegen 阶段:
  V.ops = MockHandler (默认，不再切换)
```

这个序列清晰地展示了 `V.ops` 的"换刀片"模式：同一个"刀柄"（`V.ops`），在不同的分析阶段插入不同的"刀片"（handler），完成不同的任务。分析阶段用完即弃，代码生成阶段最为复杂——嵌套了 `SimplifyIndexing` 在 `CSEProxy` 外面。

### 2.4.6 设计哲学总结

回顾本章的核心主题，`V` 的设计可以用三个关键词概括：

**1. 动态作用域（Dynamic Scoping）**

`V` 的核心实现是动态作用域——变量的值跟随运行时调用栈变化，而非源码中的词法位置。这让同一个 IR 闭包可以在不同的语义上下文中被"解释"，而不需要修改闭包本身的代码。

**2. define-by-run**

Inductor 不维护传统的 AST/SSA IR 数据结构来表示循环体。循环体是一个 Python callable（闭包），内部调用 `ops.load()`、`ops.add()`、`ops.store()` 等方法。这个 callable 被执行多次，每次安装不同的 handler——这就是"define by run"：IR 在运行时（run）中被定义（define），而不是在编译时被构建为静态数据结构。

**3. 策略模式（Strategy Pattern）**

`V.ops` 的 handler 体系是策略模式的完美实例。`inner_fn` 是上下文（Context），定义了"按什么顺序做什么运算"。Handler 是策略（Strategy），定义了"每个运算产生什么效果"。切换策略 = 切换行为，IR 代码完全不改。

这三个设计原则共同构成了 Inductor 编译器最独特的基础设施——一个用不到 50 行核心代码实现的、线程安全的、类型安全的、栈式自动恢复的动态作用域引擎。它让 Inductor 避免了维护多套 IR 遍历逻辑的复杂性，用同一个 callable 实现了依赖分析、值域推断、代码生成等多种编译阶段的核心功能。
