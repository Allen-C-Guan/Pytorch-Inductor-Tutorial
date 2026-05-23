# 第四章：IR 表示层 —— Inductor 的中间表示

> 源文件：`torch/_inductor/ir.py`（~8000 行）、`torch/_inductor/virtualized.py`

## 4.1 IR 设计哲学：为什么用 Python 闭包做 IR？

### 4.1.1 传统编译器的 IR 方式

如果你接触过 LLVM，一定熟悉这样的 IR 表示：

```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}
```

LLVM IR 是一种 **数据结构**——每个指令是一个 `Instruction` 对象，每个基本块是一个 `BasicBlock` 对象，整个函数是一个 `Function` 对象。编译器通过遍历和修改这些对象来实现优化 Pass（如 DCE、常量折叠、死代码消除）。

这种方式的优点是 **结构清晰**，缺点是 **构建 IR 的开销大**——你需要为每条指令创建对象、维护操作数列表、管理 SSA 形式的 def-use 链。

### 4.1.2 Inductor 的选择：Define-by-Run + 闭包

Inductor 走了一条完全不同的路。打开 `torch/_inductor/ir.py`，你会看到这样一段关键注释 `[Note: Inductor IR]`：

> Inductor IR uses a "define-by-run" strategy similar to PyTorch itself. Rather than constructing an explicit AST or data structure for the IR, we represent computation as **Python callables (closures)**. These closures contain calls to `V.ops` methods that get resolved to different implementations depending on the current compilation phase.

翻译过来就是：**Inductor 的 IR 不是数据结构，而是 Python 函数**。

这是什么意思？看一个具体例子。假设我们有这样一段计算：

```python
def forward(x, y):
    return x + y * 2
```

在 Inductor 中，这个计算会被表示为一个 `inner_fn`——一个 Python 闭包：

```python
def inner_fn(index):
    # V.ops.load() 不是立即加载，而是由当前 handler 决定行为
    x_val = V.ops.load("x", index)
    y_val = V.ops.load("y", index)
    # V.ops.mul/add 同理
    result = V.ops.add(x_val, V.ops.mul(y_val, V.ops.constant(2, torch.float32)))
    V.ops.store("buf0", index, result)
```

这个 `inner_fn` **本身**就是 IR。它不是一个 AST 节点，不是一个 LLVM Instruction，而是一个你可以直接调用的 Python 函数。

### 4.1.3 闭包 IR 的核心魔法：V.ops 多阶段派发

关键问题来了：如果 `inner_fn` 是一个普通 Python 函数，那它执行时 `V.ops.load()` 不就是在加载数据吗？IR 怎么能只是个函数呢？

答案是 **`V.ops` 不是固定实现**。`V.ops` 是一个 **虚拟化的处理器（Virtualized Handler）**，在不同的编译阶段被替换成不同的实现。这就像同一份菜谱（`inner_fn`），不同的厨师（handler）做出来的菜不同：

| 编译阶段 | Handler 实现 | `V.ops.load()` 的行为 |
|---------|-------------|---------------------|
| 依赖分析 | `MockHandler` | 返回一个虚拟值，记录 buffer 间的读写依赖 |
| IR 字符串化 | `IRFormatter` | 返回 `"load(buf, index)"` 字符串，用于调试 |
| 代码生成 | `KernelFormatterHandler` | 生成 Triton/C++ 代码中的加载语句 |
| 调试 | `PrintHandler` | 打印每次操作的名称和参数 |

用一个类比来理解：

> **闭包 IR 就像一张建筑蓝图。** 蓝图本身不关心你是用纸画的、用 CAD 软件画的、还是用 VR 看的——它只定义了"这里有一扇窗、那里有一扇门"。`V.ops` 就是不同的"阅读方式"：建筑师看结构，包工头看施工细节，客户看效果图。同一张蓝图，不同的阅读器给出不同的输出。

这个设计的精妙之处在于：

1. **零开销构建 IR**：不需要创建成千上万的对象节点，只需要构造一个 Python 函数。闭包自动捕获它需要的变量，Python 运行时帮你管理。

2. **天然的表达力**：Python 本身就是图灵完备的语言，闭包可以表示任意的控制流和计算逻辑。不需要设计一套新的 IR 语法。

3. **多阶段复用**：同一个 `inner_fn`，只需换一次 `V.ops` handler，就能做依赖分析、代码生成、调试输出。不需要在 IR 数据结构之间来回转换。

4. **延迟物化（Lazy Materialization）**：在 `inner_fn` 被调用之前，计算不会发生。这给了 Inductor 优化的空间——如果某个中间结果从未被使用，相应的 `inner_fn` 可能永远不需要执行。

### 4.1.4 V.ops 的实现机制

让我们深入看 `V.ops` 是如何实现多阶段派发的。核心在 `torch/_inductor/virtualized.py`：

```python
class _V:
    """Thread-local virtualized variables."""
    def __init__(self):
        self._ops_handler = OpsWrapper()  # 默认 handler

    @property
    def ops(self):
        return self._ops_handler

    def set_ops_handler(self, handler):
        """Context manager to temporarily replace the ops handler."""
        return self._ops_handler._set_handler(handler)

V = _V()
```

使用方式是 **context manager 模式**：

```python
# 在依赖分析阶段
with V.set_ops_handler(MockHandler()):
    inner_fn(index)  # V.ops.load() 现在是 MockHandler.load()

# 在代码生成阶段
with V.set_ops_handler(KernelFormatterHandler()):
    inner_fn(index)  # V.ops.load() 现在生成代码
```

这就像 **给同一个函数装上了可插拔的后端**。函数本身不关心后端是谁，它只管调用 `V.ops.load()`、`V.ops.add()` 这些接口。后端的实现在调用之前通过 `V.set_ops_handler()` 注入。

### 4.1.5 与传统 IR 的对比

| 维度 | 传统编译器 IR（LLVM） | Inductor IR |
|-----|---------------------|-------------|
| 表示形式 | 数据结构（对象图/SSA） | Python 闭包（可调用函数） |
| 构建方式 | 显式创建节点，插入基本块 | 构造函数，自动捕获变量 |
| 遍历方式 | 遍历指令链表、基本块列表 | 调用函数，handler 决定行为 |
| 优化 Pass | 修改 IR 数据结构 | 修改 IR 节点的属性（layout、name 等） |
| 多阶段复用 | 需要 pass manager 调度不同 pass | 换 V.ops handler 即可 |
| 构建开销 | 高（大量对象分配） | 低（函数闭包） |
| 语义丰富度 | 受限于 IR 定义 | Python 全部表达力 |

### 4.1.6 闭包 IR 的代价

没有银弹。闭包 IR 也有代价：

1. **不可序列化**：闭包不能直接 dump 到磁盘或通过网络传输。但 Inductor 不需要这一步——IR 只在编译过程中存在。

2. **难以做全局优化**：传统编译器可以遍历整个 IR 图做全局优化（如公共子表达式消除）。闭包 IR 的"图"是隐式的——你需要通过 handler 收集信息，然后在更高层做优化。Inductor 通过 `StorageBox.realize()` 触发的 IR pass 来实现类似功能。

3. **调试困难**：闭包是运行时构造的，你不能像看 AST 那样直接"看"IR。Inductor 提供了 `TORCH_LOGS="inductor"` 环境变量来打印 IR 字符串表示。

---

## 4.2 IRNode 继承体系

### 4.2.1 整体类层次结构

IR 表示层的核心是一棵精心设计的继承树。先看全貌：

```
IRNode (抽象基类)
├── MutableBox (包装器基类，委托给 self.data)
│   ├── TensorBox (最外层：引用语义)
│   │       └── .data → StorageBox
│   └── StorageBox (中间层：存储生命周期管理)
│           └── .data → Buffer (或 Loops，realize 后变成 ComputedBuffer)
│
├── Buffer (IRNode + CodegenSymbol：具名存储单元)
│   ├── InputBuffer (函数输入，无计算)
│   │   └── ConstantBuffer (编译期常量)
│   ├── OperationBuffer (单输出操作缓冲区)
│   │   └── ComputedBuffer (有 inner_fn 的计算缓冲区) ★核心类★
│   ├── AliasBuffer (别名缓冲区)
│   └── ReinterpretBuffer (布局重解释)
│
├── Loops (操作 IR：持有 inner_fn 闭包)
│   ├── Pointwise (逐元素操作)
│   │   └── Scatter (散射写入操作)
│   ├── Reduction (规约操作)
│   ├── Scan (前缀和操作)
│   └── Sort (排序操作)
│
├── BaseView (视图基类)
│   └── ReinterpretView (布局重解释视图)
│       ├── ExpandView (广播扩展)
│       ├── PermuteView (维度重排)
│       ├── SqueezeView (降维)
│       ├── GenericView (通用视图)
│       ├── View (reshape 操作)
│       ├── SliceView (切片操作)
│       └── DtypeView (类型转换视图)
│
└── Operation (操作基类，非 Loops 类操作)
```

### 4.2.2 三层包装模式：TensorBox → StorageBox → Buffer

这是 Inductor IR 中最精妙的设计之一。每个张量在 IR 中被三层对象包装，每层负责不同的职责：

```
┌─────────────────────────────────────────────────┐
│  TensorBox                                       │
│  职责：引用语义（多个变量可指向同一个底层数据）    │
│  关键方法：create(), __repr__()                   │
│                                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │  StorageBox                                  │ │
│  │  职责：存储生命周期管理（延迟物化）            │ │
│  │  关键方法：realize(), is_input_buffer()       │ │
│  │                                              │ │
│  │  ┌─────────────────────────────────────────┐ │ │
│  │  │  Buffer (或 ComputedBuffer)              │ │ │
│  │  │  职责：实际的 IR 节点                     │ │ │
│  │  │  关键方法：make_loader(), get_layout()    │ │ │
│  │  │  持有：name, layout, data(Loops)          │ │ │
│  │  └─────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

#### 为什么需要三层？

**问题 1：引用语义。** Python 的赋值是引用赋值。当 Dynamo 追踪到一个 FX 图中的节点时，多个输出可能共享同一个输入。比如：

```python
def forward(x):
    y = x + 1
    z = y * 2
    w = y * 3  # y 被使用了两次
    return z + w
```

在 FX 图中，`y` 是一个节点，`z` 和 `w` 都引用 `y`。Inductor 需要追踪这种共享关系。`TensorBox` 就是为了解决这个问题——它是一个轻量的 **引用包装器**，多个 FX 节点可以指向同一个 `TensorBox`，而底层的 `StorageBox` 和 `Buffer` 只有一份。

**问题 2：延迟物化。** 不是所有的中间计算都需要生成代码。如果 `y = x + 1` 的结果被内联（inline）到下游使用处，那么 `y` 不需要自己的 buffer。`StorageBox` 管理这种延迟物化——它包裹着一个 `Loops`（尚未物化的计算），只有当 `realize()` 被调用时才创建 `ComputedBuffer`。

**问题 3：实际 IR。** `Buffer` 是真正的 IR 节点，有名字、有布局、有 `inner_fn`。代码生成阶段直接操作 `Buffer`。

#### 用一个类比理解三层包装

> 想象一个快递系统：
> - **TensorBox** = 收件人地址标签。同一个包裹可以有多个标签（寄给多个人），但物理包裹只有一个。
> - **StorageBox** = 包裹的状态管理器。包裹在仓库中还没有实际发货（未 realize），直到有人要求发货时才打包。
> - **Buffer** = 实际的包裹。里面有真实的东西（计算逻辑 inner_fn），有重量（layout），有运单号（name）。

### 4.2.3 MutableBox：委托模式的基类

`TensorBox` 和 `StorageBox` 的共同基类是 `MutableBox`。它的设计极其简洁——就是 **完全委托**：

```python
class MutableBox(IRNode):
    """Mutable wrapper around an IRNode. Delegates all calls to self.data."""

    def __init__(self, data: IRNode):
        super().__init__()
        self.data = data

    def __getattr__(self, name):
        # 一切未定义的属性/方法调用，直接委托给 self.data
        return getattr(self.data, name)
```

这意味着当你调用 `tensor_box.get_size()` 时，实际的调用链是：

```
TensorBox.get_size()
  → MutableBox.__getattr__("get_size")
    → StorageBox.get_size()
      → MutableBox.__getattr__("get_size")
        → Buffer.get_size()
          → 返回 self.layout.size
```

**为什么用委托而不是继承？** 因为 `TensorBox` 和 `StorageBox` 需要在不改变底层 `Buffer` 的情况下 **替换** 它。`self.data` 是可变的——`StorageBox.realize()` 会把 `self.data` 从 `Loops` 替换成 `ComputedBuffer`。如果用继承，这种运行时类型切换是不可能的。

这和 **智能指针** 的概念类似——`std::shared_ptr<T>` 不继承自 `T`，而是包装 `T`，并在需要时替换所指向的对象。

### 4.2.4 TensorBox 详解

`TensorBox` 是三层包装的最外层：

```python
class TensorBox(MutableBox):
    """Reference-semantics wrapper around StorageBox."""

    @staticmethod
    def create(data: IRNode) -> "TensorBox":
        """Create a TensorBox wrapping a StorageBox wrapping the given IRNode."""
        if not isinstance(data, StorageBox):
            data = StorageBox(data)
        return TensorBox(data)
```

`create()` 方法是工厂方法，确保 `TensorBox` 永远直接包裹 `StorageBox`，不会跳过中间层。这是三层的 **不变量（invariant）**。

`TensorBox` 本身几乎不添加任何方法——它的职责是纯粹的引用语义。在 Inductor 的 IR 构建过程中，所有 FX 节点的输出都被包装成 `TensorBox`。当 Dynamo 的 FX 图被翻译成 Inductor IR 时，每个 `fx.Node` 对应一个 `TensorBox`。

### 4.2.5 StorageBox 详解

`StorageBox` 是最关键的一层，因为它管理 **延迟物化**：

```python
class StorageBox(MutableBox):
    """Manages storage lifetime. Wraps a Buffer or Loops."""

    def realize(self) -> str | None:
        """
        将内部未物化的 Loops (Pointwise/Reduction/...) 转化为 ComputedBuffer。
        这是 IR 从"延迟计算"变成"实际存在"的关键一步。
        """
        if isinstance(self.data, (Buffer, BaseView)):
            # 已经是 Buffer 了，不需要物化
            return self.data.name

        assert isinstance(self.data, Loops), f"Expected Loops, got {type(self.data)}"

        # 创建 ComputedBuffer，包装这个 Loops
        self.data = ComputedBuffer(
            name=None,  # 名字稍后由 V.graph.register_buffer() 分配
            layout=FlexibleLayout(
                device=self.data.get_device(),
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
            ),
            data=self.data,  # 将 Loops (Pointwise/Reduction/...) 作为 data 传入
        )

        # 向计算图注册这个新 buffer
        self.data.name = V.graph.register_buffer(self.data)
        V.graph.register_operation(self.data)

        return self.data.name

    def is_input_buffer(self) -> bool:
        return isinstance(self.data, InputBuffer)
```

`realize()` 方法做了三件事：

1. **检查**：如果 `self.data` 已经是 `Buffer`，说明已经物化，直接返回名字。
2. **包装**：如果 `self.data` 是 `Loops`（尚未物化的计算闭包），创建一个 `ComputedBuffer` 来包装它。
3. **注册**：将新创建的 `ComputedBuffer` 注册到 `V.graph`，分配名字、建立操作记录。

**什么时候会调用 `realize()`？**

- **代码生成前**：所有需要输出到内存的中间结果都必须物化。
- **需要分配 buffer 名字时**：当代码生成需要引用一个 buffer 时，必须先 realize。
- **操作融合决策后**：某些操作被决定不内联时，需要 realize 为独立 buffer。

**什么时候不需要 realize？**

- **被内联的操作**：如果一个 `Pointwise` 的结果只被一个下游操作使用，Inductor 可能选择将其 **内联**——即把它的 `inner_fn` 直接嵌入下游操作的 `inner_fn` 中，不分配独立 buffer。

这就像编译器的 **寄存器分配** 决策：变量是存在内存中（realize）还是只在寄存器中传递（inline），取决于使用情况和优化策略。

### 4.2.6 Buffer 详解

`Buffer` 是真正的 IR 节点：

```python
class Buffer(IRNode, CodegenSymbol):
    """
    A named storage unit in the IR.
    Has a name, layout, and optionally a data field containing Loops.
    """

    def __init__(self, name: str | None, layout: Layout):
        super().__init__()
        self.name = name
        self.layout = layout
        self._force_realize = False

    def get_name(self) -> str:
        return self.name

    def get_layout(self) -> Layout:
        return self.layout

    def get_device(self) -> torch.device:
        return self.layout.device

    def get_dtype(self) -> torch.dtype:
        return self.layout.dtype

    def get_size(self) -> list[Expr]:
        return self.layout.size

    def make_loader(self):
        """
        返回一个 loader 闭包。
        这个闭包在被调用时，通过 V.ops.load() 从此 buffer 加载数据。
        """
        name = self.name
        dtype = self.get_dtype()

        def load(index):
            return V.ops.load(name, index, dtype)

        return load
```

`Buffer` 的核心是 `make_loader()` 方法。它返回一个 **闭包**——这就是前面说的闭包 IR 的具体体现。这个 loader 闭包会被嵌入到下游操作的 `inner_fn` 中，形成 **闭包的闭包**——IR 节点之间通过闭包嵌套来建立依赖关系。

#### Buffer 的子类们

```
Buffer
├── InputBuffer          # 函数输入参数（如 forward(x, y) 中的 x, y）
│   └── ConstantBuffer   # 编译期常量（如 torch.tensor([1.0, 2.0])）
│
├── OperationBuffer      # 单输出操作的结果
│   └── ComputedBuffer   # 有 inner_fn 的计算结果 ★
│
├── AliasBuffer          # 别名缓冲区（多个名字指向同一块内存）
└── ReinterpretBuffer    # 布局重解释（不改变数据，只改变解读方式）
```

**InputBuffer**：表示函数的输入。它没有 `inner_fn`——数据来自外部（用户传入的 tensor）。`make_loader()` 返回的闭包会调用 `V.ops.load(name, ...)` 从输入 tensor 加载。

**ConstantBuffer**：继承自 `InputBuffer`，表示编译期已知的常量值。Inductor 可能将常量直接嵌入生成的代码中，而不是从内存加载。

**ComputedBuffer**：这是最核心的子类，它持有一个 `Loops` 对象（`Pointwise`、`Reduction` 等），包含实际的计算闭包 `inner_fn`。它有一个关键的内联优化：

```python
class ComputedBuffer(OperationBuffer):
    def make_loader(self):
        # 内联优化：如果这个 buffer 从未被写入、只被读取一次，
        # 那么不需要分配实际内存，直接返回 inner_fn 作为 loader
        if (not self.get_reduction_type()
            and self.name not in V.graph.mutated_buffers
            and self.num_reads() == 0
            and not self._force_realize):
            return self.data.make_loader()  # 返回 Pointwise.inner_fn！

        return super().make_loader()  # 正常分配 buffer
```

这个优化非常重要。如果一个计算结果从未被 realize（没有分配内存），那么它的 `inner_fn` 就会被直接嵌入到下游操作的 `inner_fn` 中。这就是 **操作融合（Operator Fusion）** 在 IR 层面的实现机制。

**AliasBuffer** 和 **ReinterpretBuffer** 用于处理内存别名和布局重解释，不引入新的计算。

### 4.2.7 Layout 系统

每个 `Buffer` 都有一个 `Layout`，描述数据在内存中的排列方式：

```python
class Layout:
    """Base layout class."""
    def __init__(self, device, dtype, size):
        self.device = device
        self.dtype = dtype
        self.size = size  # list[Expr]，每个维度的符号化大小

class FixedLayout(Layout):
    """Layout with a fixed stride pattern."""
    def __init__(self, device, dtype, size, stride):
        super().__init__(device, dtype, size)
        self.stride = stride  # list[Expr]，每个维度的步幅

class FlexibleLayout(Layout):
    """Layout where stride can be decided later by the scheduler."""
    def __init__(self, device, dtype, size):
        super().__init__(device, dtype, size)
        # stride 在调度阶段由 scheduler 决定
```

`FlexibleLayout` 的存在是因为 Inductor 的 **调度器（Scheduler）** 可能在后期决定最优的内存布局。比如，调度器可能将两个 buffer 的布局统一为相同的 stride pattern，以便生成更高效的内核。这就是 **layout propagation** 优化。

---

## 4.3 操作 IR 节点

操作 IR 节点是 `Loops` 及其子类，它们持有 `inner_fn` 闭包。这些节点代表了实际的计算——逐元素操作、规约、前缀和、排序等。

### 4.3.1 Loops：操作 IR 的基类

```python
class Loops(IRNode):
    """
    Base class for operation IR nodes.
    Holds an inner_fn (closure), ranges (loop bounds), device, and dtype.
    """
    def __init__(
        self,
        inner_fn: Callable,
        ranges: Sequence[Expr],
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.inner_fn = inner_fn  # Python 闭包，即 "IR"
        self.ranges = ranges      # 循环范围，如 [128, 64] 表示两层循环
        self.device = device
        self.dtype = dtype
```

`Loops` 是所有操作 IR 的基类。它的核心是三个属性：

- **`inner_fn`**：一个 Python 闭包，代表循环体内部的计算。接受一个 `index` 参数（当前循环索引的元组）。
- **`ranges`**：循环范围。比如 `[Expr(128), Expr(64)]` 表示一个 128x64 的两层循环。
- **`device`/`dtype`**：结果张量的设备和数据类型。

当 `Loops` 被 **执行**（即 `inner_fn` 被调用）时，它的行为取决于当前的 `V.ops` handler。这就像前面说的——同一张蓝图，不同的阅读器给出不同的输出。

### 4.3.2 Pointwise：逐元素操作

`Pointwise` 是最常见的操作类型，表示对输入张量的每个元素独立应用同一个函数：

```python
class Pointwise(Loops):
    """
    Element-wise operation.
    inner_fn maps one output index to one output value.
    No data dependencies between different indices.
    """

    def make_loader(self):
        """直接返回 inner_fn 作为 loader——支持内联融合。"""
        return self.inner_fn
```

`Pointwise.make_loader()` 直接返回 `inner_fn`。这就是内联优化的关键——当下游操作需要加载这个 `Pointwise` 的结果时，不需要从内存读取，而是直接调用 `inner_fn` 计算结果。

举个例子，`z = x + y` 会被翻译成：

```python
# 创建 Pointwise IR 节点
pw = Pointwise(
    inner_fn=lambda index: V.ops.add(
        V.ops.load("x", index),
        V.ops.load("y", index)
    ),
    ranges=[128],  # 假设张量大小为 128
    device=torch.device("cuda"),
    dtype=torch.float32,
)
```

当这个 `Pointwise` 被 realize 时，会生成类似这样的 Triton 代码：

```python
@triton.jit
def kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    y = tl.load(y_ptr + offsets, mask=offsets < n_elements)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=offsets < n_elements)
```

但如果不 realize（内联），`inner_fn` 直接被嵌入下游操作，不会生成独立的 kernel。

### 4.3.3 Reduction：规约操作

`Reduction` 表示沿某个维度将多个值合并为一个值的操作（如 sum、max、mean）：

```python
class Reduction(Loops):
    """
    Reduction operation.
    Has both iteration ranges and reduction ranges.
    inner_fn returns the value to accumulate.
    """
    def __init__(
        self,
        inner_fn: Callable,
        ranges: Sequence[Expr],         # 非规约维度的循环范围
        reduction_ranges: Sequence[Expr], # 规约维度的循环范围
        reduction_type: str,             # "sum", "max", "min", "any", "argmax", "argmin" 等
        device: torch.device,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        reduction_hint: ReductionHint = ReductionHint.NONE,
    ):
        super().__init__(inner_fn, ranges, device, dtype)
        self.reduction_ranges = reduction_ranges
        self.reduction_type = reduction_type
        self.src_dtype = src_dtype
        self.reduction_hint = reduction_hint
```

`Reduction` 比 `Pointwise` 多了几个关键属性：

- **`reduction_ranges`**：规约维度的循环范围。比如对一个 `[128, 64]` 的张量沿 dim=1 做 sum，`ranges = [128]`，`reduction_ranges = [64]`。
- **`reduction_type`**：规约方式——"sum"、"max"、"min" 等。
- **`reduction_hint`**：优化提示，帮助调度器做决策。

`Reduction` 的 `inner_fn` 结构和 `Pointwise` 不同。它不是返回一个值，而是返回一个 **待规约的值**。代码生成时，`V.ops` handler 会生成规约逻辑（如 Triton 的 `tl.sum()` 或循环累加）。

一个 `sum(x, dim=1)` 的例子：

```python
# x: [128, 64], 在 dim=1 上做 sum
# 结果: [128]
red = Reduction(
    inner_fn=lambda index: V.ops.load("x", (index[0], index[1])),
    ranges=[128],              # 外层循环：128 个输出元素
    reduction_ranges=[64],     # 内层规约：每个输出元素需要遍历 64 个输入
    reduction_type="sum",
    device=torch.device("cuda"),
    dtype=torch.float32,
)
```

在代码生成阶段，`inner_fn` 被调用时传入完整的 `(i, j)` 索引，`V.ops.load()` 加载对应的输入值，然后 `Reduction` 的 handler 将这些值累加起来。

### 4.3.4 Scan：前缀和操作

`Scan` 表示前缀和（prefix sum / scan）类操作，如 `torch.cumsum()`：

```python
class Scan(Loops):
    """
    Prefix sum / scan operation.
    """
    def __init__(
        self,
        inner_fn: Callable,
        combine_fn: Callable | None,
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        device: torch.device,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        reindex: dict | None = None,
    ):
        super().__init__(inner_fn, ranges, device, dtype)
        self.combine_fn = combine_fn  # 合并函数（如加法）
        self.reduction_ranges = reduction_ranges
        self.src_dtype = src_dtype
        self.reindex = reindex
```

`Scan` 和 `Reduction` 类似，但有一个关键区别：`Reduction` 将一组值合并为一个值，而 `Scan` 将一组值变换为 **一组等长的值**——每个输出是前面所有输入的累积结果。

`combine_fn` 是一个额外的闭包，定义了如何将两个部分结果合并。对于 `cumsum`，`combine_fn` 就是加法。

### 4.3.5 Sort：排序操作

`Sort` 表示排序类操作（如 `torch.sort()`、`torch.topk()`）：

```python
class Sort(Loops):
    """
    Sorting operation.
    """
    def __init__(
        self,
        inner_fn: Callable,
        ranges: Sequence[Expr],
        reduction_ranges: Sequence[Expr],
        device: torch.device,
        dtype: torch.dtype,
        stable: bool = False,
        descending: bool = False,
    ):
        super().__init__(inner_fn, ranges, device, dtype)
        self.reduction_ranges = reduction_ranges
        self.stable = stable
        self.descending = descending
```

`Sort` 有 `stable`（是否保持相等元素的原始顺序）和 `descending`（是否降序）两个布尔标志。

### 4.3.6 Scatter：散射写入操作

`Scatter` 继承自 `Pointwise`，但增加了散射写入的语义：

```python
class Scatter(Pointwise):
    """
    Scatter write operation.
    Writes to non-contiguous output locations.
    """
    def __init__(
        self,
        inner_fn: Callable,
        ranges: Sequence[Expr],
        output_indexer,         # 输出索引映射
        scatter_mode: str | None = None,  # "scatter", "scatter_add" 等
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(inner_fn, ranges, device, dtype)
        self.output_indexer = output_indexer
        self.scatter_mode = scatter_mode
```

`Scatter` 的关键是 `output_indexer`——它定义了如何将计算结果的索引映射到输出的实际存储位置。普通的 `Pointwise` 假设输出索引和存储位置是一一对应的，而 `Scatter` 允许不规则的映射（如 `torch.scatter()`、`torch.index_copy()` 等）。

### 4.3.7 操作 IR 节点对比

| 类别 | 类名 | 循环结构 | inner_fn 语义 | 典型操作 |
|-----|------|---------|-------------|---------|
| 逐元素 | `Pointwise` | 单层循环 | `index → value` | add, mul, relu |
| 规约 | `Reduction` | 循环 + 规约循环 | `(i, j) → value` | sum, max, mean |
| 前缀和 | `Scan` | 循环 + 扫描循环 | `(i, j) → value` | cumsum, cumprod |
| 排序 | `Sort` | 循环 + 排序循环 | `(i, j) → value` | sort, topk |
| 散射 | `Scatter` | 单层循环 + 输出映射 | `index → value` | scatter, index_add |

---

## 4.4 IR Passes

IR Pass 是编译器在 IR 上执行的变换和优化。Inductor 的 IR Pass 不是像 LLVM 那样遍历指令链表，而是通过 **属性查询** 和 **handler 派发** 来实现的。

### 4.4.1 IR Pass 的触发时机

IR Pass 的执行集中在两个阶段：

**阶段 1：IR 构建期（Dynamo → Inductor IR 翻译）**
- 操作融合（Operator Fusion）：将多个 `Pointwise` 合并为一个
- 布局推断（Layout Inference）：确定每个 buffer 的 layout
- 死代码消除（DCE）：移除未使用的中间结果

**阶段 2：调度 + 代码生成期**
- 调度决策：决定哪些 buffer 需要 realize，哪些可以内联
- 代码生成：调用 `inner_fn`，通过 `V.ops` handler 生成 Triton/C++ 代码
- 内存分配：为 realize 的 buffer 分配内存

### 4.4.2 延迟物化 Pass

延迟物化是 Inductor 最核心的 IR Pass 之一。它的工作原理很简单：

```
初始状态：所有操作都是 Loops (Pointwise/Reduction/...)，没有 Buffer
    ↓
realize() 被调用
    ↓
检查：inner_fn 是否需要独立 buffer？
    ├─ 是：创建 ComputedBuffer，分配名字，注册到 V.graph
    └─ 否：保持 Loops，等待内联到下游操作
```

这个决策的依据包括：

1. **使用次数**：如果一个 `Pointwise` 的结果被多个下游操作使用，它必须 realize（否则需要重复计算）。
2. **是否是规约**：`Reduction` 的结果几乎总是需要 realize（规约是一个独立的 kernel）。
3. **是否被 mutate**：如果一个 buffer 会被原地修改（如 `+=` 操作），它必须 realize。
4. **调度器决策**：调度器可能根据内存压力、kernel 大小等因素强制 realize 某些 buffer。

### 4.4.3 操作融合 Pass

操作融合是 Inductor 性能优化的关键。它的核心机制已经在 IR 层面设计好了——通过 `ComputedBuffer.make_loader()` 的内联优化：

```python
# 假设有三个操作：a = x + 1, b = a * 2, c = b - 3
# 如果 b 从不 realize，那么 c 的 inner_fn 会是：

def inner_fn_c(index):
    # b 的 inner_fn 被内联到这里
    b_val = (
        # a 的 inner_fn 被内联到这里
        V.ops.add(
            V.ops.load("x", index),
            V.ops.constant(1, torch.float32)
        )
        * V.ops.constant(2, torch.float32)
    )
    return V.ops.sub(b_val, V.ops.constant(3, torch.float32))
```

三个操作被融合成了一个 `inner_fn`，最终只生成一个 Triton kernel。

融合的规则大致是：

- `Pointwise + Pointwise → Pointwise`（逐元素操作可以任意融合）
- `Pointwise + Reduction → Reduction`（逐元素操作可以融合到后续的 Reduction）
- `Reduction + Pointwise → 两个独立 kernel`（规约结果需要先写出，再逐元素处理）
- 跨设备边界必须断开融合

### 4.4.4 布局传播 Pass

布局传播发生在调度阶段。`FlexibleLayout` 允许调度器在后期决定最优的 stride pattern：

```python
# 初始创建时使用 FlexibleLayout
buffer = ComputedBuffer(
    name="buf0",
    layout=FlexibleLayout(device, dtype, size=[128, 64]),
    data=pointwise_op,
)

# 调度器可能决定改为行优先（contiguous）布局
buffer.layout = FixedLayout(device, dtype, size=[128, 64], stride=[64, 1])
```

布局传播的目标是减少内存访问的跳跃（stride 不连续），提高缓存命中率。

### 4.4.5 V.ops Handler 派发 Pass

这是 Inductor 独有的 Pass 机制——通过切换 `V.ops` handler 来实现不同的分析和代码生成阶段。核心的 handler 类型：

```
V.ops Handler 层次：
├── OpsWrapper (默认，包装所有返回值为 OpsValue)
│
├── MockHandler (依赖分析阶段)
│   功能：记录 buffer 间的读写依赖关系
│   V.ops.load() → 返回虚拟值，记录 "读 buffer X"
│   V.ops.store() → 记录 "写 buffer Y"
│
├── IRFormatter (IR 字符串化)
│   功能：将 IR 转化为可读字符串
│   V.ops.load() → 返回 "load(buf_name, index)"
│
├── KernelFormatterHandler (代码生成阶段)
│   功能：生成 Triton/C++ 代码
│   V.ops.load() → 生成实际的加载指令
│
└── WrapperHandler (外层代码生成)
    功能：生成 Python wrapper 代码
```

每个 handler 必须实现 `V.ops` 的全部接口（load、store、add、mul、reduce 等）。handler 之间的切换通过 context manager 实现：

```python
# 依赖分析
with V.set_ops_handler(MockHandler()):
    buffer.data.inner_fn(indices)  # 执行 inner_fn，MockHandler 收集依赖

# 代码生成
with V.set_ops_handler(KernelFormatterHandler()):
    buffer.data.inner_fn(indices)  # 执行 inner_fn，生成 Triton/C++ 代码
```

### 4.4.6 OpsValue：流式表达式构建

`OpsWrapper` 包装所有 `V.ops` 的返回值为 `OpsValue`，支持 Python 的算术运算符重载：

```python
class OpsValue:
    """Wrapper that allows fluent arithmetic expression writing."""
    def __init__(self, value):
        self.value = value

    def __add__(self, other):
        return OpsValue(V.ops.add(self.value, unwrap(other)))

    def __mul__(self, other):
        return OpsValue(V.ops.mul(self.value, unwrap(other)))

    # ... 其他算术运算符
```

这使得 `inner_fn` 可以写成更自然的 Python 表达式：

```python
# 没有 OpsValue 时：
def inner_fn(index):
    return V.ops.add(
        V.ops.mul(V.ops.load("x", index), V.ops.load("y", index)),
        V.ops.constant(1, torch.float32)
    )

# 有 OpsValue 时（更自然）：
def inner_fn(index):
    return V.ops.load("x", index) * V.ops.load("y", index) + 1
```

---

## 4.5 追踪示例：从 Python 到 IR 到代码

让我们用一个完整的例子追踪 Inductor IR 的生命周期。考虑这个简单的函数：

```python
import torch

@torch.compile
def forward(x, y):
    z = x + y       # 逐元素加法
    w = z * 2       # 逐元素乘法
    return w.sum()  # 全局规约
```

### 4.5.1 阶段 1：Dynamo 追踪 → FX 图

Dynamo 首先将 Python 函数追踪为 FX 图：

```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=1] = placeholder[target=y]
    %z : [num_users=1] = call_function[target=torch.add](args = (%x, %y), kwargs = {})
    %w : [num_users=1] = call_function[target=torch.mul](args = (%z, 2), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%w,), kwargs = {})
    return sum_1
```

### 4.5.2 阶段 2：FX 图 → Inductor IR

Inductor 将每个 FX 节点翻译为 Inductor IR。这里是逐步翻译过程：

**步骤 1：输入参数**

```python
# x 和 y 被翻译为 InputBuffer
x_buffer = InputBuffer(name="x", layout=FixedLayout("cuda", torch.float32, [128]))
y_buffer = InputBuffer(name="y", layout=FixedLayout("cuda", torch.float32, [128]))

# 包装为三层结构
x_ir = TensorBox(StorageBox(x_buffer))  # x → StorageBox(InputBuffer)
y_ir = TensorBox(StorageBox(y_buffer))  # y → StorageBox(InputBuffer)
```

此时在 debugger 中的数据结构：

```
x_ir: TensorBox
  .data: StorageBox
    .data: InputBuffer
      .name: "x"
      .layout: FixedLayout(device=cuda, dtype=float32, size=[128], stride=[1])

y_ir: TensorBox
  .data: StorageBox
    .data: InputBuffer
      .name: "y"
      .layout: FixedLayout(device=cuda, dtype=float32, size=[128], stride=[1])
```

**步骤 2：z = x + y**

```python
# 创建 Pointwise IR 节点
z_loops = Pointwise(
    inner_fn=lambda index: V.ops.add(
        x_ir.make_loader()(index),  # 调用 InputBuffer.make_loader()() → V.ops.load("x", index)
        y_ir.make_loader()(index),  # 调用 InputBuffer.make_loader()() → V.ops.load("y", index)
    ),
    ranges=[128],
    device=torch.device("cuda"),
    dtype=torch.float32,
)

# 包装为三层结构
z_ir = TensorBox(StorageBox(z_loops))
```

注意：此时 `z` 还没有被 realize。`StorageBox.data` 是 `Pointwise`（一个 `Loops`），不是 `Buffer`。

```
z_ir: TensorBox
  .data: StorageBox
    .data: Pointwise  ← 尚未物化！
      .inner_fn: <closure>
      .ranges: [128]
      .device: cuda
      .dtype: float32
```

**步骤 3：w = z * 2**

```python
# z 的 make_loader() 返回 inner_fn（因为 z 还没有 realize）
# 所以 w 的 inner_fn 会内联 z 的计算
w_loops = Pointwise(
    inner_fn=lambda index: V.ops.mul(
        z_ir.make_loader()(index),  # 内联了 z 的 inner_fn！
        V.ops.constant(2, torch.float32),
    ),
    ranges=[128],
    device=torch.device("cuda"),
    dtype=torch.float32,
)

w_ir = TensorBox(StorageBox(w_loops))
```

此时 `w` 的 `inner_fn` 展开后等价于：

```python
def w_inner_fn(index):
    # z 的 inner_fn 被内联到这里
    z_val = V.ops.add(
        V.ops.load("x", index),
        V.ops.load("y", index),
    )
    return V.ops.mul(z_val, V.ops.constant(2, torch.float32))
```

**z + z*2 两个 Pointwise 已经融合成了 w 的一个 Pointwise！** 这就是操作融合。

**步骤 4：sum_1 = w.sum()**

```python
# sum 是一个 Reduction 操作
sum_loops = Reduction(
    inner_fn=lambda index: w_ir.make_loader()(index[0]),  # 内联 w（以及 z）的计算
    ranges=[],              # 全局规约：输出是标量，没有非规约维度
    reduction_ranges=[128], # 规约维度：遍历所有 128 个元素
    reduction_type="sum",
    device=torch.device("cuda"),
    dtype=torch.float32,
)

sum_ir = TensorBox(StorageBox(sum_loops))
```

由于 `w` 也没有 realize，`w_ir.make_loader()` 返回 `w_loops.inner_fn`。所以 `sum` 的 `inner_fn` 展开后：

```python
def sum_inner_fn(index):
    # w 的 inner_fn 被内联
    w_val = V.ops.mul(
        # z 的 inner_fn 被内联
        V.ops.add(
            V.ops.load("x", index),
            V.ops.load("y", index),
        ),
        V.ops.constant(2, torch.float32),
    )
    return w_val
```

### 4.5.3 阶段 3：IR Passes（延迟物化 + 融合）

在调度阶段，调度器分析 IR 并做决策：

1. **z**：只被 w 使用，且 w 是 Pointwise → **不 realize，保持内联**
2. **w**：只被 sum 使用，但 sum 是 Reduction → **Pointwise + Reduction 融合** → w 不 realize
3. **sum**：是最终输出 → **必须 realize**

最终只有 `sum` 被物化为 `ComputedBuffer`：

```
sum_ir: TensorBox
  .data: StorageBox
    .data: ComputedBuffer  ← realize 后
      .name: "buf0"  (由 V.graph.register_buffer 分配)
      .layout: FlexibleLayout(cuda, float32, size=[])
      .data: Reduction
        .inner_fn: <融合了 x+y*2 计算的闭包>
        .ranges: []
        .reduction_ranges: [128]
        .reduction_type: "sum"
```

z 和 w 始终保持为 `Pointwise`（未物化），它们的计算被融合进了 `sum` 的 `inner_fn`。

### 4.5.4 阶段 4：代码生成

在代码生成阶段，`V.ops` handler 被替换为 `KernelFormatterHandler`，然后 `inner_fn` 被调用：

```python
with V.set_ops_handler(KernelFormatterHandler()):
    # 遍历 reduction 维度
    for r_index in reduction_indices:
        # 调用 inner_fn，此时 V.ops.add/mul/load 生成实际代码
        sum_buffer.data.inner_fn((r_index,))
```

生成的 Triton 代码大致为：

```python
@triton.jit
def kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # 内联了 z = x + y 和 w = z * 2
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)  # 等等，这里 x 和 y 都需要...
    # 实际上，y 是第二个输入

    # 融合后的计算
    tmp = x + y        # z = x + y (内联)
    tmp = tmp * 2.0    # w = z * 2 (内联)

    # Reduction: sum
    acc = tl.sum(tmp)   # sum_1 = w.sum()
    tl.store(out_ptr, acc)
```

（注意：实际的 Triton 代码生成会更复杂，包括 block tiling、vectorization 等，这里简化展示。）

### 4.5.5 完整数据流图

```
FX Graph:
  x ──┐
       ├── torch.add ── z ──┐
  y ──┘                     ├── torch.mul ── w ── torch.sum ── sum_1
                             2 ─────────────┘

Inductor IR:
  InputBuffer("x") ──┐
                      ├── Pointwise(inner_fn=add) ── z (未 realize，内联)
  InputBuffer("y") ──┘                                  │
                                                         ↓ (内联)
                               Pointwise(inner_fn=mul(add(...), 2)) ── w (未 realize，内联)
                                                                              │
                                                                              ↓ (内联)
  Reduction(inner_fn=mul(add(load_x, load_y), 2), type="sum") ── ComputedBuffer("buf0")

Generated Code:
  一个 Triton kernel = load x, load y → add → mul 2 → sum → store buf0
```

### 4.5.6 数据流中的关键观察

1. **闭包嵌套 = 操作融合**：`w` 的 `inner_fn` 嵌套了 `z` 的 `inner_fn`，`sum` 的 `inner_fn` 嵌套了 `w` 的 `inner_fn`。嵌套层数就是融合的操作数。

2. **未物化的操作不占内存**：`z` 和 `w` 没有对应的 buffer 名字，不会分配内存。只有最终结果 `buf0` 占用内存。

3. **`make_loader()` 是融合的关键入口**：当一个操作调用上游的 `make_loader()` 时，如果上游是 `Pointwise`（未 realize），返回的是 `inner_fn`（内联）；如果是 `Buffer`（已 realize），返回的是 `V.ops.load()`（从内存读取）。

4. **`V.ops` 是多阶段的桥梁**：同一个 `inner_fn`，在依赖分析时记录依赖关系，在代码生成时生成 Triton/C++ 代码。闭包本身从不改变，改变的只是执行闭包时的 handler。

---

## 4.6 本章总结

Inductor 的 IR 表示层是一个精心设计的系统，核心理念可以概括为三个要点：

1. **闭包即 IR**：用 Python 闭包（`inner_fn`）代替传统的数据结构 IR。闭包天然支持延迟计算和多阶段复用，且构建开销极低。

2. **三层包装**：`TensorBox → StorageBox → Buffer` 分别负责引用语义、延迟物化和实际 IR 节点。这种分层使得不同关注点被清晰地分离。

3. **Handler 派发**：通过 `V.ops` 的可插拔 handler 机制，同一个闭包在不同编译阶段展现出不同的行为，实现了依赖分析、代码生成、调试输出等功能的无缝切换。

这三个设计选择共同构成了 Inductor IR 的独特风格——它是 Python 原生的、闭包驱动的、延迟物化的中间表示。理解了这三个要点，就理解了 Inductor IR 层的核心架构。
