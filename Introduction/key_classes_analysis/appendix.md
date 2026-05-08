# 附录

---

## 附录 A：完整类名索引

以下按照字母顺序列出本书涉及的所有核心类，每个条目包含完整的模块路径、源文件位置（相对于 `torch/_inductor/`）、简要描述以及相关章节引用。

> **说明：** 源文件路径基于 PyTorch 2.x 的目录结构。`codegen/` 子目录下的文件对应 `torch/_inductor/codegen/` 路径。

| # | 类名 | 模块路径 | 源文件 | 简要描述 | 章节 |
|---|------|---------|--------|---------|------|
| 1 | `BaseScheduling` | `torch._inductor.scheduler` | `scheduler.py` | 所有后端调度策略的抽象基类，定义 `codegen_node()`、`group_fn` 等接口 | Ch1, Ch6 |
| 2 | `BaseSchedulerNode` | `torch._inductor.scheduler` | `scheduler.py` | 调度器节点基类，封装 IR 节点并提供依赖追踪与融合查询接口 | Ch1, Ch6 |
| 3 | `BracesBuffer` | `torch._inductor.codegen.common` | `codegen/common.py` | 带花括号缩进的代码缓冲区，继承自 `IndentedBuffer`，用于生成结构化代码块 | Ch7 |
| 4 | `Buffer` | `torch._inductor.ir` | `ir.py` | 核心缓冲区 IR 节点，同时实现 `IRNode` 和 `CodegenSymbol`，持有名称与布局信息 | Ch1, Ch4 |
| 5 | `ComputedBuffer` | `torch._inductor.ir` | `ir.py` | 由 Inductor 计算产生的缓冲区，继承 `OperationBuffer`，持有 `Pointwise`/`Reduction` 等操作描述 | Ch4, Ch6 |
| 6 | `ConstantBuffer` | `torch._inductor.ir` | `ir.py` | 常量缓冲区，继承 `InputBuffer`，表示编译期已知的常量张量 | Ch4 |
| 7 | `CppKernel` | `torch._inductor.codegen.cpp` | `codegen/cpp.py` | C++ 后端 Kernel 代码生成器，继承 `Kernel`，生成标量 C++ 循环代码 | Ch7, Ch8 |
| 8 | `CppKernelProxy` | `torch._inductor.codegen.cpp` | `codegen/cpp.py` | `CppKernel` 的代理类，用于在外层循环融合中委托内核调用 | Ch8 |
| 9 | `CppOverrides` | `torch._inductor.codegen.cpp` | `codegen/cpp.py` | C++ 后端 OpOverrides 实现，将标量 ops 翻译为 C++ 表达式字符串 | Ch3, Ch7 |
| 10 | `CppScheduling` | `torch._inductor.codegen.cpp` | `codegen/cpp.py` | C++ 后端调度策略，继承 `BaseScheduling`，处理 CPU 内核的代码生成与调度 | Ch6, Ch8 |
| 11 | `CppVecKernel` | `torch._inductor.codegen.cpp` | `codegen/cpp.py` | C++ 向量化 Kernel，继承 `CppKernel`，生成 SIMD intrinsics 代码 | Ch7, Ch8 |
| 12 | `CppVecOverrides` | `torch._inductor.codegen.cpp` | `codegen/cpp.py` | C++ 向量化 OpOverrides，继承 `CppOverrides`，将 ops 翻译为 SIMD intrinsics 表达式 | Ch3, Ch7 |
| 13 | `CppWrapperCpu` | `torch._inductor.codegen.cpp_wrapper_cpu` | `codegen/cpp_wrapper_cpu.py` | CPU 端 C++ Wrapper 代码生成器，继承 `PythonWrapperCodegen` | Ch7, Ch8 |
| 14 | `CppWrapperGpu` | `torch._inductor.codegen.cpp_wrapper_gpu` | `codegen/cpp_wrapper_gpu.py` | GPU 端 C++ Wrapper 代码生成器，继承 `CppWrapperCpu`，处理 CUDA/ROCm 相关包装 | Ch8 |
| 15 | `CSE` | `torch._inductor.codegen.common` | `codegen/common.py` | 公共子表达式消除引擎，泛型类 `CSE[CSEVariableType]`，在代码生成期间去重重复表达式 | Ch3, Ch7 |
| 16 | `CSEProxy` | `torch._inductor.codegen.common` | `codegen/common.py` | CSE 代理 Handler，继承 `DefaultHandler`，拦截 ops 调用并委托给底层 `CSE` 实例 | Ch3, Ch7 |
| 17 | `CSEVariable` | `torch._inductor.codegen.common` | `codegen/common.py` | CSE 变量包装器，持有变量名、数据类型和表达式等元信息 | Ch3, Ch7 |
| 18 | `CUDACombinedScheduling` | `torch._inductor.codegen.cuda_combined_scheduling` | `codegen/cuda_combined_scheduling.py` | CUDA 组合调度策略，继承 `BaseScheduling`，通过委托模式组合 Triton 和 C++ 调度 | Ch6 |
| 19 | `DefaultHandler` | `torch._inductor.ops_handler` | `ops_handler.py` | 默认 Handler 基类，提供 `_default()` 回退机制，所有未覆盖的 ops 走默认路径 | Ch3 |
| 20 | `DtypePropagationOpsHandler` | `torch._inductor.dtype_propagation` | `dtype_propagation.py` | 数据类型传播 Handler，遍历 IR 闭包推断每个操作的输出 dtype | Ch3, Ch5 |
| 21 | `ExternKernelSchedulerNode` | `torch._inductor.scheduler` | `scheduler.py` | 外部内核调度节点，继承 `BaseSchedulerNode`，封装 `aten::` 等不可融合的外部算子调用 | Ch6 |
| 22 | `FallbackKernel` | `torch._inductor.ir` | `ir.py` | 回退内核 IR 节点，表示无法被 Inductor 融合而直接调用 ATen 实现的操作 | Ch4, Ch5 |
| 23 | `FreeSymbolsOpsHandler` | `torch._inductor.dependencies` | `dependencies.py` | 自由符号收集 Handler，遍历 IR 闭包收集其中引用的所有 SymPy 符号变量 | Ch3, Ch6 |
| 24 | `FusedSchedulerNode` | `torch._inductor.scheduler` | `scheduler.py` | 融合调度节点，继承 `BaseSchedulerNode`，表示多个 `SchedulerNode` 经融合后的复合节点 | Ch6 |
| 25 | `GraphLowering` | `torch._inductor.graph` | `graph.py` | 图级 Lowering 入口，继承 `torch.fx.Interpreter`，将 FX Graph 转换为 Inductor IR | Ch1, Ch5 |
| 26 | `HalideScheduling` | `torch._inductor.codegen.halide` | `codegen/halide.py` | Halide 后端调度策略，继承 `SIMDScheduling`，生成 Halide 调度描述 | Ch6 |
| 27 | `IndentedBuffer` | `torch._inductor.utils` | `utils.py` | 带缩进管理的文本缓冲区，用于所有代码生成场景中的源码拼接 | Ch7 |
| 28 | `InputBuffer` | `torch._inductor.ir` | `ir.py` | 输入缓冲区，继承 `Buffer`，表示编译期的外部输入张量 | Ch4 |
| 29 | `IRNode` | `torch._inductor.ir` | `ir.py` | 所有 Inductor IR 节点的基类，提供起源追踪（origin tracking）等通用基础设施 | Ch4 |
| 30 | `Kernel` | `torch._inductor.codegen.common` | `codegen/common.py` | 设备端 Kernel 代码生成器基类，继承 `CodeGen`，管理迭代范围、参数绑定和 CSE 上下文 | Ch7 |
| 31 | `KernelArgs` | `torch._inductor.codegen.common` | `codegen/common.py` | Kernel 参数管理器，负责为生成的内核函数分配参数名、类型和缓冲区映射 | Ch7 |
| 32 | `MetalScheduling` | `torch._inductor.codegen.mps` | `codegen/mps.py` | Metal (MPS) 后端调度策略，继承 `SIMDScheduling`，面向 Apple GPU 生成 Metal 着色器代码 | Ch6 |
| 33 | `MockHandler` | `torch._inductor.ops_handler` | `ops_handler.py` | 模拟 Handler，将所有 ops 调用转换为字符串表示，用于调试和 IR 可视化 | Ch3 |
| 34 | `NoopHandler` | `torch._inductor.ops_handler` | `ops_handler.py` | 空操作 Handler，所有 ops 调用均无副作用，用于仅需遍历而不需要结果的场景 | Ch3 |
| 35 | `NullHandler` | `torch._inductor.virtualized` | `virtualized.py` | 空值守卫（Sentinel），表示 Virtualized 变量未设置时的默认值，访问任意属性即报错 | Ch2 |
| 36 | `NullKernelHandler` | `torch._inductor.virtualized` | `virtualized.py` | 内核空值守卫，继承 `NullHandler`，提供 `removed_buffers` 等属性的默认空集合 | Ch2 |
| 37 | `OpOverrides` | `torch._inductor.codegen.common` | `codegen/common.py` | 后端 OpOverrides 基类，继承 `BasicMathOpsMixin`、`OpDecompositions` 和 `OpsHandler[Any]`，提供数学运算的元编程注册 | Ch3, Ch7 |
| 38 | `OpsHandler` | `torch._inductor.ops_handler` | `ops_handler.py` | Handler 协议的泛型基类 `OpsHandler[T]`，定义约 100 余个标量操作方法，T 决定语义域 | Ch3 |
| 39 | `OpsValue` | `torch._inductor.virtualized` | `virtualized.py` | ops 返回值的包装器，重载算术运算符以支持 `a + b` 语法糖替代 `ops.add(a, b)` | Ch2 |
| 40 | `OpsWrapper` | `torch._inductor.virtualized` | `virtualized.py` | ops 调用包装器，继承 `DefaultHandler`，将所有返回值包装为 `OpsValue` 实例 | Ch2 |
| 41 | `OuterLoopFusedKernel` | `torch._inductor.codegen.cpp` | `codegen/cpp.py` | 外层循环融合 Kernel，继承 `CppKernel`，将多个内核嵌套到统一的外层循环中 | Ch8 |
| 42 | `Pointwise` | `torch._inductor.ir` | `ir.py` | 逐点操作 IR 节点，继承 `Loops`，表示对张量每个元素独立执行同一段计算 | Ch4, Ch5 |
| 43 | `PythonWrapperCodegen` | `torch._inductor.codegen.wrapper` | `codegen/wrapper.py` | Python Wrapper 代码生成器，继承 `CodeGen`，生成调用内核的 Python 包装函数 | Ch7, Ch8 |
| 44 | `Reduction` | `torch._inductor.ir` | `ir.py` | 规约操作 IR 节点，继承 `Loops`，表示含规约维度（sum/max/argmax 等）的计算 | Ch4, Ch5 |
| 45 | `ReinterpretView` | `torch._inductor.ir` | `ir.py` | 重解释视图，继承 `BaseView`，在不拷贝数据的情况下以新的数据类型重新解释缓冲区 | Ch4 |
| 46 | `Scheduler` | `torch._inductor.scheduler` | `scheduler.py` | 调度器主类，负责依赖分析、拓扑排序、融合决策和代码生成的总协调 | Ch1, Ch6 |
| 47 | `SchedulerNode` | `torch._inductor.scheduler` | `scheduler.py` | 基础调度节点，继承 `BaseSchedulerNode`，封装单个 `ComputedBuffer` 或 `TemplateBuffer` | Ch6 |
| 48 | `SIMDKernel` | `torch._inductor.codegen.simd` | `codegen/simd.py` | SIMD Kernel 基类，继承 `Kernel[CSEVariableType]`，管理迭代范围分块和索引计算 | Ch7 |
| 49 | `SIMDScheduling` | `torch._inductor.codegen.simd` | `codegen/simd.py` | SIMD 调度策略基类，继承 `BaseScheduling`，为 Triton/Metal/Halide 等后端提供公共调度逻辑 | Ch6, Ch7 |
| 50 | `SimplifyIndexing` | `torch._inductor.sizevars` | `sizevars.py` | 索引简化 Handler，继承 `WrapperHandler`，利用 `SizeVarAllocator` 化简 SymPy 索引表达式 | Ch3, Ch6 |
| 51 | `SliceView` | `torch._inductor.ir` | `ir.py` | 切片视图，继承 `View`（-> `GenericView` -> `BaseView`），表示对缓冲区的切片操作 | Ch4 |
| 52 | `StorageBox` | `torch._inductor.ir` | `ir.py` | 存储盒，继承 `MutableBox(IRNode)`，包装 `Buffer` 并提供就地更新和别名管理 | Ch4 |
| 53 | `SubgraphPythonWrapperCodegen` | `torch._inductor.codegen.wrapper` | `codegen/wrapper.py` | 子图 Python Wrapper，继承 `PythonWrapperCodegen`，支持子图（如 autograd）的独立包装 | Ch8 |
| 54 | `SymPyOps` | `torch._inductor.index_propagation` | `index_propagation.py` | SymPy 符号操作 Handler，将所有 ops 翻译为 SymPy 表达式，用于索引传播与常量折叠 | Ch3 |
| 55 | `TemplateSchedulerNode` | `torch._inductor.scheduler` | `scheduler.py` | 模板调度节点（通过 `SchedulerNode` 统一处理 `TemplateBuffer`），用于手写内核模板的调度 | Ch6 |
| 56 | `TensorBox` | `torch._inductor.ir` | `ir.py` | 张量盒，继承 `MutableBox(IRNode)`，最外层 IR 包装器，提供视图操作的透明代理 | Ch4, Ch5 |
| 57 | `TritonKernel` | `torch._inductor.codegen.triton` | `codegen/triton.py` | Triton 后端 Kernel 代码生成器，继承 `SIMDKernel`，生成 `@triton.jit` 装饰的 GPU 内核 | Ch7, Ch8 |
| 58 | `TritonKernelOverrides` | `torch._inductor.codegen.triton` | `codegen/triton.py` | Triton 内核级 OpOverrides，继承 `TritonOverrides`，在内核上下文中生成 Triton 内建函数调用 | Ch3, Ch7 |
| 59 | `TritonOverrides` | `torch._inductor.codegen.triton` | `codegen/triton.py` | Triton 后端 OpOverrides，继承 `OpOverrides`，将标量 ops 翻译为 Triton 语言表达式 | Ch3, Ch7 |
| 60 | `TritonScheduling` | `torch._inductor.codegen.triton` | `codegen/triton.py` | Triton 后端调度策略，继承 `SIMDScheduling`，处理 GPU 内核的融合、分块和代码生成 | Ch6, Ch7 |
| 61 | `ValueRangeAnalysis` | `torch._inductor.bounds` | `bounds.py` | 值范围分析 Handler，继承 `SymPyValueRangeAnalysis` 和 `DefaultHandler`，推断每个操作的值域区间 | Ch3, Ch6 |
| 62 | `Virtualized` | `torch._inductor.virtualized` | `virtualized.py` | 线程局部动态作用域容器 `Virtualized[T]`，通过 context manager 实现安全的全局状态切换 | Ch2 |
| 63 | `WrapperCodegen` | *(见 `PythonWrapperCodegen`)* | `codegen/wrapper.py` | Wrapper 代码生成概念统称，在代码中由 `PythonWrapperCodegen` 及其子类承担此角色 | Ch7, Ch8 |
| 64 | `WrapperHandler` | `torch._inductor.ops_handler` | `ops_handler.py` | Handler 包装器基类，持有 `_inner` 引用，将未覆盖的 ops 委托给内部 Handler | Ch3 |
| 65 | `_RecordLoadStoreInner` | `torch._inductor.dependencies` | `dependencies.py` | 内存依赖记录 Handler，遍历 IR 闭包收集 `load`/`store` 操作对应的 `MemoryDep` 依赖 | Ch3, Ch6 |
| 66 | `_V` | `torch._inductor.virtualized` | `virtualized.py` | Virtualized 门面类，汇聚所有 `Virtualized` 实例的 getter/setter，提供 `V.ops`、`V.kernel` 等统一访问入口 | Ch2 |

---

## 附录 B：继承树快速参考图

### B.1 Virtualized 上下文树

`_V` 是 Virtualized 子系统的门面（Facade），它本身不是基类，而是将所有 `Virtualized[T]` 实例汇聚为一组属性和方法。以下是 `_V` 暴露的全部 11 个动态作用域上下文（每个底层都是一个 `Virtualized[T]` 实例）：

```
_V (门面类，单例 V = _V())
│
├── V.ops                  : Virtualized[OpsHandler[Any]]
│   └── 当前活跃的操作 Handler（CSEProxy / _RecordLoadStoreInner / ...）
│
├── V.graph                : Virtualized[GraphLowering]
│   └── 当前正在编译的 GraphLowering 实例
│
├── V.kernel               : Virtualized[NullKernelHandler]
│   └── 当前活跃的 Kernel 代码生成器（TritonKernel / CppKernel / ...）
│
├── V.fake_mode            : Virtualized[FakeTensorMode]
│   └── 当前编译使用的 FakeTensorMode（用于 shape 推断）
│
├── V.debug                : Virtualized[DebugContext]
│   └── 调试上下文（控制调试输出目录和日志级别）
│
├── V.interpreter          : Virtualized[InterpreterShim]
│   └── FX Interpreter 封装（Lowering 阶段遍历 FX Graph）
│
├── V.real_inputs          : Virtualized[list[torch.Tensor]]
│   └── 真实输入张量（非 fake），用于某些需要真实数据的场景
│
├── V.aot_compilation      : Virtualized[bool]
│   └── 是否处于 AOT 编译模式
│
├── V.current_node         : Virtualized[torch.fx.Node]
│   └── 当前正在处理的 FX 节点（用于起源追踪）
│
├── V.local_buffer_context : Virtualized[LocalBufferContext]
│   └── 局部缓冲区管理上下文（C++ 后端使用）
│
└── V.choices              : Virtualized[InductorChoices]
    └── 启发式策略配置（延迟初始化，控制融合/分块等决策参数）
```

**典型生命周期：**

```
compile_fx.compile_fx_inner()
  │
  ├── V.set_graph_handler(graph)           # 安装 GraphLowering
  ├── V.set_fake_mode(fake_mode)           # 安装 FakeTensorMode
  ├── V.set_choices_handler(choices)       # 安装启发式策略
  │
  ├── [Lowering 阶段]
  │     └── Interpreter.run()
  │           └── V.set_ops_handler(...)    # 不同算子可能临时切换 handler
  │
  ├── [调度阶段]
  │     └── V.set_ops_handler(_RecordLoadStoreInner())  # 依赖分析
  │
  ├── [代码生成阶段]
  │     ├── V.set_kernel_handler(triton_kernel)          # 安装 Kernel
  │     └── V.set_ops_handler(CSEProxy(...))             # 安装代码生成 handler
  │
  └── [Wrapper 生成]
        └── V.set_kernel_handler(NullKernelHandler())    # 清除 kernel 上下文
```

### B.2 OpsHandler 继承树

```
OpsHandler[T] (泛型协议基类, ops_handler.py)
│   定义 ~100 个标量操作方法：load, store, add, relu, ...
│   T 决定语义域（str / torch.dtype / ValueRanges[Expr] / ...）
│
├── DefaultHandler (ops_handler.py)
│   │   提供 _default() 回退机制：未覆盖的 ops 通过 getattr 动态路由
│   │
│   ├── MockHandler (ops_handler.py)
│   │   └── T = str：将所有 ops 转为字符串表示，用于调试和 IR 打印
│   │
│   ├── NoopHandler (ops_handler.py)
│   │   └── 所有 ops 无副作用，返回 None
│   │
│   ├── DtypePropagationOpsHandler (dtype_propagation.py)
│   │   └── T = torch.dtype：推断每个操作的输出数据类型
│   │
│   ├── ValueRangeAnalysis (bounds.py)
│   │   └── T = ValueRanges[Expr]：推断每个操作的值范围区间
│   │       继承链：SymPyValueRangeAnalysis -> ValueRangeAnalysis
│   │
│   ├── SymPyOps / IndexPropagation (index_propagation.py)
│   │   └── T = TypedExpr：将 ops 翻译为 SymPy 符号表达式
│   │       SymPyOps -> IndexPropagation(DefaultHandler)
│   │
│   ├── FreeSymbolsOpsHandler (dependencies.py)
│   │   └── T = None（副作用型）：收集 IR 闭包中的所有自由符号
│   │
│   ├── CountOps (loop_body.py)
│   │   └── T = Counter：统计 IR 闭包中各类操作的出现次数
│   │
│   ├── OpCounterCSE (ops_handler.py)
│   │   └── T = CSEVariable：带 CSE 去重的操作计数器
│   │
│   ├── KernelFormatterHandler (ops_handler.py)
│   │   └── 格式化内核操作日志，被 RecordLoadStore 使用
│   │
│   └── ExtractConstantsHandler (ops_handler.py)
│       └── 继承 NoopHandler，提取常量表达式
│
├── WrapperHandler (ops_handler.py)
│   │   持有 _inner Handler 引用，将未覆盖的 ops 委托给 _inner
│   │
│   ├── SimplifyIndexing (sizevars.py)
│   │   └── 利用 SizeVarAllocator 化简索引表达式
│   │
│   ├── CSEProxy (codegen/common.py)
│   │   └── 公共子表达式消除代理（实际继承 DefaultHandler，但行为等价 Wrapper）
│   │
│   ├── CaptureIndexing (loop_body.py)
│   │   └── 捕获索引表达式，用于优化
│   │
│   ├── LocalizeBufferHandler (codegen/cpp_utils.py)
│   │   └── C++ 后端专用：将缓冲区引用重定位到本地变量
│   │
│   ├── AddParenHandler (ops_handler.py)
│   │   └── 为所有表达式添加括号（调试用）
│   │
│   └── SimpleCSEHandler (ops_handler.py)
│       └── 简化版 CSE 处理器（调试用）
│
├── OpOverrides (codegen/common.py)
│   │   后端 OpOverrides 基类，通过元编程自动注册数学运算
│   │   继承链：BasicMathOpsMixin -> OpDecompositions -> OpOverrides
│   │
│   ├── TritonOverrides (codegen/triton.py)
│   │   └── Triton 表达式翻译（标量级）
│   │       └── TritonKernelOverrides (codegen/triton.py)
│   │           └── Triton 内核级翻译，增加 load/store 的 tl.load/tl.store 支持
│   │
│   ├── CppOverrides (codegen/cpp.py)
│   │   └── C++ 标量表达式翻译
│   │       └── CppVecOverrides (codegen/cpp.py)
│   │           └── C++ SIMD intrinsics 翻译
│   │               └── CppTile2DOverrides (codegen/cpp.py)
│   │                   └── C++ 2D 分块 SIMD 翻译
│   │
│   ├── HalideOverrides (codegen/halide.py)
│   │   └── Halide 表达式翻译
│   │
│   └── MetalOverrides (codegen/mps.py)
│       └── Metal 着色器表达式翻译
│
└── _RecordLoadStoreInner (dependencies.py)
    └── 遍历 IR 闭包，记录每个 load/store 操作产生 MemoryDep 依赖
        实际继承 V.MockHandler（即 MockHandler）
```

### B.3 Buffer 继承树

```
IRNode (ir.py)                           # 所有 IR 节点的基类
│
├── MutableBox (ir.py)                   # 可变盒子，提供 data 属性的透明代理
│   │
│   ├── TensorBox (ir.py)                # 最外层包装：张量级接口
│   │   └── TensorBox.create(data) -> TensorBox(StorageBox(data))
│   │       用户层面的 IR 入口，所有 lowering 操作产出的最外层对象
│   │
│   └── StorageBox (ir.py)               # 中间层包装：存储管理
│       └── StorageBox(data)             # data 是 Buffer 或其子类
│           提供就地更新（realize()）、别名管理和缓冲区复用
│
└── Buffer (ir.py)                       # 核心缓冲区节点
    │   同时实现 IRNode 和 CodegenSymbol
    │   持有 name (str) 和 layout (OutputSpec)
    │
    ├── InputBuffer (ir.py)              # 外部输入缓冲区
    │   ├── DonatedBuffer (ir.py)        # 可捐赠的输入（编译后释放）
    │   └── ConstantBuffer (ir.py)       # 编译期常量缓冲区
    │
    ├── OperationBuffer (ir.py)          # 操作缓冲区 = Buffer + Operation
    │   ├── ComputedBuffer (ir.py)       # Inductor 计算生成的缓冲区
    │   │   持有 Pointwise / Reduction / Scan 等操作描述
    │   │
    │   └── TemplateBuffer (ir.py)       # 手写内核模板缓冲区
    │       ├── TritonTemplateBuffer (ir.py)    # Triton 模板缓冲区
    │       ├── MultiTemplateBuffer (ir.py)     # 多模板选择缓冲区
    │       ├── CUDATemplateBuffer (ir.py)      # CUDA 模板缓冲区
    │       └── CppTemplateBuffer (ir.py)       # C++ 模板缓冲区
    │
    ├── BaseView (ir.py) → IRNode        # 视图基类
    │   ├── ExpandView                   # 广播视图
    │   ├── PermuteView                  # 转置视图
    │   ├── SqueezeView                  # 压缩维度视图
    │   ├── GenericView → View           # 通用视图
    │   │   └── SliceView (ir.py)        # 切片视图
    │   ├── ReinterpretView (ir.py)      # 重解释类型视图
    │   └── DtypeView                    # 数据类型转换视图
    │
    └── InputsKernel (ir.py)             # 多输入合并内核缓冲区
```

**三层层级关系总结：**

```
用户层:   TensorBox    ← 对外统一接口，承载视图操作
            │
存储层:   StorageBox   ← 管理缓冲区生命周期、别名、就地更新
            │
数据层:   Buffer       ← 持有实际的名称、布局、数据类型信息
```

### B.4 SchedulerNode 继承树

```
BaseSchedulerNode (scheduler.py)
│   持有 node (ir.Buffer)、users (set)、dep_edges (list) 矢量
│   定义 get_name(), get_size(), get_device(), mark_fusable() 等接口
│
├── SchedulerNode (scheduler.py)
│   │   封装 ComputedBuffer 或 TemplateBuffer
│   │   管理 read_writes (ReadWrites)、group (循环范围) 等调度属性
│   │
│   ├── 用于 Pointwise / Reduction / Scan 等可融合节点
│   └── 当 node 是 TemplateBuffer 时，行为等同于 TemplateSchedulerNode
│
├── FusedSchedulerNode (scheduler.py)
│   │   融合节点，包含多个 SchedulerNode
│   │   持有 snodes: list[SchedulerNode] 属性
│   │
│   ├── ForeachKernelSchedulerNode (scheduler.py)
│   │   └── torch.stack / torch.cat 等批量操作的融合节点
│   │
│   └── OuterLoopFusedSchedulerNode (codegen/cpp.py)
│       └── C++ 后端外层循环融合专用节点
│
├── ExternKernelSchedulerNode (scheduler.py)
│   └── 不可融合的外部算子（如 aten::matmul），直接调用 ATen 实现
│       不参与融合，但参与依赖分析和拓扑排序
│
├── NopKernelSchedulerNode (scheduler.py)
│   └── 空操作节点，用于占位或已被优化的节点
│
└── GroupedSchedulerNode (scheduler.py)
    └── 分组调度节点，用于 grouped reduce-scatter 等集合通信操作
```

### B.5 BaseScheduling 继承树

```
BaseScheduling (scheduler.py)
│   定义调度策略接口：
│   - codegen_node(node)           # 为单个节点生成代码
│   - codegen_nodes(nodes)         # 为节点组生成代码
│   - group_fn                     # 节点分组函数
│   - can_fuse_vertical/horizontal # 融合可行性判断
│
├── CppScheduling (codegen/cpp.py)
│   └── 纯 C++ 标量后端（非向量化）
│       生成嵌套 for 循环 + 标量运算的 C++ 代码
│       适用场景：不支持 SIMD 的 CPU 或 fallback
│
├── SIMDScheduling (codegen/simd.py)
│   │   SIMD 调度策略基类
│   │   提供通用的分块、融合、代码生成逻辑
│   │   管理 IterationRanges 和迭代层次
│   │
│   ├── TritonScheduling (codegen/triton.py)
│   │   └── Triton GPU 后端调度
│   │       负责 Triton 内核的融合决策、config 选择、代码生成
│   │
│   ├── MetalScheduling (codegen/mps.py)
│   │   └── Apple Metal (MPS) 后端调度
│   │       生成 Metal 计算着色器代码
│   │
│   └── HalideScheduling (codegen/halide.py)
│       └── Halide 后端调度
│           生成 Halide 调度描述和管道代码
│
├── CUDACombinedScheduling (codegen/cuda_combined_scheduling.py)
│   └── CUDA 组合调度策略
│       不直接生成代码，而是通过委托模式将节点分发给
│       TritonScheduling 或 CUDACPPScheduling
│       适用场景：同一计算图中混合 Triton 和 C++ 内核
│
├── CUDACPPScheduling (codegen/cuda/cuda_cpp_scheduling.py)
│   └── CUDA C++ 后端调度（用于 CUDA 上的 C++ 内核模板）
│
└── ROCmCPPScheduling (codegen/rocm/rocm_cpp_scheduling.py)
    └── ROCm C++ 后端调度（AMD GPU 上的 C++ 内核模板）
```

**调度策略选择逻辑（简化）：**

```
Scheduler.__init__()
  └── get_scheduling_for_device(device)
        │
        ├── CPU 设备:
        │   ├── 支持 SIMD → SIMDScheduling (或 CppScheduling)
        │   └── 无 SIMD  → CppScheduling
        │
        ├── CUDA 设备:
        │   ├── Triton 可用 → TritonScheduling
        │   ├── 混合后端   → CUDACombinedScheduling
        │   └── C++ 模板   → CUDACPPScheduling
        │
        ├── ROCm 设备:
        │   └── ROCmCPPScheduling
        │
        └── MPS 设备:
            └── MetalScheduling
```

### B.6 Kernel 继承树

```
CodeGen (codegen/common.py)
│   代码生成器的最底层基类，持有 _emit()
│
└── Kernel[CSEVariableType] (codegen/common.py)
    │   设备端 Kernel 代码生成器基类
    │   管理迭代范围 (IterationRanges)、CSE 上下文、参数绑定
    │   定义 load/store/reduction 等核心操作的骨架
    │
    ├── CppKernel (codegen/cpp.py)
    │   │   C++ 后端内核，生成嵌套 for 循环代码
    │   │
    │   ├── CppVecKernel (codegen/cpp.py)
    │   │   │   C++ SIMD 向量化内核
    │   │   │   管理向量宽度、SIMD 寄存器分配
    │   │   │
    │   │   └── CppTile2DKernel (codegen/cpp.py)
    │   │       └── 2D 分块向量化内核，用于矩阵乘法等 2D 计算模式
    │   │
    │   ├── OuterLoopFusedKernel (codegen/cpp.py)
    │   │   └── 外层循环融合内核，将多个内核嵌套到统一循环中
    │   │
    │   └── CppKernelProxy (codegen/cpp.py)
    │       └── CppKernel 代理，用于委托式内核调用
    │
    └── SIMDKernel[CSEVariableType] (codegen/simd.py)
        │   SIMD 内核基类
        │   管理 IterationRangesRoot/Entry 层次结构
        │   提供通用的索引计算和迭代范围管理
        │
        ├── TritonKernel (codegen/triton.py)
        │   │   Triton GPU 内核，生成 @triton.jit Python 函数
        │   │   管理 block 大小、线程层次、共享内存等
        │   │
        │   └── TritonSplitScanKernel (codegen/triton_split_scan.py)
        │       └── Triton 分段扫描内核，用于 scan 操作的分块实现
        │
        ├── MetalKernel (codegen/mps.py)
        │   └── Metal 计算着色器内核，生成 MSL (Metal Shading Language) 代码
        │
        └── HalideKernel (codegen/halide.py)
            └── Halide 管道内核，生成 Halide 调度描述

── 模板内核（独立继承链） ──────────────────────────

TritonKernel (codegen/triton.py)
└── TritonTemplateKernel (select_algorithm.py)
    └── Triton 手写模板内核（如融合 attention、matmul 模板）

CUDAKernel (codegen/cuda/cuda_kernel.py)
└── CUDATemplateKernel (codegen/cuda/cuda_kernel.py)
    └── CUDA 手写模板内核（如 CUTLASS-based matmul 模板）
```

### B.7 WrapperCodegen 继承树

```
CodeGen (codegen/common.py)
│
└── PythonWrapperCodegen (codegen/wrapper.py)
    │   Python 包装代码生成器基类
    │   生成调用各内核的 Python 函数源码
    │   管理 内存规划、缓冲区分配/释放、内核调用序列
    │   核心方法：generate(), call_kernel(), codegen_input_buffer()
    │
    ├── SubgraphPythonWrapperCodegen (codegen/wrapper.py)
    │   └── 子图包装生成器（用于 autograd 前向/反向分离场景）
    │
    ├── WrapperFxCodegen (codegen/wrapper_fxir.py)
    │   └── FX IR 包装生成器，将内核调用转换为 FX Graph 表示
    │
    └── CppWrapperCpu (codegen/cpp_wrapper_cpu.py)
        │   CPU C++ 包装生成器，继承 PythonWrapperCodegen
        │   生成 C++ 代码替代 Python 包装，减少 Python 开销
        │
        ├── CppWrapperCpuArrayRef (codegen/cpp_wrapper_cpu_array_ref.py)
        │   └── CPU C++ 包装（ArrayRef 接口版本）
        │
        └── CppWrapperGpu (codegen/cpp_wrapper_gpu.py)
            │   GPU C++ 包装生成器，继承 CppWrapperCpu
            │   增加 CUDA/ROCm 相关的设备管理和流同步
            │
            └── CppWrapperMps (codegen/cpp_wrapper_mps.py)
                └── MPS C++ 包装生成器，处理 Metal 命令缓冲区提交
```

---

## 附录 C：Triton 后端完整调用栈参考

本附录展示从 `torch.compile(fn)` 到最终 Triton 内核执行的完整调用链，覆盖 Dynamo、AOTAutograd 和 Inductor 的所有关键阶段。

### C.1 顶层入口

```
torch.compile(fn, backend="inductor")
  │
  └── torch._dynamo.optimize(backend="inductor")(fn)
        │   返回一个 wrapped 函数，首次调用时触发编译
        │
        └── [首次调用时]
```

### C.2 TorchDynamo 阶段 —— FX Graph 捕获

```
torch._dynamo.optimize()
  └── _dynamo.utils.compile_and_call_fn()
        │
        ├── [Frame Evaluation]
        │     CPython PEP 659 字节码监控机制
        │     在函数执行时拦截每条字节码，记录张量操作
        │
        ├── [Tracing]
        │     执行用户函数，收集所有 torch 操作为 FX Node
        │     遇到不支持的操作时 graph break
        │
        └── _dynamo.output_graph.OutputGraph.compile()
              │   将收集的 FX Node 组装为 torch.fx.Graph
              │   输出：Joint Graph（前向+反向交织）
              │
              └── 调用 backend（即 AOTAutograd）
```

### C.3 AOTAutograd 阶段 —— 前向/反向分离

```
AOTAutograd (torch._functorch/aot_autograd.py)
  └── aot_autograd(fw_compiler, bw_compiler)
        │
        ├── [Graph Partition]
        │     将 Joint Graph 分离为：
        │     - forward_graph: 仅前向计算节点
        │     - backward_graph: 仅反向梯度计算节点
        │     保存中间激活值用于反向传播
        │
        ├── [Decomposition]
        │     将高层 PyTorch 操作分解为 ATen 基础操作
        │     例如: cross_entropy → log_softmax + nll_loss
        │     输出：两个仅包含 aten:: 命名空间的 FX Graph
        │
        └── fw_compiler(forward_graph)    # 调用 Inductor 编译前向图
            bw_compiler(backward_graph)   # 调用 Inductor 编译反向图
```

### C.4 Inductor 入口 —— compile_fx

```
compile_fx.compile_fx_inner(graph, example_inputs, ...)
  │   文件：torch/_inductor/compile_fx.py
  │
  ├── ==========================================
  ├── [Phase 0] 常量折叠 (Constant Folding)
  ├── ==========================================
  │
  ├── constant_folding(graph)
  │   └── GraphLowering(const_graph).run()
  │         │   文件：torch/_inductor/graph.py
  │         │   继承 torch.fx.Interpreter
  │         │
  │         ├── __init__()
  │         │     创建缓冲区映射、dce 管道
  │         │
  │         └── run()
  │               │   Interpreter 的主循环
  │               │
  │               ├── placeholder()    → 创建 InputBuffer + TensorBox
  │               ├── get_attr()       → 创建 ConstantBuffer
  │               ├── call_function()  → 执行常量折叠 lowering
  │               └── output()         → 收集输出
  │
  ├── ==========================================
  ├── [Phase 1] 主 Lowering
  ├── ==========================================
  │
  ├── GraphLowering(main_graph).run()
  │     │
  │     ├── placeholder(node)
  │     │     为每个输入创建：
  │     │     InputBuffer(name, layout) → StorageBox → TensorBox
  │     │
  │     ├── get_attr(node)
  │     │     为每个参数/常量创建：
  │     │     ConstantBuffer(name, layout) → StorageBox → TensorBox
  │     │
  │     ├── call_function(node, args, kwargs)
  │     │     │   根据 node.target 查找对应的 lowering 函数
  │     │     │
  │     │     ├── aten.add.Tensor → lowering.add()
  │     │     │     └── create_pointwise(ops.add, ...)
  │     │     │           定义 inner_fn = lambda: ops.add(ops.load(...), ...)
  │     │     │           返回 Pointwise(inner_fn=inner_fn) → ComputedBuffer → StorageBox → TensorBox
  │     │     │
  │     │     ├── aten.relu.default → lowering.relu()
  │     │     │     └── create_pointwise(ops.relu, ...)
  │     │     │
  │     │     ├── aten.mm.default → lowering.mm()
  │     │     │     └── 选择模板内核或回退到 ATen
  │     │     │           可能创建 TemplateBuffer 或 FallbackKernel
  │     │     │
  │     │     ├── aten.sum.dim_IntList → lowering.sum()
  │     │     │     └── create_reduction(ops.add, ...)
  │     │     │           定义 inner_fn 含 reduction 维度处理
  │     │     │           返回 Reduction(inner_fn=inner_fn) → ComputedBuffer
  │     │     │
  │     │     └── [~200+ 个其他 aten 算子的 lowering]
  │     │
  │     └── output(node)
  │           收集所有输出 TensorBox，记录输出缓冲区映射
  │
  ├── graph.finalize()  # 后处理：死代码消除、缓冲区复用等
  │
  ├── ==========================================
  ├── [Phase 2] 调度 (Scheduling)
  ├── ==========================================
  │
  ├── Scheduler.__init__(graph)
  │     │   文件：torch/_inductor/scheduler.py
  │     │
  │     ├── [Step 2a] 创建 SchedulerNode
  │     │     遍历 graph.buffers，为每个 Buffer 创建对应的 SchedulerNode：
  │     │     - ComputedBuffer → SchedulerNode
  │     │     - TemplateBuffer → SchedulerNode (模板模式)
  │     │     - FallbackKernel → ExternKernelSchedulerNode
  │     │
  │     ├── [Step 2b] 依赖分析
  │     │     对每个 SchedulerNode 执行依赖提取：
  │     │
  │     │     V.set_ops_handler(_RecordLoadStoreInner())
  │     │     node.inner_fn()  # 执行 IR 闭包
  │     │       └── V.ops.load(name, index)  → 记录 MemoryDep(name, index)
  │     │       └── V.ops.store(name, index) → 记录 MemoryDep(name, index)
  │     │
  │     │     结果：每个节点获得 read_writes = ReadWrites(
  │     │       reads={MemoryDep(buf1, idx1), MemoryDep(buf2, idx2)},
  │     │       writes={MemoryDep(buf_out, idx_out)}
  │     │     )
  │     │
  │     ├── [Step 2c] 值范围分析
  │     │     V.set_ops_handler(ValueRangeAnalysis())
  │     │     node.inner_fn()
  │     │       └── 推断每个中间值的范围，用于优化决策
  │     │
  │     ├── [Step 2d] 索引传播
  │     │     V.set_ops_handler(IndexPropagation())
  │     │     node.inner_fn()
  │     │       └── 计算索引表达式，用于常量折叠和化简
  │     │
  │     ├── [Step 2e] 拓扑排序
  │     │     topological_sort()
  │     │       基于 dep_edges 构建依赖有向图
  │     │       按依赖关系排序节点，确保被依赖者在前
  │     │
  │     └── [Step 2f] 融合决策
  │           fusion_pass()
  │             │   遍历拓扑排序后的节点列表，尝试融合
  │             │
  │             ├── can_fuse_vertical(node1, node2)
  │             │   判断 node2 是否可以内联到 node1 的循环体中
  │             │   条件：node1 读取 node2，且两者循环范围兼容
  │             │
  │             ├── can_fuse_horizontal(node1, node2)
  │             │   判断 node1 和 node2 是否可以合并为同一个内核
  │             │   条件：两者读取相同输入，循环范围相同
  │             │
  │             └── 融合结果：
  │                 SchedulerNode + SchedulerNode → FusedSchedulerNode
  │                 原始节点被标记为 fused，FusedSchedulerNode 取代其位置
  │
  ├── ==========================================
  ├── [Phase 3] 代码生成 (Code Generation)
  ├── ==========================================
  │
  ├── Scheduler.codegen()
  │     │
  │     └── [为每个后端设备选择 Scheduling 策略]
  │           │
  │           └── TritonScheduling.codegen_nodes(nodes)
  │                 │   文件：torch/_inductor/codegen/triton.py
  │                 │
  │                 ├── [对每个 FusedSchedulerNode / SchedulerNode]
  │                 │
  │                 └── TritonScheduling.codegen_node(node)
  │                       │
  │                       ├── TritonKernel.__init__(...)
  │                       │     │   创建 Triton 内核实例
  │                       │     │   设置迭代范围、block 大小参数
  │                       │     │
  │                       │     └── IterationRangesRoot(itervars, ranges)
  │                       │           创建迭代范围层次结构
  │                       │
  │                       ├── with V.set_kernel_handler(triton_kernel):
  │                       │     │   将 TritonKernel 安装为 V.kernel
  │                       │     │
  │                       │     ├── with V.set_ops_handler(...):
  │                       │     │     │   安装代码生成 Handler 链
  │                       │     │     │
  │                       │     │     └── V.ops = CSEProxy(
  │                       │     │           SimplifyIndexing(
  │                       │     │             TritonKernelOverrides()))
  │                       │     │
  │                       │     │     Handler 链调用流程：
  │                       │     │     V.ops.add(a, b)
  │                       │     │       → CSEProxy.add(a, b)
  │                       │     │         → 查询 CSE 缓存
  │                       │     │         → 若未命中: SimplifyIndexing.add(a, b)
  │                       │     │           → 化简索引表达式
  │                       │     │           → TritonKernelOverrides.add(a, b)
  │                       │     │             → 返回 "tl.add(a_var, b_var)"
  │                       │     │         → CSE 缓存结果
  │                       │     │         → 返回 CSEVariable("tmp0", dtype)
  │                       │     │
  │                       │     └── inner_fn()  # 执行 IR 闭包
  │                       │           │   在 TritonKernelOverrides 上下文中
  │                       │           │   逐个调用 ops 方法
  │                       │           │
  │                       │           ├── V.ops.load("buf0", index_expr)
  │                       │           │     → CSEProxy → SimplifyIndexing → TritonKernelOverrides.load()
  │                       │           │     → 生成 "tl.load(buf0_ptr + offset, mask=...)"
  │                       │           │     → 返回 CSEVariable("load_tmp0", dtype)
  │                       │           │
  │                       │           ├── V.ops.add(load_a, load_b)
  │                       │           │     → 生成 "tl.add(load_tmp0, load_tmp1)"
  │                       │           │     → 返回 CSEVariable("add_tmp0", dtype)
  │                       │           │
  │                       │           ├── V.ops.relu(add_result)
  │                       │           │     → 生成 "tl.maximum(add_tmp0, 0)"
  │                       │           │     → 返回 CSEVariable("relu_tmp0", dtype)
  │                       │           │
  │                       │           └── V.ops.store("buf_out", index_expr, relu_result)
  │                       │                 → 生成 "tl.store(buf_out_ptr + offset, relu_tmp0, mask=...)"
  │                       │
  │                       └── TritonKernel.codegen()
  │                             │   将收集的所有 CSE 变量和操作
  │                             │   组装为完整的 @triton.jit 内核函数
  │                             │
  │                             ├── 生成内核签名
  │                             │     @triton.jit
  │                             │     def triton_(buf0_ptr, buf1_ptr, buf_out_ptr,
  │                             │                 xnumel, ynumel, ...,
  │                             │                 BLOCK_X: tl.constexpr, ...):
  │                             │
  │                             ├── 生成迭代循环
  │                             │     xoffset = tl.program_id(0) * BLOCK_X
  │                             │     xindex = xoffset + tl.arange(0, BLOCK_X)
  │                             │     xmask = xindex < xnumel
  │                             │
  │                             ├── 插入所有 load/store/compute 语句
  │                             │     (从 CSE 缓存中提取去重后的代码)
  │                             │
  │                             └── 返回完整内核源码字符串
  │
  ├── ==========================================
  ├── [Phase 4] Wrapper 代码生成
  ├── ==========================================
  │
  ├── PythonWrapperCodegen.generate()
  │     │   文件：torch/_inductor/codegen/wrapper.py
  │     │
  │     ├── [为每个内核生成调用代码]
  │     │
  │     ├── call_kernel(kernel_name, kernel_obj, ...)
  │     │     │   生成 Python 代码调用已编译的内核
  │     │     │
  │     │     ├── Triton 内核调用：
  │     │     │     triton_.run(
  │     │     │       buf0.data_ptr(), buf1.data_ptr(), buf_out.data_ptr(),
  │     │     │       xnumel=1024, ynumel=512,
  │     │     │       BLOCK_X=128, ...,
  │     │     │       stream=torch.cuda.current_stream().cuda_stream
  │     │     │     )
  │     │     │
  │     │     └── ATen 回退调用：
  │     │           torch.ops.aten.mm(input, weight)
  │     │
  │     ├── codegen_input_buffer(...)
  │     │     生成输入缓冲区的初始化和内存分配代码
  │     │
  │     ├── codegen_output_buffer(...)
  │     │     生成输出缓冲区的返回代码
  │     │
  │     └── 生成最终 Python 函数源码
  │           │
  │           └── def forward(self, arg0, arg1, ...):
  │                 buf0 = arg0
  │                 buf1 = arg1
  │                 # kernel: triton_0
  │                 triton_0.run(buf0.data_ptr(), buf1.data_ptr(), ...)
  │                 buf_out = ...
  │                 return (buf_out,)
  │
  └── [Phase 5] 编译与缓存
        │
        ├── kernel_code = compile_kernel(triton_source)
        │     将 Triton Python 源码编译为可执行内核
        │     缓存到 disk（torch_compile_debug/ 目录）
        │
        ├── wrapper_code = compile_wrapper(python_source)
        │     将 Python wrapper 源码编译为可调用函数
        │
        └── 返回编译后的函数对象
              首次调用时编译，后续调用直接使用缓存
```

### C.2 完整调用栈（线性视图）

以下将上述过程压缩为单一线性调用栈，便于快速定位特定函数的调用位置：

```
torch.compile(model)
│
├── torch._dynamo.optimize("inductor")(model)
│     ├── _dynamo.utils.compile_and_call_fn()
│     │     └── _dynamo.output_graph.OutputGraph.compile()
│     │           └── 输出：torch.fx.Graph（Joint Graph）
│     │
│     └── aot_autograd(fw_compiler=compile_fx, bw_compiler=compile_fx)
│           ├── make_fx(fw_graph) → forward FX Graph
│           ├── make_fx(bw_graph) → backward FX Graph
│           └── fw_compiler(forward_graph)
│
├── compile_fx.compile_fx_inner(graph, inputs)
│     │
│     ├── [Phase 0] constant_folding(graph)
│     │     └── GraphLowering(const_graph).run()
│     │
│     ├── [Phase 1] GraphLowering(main_graph).run()
│     │     ├── placeholder() → InputBuffer → TensorBox
│     │     ├── call_function() → lowering.xxx() → Pointwise/Reduction → ComputedBuffer
│     │     └── output() → 收集输出
│     │
│     ├── graph.finalize()
│     │
│     ├── [Phase 2] Scheduler.__init__(graph)
│     │     ├── 为每个 Buffer 创建 SchedulerNode
│     │     ├── V.set_ops_handler(_RecordLoadStoreInner())
│     │     │     node.inner_fn() → 收集 MemoryDep
│     │     ├── V.set_ops_handler(ValueRangeAnalysis())
│     │     │     node.inner_fn() → 推断值范围
│     │     ├── V.set_ops_handler(IndexPropagation())
│     │     │     node.inner_fn() → 计算索引表达式
│     │     ├── topological_sort()
│     │     └── fusion_pass()
│     │           ├── can_fuse_vertical()
│     │           └── can_fuse_horizontal()
│     │
│     ├── [Phase 3] Scheduler.codegen()
│     │     └── TritonScheduling.codegen_node(fused_node)
│     │           ├── TritonKernel.__init__()
│     │           ├── V.set_kernel_handler(triton_kernel)
│     │           ├── V.set_ops_handler(CSEProxy(SimplifyIndexing(TritonKernelOverrides())))
│     │           ├── inner_fn()  # 执行闭包，生成 Triton 代码
│     │           │     ├── V.ops.load() → CSEProxy → SimplifyIndexing → TritonKernelOverrides.load()
│     │           │     ├── V.ops.add()  → CSEProxy → SimplifyIndexing → TritonKernelOverrides.add()
│     │           │     ├── V.ops.relu() → CSEProxy → SimplifyIndexing → TritonKernelOverrides.relu()
│     │           │     └── V.ops.store() → TritonKernelOverrides.store()
│     │           └── TritonKernel.codegen() → @triton.jit 源码
│     │
│     └── [Phase 4] PythonWrapperCodegen.generate()
│           ├── call_kernel() → 内核调用代码
│           ├── codegen_input_buffer() → 输入分配
│           └── 输出：完整 Python wrapper 函数源码
│
└── [Phase 5] 编译缓存
      ├── compile_kernel() → 可执行 Triton 内核
      └── compile_wrapper() → 可调用 Python 函数
```

### C.3 关键文件索引

| 阶段 | 关键文件 | 主要类/函数 |
|------|---------|-----------|
| Dynamo | `torch/_dynamo/` | `optimize()`, `OutputGraph` |
| AOTAutograd | `torch/_functorch/aot_autograd.py` | `aot_autograd()`, `make_fx()` |
| Inductor 入口 | `torch/_inductor/compile_fx.py` | `compile_fx_inner()` |
| Graph Lowering | `torch/_inductor/graph.py` | `GraphLowering` |
| 算子 Lowering | `torch/_inductor/lowering.py` | `add()`, `relu()`, `mm()`, ... |
| IR 定义 | `torch/_inductor/ir.py` | `IRNode`, `Buffer`, `Pointwise`, `Reduction` |
| 调度器 | `torch/_inductor/scheduler.py` | `Scheduler`, `BaseSchedulerNode`, `BaseScheduling` |
| 依赖分析 | `torch/_inductor/dependencies.py` | `_RecordLoadStoreInner`, `MemoryDep`, `ReadWrites` |
| 值域分析 | `torch/_inductor/bounds.py` | `ValueRangeAnalysis` |
| 索引传播 | `torch/_inductor/index_propagation.py` | `SymPyOps`, `IndexPropagation` |
| 索引化简 | `torch/_inductor/sizevars.py` | `SimplifyIndexing`, `SizeVarAllocator` |
| Virtualized | `torch/_inductor/virtualized.py` | `Virtualized`, `_V`, `OpsValue` |
| Handler 协议 | `torch/_inductor/ops_handler.py` | `OpsHandler[T]`, `DefaultHandler`, `WrapperHandler` |
| 类型推断 | `torch/_inductor/dtype_propagation.py` | `DtypePropagationOpsHandler` |
| CSE 引擎 | `torch/_inductor/codegen/common.py` | `CSE`, `CSEProxy`, `CSEVariable`, `Kernel` |
| Triton 后端 | `torch/_inductor/codegen/triton.py` | `TritonKernel`, `TritonScheduling`, `TritonOverrides` |
| SIMD 基类 | `torch/_inductor/codegen/simd.py` | `SIMDKernel`, `SIMDScheduling` |
| C++ 后端 | `torch/_inductor/codegen/cpp.py` | `CppKernel`, `CppScheduling`, `CppOverrides` |
| Wrapper | `torch/_inductor/codegen/wrapper.py` | `PythonWrapperCodegen`, `SubgraphPythonWrapperCodegen` |
| 工具 | `torch/_inductor/utils.py` | `IndentedBuffer` |

---

## 附录 D：调试指南

### D.1 TORCH_LOGS 环境变量

PyTorch Inductor 提供了丰富的日志系统，通过 `TORCH_LOGS` 环境变量控制输出内容。以下列出理解 Inductor 编译过程最常用的日志类别：

| TORCH_LOGS 值 | 显示内容 | 用途说明 | 章节引用 |
|---------------|---------|---------|---------|
| `dynamo` | TorchDynamo 字节码追踪详情 | 查看 Dynamo 如何捕获 Python 字节码、何时发生 graph break | Ch1 |
| `aot` | AOTAutograd 前向/反向分离过程 | 查看联合图如何被分解、算子分解链 | Ch1, Ch5 |
| `inductor` | Inductor 总体日志 | 编译各阶段的概览信息 | 全部 |
| `inductor.codegen` | 代码生成详情 | 查看生成的内核代码、wrapper 代码 | Ch7, Ch8 |
| `inductor.kernel` | Kernel 创建和配置 | 查看 block 大小选择、内核实例化参数 | Ch7 |
| `inductor.scheduler` | 调度决策日志 | 查看融合判断、节点分组、拓扑排序 | Ch6 |
| `inductor.lowering` | 算子 lowering 过程 | 查看 FX 算子如何转换为 Inductor IR | Ch5 |
| `inductor.ir` | IR 节点创建和变换 | 查看 Buffer、Pointwise、Reduction 的创建 | Ch4 |
| `inductor.handlers` | Handler 安装和切换 | 查看 V.ops 的上下文切换过程 | Ch3 |
| `inductor.fusion` | 融合决策详情 | 查看哪些节点被融合、融合原因 | Ch6 |
| `inductor.autotune` | Triton config 自动调优 | 查看 block 大小的搜索空间和选择结果 | Ch7 |
| `inductor.memory` | 内存规划信息 | 查看缓冲区分配、复用和释放决策 | Ch7 |

**使用方式：**

```bash
# 单一日志类别
TORCH_LOGS="inductor.scheduler" python my_script.py

# 多个日志类别（逗号分隔）
TORCH_LOGS="inductor.scheduler,inductor.codegen" python my_script.py

# 使用 "+" 前缀启用详细模式（输出更多内部细节）
TORCH_LOGS="+inductor" python my_script.py
```

### D.2 关键调试命令

```bash
# ===== 编译流程日志 =====

# 查看 Inductor 完整编译输出（含各阶段摘要）
TORCH_LOGS="+inductor" python my_script.py

# 查看生成的 Triton 内核源码
TORCH_LOGS="+inductor.codegen" python my_script.py

# 查看调度决策（哪些节点被融合、融合方式）
TORCH_LOGS="+inductor.scheduler" python my_script.py

# 查看 Lowering 过程（FX 算子到 Inductor IR 的转换）
TORCH_LOGS="+inductor.lowering" python my_script.py

# ===== Dynamo 追踪日志 =====

# 查看 Dynamo 字节码追踪详情
TORCH_LOGS="dynamo" python my_script.py

# 查看 FX Graph 结构
TORCH_LOGS="+dynamo" python my_script.py

# ===== 调试输出文件 =====

# 将所有编译产物保存到磁盘
TORCH_COMPILE_DEBUG=1 python my_script.py
# 自动创建 torch_compile_debug/ 目录

# 同时启用日志和文件输出
TORCH_COMPILE_DEBUG=1 TORCH_LOGS="+inductor" python my_script.py

# ===== 性能分析 =====

# 查看 Triton autotune 过程（block 大小搜索）
TORCH_LOGS="inductor.autotune" python my_script.py

# 查看内存分配规划
TORCH_LOGS="inductor.memory" python my_script.py

# ===== 编译缓存控制 =====

# 禁用编译缓存（每次重新编译）
torch._inductor.config.triton.unique_kernel_names = True

# 清除编译缓存
rm -rf ~/.cache/torchinductor/

# ===== Python 代码调试 =====

# 在 Inductor 编译入口设置断点
TORCH_LOGS="+inductor" python -m pdb my_script.py
```

### D.3 输出目录结构

当设置 `TORCH_COMPILE_DEBUG=1` 时，Inductor 会将编译过程中的所有中间产物保存到 `torch_compile_debug/` 目录：

```
torch_compile_debug/
└── <run_id>/                           # 每次脚本执行的唯一标识
    └── <compile_id>/                   # 每次编译调用的唯一标识
        │
        ├── graph.py                    # 输入 FX Graph 的 Python 源码表示
        │     包含完整的 FX Graph 节点定义
        │     可独立运行以复现编译输入
        │
        ├── ir.py                       # Inductor IR 的文本表示
        │     列出所有 Buffer 及其类型、形状、操作
        │     格式：buffer_name = Buffer(type, shape, dtype, operation)
        │
        ├── ir_post_grad.py             # 经过 AOTAutograd 分解后的 IR
        │     仅包含 aten 级别算子
        │
        ├── fused_functions.py          # 融合后的 IR 闭包
        │     展示每个融合组的 inner_fn 内容
        │
        ├── fx_graph_runnable.py        # 可运行的 FX Graph
        │     包含完整的 GraphModule 定义
        │
        ├── kernels/                    # 生成的设备端内核代码
        │   ├── triton_0.py             # 第 1 个 Triton 内核源码
        │   ├── triton_1.py             # 第 2 个 Triton 内核源码
        │   ├── triton_2.py
        │   └── ...
        │     每个文件包含完整的 @triton.jit 函数定义
        │     可直接用于调试和性能分析
        │
        ├── output_code.py              # 最终生成的 Python wrapper 代码
        │     包含 forward() 函数的完整源码
        │     展示所有内核调用序列和缓冲区管理
        │
        ├── cudagraphs/                 # CUDA Graph 相关（如启用）
        │   └── ...
        │
        └── torchbind_overrides.log     # 自定义算子覆盖日志
```

**关键文件解读：**

| 文件 | 作用 | 调试价值 |
|------|------|---------|
| `graph.py` | 查看输入 FX Graph 是否正确捕获了所有操作 | 排查 Dynamo graph break 问题 |
| `ir.py` | 查看 Inductor IR 的缓冲区列表和操作类型 | 确认 Lowering 是否正确 |
| `fused_functions.py` | 查看融合后的操作分组 | 理解融合决策结果 |
| `kernels/triton_*.py` | 查看生成的 Triton 内核源码 | 分析内核性能、验证代码正确性 |
| `output_code.py` | 查看最终的可执行 Python 代码 | 理解整体执行流程 |

### D.4 常见调试场景

**场景 1：编译结果不符合预期（数值错误）**

```bash
# 步骤 1：查看 FX Graph 是否正确
TORCH_COMPILE_DEBUG=1 python my_script.py
# 检查 torch_compile_debug/*/graph.py

# 步骤 2：查看 IR 是否正确
# 检查 ir.py，确认 Buffer 操作链

# 步骤 3：查看生成的内核代码
# 检查 kernels/triton_*.py，确认内核逻辑

# 步骤 4：禁用 Inductor 优化，回退到 eager
TORCHDYNAMO_DISABLE=1 python my_script.py  # 完全禁用编译
```

**场景 2：融合不充分（性能低于预期）**

```bash
# 查看调度决策
TORCH_LOGS="+inductor.scheduler" python my_script.py
# 关注 "can_fuse" 相关日志，查看哪些节点未被融合及原因

# 查看 IR 结构
TORCH_COMPILE_DEBUG=1 python my_script.py
# 检查 ir.py 中是否有过多的独立 Buffer
```

**场景 3：理解特定操作的编译过程**

```bash
# 启用全量日志
TORCH_LOGS="+inductor,+inductor.codegen,+inductor.scheduler,+inductor.lowering" \
  python my_script.py 2>&1 | tee compile_log.txt

# 在日志中搜索目标操作
grep "aten.relu" compile_log.txt  # 查看 relu 的 lowering
grep "triton_" compile_log.txt    # 查看生成的 Triton 内核
```

---

## 附录 E：Handler 分类速查表

### E.1 Handler 分类总览

Handler 体系是 Inductor 实现 "define-by-run" 编译范式的核心机制。每个 Handler 都是对同一组 ops 操作的不同解读，对应抽象解释理论中不同的"抽象域"。下表按照用途分类列出所有 Handler。

| 分类 | Handler | 输入域 (T) | 输出域 | 使用阶段 | 源文件 |
|------|---------|-----------|--------|---------|--------|
| **核心基类** | `OpsHandler[T]` | T（泛型） | T | 全阶段 | `ops_handler.py` |
| **核心基类** | `DefaultHandler` | Any | via `_default()` | 全阶段 | `ops_handler.py` |
| **核心基类** | `WrapperHandler` | Any | via `_inner` | 全阶段 | `ops_handler.py` |
| **核心基类** | `MockHandler` | str | str（操作名表示） | 调试 | `ops_handler.py` |
| **核心基类** | `NoopHandler` | Any | None | 工具 | `ops_handler.py` |
| **分析类** | `DtypePropagationOpsHandler` | ops | `torch.dtype` | Lowering / Codegen | `dtype_propagation.py` |
| **分析类** | `ValueRangeAnalysis` | ops | `ValueRanges[Expr]` | Scheduling | `bounds.py` |
| **分析类** | `SymPyOps` | ops | `TypedExpr` | Scheduling | `index_propagation.py` |
| **分析类** | `IndexPropagation` | ops | `TypedExpr` | Scheduling | `index_propagation.py` |
| **分析类** | `FreeSymbolsOpsHandler` | ops | None（收集符号集合） | Scheduling | `dependencies.py` |
| **分析类** | `_RecordLoadStoreInner` | ops | None（收集 `MemoryDep`） | Scheduling | `dependencies.py` |
| **分析类** | `CountOps` | ops | `Counter` | Scheduling | `loop_body.py` |
| **分析类** | `CaptureIndexing` | ops | `TypedExpr` | Scheduling | `loop_body.py` |
| **分析类** | `KernelFormatterHandler` | ops | 格式化字符串 | Scheduling | `ops_handler.py` |
| **代码生成** | `TritonOverrides` | ops | str（Triton 表达式） | Codegen (GPU) | `codegen/triton.py` |
| **代码生成** | `TritonKernelOverrides` | ops | str（Triton 内核代码） | Codegen (GPU) | `codegen/triton.py` |
| **代码生成** | `CppOverrides` | ops | str（C++ 表达式） | Codegen (CPU) | `codegen/cpp.py` |
| **代码生成** | `CppVecOverrides` | ops | str（C++ SIMD intrinsics） | Codegen (CPU) | `codegen/cpp.py` |
| **代码生成** | `CppTile2DOverrides` | ops | str（C++ 2D 分块 SIMD） | Codegen (CPU) | `codegen/cpp.py` |
| **代码生成** | `MetalOverrides` | ops | str（Metal MSL 表达式） | Codegen (MPS) | `codegen/mps.py` |
| **代码生成** | `HalideOverrides` | ops | str（Halide 表达式） | Codegen (Halide) | `codegen/halide.py` |
| **工具类** | `CSEProxy` | ops | `CSEVariable` | Codegen | `codegen/common.py` |
| **工具类** | `SimplifyIndexing` | ops | 化简后的 ops | Scheduling | `sizevars.py` |
| **工具类** | `LocalizeBufferHandler` | ops | 重命名的 ops | C++ codegen | `codegen/cpp_utils.py` |
| **工具类** | `AddParenHandler` | ops | 括号包裹的 ops | 调试 | `ops_handler.py` |
| **工具类** | `OpCounterCSE` | ops | `CSEVariable` + 计数 | Scheduling | `ops_handler.py` |
| **工具类** | `SimpleCSEHandler` | ops | `CSEVariable` | 调试 | `ops_handler.py` |
| **工具类** | `ExtractConstantsHandler` | ops | None（提取常量） | 工具 | `ops_handler.py` |

### E.2 典型 Handler 链配置

在不同编译阶段和后端中，Handler 以链式组合的方式安装到 `V.ops`。以下是各场景下的典型配置。

#### E.2.1 Triton GPU 代码生成

这是最复杂也是最常用的 Handler 链，用于生成 Triton GPU 内核代码。

```
V.ops = CSEProxy(
          SimplifyIndexing(
            TritonKernelOverrides()))
```

**调用流程：**

```
inner_fn() 执行
  │
  ├── V.ops.load("buf0", [i0 + i1 * N])
  │     │
  │     ├── CSEProxy.load("buf0", [i0 + i1 * N])
  │     │     │   计算 CSE key = ("load", "buf0", "i0 + i1*N")
  │     │     │   查询 CSE 缓存
  │     │     │
  │     │     ├── [缓存未命中]
  │     │     │     SimplifyIndexing.load("buf0", [i0 + i1 * N])
  │     │     │       │   使用 SizeVarAllocator 化简索引
  │     │     │       │   例如：i0 + i1 * N → 存储为 sizevar "S0"
  │     │     │       │
  │     │     │       TritonKernelOverrides.load("buf0", simplified_index)
  │     │     │         │   生成 Triton 代码
  │     │     │         │   返回 "tl.load(buf0 + tl.arange(0, BLOCK) + offset, mask=...)"
  │     │     │
  │     │     └── [缓存命中]
  │     │           直接返回已缓存的 CSEVariable
  │     │
  │     └── 返回 CSEVariable("tmp0", torch.float32)
  │
  ├── V.ops.add(tmp0, tmp1)
  │     ├── CSEProxy.add(tmp0, tmp1)
  │     │     ├── SimplifyIndexing.add(tmp0, tmp1)
  │     │     │     └── TritonKernelOverrides.add(tmp0, tmp1)
  │     │     │           └── 返回 "tl.add(tmp0, tmp1)"
  │     │     └── 缓存并返回 CSEVariable("tmp2", torch.float32)
  │     └── 返回 CSEVariable("tmp2", torch.float32)
  │
  └── V.ops.store("buf_out", index, tmp2)
        └── 生成 "tl.store(buf_out + offset, tmp2, mask=...)"
```

#### E.2.2 CPU C++ 代码生成

用于生成 C++ 标量或 SIMD 代码的 Handler 链。

```
V.ops = CSEProxy(
          SimplifyIndexing(
            CppOverrides()))
```

**与 Triton 链的对比：**

| 属性 | Triton 链 | C++ 链 |
|------|----------|--------|
| 底层 Overrides | `TritonKernelOverrides` | `CppOverrides` |
| load 输出 | `tl.load(ptr + offset, mask=...)` | `buf[index]` |
| add 输出 | `tl.add(a, b)` | `a + b` |
| store 输出 | `tl.store(ptr + offset, val, mask=...)` | `buf[index] = val` |
| CSE Variable 类型 | `TritonCSEVariable` | `CSEVariable` |

#### E.2.3 CPU C++ 向量化代码生成

用于生成 C++ SIMD intrinsics 代码。

```
V.ops = CSEProxy(
          SimplifyIndexing(
            CppVecOverrides()))
```

**VecOverrides 的特殊行为：**

- `load` 生成 `_mm256_loadu_ps(&buf[index])`（AVX2）或类似 intrinsic
- `add` 生成 `_mm256_add_ps(a, b)`
- `store` 生成 `_mm256_storeu_ps(&buf[index], val)`
- 数据类型决定使用哪种 SIMD 指令集（SSE/AVX/AVX512/NEON）

#### E.2.4 依赖分析

调度阶段用于收集内存依赖关系的 Handler。

```
V.ops = _RecordLoadStoreInner()
```

**行为说明：**

```
_RecordLoadStoreInner 是一个 "副作用型" Handler：
- 不产生有意义的返回值（返回 CSEVariable 占位符）
- 每个 load() 调用记录一个 MemoryDep(buffer_name, index_expr)
- 每个 store() 调用记录一个 MemoryDep(buffer_name, index_expr)
- 最终从 self.reads / self.writes 集合提取依赖信息
```

#### E.2.5 值范围分析

调度阶段用于推断操作结果值域区间的 Handler。

```
V.ops = ValueRangeAnalysis()
```

**行为示例：**

```
ValueRangeAnalysis 将每个操作映射为区间运算：
- constant(5, float32)       → ValueRanges[5, 5]
- load("buf0", ...)          → ValueRanges[0, 1]  (假设 sigmoid 输出)
- add(ranges_a, ranges_b)    → ValueRanges[a_lo + b_lo, a_hi + b_hi]
- relu(ranges)               → ValueRanges[max(0, lo), max(0, hi)]
- mul(ranges_a, ranges_b)    → ValueRanges[min(乘积组合), max(乘积组合)]

用途：
- 判断 index 是否可能越界 → 决定是否插入 bounds check
- 判断值是否恒为正/负 → 决定是否可以省略 abs/neg
- 判断 reduction 值域 → 选择合适的 reduction 算法
```

#### E.2.6 数据类型传播

Lowering 阶段用于推断操作输出数据类型的 Handler。

```
V.ops = DtypePropagationOpsHandler()
```

**行为示例：**

```
DtypePropagationOpsHandler 将每个操作映射为类型推断规则：
- constant(5, torch.float32)     → torch.float32
- load("buf0", ..., torch.float16) → torch.float16
- add(dtype_a, dtype_b)          → promote(dtype_a, dtype_b)  (通常保持同类型)
- to_dtype(x, torch.float32)     → torch.float32
- relu(x, src_dtype=torch.float16) → torch.float16

用途：
- 确保生成的代码使用正确的数据类型
- 在代码生成阶段为变量分配正确的 dtype
```

#### E.2.7 索引传播与化简

调度阶段用于计算和化简索引表达式的 Handler。

```
V.ops = SymPyOps()       # 索引传播：生成 SymPy 表达式
V.ops = IndexPropagation()  # 带类型的索引传播
```

**与 SimplifyIndexing 的配合：**

```
SimplifyIndexing 是 WrapperHandler，包装在任意 Handler 之前：
SimplifyIndexing(inner_handler)

- 拦截所有 index_expr() 调用
- 使用 SizeVarAllocator 化简 SymPy 表达式
- 将化简后的表达式传递给 inner_handler
- 例如：2*i + 3*i → 5*i（合并同类项）
- 例如：i*N + j → 存储为 sizevar，减少重复计算
```

### E.3 Handler 生命周期总结

```
编译阶段          安装的 Handler                            用途
────────────────────────────────────────────────────────────────────────
Phase 1           (无 Handler)                              Lowering 使用
Lowering          直接调用 lowering 函数                      lowering 函数内部
                  创建 Pointwise/Reduction IR 节点            不经过 V.ops

Phase 2           _RecordLoadStoreInner()                   依赖分析
Scheduling        │→ 收集 MemoryDep                         (每个 SchedulerNode)
                  │
                  ValueRangeAnalysis()                       值范围分析
                  │→ 推断值域区间                            (每个 SchedulerNode)
                  │
                  IndexPropagation()                         索引传播
                  │→ 计算 SymPy 表达式                       (每个 SchedulerNode)
                  │
                  FreeSymbolsOpsHandler()                    符号收集
                  │→ 收集自由符号                            (部分节点)

Phase 3           CSEProxy(                                 代码生成
Codegen             SimplifyIndexing(                       (每个 Kernel)
                      TritonKernelOverrides()))     [GPU]
                  CSEProxy(
                    SimplifyIndexing(
                      CppOverrides()))              [CPU 标量]
                  CSEProxy(
                    SimplifyIndexing(
                      CppVecOverrides()))           [CPU SIMD]
                  CSEProxy(
                    SimplifyIndexing(
                      MetalOverrides()))            [MPS]

Phase 4           (无 Handler)                              Wrapper 使用
Wrapper           PythonWrapperCodegen                       内部代码拼接
                  不经过 V.ops
```
