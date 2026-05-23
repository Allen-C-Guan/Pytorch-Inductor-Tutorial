# 附录 E：术语表（中英对照）

本附录收录本书中出现的核心术语的中英文对照及简要释义。术语按英文字母顺序排列。中文译文采用本书全书的统一翻译规范。

---

## E.1 编译器术语

| 英文术语 | 中文翻译 | 释义 |
|---------|---------|------|
| **Abstract Syntax Tree (AST)** | 抽象语法树 | 源代码经过语法分析后生成的树状中间表示，每个节点代表源码中的一个语法结构。Triton 中 Python `ast` 模块生成的 AST 是编译的起点。 |
| **Ahead-Of-Time Compilation (AOT)** | 预编译 / 静态编译 | 在程序运行之前将源代码编译为机器码的编译模式。与 JIT 相对。 |
| **Alias Analysis** | 别名分析 | 分析程序中多个指针是否指向同一内存地址。Triton 中用于判断 buffer 是否可以共享内存空间。 |
| **Attribute** | 属性 | MLIR 中附加在 Operation 上的编译期常量元数据，例如缓存策略、舍入模式等。在 TableGen 中用 `I32EnumAttr` 等定义。 |
| **Autotuning** | 自动调优 | 通过搜索不同的编译配置参数（如 BLOCK_SIZE, num_warps）来找到最优性能的过程。Triton 的 `autotuner.py` 实现此功能。 |
| **Back-End** | 后端 | 编译器中将优化后的 IR 翻译为目标机器码的阶段。Triton 的后端包括 TTGIR -> LLVM IR -> PTX -> CUBIN。 |
| **Basic Block** | 基本块 | 一个线性的指令序列，只有一个入口和一个出口。是控制流图（CFG）的基本构建单元。 |
| **Block** | 基本块 | 见 Basic Block。在 MLIR 中，Block 是 Operation 的有序列表，以 Terminator Operation 结尾，可以拥有 Block Arguments。 |
| **Block Argument** | 基本块参数 | MLIR 中 Block 的参数，类似于函数参数，用于向 Block 中传入值（替代了 LLVM IR 中的 phi 节点）。 |
| **Calling Convention** | 调用约定 | 定义函数调用时参数如何传递、返回值如何返回、哪些寄存器需要保存和恢复的一组规则。 |
| **Canonicalization** | 规范化 | 将 IR 表达式化简为标准（规范）形式的过程。例如 `x + 0 -> x`。由 Pattern Rewrite 规则实现。 |
| **Code Generation (Codegen)** | 代码生成 | 编译器后端中将高层 IR 翻译为目标机器码（或汇编）的过程。 |
| **Common Subexpression Elimination (CSE)** | 公共子表达式消除 | 一种编译器优化，如果同一个表达式被计算多次，则只计算一次并复用结果。 |
| **Compilation Pipeline** | 编译管线 | 编译器各阶段按顺序组成的处理链条。Triton 的管线为：TTIR -> TTGIR -> LLVM IR -> PTX -> CUBIN。 |
| **Compiler** | 编译器 | 将一种编程语言（源语言）编写的程序翻译为另一种语言（目标语言）的等价程序的软件。 |
| **Constant Folding** | 常量折叠 | 在编译期计算常量表达式的值，避免在运行时重复计算。Triton 中的 `constexpr` 机制实现此功能。 |
| **Control Flow Graph (CFG)** | 控制流图 | 以基本块为节点、以控制流转移为边的有向图，表示程序的执行路径。 |
| **Dead Code Elimination (DCE)** | 死代码消除 | 移除对程序输出无影响的代码（计算结果从未被使用的操作）。 |
| **Dependence Analysis** | 依赖分析 | 分析程序中语句之间的数据依赖关系（读后写、写后读、写后写），是循环优化等变换的前置条件。 |
| **Dialect** | 方言 | MLIR 的核心可扩展机制，定义一组相关的 Operation、Type、Attribute 的命名空间。Triton 定义了 `tt`（TTIR）和 `ttg`（TTGIR）等方言。 |
| **DialectConversion** | 方言转换 | MLIR 提供的框架，支持将一种方言的 Operation 系统性地转换为另一种方言的 Operation。Triton 的 TTIR -> TTGIR lowering 即基于此实现。 |
| **Dominator Tree** | 支配树 | 描述基本块之间支配关系的树结构。基本块 A 支配 B 表示从入口到 B 的每条路径都经过 A。 |
| **Front-End** | 前端 | 编译器中将源代码翻译为中间表示（IR）的阶段。Triton 的前端将 Python Triton DSL 翻译为 TTIR。 |
| **Graph Coloring** | 图着色 | 一种寄存器分配算法（Chaitin-Briggs），将变量映射到寄存器的问题转化为对干涉图的 k-着色问题。 |
| **Inline Expansion** | 内联展开 | 将被调用函数的函数体复制到调用点，消除函数调用开销。Triton 中 Python 函数调用被 inline 为 TTIR 操作序列。 |
| **Instruction Selection** | 指令选择 | 将编译器 IR 中的操作映射为目标机器指令的过程。Triton 中表现为 TTGIR -> LLVM IR 的转换。 |
| **Instruction Scheduling** | 指令调度 | 重新排列指令的执行顺序以最大化硬件单元利用率并减少流水线停顿。Triton 中的软件流水线（Pipelining）是一种指令调度技术。 |
| **Interference Graph** | 干涉图 | 寄存器分配中使用的图，节点为变量，边表示两个变量同时活跃（不能分配同一寄存器）。 |
| **Intermediate Representation (IR)** | 中间表示 | 编译器中连接前端与后端的程序表示形式，位于源代码与目标代码之间的抽象层次。Triton 使用两级 IR：TTIR 和 TTGIR。 |
| **Just-In-Time Compilation (JIT)** | 即时编译 | 在程序运行时（而非运行前）进行编译的模式。Triton 默认使用 JIT 编译，每次 kernel launch 前编译或从缓存加载。 |
| **Liveness Analysis** | 活跃范围分析 | 数据流分析的一种，确定每个变量在程序中的哪些点"活跃"（其值可能被后续使用）。是寄存器分配的基础。 |
| **Loop Peeling** | 循环剥离 | 将循环的前几次或后几次迭代分离出来成为独立的基本块，以消除边界条件检查或为后续优化创造条件。 |
| **Loop Tiling (Strip-Mining)** | 循环分块 | 将大循环分解为两层嵌套：外层遍历"块"（tile），内层在块内迭代。Triton 的 tile 编程模型与此天然对应。 |
| **Loop Unrolling** | 循环展开 | 将循环体复制多份以减少循环控制开销并增加指令级并行度。 |
| **Lowering** | 降级 | 将高层 IR 操作翻译为低层 IR 操作序列的过程。例如 TTIR -> TTGIR lowering。 |
| **MLIR** | MLIR（多层次中间表示） | LLVM 社区开发的编译器基础设施框架。核心特征是可扩展的 Dialect 系统和渐进式 Lowering 支持。 |
| **Operation (Op)** | 操作 | MLIR 中最基本的计算单元。每个 Op 属于一个 Dialect，有唯一的名称（如 `tt.load`）、输入 Operand、输出 Result、Attribute 和 Region。 |
| **Optimizer** | 优化器 | 编译器中对 IR 进行变换以提高最终代码运行效率的阶段。 |
| **Pass** | 通道 | MLIR 中对 IR 进行一次转换或分析的执行单元。一个编译管线由多个 pass 按顺序执行组成。 |
| **Pattern Rewrite** | 模式重写 | MLIR 的声明式 IR 重写机制。开发者定义"匹配什么 IR 模式"和"如何重写"，MLIR 自动在 IR 上应用这些规则。 |
| **Peephole Optimization** | 窥孔优化 | 在一个小的指令窗口（"窥孔"）内进行局部替换的优化技术。 |
| **Register Allocation** | 寄存器分配 | 将活跃变量映射到处理器数量有限的物理寄存器的过程。Triton 将此任务委托给 LLVM 后端。 |
| **Region** | 区域 | MLIR 中附属于 Operation 的基本块列表容器。用于表示循环体（`scf.for`）、条件分支（`scf.if`）、归约组合函数（`tt.reduce`）等嵌套结构。 |
| **SSA (Static Single Assignment)** | 静态单赋值 | 一种 IR 属性：程序中的每个变量仅被赋值一次。MLIR 的 SSA 值（Value）遵守此属性。 |
| **Software Pipelining** | 软件流水线 | 一种指令调度技术，通过重叠不同循环迭代的执行来隐藏指令延迟。Triton 中用于隐藏 global -> shared 的数据搬运延迟。 |
| **Spilling** | 溢出 | 寄存器分配中当物理寄存器不足时，将变量的值临时写入内存（stack）的操作。 |
| **TableGen** | TableGen | LLVM/MLIR 的声明式元编程语言。用于定义 Operation、Type、Attribute 等，`.td` 文件由 `mlir-tblgen` 处理生成 C++ 代码。 |
| **Terminator Operation** | 终止操作 | Block 中最后一个必须有后继的 Operation，如 `tt.return`、`scf.yield`、`cf.br` 等。 |
| **Type Inference** | 类型推断 | 编译器自动推导变量或表达式的类型，无需程序员显式标注。 |
| **Type System** | 类型系统 | 编程语言或 IR 中用于分类值的规则集合。Triton 的类型系统包括浮点、整数、指针、张量、内存描述符等。 |
| **Use-Def Chain** | 用-定链 | 数据流分析中的数据结构，记录每个变量使用点对应的定义点。Layout 传播算法即基于 use-def 链。 |

---

## E.2 GPU / 并行计算术语

| 英文术语 | 中文翻译 | 释义 |
|---------|---------|------|
| **Bank Conflict** | Bank 冲突 | Shared Memory 中多个线程同时访问同一 bank 中的不同地址时产生的冲突，导致访问序列化，降低带宽。 |
| **Barrier** | 屏障 / 同步栅 | GPU 编程中的同步原语，确保 CTA 内所有线程到达同一点后才能继续执行。Triton 中为 `ttg.barrier`。 |
| **Coalesced Access** | 合并访问 | GPU 全局内存访问优化模式：一个 warp 内的线程访问连续的全局内存地址，从而将多次访问合并为一次或几次内存事务以最大化带宽利用率。 |
| **Compute Unit (CU)** | 计算单元 | AMD GPU 的核心计算单元，等价于 NVIDIA 的 SM。 |
| **Constant Memory** | 常量内存 | GPU 上只读的、带缓存的全局内存区域，适合所有线程读取相同值的场景。 |
| **Cooperative Thread Array (CTA)** | 协作线程阵列 | 见 Thread Block。CUDA 术语，一组共享 Shared Memory 并可同步的线程集合。 |
| **CUDA** | CUDA | NVIDIA 的并行计算平台和编程模型，允许开发者使用 C++ 编写 GPU 内核。 |
| **CUDA Core** | CUDA 核心 | NVIDIA GPU 中执行标量算术运算（FP32/INT32）的硬件单元。 |
| **Double Buffering** | 双缓冲 / 乒乓缓冲 | 使用两块 buffer 交替使用：一块用于计算、另一块用于数据加载，以隐藏数据传输延迟。 |
| **DRAM** | 动态随机存取存储器 | GPU 上用作全局内存的大容量、高延迟内存（通常是 HBM）。 |
| **Grid** | 网格 | 一个 GPU Kernel Launch 中所有 CTA 的集合。 |
| **HBM (High Bandwidth Memory)** | 高带宽内存 | 现代 GPU 使用的高带宽 DRAM，通过硅中介层与 GPU 芯片互连，提供 TB/s 级带宽。 |
| **Kernel** | 内核函数 | 在 GPU 上由大量线程并行执行的函数。Triton 中通过 `@triton.jit` 装饰器定义。 |
| **L1 Cache** | 一级缓存 | GPU SM 内部的底层数据缓存（通常与 Shared Memory 共享同一片上 SRAM，可配置划分比例）。 |
| **L2 Cache** | 二级缓存 | GPU 上所有 SM 共享的更大容量缓存（A100: 40MB, H100: 50MB）。 |
| **Latency Hiding** | 延迟隐藏 | GPU 通过在一个 warp 等待内存时切换到另一个就绪 warp 执行，从而隐藏内存访问延迟的机制。 |
| **Lockstep Execution** | 锁步执行 | SIMT 架构中一个 warp 内的所有线程在同一个周期执行同一条指令。 |
| **Memory Hierarchy** | 内存层次 | GPU 中从寄存器（最快、最小）到 HBM（最慢、最大）的分层存储结构。 |
| **Occupancy** | 占用率 | GPU SM 上实际调度的活跃 warp 数与理论上最大可调度 warp 数的比值。高占用率有助于更好的延迟隐藏。 |
| **PTX (Parallel Thread Execution)** | 并行线程执行 | NVIDIA GPU 的虚拟指令集架构（ISA），位于 CUDA C++ 和机器码（SASS）之间的抽象层。 |
| **Register File** | 寄存器文件 | GPU SM 上的高速存储，每个线程拥有私有的寄存器，访问延迟接近零。 |
| **SASS (Streaming Assembler)** | 流式汇编器 / 机器码 | NVIDIA GPU 的实际底层机器指令，由 PTX 汇编器（`ptxas`）生成。通常不公开文档化。 |
| **Shared Memory** | 共享内存 | GPU SM 中的片上 SRAM，可被同一 CTA 内的所有线程访问。通过 `__shared__`（CUDA）或 `ttg.local_alloc` + `ttg.local_load` (Triton) 使用。 |
| **SIMT (Single Instruction, Multiple Thread)** | 单指令多线程 | NVIDIA GPU 采用的执行模型：多个线程执行同一条指令，但各自拥有独立的寄存器和程序计数器。 |
| **SM (Streaming Multiprocessor)** | 流式多处理器 | NVIDIA GPU 的核心计算单元。每个 SM 包含 Warp Scheduler、CUDA Core、Tensor Core、Register File、Shared Memory/L1 Cache。 |
| **Tensor Core** | 张量核心 | NVIDIA GPU 中专用于矩阵乘加（D=A*B+C）的硬件单元，可在单周期内完成多次乘加运算。 |
| **Thread** | 线程 | GPU 执行的最小单元。在 SIMT 模型下，一组线程（warp）共同执行同一条指令。 |
| **Thread Block** | 线程块 | 一组可同步和通过 Shared Memory 通信的线程集合。见 CTA。 |
| **TMA (Tensor Memory Accelerator)** | 张量内存加速器 | NVIDIA Hopper 架构引入的硬件单元，支持异步张量数据搬运，可处理复杂的多维数据复制模式。 |
| **Warp** | 线程束 | NVIDIA GPU 中 32 个线程为一组的调度和执行基本单元。一个 warp 内的线程以锁步方式执行。 |
| **Warp Divergence** | Warp 分支发散 | 当 warp 内的线程因条件分支走向不同路径时，部分线程被屏蔽，导致并行效率降低。 |
| **Warp Scheduler** | Warp 调度器 | SM 中的硬件单元，负责在每个周期从就绪的 warp 中选择一条指令发射执行。 |
| **Warp Shuffle** | Warp 洗牌 | CUDA 提供的 warp 内线程间直接交换寄存器数据的高效机制（`__shfl_sync`）。Triton 的 `tt.reduce` 使用此机制。 |
| **Warp Specialization** | Warp 特化 | 将 warp 划分为不同角色（如生产者/消费者）以利用异步执行提升并行度的优化技术。 |
| **Wavefront** | 波前 | AMD GPU 的线程调度单元，等价于 NVIDIA 的 Warp。通常大小为 64 个 work-item。 |

---

## E.3 Triton 专有术语

| 英文术语 | 中文翻译 | 释义 |
|---------|---------|------|
| **Async Copy** | 异步拷贝 | Triton 中从全局内存到共享内存的异步数据搬运机制（对应 CUDA `cp.async` 指令）。由 `ttg.async_copy_global_to_local` 等 Op 实现。 |
| **Block (Triton Programming Model)** | 块（Triton 编程模型） | 一个 Triton kernel 的一次调用实例，对应一个 CTA。通过 `tl.program_id(axis)` 获取在 grid 中的坐标。 |
| **Blocked Layout** | 块状布局 | 最基本的 Distributed Encoding：每个 warp 持有张量的一个连续子块。有利于全局内存合并访问。 |
| **constexpr** | 编译期常量 | Triton DSL 中的编译期常量机制：参数标注 `tl.constexpr` 后，其值在编译期确定，支持特化（specialization）。 |
| **ConvertLayoutOp** | Layout 转换操作 | TTGIR 的核心操作：`ttg.convert_layout`。改变张量的 Encoding（数据分布），在代码生成时转换为 shared memory 中转和 warp shuffle 指令。 |
| **Distributed Encoding** | 分布式编码 | TTGIR 中描述数据在 GPU 线程上分布方式的 Layout 属性总称。包括 Blocked、MMA、DotOperand、Slice、Linear 等。 |
| **DotOperand Encoding** | 点积操作数编码 | TTGIR 中 `tt.dot` 操作数（A 矩阵或 B 矩阵）所需的特定 Layout，将矩阵元素分配到线程以匹配 Tensor Core MMA 指令的输入格式。 |
| **Encoding (Layout)** | 编码（布局） | TTGIR 中附加在 Tensor 或 MemDesc 类型上的属性，描述数据在 GPU 线程、warp、CTA 和内存空间上的分布方式。是 Triton 编译器最核心的设计抽象。 |
| **Layout** | 布局 | 见 Encoding。 |
| **Layout Propagation** | Layout 传播 | TTIR -> TTGIR lowering 中的核心算法。根据每个操作的输入/输出约束，通过前向和后向传播的迭代求解为所有张量分配 Layout。 |
| **libdevice** | libdevice 数学库 | Triton 的数学函数库（`triton/language/extra/libdevice.py`），对标 NVIDIA CUDA 的 libdevice 数学库。 |
| **Memory Descriptor (MemDesc)** | 内存描述符 | TTGIR 的 MemDescType：描述一块 shared memory buffer 的形状、元素类型、Layout和内存空间属性。通过 `ttg.local_alloc` 创建。 |
| **Memory Space** | 内存空间 | 描述数据所在的物理存储层次（Global、Shared、Register）。在 TTGIR 中作为 MemDescType 的属性。 |
| **MMA (Matrix Multiply-Accumulate)** | 矩阵乘累加 | Tensor Core 的指令类别。Triton 的 MMAv1 (Volta)、MMAv2 (Ampere)、MMAv3 (Hopper wgmma) 对应不同代次的 Tensor Core。 |
| **NVMMASharedEncoding** | NVIDIA MMA 共享编码 | MMAv3/MMAv5 的共享内存输入布局规范，定义 Shared Memory 中的数据排列方式以匹配 `wgmma` 指令的输入格式。 |
| **Pipelining** | （软件）流水线 | Triton 的编译优化技术：将循环中的数据拷贝（global -> shared）与计算重叠执行，通过双缓冲隐藏内存延迟。 |
| **Program (Triton)** | 程序（Triton） | Triton kernel grid 中的一个实例，等价于 CUDA 的 CTA / Thread Block。`tl.program_id` 返回其坐标。 |
| **Shared Encoding** | 共享编码 | TTGIR 中描述数据在 Shared Memory 中排列方式的 Layout 属性。包括 SwizzledShared、PaddedShared、NVMMAShared 等。 |
| **Swizzling** | Swizzle 模式 | Shared Memory 中通过 XOR 地址变换重新排列元素的方法，用于减少 bank conflict。由 `swizzled_shared` 或 `amd_rotating_shared` Layout 描述。 |
| **Tile** | 瓦片 | Triton 编程模型的基本数据单元。一个 tile 是一个多维数据块，被分配给一个 CTA 处理。Triton 以 tile（而非 thread）为第一公民。 |
| **Triton DSL** | Triton 领域特定语言 | Triton 的 Python 前端（`triton.language`），提供 `tl.load`、`tl.store`、`tl.arange` 等核心原语的 GPU 编程接口。 |
| **Triton** | Triton 编译器 | 一个专为 GPU 设计的领域特定语言和编译器，以 tile 为基本编程单元，采用基于 MLIR 的两级 IR（TTIR + TTGIR）架构。 |
| **TTIR (Triton IR)** | Triton 中间表示 | Triton 的第一级 IR：硬件无关的数据流方言。定义在 `triton/include/triton/Dialect/Triton/IR/`。核心 Op 包括 `tt.load`、`tt.store`、`tt.reduce`、`tt.dot` 等。 |
| **TTGIR (TritonGPU IR)** | Triton GPU 中间表示 | Triton 的第二级 IR：GPU 硬件感知的并行方言。定义在 `triton/include/triton/Dialect/TritonGPU/IR/`。核心特征是 Layout 编码系统。 |

---

## E.4 Ascend / NPU 专有术语

| 英文术语 | 中文翻译 | 释义 |
|---------|---------|------|
| **Ascend** | 昇腾 | 华为自研的 AI 处理器系列（NPU），采用达芬奇（Da Vinci）架构。 |
| **Da Vinci Architecture** | 达芬奇架构 | 华为昇腾 NPU 的底层硬件架构，专为神经网络计算设计。 |
| **AI Core** | AI 核心 | 昇腾 NPU 的核心计算单元，包含 Cube Unit（矩阵运算）、Vector Unit（向量运算）和 Scalar Unit（标量运算）。 |
| **Cube Unit** | Cube 单元 | 昇腾 NPU 中的矩阵乘法加速硬件，类似 NVIDIA Tensor Core。 |
| **Vector Unit** | 向量单元 | 昇腾 NPU 中的逐元素向量运算硬件。 |
| **AscendCL (Ascend Computing Language)** | 昇腾计算语言 | 华为昇腾 NPU 的编程 API，是 Triton Ascend 后端的最终目标语言。 |
| **triton-ascend** | Triton 昇腾后端 | Triton 编译器为昇腾 NPU 开发的后端插件，通过自有的 Dialect 和代码生成将 Triton kernel 编译为 Ascend 可执行代码。 |

---

## E.5 缩写速查

| 缩写 | 全称 | 中文 |
|------|------|------|
| **AOT** | Ahead-Of-Time | 预编译 |
| **AST** | Abstract Syntax Tree | 抽象语法树 |
| **CFG** | Control Flow Graph | 控制流图 |
| **CGA** | Cooperative Grid Array | 协作网格阵列 |
| **CTA** | Cooperative Thread Array | 协作线程阵列 |
| **CUBIN** | CUDA Binary | CUDA 二进制 |
| **DRAM** | Dynamic Random-Access Memory | 动态随机存取存储器 |
| **DSL** | Domain-Specific Language | 领域特定语言 |
| **EaC** | Engineering a Compiler | 《编译器工程》（参考教材） |
| **HBM** | High Bandwidth Memory | 高带宽内存 |
| **IR** | Intermediate Representation | 中间表示 |
| **JIT** | Just-In-Time | 即时编译 |
| **LLVM** | Low Level Virtual Machine | 底层虚拟机（编译器基础设施） |
| **MLIR** | Multi-Level Intermediate Representation | 多层次中间表示 |
| **MMA** | Matrix Multiply-Accumulate | 矩阵乘累加 |
| **NPU** | Neural Processing Unit | 神经网络处理单元 |
| **PTX** | Parallel Thread Execution | 并行线程执行（NVIDIA 虚拟 ISA） |
| **ROCm** | Radeon Open Compute Platform | AMD 开源计算平台 |
| **SASS** | Streaming Assembler | 流式汇编器（NVIDIA 机器码） |
| **SIMT** | Single Instruction, Multiple Threads | 单指令多线程 |
| **SM** | Streaming Multiprocessor | 流式多处理器 |
| **SSA** | Static Single Assignment | 静态单赋值 |
| **TMA** | Tensor Memory Accelerator | 张量内存加速器 |
| **TTIR** | Triton Intermediate Representation | Triton 中间表示 |
| **TTGIR** | TritonGPU Intermediate Representation | Triton GPU 中间表示 |
| **WGMMA** | Warp Group Matrix Multiply-Accumulate | Warp 组矩阵乘累加（Hopper 指令） |
| **WMMA** | Wave Matrix Multiply-Accumulate | Wave 矩阵乘累加（AMD RDNA 指令） |
