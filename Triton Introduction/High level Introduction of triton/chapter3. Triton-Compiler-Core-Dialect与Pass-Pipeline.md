# Triton Compiler Core：Dialect 与 Pass Pipeline

2026年2月2日 · 阅读需 130 分钟
![图片]()


在小说阅读器读本章


去阅读


![图片]()


在小说阅读器中沉浸阅读


## 引言


人工智能时代的浪潮汹涌澎湃，特别是大语言模型与生成式 AI 的迅猛崛起，将深度学习对计算能力的需求推向前所未有的巅峰。GPU 作为驱动这一革命的核心引擎，其编程范式与优化技术已成为制约模型规模扩展与推理效率的关键瓶颈。传统 CUDA 编程虽能精雕细琢般榨取硬件极致性能，却迫使开发者深陷线程层次、内存管理与同步机制的繁琐细节之中，代码冗长且维护艰难；与之相对，PyTorch、TensorFlow 等高层次框架虽赋予开发者优雅简洁的表达，却往往在底层内核生成中牺牲性能，无法彻底释放 GPU 的潜能。


Triton——这一由 OpenAI 开源的 GPU 编程语言与编译器应运而生，化解了易用性与高性能间的长期张力。它依托 MLIR 这一富有层次感的中间表示生态，构建了 Triton 方言与多阶段 Pass Pipeline，铸就了一种既亲和开发者又忠于硬件的 AI 内核开发新范式。开发者只需以近似 NumPy 的 Python 代码勾勒计算逻辑，便可收获接近手写 CUDA 的性能，代码量大幅精简，同时保有高度灵活的硬件适配与优化扩展空间。正因如此，Triton 迅速成为 FlashAttention、xFormers 内存高效注意力以及诸多前沿 AI 算子的实现平台，并在 PyTorch 生态、学术研究与工业部署中赢得广泛赞誉与深度采纳。


Triton 编译器的精髓在于其渐进式 Lowering 哲学：从高抽象的 Triton IR 起步，层层递进至 GPU 特定代码，并在每一阶段施加精准而有力的针对性优化。本文将深入剖析 Triton 方言的设计理念与核心机制，以及完整的 Pass Pipeline 架构。首先剖析方言如何抽象 AI 算子的计算逻辑与数据流动；继而详述前端解析、中间优化与后端代码生成的完整流程，以及其与 MLIR、LLVM 生态的融合，最后通过一个简单的自定义方言与优化 Pass 的案例，展现 Triton 的扩展活力。


## Triton 方言


Triton 方言的设计理念体现了 MLIR 框架下专用中间表示的精髓：它在 GPU 编程易用性与高性能的固有张力中寻求平衡，既通过高级抽象提升开发者生产力，又通过底层控制保障硬件性能潜力，同时特别针对计算密集型 AI 算子提供高效表达范式。


### Triton 方言的设计理念


Triton 方言的核心追求，是将 GPU 编程从繁琐的低层细节中解放出来，同时保留足够的控制阀门以榨取硬件性能。传统 CUDA 迫使开发者手动编排线程索引与同步机制，IR 表达冗长而复杂；高层次抽象虽简化描述，却往往丢失优化机会，导致性能缺口。Triton 方言则以块级并行抽象为支点：通过`tt.program_id`与`tt.make_range`等操作，隐式管理网格级并行与块内偏移生成，开发者只需描述单一计算块的逻辑。这种抽象使方言表达更接近声明式张量操作，不仅大幅简化语义描述，还显著提升了生产力，同时为后续优化阶段提供了清晰、可分析的并行结构。


与此同时，Triton 方言并未牺牲底层控制，而是通过显式机制为性能调优保留了必要空间。指针空间（pointer space）建模明确区分内存层次（全局、共享、寄存器），允许在 IR 层面精准控制数据驻留与流动；掩码与边界检查机制深度集成于`tt.load/tt.store`操作，确保安全访问的同时减少无效计算。这些机制在保持方言硬件无关性的前提下，为开发者提供了领域知识驱动的调优手段，同时为优化阶段的硬件特定变换（如布局注入与调度重排）奠定了基础。


### Triton 方言的核心内容解析


Triton 方言（TTIR）是 Triton 编译器的中间表示，提供了一套丰富的操作符来支持高性能 GPU 编程，Triton 方言的设计特点包括：


**1. 灵活的张量形状变换**


Triton 方言的形状操作核心在于将张量的逻辑视图与物理存储高效解耦。以`reshape`操作为代表，它允许在不改变底层数据的前提下重新解释张量的维度结构，这种“视图变换”避免了实际的数据重排开销。编译器可根据`allow_reorder`等属性提示，在保持语义的前提下自主选择是否调整内存布局以优化访问模式。


`transpose`操作则专门处理维度的重排列，其智能之处在于能识别何时仅需调整内存访问的“步幅”等元数据即可实现维度交换，而非进行昂贵的数据搬运。这种设计使得矩阵转置等常用操作能最大程度地利用现有数据布局，减少实际的内存带宽消耗。


**2. 高性能计算支持：包含矩阵乘法、归约和扫描等高性能操作**


`TT_DotOp`是 Triton 方言中封装标准矩阵乘加运算的粗粒度原语。它将矩阵乘法与累加两个关键步骤融合为**一个单一的高级操作**，这种大颗粒度的设计允许编译器将整个计算块作为原子单元进行调度与优化。该操作通过 inputPrecision 属性支持 TF32 等面向 Tensor Core 的精度控制，能根据硬件能力自动选择最优实现路径，从而在保持接口统一的前提下，为编译器高效映射至硬件的大块矩阵计算指令提供了核心抽象。


归约操作`ReduceOp`则将跨维度的聚合计算（如求和、求最大值）提升为编译器可显式分析与优化的独立原语。其关键设计在于允许开发者自定义归约的具体组合算法，同时由编译器基于此高层语义自动选择最优的并行执行策略（如树状归约），高效解决数据聚合带来的同步挑战。


扫描操作`ScanOp`专门处理像前缀和这类具有数据依赖性的计算模式。它将序列上的关联操作抽象为一个可定制的内核，使编译器能够理解其数学上的并行潜力，从而将原本串行的累积计算转化为高效的分层或分块并行实现，在保持逻辑正确性的同时最大化硬件利用率。


**3. 指针系统**


**通用指针类型**：定义了统一的`TT_PtrType`，能够指向标量或张量，并通过`addressSpace`参数显式区分内存区域，为跨层次内存访问提供了类型安全且支持完整指针运算的基础抽象。


**多样化的指针变体**：在通用指针基础上，系统性地引入了`TT_PtrTensor`（指针张量）与`TT_TensorPtr`（指向张量的指针）等高级变体，专门用以高效表达间接访问、批量内存操作以及复杂数据结构。`ptr<tensor<...>>`（指向张量的指针）将单个指针与一个完整的数据块形状绑定，使得内存操作能够以“块”而非离散标量为粒度进行寻址与传输，这为编译器识别和优化连续的、规整的块状访问（如合并加载）提供了清晰的类型化依据。相反，`tensor<...xptr<>>`（元素为指针的张量）则构成了一个指针数组，允许每个元素独立寻址，从而天然地用于表达稀疏数据结构和间接的、不规则的访问模式。


**4. 指针系统与内存操作协同工作的完整体系**


TTIR 构建了一套以类型为中心的内存访问抽象体系。在这一体系中，如`ptr<tensor<128x128xf16>>`这类具体化的指针类型，静态地定义了目标数据块的形状与布局，成为所有内存访问操作的**根基性约束与上下文**。高级内存操作的设计完全建立在此类型系统之上：它们接收这类携带完整形状信息的指针作为操作数，其本身的语义（如访问粒度、边界行为）也由指针所指向的类型来定义。


协同工作的典型模式是，针对不同类型的指针使用相应的算术操作：对于标量指针和指针张量（`TT_PtrLike`），使用 TT_AddPtrOp 操作，该操作接收指针（TT_PtrLike 类型）和整数偏移量（TT_IntLike 类型）作为输入，输出与原指针类型完全一致的新指针，并支持元素级操作和形状/编码一致性检查；而对于指向张量的指针（`ptr<tensor<...>>`），则使用专门设计的 TT_AdvanceOp 操作，该操作接收 TT_TensorPtr 类型指针和变长的 32 位整数偏移量数组作为输入，通过`advance %ptr, [offsets]`表示多维偏移，同样输出与原指针类型一致的新指针。这两种操作都被标记为纯函数（Pure）并支持编译器折叠优化，确保从基础指针派生出的新指针能完整保留目标块的形状语义，可被直接传递给后续的内存操作。


`TT_LoadOp`等内存操作可以实现形状感知与安全可控的高层**块状访问**。`TT_TensorPtr`所携带的形状信息（如 ptr<tensor<128x128xf16>>中的 128x128 维度与 f16 数据类型），使得 Load 操作在语义上便明确了待传输数据的整体布局，编译器可据此预先规划寄存器分配与内存访问模式。


同时，`TT_LoadOp`通过内嵌的多种属性实现了安全与性能的精细控制：boundaryCheck 属性（DenseI32ArrayAttr 类型）支持按维度指定边界检查策略，padding 属性（TT_PaddingOptionAttr 类型）提供了边界访问时的填充选项，cache 属性（TT_CacheModifierAttr 类型，默认值为 NONE）允许显式控制缓存行为等等，这些属性通过声明式方式将安全访问与性能调优策略融入操作定义中，配合 TT_LoadOp 支持的掩码（TT_BoolLike 类型），进一步增强了访问灵活性。编译器利用这些丰富的语义信息，能在保障访问正确性的前提下，对块状内存访问进行深度优化，如自动合并相邻访问、调整访问顺序以提高缓存命中率，或根据边界检查结果生成更高效的代码路径，从而实现了高层抽象与底层性能的有机统一。


## 多层级 Pass Pipeline 与集成流程


### 总体架构：三层式 Pass Pipeline 设计


Triton 的 Pass Pipeline 常见的设计可以划分为三个阶段，各阶段目标与分工如下。


**前端阶段**主要负责从 Python 源代码到初始 Triton IR（TTIR）的转换。其目标是忠实捕捉用户内核的计算语义与数据流抽象，同时进行初步规范化。前端 Pass 包括 Python AST 解析、类型推断、参数绑定以及初始 Dialect 转换（如 triton.language 到 tt.func 与 tt.ops）。这一阶段输出高层次 TTIR，保留块级并行、指针空间和张量布局等抽象，避免过早引入硬件细节。分工重点在于正确性与易调试性，确保生成的 IR 便于后续优化。


**优化阶段**是 Triton Pipeline 的核心，目标是通过一系列针对性 Pass 最大化内核性能。该阶段主要在 Triton IR 层面展开，从 TTIR 逐步 Lowering。


**后端阶段**聚焦于硬件特定代码生成，目标是将优化后的 IR 降低至可执行机器码。该阶段从优化阶段后的 IR 起始，转换至 LLVM Dialect 的 ptx 汇编格式等。其中转换 Pass 负责主体语义的映射，而寄存器分配与指令调度等 Pass 则在此之后进行细粒度的资源管理与性能优化


通过三个阶段的分工，将语义保持、效能优化和硬件适配这些复杂关注点进行分离，体现了渐进式 Lowering 的设计哲学。


### 前端 Pass Pipeline：从高级抽象到 Triton IR


Triton 编译器的前端 Pass Pipeline 承担着关键的桥梁角色，其核心使命是将开发者以 Python 编写的高级抽象内核，系统性地转换为规范化、富含语义的 Triton IR（TTIR）。这一阶段的设计强调**语义保真与初步规范化**：在保留块级并行、数据流与计算意图的同时，完成类型推导、常量传播与高级抽象的显式化，刻意避免在此阶段引入硬件相关的底层细节。前端 Pipeline 虽然相对精简，却为整个编译流程奠定了清晰的基础——它所生成的高层次 TTIR，作为后续优化 Pass 施展针对性变换（如内存提升、循环流水线化）的理想载体，同时也支持快速 JIT 编译，确保了开发调试阶段的高效迭代。


前端 Pipeline 的输入是经`@triton.jit`装饰的 Python 函数对象，输出则为嵌入 MLIR Module 中的`tt.func`函数，其函数体由`tt.`命名空间下的高级操作（如`tt.dot`，`tt.load`）构成。该过程在内核首次被调用时触发，由 Triton 的内部编译器驱动。其工作流程始于对 Python 函数对象的**静态解析与中间表示构建**：


- **构建 AST 树**：编译器分析函数字节码，识别对`triton.language`（`tl.`）模块的调用（如`tl.load`，`tl.dot`）。这些调用在编译时实质上是**IR 构建器**，它们共同定义了一个**AST 树** 。
- **执行上下文关联** ：编译器将用户调用内核时提供的`grid`、`num_warps`等**执行配置**，与上一步构建的计算图进行绑定。同时，函数的形式参数（如指向数据的指针、步长`stride`）被赋予具体的类型和属性，为后续优化提供上下文。
- **高级 TTIR 生成**：基于以上信息，编译器生成初始的 MLIR Module，其中包含`tt.func`。此时 IR 中的操作仍保持高级抽象，但所有动态的 Python 语言特性已被固化为静态的、可分析的数据流和控制流图。


```
import triton
import triton.language as tl
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```


对应生成的初始 TTIR 片段:


```
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/stx/workspace/test1.py":6:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/stx/workspace/test1.py":6:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/stx/workspace/test1.py":6:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/home/stx/workspace/test1.py":6:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32> loc(#loc5)
    %4 = arith.addi %3, %2 : tensor<1024xi32> loc(#loc5)
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32> loc(#loc6)
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32> loc(#loc6)
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc7)
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc7)
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc8)
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc9)
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc9)
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc10)
    %13 = arith.addf %9, %12 : tensor<1024xf32> loc(#loc11)
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc12)
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc12)
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
    tt.return loc(#loc14)
  }
```


### 优化 Pass Pipeline：保持语义的深度优化


Triton 编译器的优化 Pass Pipeline 是整个流程的**核心性能引擎**，其任务是在**Triton IR 层面**实施一系列渐进式、具有针对性的高级变换。这一阶段的设计哲学是**保持语义的抽象层上进行深度优化**：从前端输出的、相对纯净的 TTIR 出发，逐步且系统地注入适用于通用计算模型的优化（例如**数据局部性提升、计算与内存访问的重叠、指令级并行挖掘**）。


Triton 编译器优化 Pass Pipeline 的设计聚焦于**数据局部性、并行执行单元协作、分层存储系统**等并行计算架构的通用抽象，它确保生成的内核代码在 GPU 乃至其他支持并行计算范式的加速器上均具备良好的性能可移植性。此 Pipeline 的最终产出是一个经过充分优化、蕴含丰富并行与访存信息的中间表示（IR），为后续各类硬件特定的后端（例如针对 GPU 的指令映射与 Warp 调度，或针对 CPU 的向量化与缓存优化）提供了高性能且语义明确的共同起点。


整个优化 Pipeline 由一个智能的 PassManager 动态调度，支持复杂的 Pass 间依赖与多次迭代。尤为关键的是，这一阶段与**自动调优框架深度集成**：调优器会驱动 Pipeline 以不同的编译时常量（如分块大小、流水线阶段数）反复执行。


下面举一个优化的例子，这段代码列出了当前阶段执行的 pass：


```
def _ttir_to_coreir(mod):
 # Get Triton-MLIR as string
    ttir_code = str(mod)
 with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, "tt.mlir")
        dst_path = os.path.join(tmpdir, "core.mlir")
        Path(src_path).write_text(ttir_code)
        triton_opt_path = _get_triton_opt_path()
        _dump_ir_if_needed([src_path])

        args = [triton_opt_path, src_path,
 "--triton-to-core-dialects",
 "--linalg-tiling",
 "--legalize-tensor-form-loops",
 "--one-shot-bufferize",
 "--convert-bufferization-to-memref",
 "--cse",
 "--canonicalize"]
 if os.getenv("TRITON_DEBUG", "0") == "1":
            args.append("--mlir-print-debuginfo")
        args += ["-o",
            dst_path]
        subprocess.check_call(args)
        _dump_ir_if_needed([dst_path])
 return Path(dst_path).read_text()
```


`--triton-to-core-dialects`是自定义转换 Pass，负责将 Triton 方言操作（如`tt.dot`、`tt.reduce`）重写为核心 MLIR 方言操作，主要映射到**`linalg`线性代数）、`arith`（算术）、`scf`（结构化控制流）**等方言。这一步是关键桥接：它在保留 TTIR 计算语义的同时，将程序引入通用的 MLIR 优化生态系统，从而能够复用后端的标准优化工具链。


`--linalg-tiling`对 Linalg 操作应用分块变换，将大张量计算分解为小块上的循环嵌套（分块尺寸可配置，通常由自动调优器驱动）。这直接提升了数据局部性，是核心性能来源。比如，在 GEMM 中，它可将单个`linalg.matmul`拆分为外层分块循环与内层小块乘法，直接对应 Triton 源码中的阻塞抽象。


`--legalize-tensor-form-loops`负责合法化分块后生成的、仍包含张量操作的循环结构，确保其符合 MLIR 张量方言的语义规范，为后续的缓冲化转换铺平道路。


`--one-shot-bufferize`应用 MLIR 的“一次性缓冲化”策略，将张量操作整体转换为基于内存缓冲区（`memref`）的操作。其全局分析能力能最大程度实现原位更新并消除临时拷贝，对于降低 AI 算子（如 FlashAttention）的内存峰值使用至关重要。


`--convert-bufferization-to-memref`将缓冲化结果进一步标准化为纯粹的`memref`方言操作，确保内存视图明确，为最终 Lowering 到 LLVM IR 或 GPU 后端做好准备。


`--cse`（公共子表达式消除）与`--canonicalize`（规范化）作为收尾清理 Pass，负责消除冗余计算、折叠常量并简化 IR 形式，确保传递给后端的代码简洁高效。


这个例子展示了 Triton 优化流水线的一个核心设计原则：它并非固定不变，而是一个由可重用、可配置的优化模块灵活组合而成的框架，这种设计为应对不同的硬件架构与计算模式提供了内在的适应性。


### 后端 Pass Pipeline：硬件适配与代码生成


Triton 编译器的后端 Pass Pipeline 是编译流程的最终阶段，负责将经过优化的中间表示（IR）转换为目标硬件的可执行代码。这一阶段的设计聚焦于硬件映射与最终优化：从已包含高级优化属性（如分块、流水线、特定内存布局）但保持相对硬件中立的 IR 出发，通过一系列 Lowering Pass，逐步将其适配到特定硬件架构，最终生成能够充分利用目标平台专用计算单元（如 NVIDIA Tensor Core、AMD Matrix Core）的高效代码。


这里举一个针对 CPU 的例子来说明这一阶段具体可以做哪些事情：


```
def _coreir_to_llir(mod, metadata):
    coreir_code = str(mod)
 with tempfile.TemporaryDirectory() as tmpdir:
        coreir_path = os.path.join(tmpdir, "core.mlir")
        llvmir_path = os.path.join(tmpdir, "ll.mlir")
        llir_path = os.path.join(tmpdir, "ll.ir")
        Path(coreir_path).write_text(coreir_code)
        mlir_opt_path = _get_llvm_bin_path("mlir-opt")
        args = [mlir_opt_path, coreir_path,
 "--convert-linalg-to-affine-loops",
 "--lower-affine",
 "--convert-linalg-to-loops",
 "--expand-strided-metadata",
 "--convert-scf-to-cf",
 "--convert-arith-to-llvm",
 "--convert-math-to-llvm",
 "--convert-complex-to-llvm",
 "--convert-vector-to-llvm",
 "--convert-index-to-llvm",
 "--memref-expand",
 "--finalize-memref-to-llvm",
 "--convert-func-to-llvm",
 "--convert-cf-to-llvm",
 "--lower-affine",
 "--convert-arith-to-llvm",
 "--canonicalize",
 "--reconcile-unrealized-casts"]
 ......
```


这些 pass 中包含了多种优化方法：


**高层方言到低级循环的转换**


```
"--convert-linalg-to-affine-loops",  # Linalg→Affine循环
"--lower-affine",                    # Affine→标准循环
"--convert-linalg-to-loops",         # 剩余Linalg→循环
```


这一阶段将声明式的线性代数操作转换为结构化控制流，是计算语义从**代数描述**到**执行流程**的关键转变。这种转换保留了前序优化的分块策略，为后续硬件特定优化奠定基础。


**内存抽象的低级化转换**


```
"--memref-expand",
"--finalize-memref-to-llvm",
```


这一转换过程的核心目标是在保持原有计算语义的前提下，将平内存操作描述逐步具体化为面向特定硬件架构的低级表示。它系统地将抽象的内存空间、布局和访问模式翻译为 LLVM 能够理解和优化的显式内存操作序列，包括地址计算、指针操作和内存屏障等底层原语。


**控制流和计算原语的统一化**


```
"--convert-scf-to-cf",                # 结构化控制流→基础控制流
"--convert-arith-to-llvm",            # 算术运算→LLVM运算
"--convert-vector-to-llvm",           # 向量操作→LLVM向量指令
```


将结构化控制流 Lowering 为基础控制流，并同步将算术与向量操作分别转换为 LLVM 方言的对应形式。此过程将所有控制流与计算原语统一至 LLVM 框架，为后续跨硬件平台的指令生成提供了语义一致的中间表示基础。


**迭代清理与最终合法化**


```
# 二次清理确保转换完整性
"--lower-affine",
"--convert-arith-to-llvm",
"--canonicalize",
"--reconcile-unrealized-casts"
```


由于转换过程可能产生新的中间表示，需要多次清理确保 IR 的合法性。这体现了编译器降低过程的复杂性——转换不是线性单向的，而是需要迭代协调的循环过程。


在获得 LLVM IR 后，Triton 编译器后端就可以通过调用 Clang 等编译器将其进一步转换为目标硬件平台上的二进制代码。


## 自定义方言与优化 Pass


自定义方言与优化 Pass 的引入，是 Triton 框架实现可扩展性与硬件泛化能力的核心机制。其设计在于遵循一套连贯的工程范式：首先通过自定义方言为新的硬件特性或计算模式建立抽象模型，将其集成到 MLIR 的多层 IR 生态中；随后，围绕该方言设计针对性的优化 Pass，在编译流水线的适当时机，将高级抽象逐步“翻译”并“优化”为具体的硬件指令与资源调度策略。接下来将举一个简单的例子。


### 自定义 Dialect 的定义


这里我们定义一个简单的 Calculator 方言，实现标量加减乘除的功能，创建这个方言的第一步，是编写其 TableGen（.td）定义文件。


```
calculator/Dialect/IR/
├── CMakeLists.txt
├── CalculatorDialect.h
├── CalculatorDialect.td
├── CalculatorOps.h
└── CalculatorOps.td
```


首先为 Calculator 方言在 CalculatorDialect.td 中建立身份标识和基础框架，它定义了该方言在 MLIR 系统中的唯一名称（`calc`）、C++命名空间及描述性文字，是后续所有具体操作和类型定义的容器与入口。


```
#ifndef CALCULATOR_DIALECT
#define CALCULATOR_DIALECT

include "mlir/IR/DialectBase.td"

// Defines the calculator dialect
def CalculatorDialect : Dialect {
  let name = "calc";
  let cppNamespace = "::mlir::calculator";
  let summary = "Calculator dialect for basic arithmetic operations";
  let description = [{
    The Calculator dialect provides basic arithmetic operations
    as a simple example of how to define a custom MLIR dialect.
  }];
}

#endif
```


定义方言功能的核心在于实现其具体操作， 接下来，在 CalculatorOps.td 文件中，我们将为 Calculator 方言定义其基础算术运算操作，将抽象的方言概念转化为 MLIR 编译器能够识别和处理的具体指令节点。


```
def AddOp : CalculatorOp<"add", []> {
  let summary = "integer addition operation";
  let description = [{
    Performs integer addition on two operands.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def SubOp : CalculatorOp<"sub", []> {
  let summary = "integer subtraction operation";
  let description = [{
    Performs integer subtraction on two operands.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def MulOp : CalculatorOp<"mul", []> {
  let summary = "integer multiplication operation";
  let description = [{
    Performs integer multiplication on two operands.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}

def DivOp : CalculatorOp<"div", []> {
  let summary = "integer division operation";
  let description = [{
    Performs integer division on two operands.
  }];
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
}
```


其中，`summary`和`description`共同构成了操作的语言文档，其中`summary`提供简短的功能摘要，通常用于自动生成的文档索引；`description`则提供详细语义说明，包括操作行为和边界条件。`arguments`定义了操作的输入契约，通过`(ins ...)`列出所有输入操作数及其类型和名称，构成操作的类型签名一部分；而`results`通过`(outs ...)`类似地定义了输出的类型和名称。这些字段共同完整地描述了一个操作的接口，为后续的编译器验证、转换和代码生成提供了完整的结构化信息。


完成.td 定义文件的编写后，需通过 MLIR 的构建系统生成对应的 C++ 接口代码。这一过程由 Tablegen 工具完成，它根据声明式规范自动生成的 C++ 类定义、方法声明及方言注册代码。通常在项目的 CMakeLists.txt 中配置，将 .td 文件转换为编译所需的 .inc 头文件与实现文件：


```
set(LLVM_TARGET_DEFINITIONS CalculatorOps.td)
mlir_tablegen(CalculatorDialect.h.inc -gen-dialect-decls -dialect=calc)
mlir_tablegen(CalculatorDialect.cpp.inc -gen-dialect-defs -dialect=calc)
mlir_tablegen(CalculatorOps.h.inc -gen-op-decls)
mlir_tablegen(CalculatorOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(CalculatorTableGen)
```


通过 TableGen 这一声明式代码生成工具，我们得以将简洁的`.td`文件中的高层语义定义自动转换为完备而复杂的 C++ 基础设施代码。这一自动化过程不仅生成了每个操作对应的类声明与实现骨架，还内嵌了类型验证、属性存储等标准化逻辑，从而将开发者从大量重复且易出错的样板代码编写中解放出来，使其能够专注于此方言独有的语义与优化逻辑实现。


### 自定义优化 Pass 的实现


在完成 Calculator 方言的抽象定义后，下一步便是构建其在 MLIR 编译流水线中的 Conversion Pass。这一步骤的核心目标是将我们自定义的、领域特定的 Calculator 操作，系统性地 Lowering 至 MLIR 内置的、更底层且已被广泛支持的 arith 标准方言。先定义以下文件：


```
├── include/                         //接口层
│   └── calculator/
│       └── Conversion/
│           └── CalculatorToArith/
│               ├── Passes.td
│               ├── Passes.h
│               ├── CalculatorToArith.h
│               └── CMakeLists.txt
└── lib/                             //实现层
    └── Conversion/
        └── CalculatorToArith/
            ├── CalculatorToArithPass.cpp
            ├── CalculatorToArith.cpp
            └── CMakeLists.txt
```


include 下的文件定义了转换的公共接口层，而 lib 下的文件则是转换的具体实现层，接下来，我们将首先深入 lib/下的具体实现层，从 CalculatorToArith.cpp 入手，剖析转换逻辑如何将高层的领域特定操作 Lowering 至 MLIR 的标准中间表示。


```
#define GEN_PASS_CLASSES
#include "calculator/Conversion/CalculatorToArith/Passes.h.inc"

usingnamespace mlir;
usingnamespace mlir::calculator;

namespace {

struct AddOpConversion :public OpConversionPattern<AddOp> {
using OpConversionPattern::OpConversionPattern;

LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};
...
//其他Op均要实现matchAndRewrite
```


转换过程的核心机制在于模式匹配与重写：每个转换模式（如 AddOpConversion）均继承自 MLIR 的 OpConversionPattern 模板，并通过重写其 matchAndRewrite 方法，捕获源方言中的特定操作节点，然后利用 ConversionPatternRewriter 将其原位替换为目标方言中语义等价的操作节点。


在定义并实现各操作的匹配重写逻辑后，必须将这些独立的转换模式系统性地封装并注册为一个标准的 MLIR Pass，才能将其集成到编译流程中。我们构建一个 CalculatorToArithPass.cpp ，创建继承自自动生成的 Pass 基类的具体 Pass 类来完成这一过程，在其核心的 runOnOperation 方法中，设定转换的合法目标（允许出现的 arith 方言）与非法目标（需要被完全转换掉的 calculator 方言），以此划定重写的边界。通过调用 populateCalculatorToArithConversionPatterns 将之前定义的所有转换模式收集到统一的模式集中，最终由 applyPartialConversion 驱动执行整个模块的渐进式、部分转换。


```
namespace mlir {
namespace calculator {
#define GEN_PASS_DEF_CONVERTCALCULATORTOARITH
#include "calculator/Conversion/CalculatorToArith/Passes.h.inc"
} // namespace calculator
} // namespace mlir

namespace {
usingnamespace mlir;
usingnamespace mlir::calculator;

class ConvertCalculatorToArithPass :public calculator::impl::ConvertCalculatorToArithBase<ConvertCalculatorToArithPass> {
public:
using calculator::impl::ConvertCalculatorToArithBase<ConvertCalculatorToArithPass>::ConvertCalculatorToArithBase;
void runOnOperation() override {
    MLIRContext *context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    // 定义转换目标
    target.addLegalDialect<mlir::arith::ArithDialect,
                           mlir::func::FuncDialect>();
    target.addIllegalDialect<mlir::calculator::CalculatorDialect>();
    // 添加转换模式
    populateCalculatorToArithConversionPatterns(patterns);
    // 执行转换
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
```


完成可执行的转换 Pass 实现后，需要为其建立规范的接口抽象层，以使其能够被 MLIR 的 Pass 管理器统一识别、调度和调用。这一接口层同样可以通过声明式的 TableGen 语法高效定义：在 Passes.td 文件中，我们使用 Pass 类模板来声明一个通行证，指定其在命令行中的调用标识符、所操作的顶级 IR 对象类型、可读的描述信息，以及关键的工厂函数指针。


```
ifndef CALCULATOR_TO_ARITH_PASSES
#define CALCULATOR_TO_ARITH_PASSES

include "mlir/Pass/PassBase.td"

def ConvertCalculatorToArith : Pass<"convert-calculator-to-arith", "mlir::ModuleOp"> {
let summary = "Convert Calculator dialect to Arith dialect";
let description = "This pass converts operations from the Calculator dialect to the Arith dialect.";
let constructor = "mlir::calculator::createConvertCalculatorToArithPass()";
let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif
```


最后，我们需要在 CalculatorToArith.h 头文件中建立公共接口声明，通过预定义宏 GEN_PASS_DECL 触发对 TableGen 生成的 Pass 声明代码（Passes.h.inc）的包含，从而自动获得 ConvertCalculatorToArithPass 类的正式声明。在此基础上，我们显式声明两个关键函数：populateCalculatorToArithConversionPatterns 函数用于向外部暴露转换模式的注册接口，允许其他转换流程复用这些模式；createConvertCalculatorToArithPass 函数则提供标准的 Pass 构建入口，返回一个包装好的 std::unique_ptrmlir::Pass 对象。


```
namespace mlir {
namespace calculator {

#define GEN_PASS_DECL
#include "calculator/Conversion/CalculatorToArith/Passes.h.inc"

void populateCalculatorToArithConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<mlir::Pass> createConvertCalculatorToArithPass();

} // namespace calculator
} // namespace mlir
```


至此，一个完整的、可被 MLIR Pass 管理器调度和执行的方言转换 Pass 便构建完成，接下来举一个例子展现转换的效果，我们首先写一个 Calculator 方言的例子：


```
func.func @main(%arg0: i32, %arg1: i32) -> i32 {
  %0 = "calc.add"(%arg0, %arg1) : (i32, i32) -> i32
  %1 = "calc.sub"(%0, %arg0) : (i32, i32) -> i32
  %2 = "calc.mul"(%1, %arg1) : (i32, i32) -> i32
  %3 = "calc.div"(%2, %arg1) : (i32, i32) -> i32
  return %3 : i32
}
```


通过调用标准工具链（如 triton-opt）并传入我们已定义好的 --convert-calculator-to-arith 转换 Pass，编译器会自动将模块中所有的 Calculator 方言操作转换为等价的 Arith 方言表示。


```
module {
  func.func @main(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.subi %0, %arg0 : i32
    %2 = arith.muli %1, %arg1 : i32
    %3 = arith.divsi %2, %arg1 : i32
    return %3 : i32
  }
}
```


获得 Arith 这一广泛支持的标准化中间表示后，便接入了 MLIR 庞大且成熟的优化生态系统，可应用一系列通用的优化 Pass（如常量折叠、代数简化、死代码消除等），从而为后续向更低层硬件指令的进一步 Lowering 和性能优化打下基础。


## 总结


本文系统性地剖析了 Triton 编译器核心的设计哲学与工程实现。首先，阐述了 Triton 方言如何通过其类型系统、操作集与属性，在高级编程友好性与底层硬件控制力之间取得了平衡。介绍了其多层次 Pass Pipeline 的完整流程：从前端解析与算子融合，到平台无关的中间优化，再到针对特定 GPU 架构的后端代码生成与指令调度，并完整介绍了 Triton IR 与 MLIR 标准方言及 LLVM IR 生态的集成 Lowering 路径。最后，通过一个从定义、实现到集成自定义方言与优化 Pass 的完整案例，展现了 Triton 编译器框架的可扩展性与硬件适配活力，为构建高性能、可移植的 AI 算子供给了坚实且灵活的编译基础设施。


**--------END--------**


![](http://mmbiz.qpic.cn/sz_mmbiz_png/zxGibZiawY03lUk44Lq2lGmalwJ267LXTa5L4bFK8Jl4ba4W5frn7oLjvISDcb90v8NtTlZmK1bP31BGS9JgOuiaQ/300?wx_fmt=png&wxfrom=19)


公众号


![图片](https://www.terapines.com/articles-images/2026-02-02-Triton-Compiler-Core：Dialect-与-Pass-Pipeline/4.png)


**点击阅读原文 加入1nfinite**


![图片](https://www.terapines.com/articles-images/2026-02-02-Triton-Compiler-Core：Dialect-与-Pass-Pipeline/5.png)


预览时标签不可点


阅读原文


该账号因违规无法跳转

**标签：**
- [Triton on RISC-V](/articles/tags/triton-on-risc-v)



---

> **原文链接**: https://www.terapines.com/articles/2026/02/02/Triton-Compiler-Core%EF%BC%9ADialect-%E4%B8%8E-Pass-Pipeline/
>
> 本文档仅供学习研究使用，版权归原作者及[兆松科技（武汉）有限公司](https://www.terapines.com)所有。如涉及侵权，请联系删除。
