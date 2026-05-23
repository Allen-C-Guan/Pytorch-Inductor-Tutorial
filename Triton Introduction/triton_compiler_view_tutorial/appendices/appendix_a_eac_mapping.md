# 附录 A：Engineering a Compiler 章节完整映射

本附录提供 *Engineering a Compiler*（EaC, Cooper & Torczon, 第三版）与本书各章节及 Triton 编译器源码模块的完整对应关系。读者可将 EaC 作为编译器理论的主教材，本书作为其与真实编译器实现之间的桥梁。

---

## A.1 全书映射总表

| EaC 章节 | EaC 主题 | 本书章节 | Triton 映射 | 关键 Triton 源文件 |
|----------|---------|---------|------------|------------------|
| **Ch.1** Overview of Compilation | 编译器概述：前端、优化器、后端 | **第1章** 编译器设计导论与 Triton 全景 | Triton 编译管线总览：`Triton DSL → TTIR → TTGIR → LLVM IR → PTX → CUBIN` | `triton/python/triton/compiler/compiler.py` |
| **Ch.2** Scanners | 词法分析：正则表达式、DFA、NFA、扫描器生成器 | **第2章** Triton 编程语言设计 | Python 作为 DSL 宿主语言取代传统扫描器角色；`@triton.jit` 装饰器捕获 Python AST 作为词法/语法分析的起点 | `triton/python/triton/language/core.py` |
| **Ch.3** Parsers | 语法分析：上下文无关文法、LL/LR 解析、抽象语法树 | **第2章** Triton 编程语言设计 | `triton.language` 的 Python 函数体即 AST；`code_generator.py` 的 `visit` 方法实现 AST 遍历 | `triton/python/triton/compiler/code_generator.py` |
| **Ch.4** Intermediate Representations | 中间表示设计：图形 IR、线性 IR、SSA 形式、符号表 | **第3章** MLIR 基础设施与 TTIR 设计；**第4章** TTGIR 设计 | TTIR（硬件无关数据流 IR）与 TTGIR（GPU 硬件感知并行 IR）的两级设计；MLIR 的 SSA 值、Block、Region | `triton/include/triton/Dialect/Triton/IR/TritonOps.td`；`triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` |
| **Ch.5** Syntax-Driven Translation / Semantic Analysis | 语义分析：类型检查、类型推断、属性文法、语法制导翻译 | **第5章** 语义分析与类型系统；**第6章** Lowering — TTIR to TTGIR | `triton.language.semantic` 的类型检查；`constexpr` 机制；MLIR DialectConversion 框架实现的语法制导 Lowering | `triton/python/triton/language/semantic.py`；`triton/lib/Conversion/TritonToTritonGPU/` |
| **Ch.6** The Procedure Abstraction | 过程抽象：调用约定、栈帧、参数传递、作用域 | **第13章** JIT 编译系统与缓存管理 | `tt.call`、`tt.func`、`tt.return` 的过程抽象；JIT 编译时的 kernel launch 调用约定 | `triton/include/triton/Dialect/Triton/IR/TritonOps.td` (`CallOp`, `FuncOp`, `ReturnOp`) |
| **Ch.7** Code Shape / Memory Hierarchy | 代码形态与内存层次：局部性、缓存、地址空间 | **第8章** 内存优化 — Coalescing, Layout 与 Shared Memory | GPU 内存层次：Global(HBM) → L2 → L1/Shared → Register；`ttg.local_alloc`、`ttg.async_copy_global_to_local` 的内存操作；Memory Space 属性 | `triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` (`LocalAllocOp`, `AsyncCopyGlobalToLocalOp`)；`triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` |
| **Ch.8** Introduction to Optimization | 优化概论：局部优化、全局优化、数据流分析 | **第7章** 循环优化 — Tiling, Peeling 与展开 | TTIR 和 TTGIR 层的各类优化 pass：loop peeling、coalescing、layout 优化、instruction reordering | `triton/include/triton/Dialect/Triton/Transforms/`；`triton/include/triton/Dialect/TritonGPU/Transforms/` |
| **Ch.9** Data-Flow Analysis / Loop Optimizations | 数据流分析与循环优化：支配树、SSA、循环分块、循环展开、依赖分析 | **第7章** 循环优化 — Tiling, Peeling 与展开 | Loop Peeling（`LoopPeeling` pass）；Tiling（Triton tile-based 编程模型的天然对应）；依赖分析（`AxisInfo` 分析） | `triton/lib/Dialect/Triton/Transforms/LoopPeeling.cpp`；`triton/lib/Analysis/AxisInfo.cpp` |
| **Ch.10** Instruction Selection | 指令选择：树模式匹配、DAG 覆盖、Peephole 优化 | **第9章** 指令选择 — TTGIR to LLVM IR | `TritonGPUToLLVM` 转换：`ElementwiseOpToLLVM`、`ReduceOpToLLVM`、`DotOpToLLVM`（Tensor Core MMA）、`LoadOpToLLVM`/`StoreOpToLLVM`、`ConvertLayoutOpToLLVM` | `triton/lib/Conversion/TritonGPUToLLVM/` |
| **Ch.11** Instruction Scheduling | 指令调度：列表调度、软件流水线、模调度 | **第10章** 软件流水线与 Warp Specialization | `PipelineExpander`（异步拷贝流水线）、`PipeliningUtility`（阶段划分）、乒乓 buffer 策略；`WarpSpecialization` pass | `triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/`；`triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/` |
| **Ch.12** Register Allocation | 寄存器分配：活跃范围分析、干涉图、图着色（Chaitin-Briggs）、线性扫描 | **第11章** 寄存器分配与内存管理 | `Allocation` 分析（buffer 活跃范围、内存复用）；`Alias` 分析（指针别名）；`AllocateSharedMemory`；`GlobalScratchMemoryAllocation` | `triton/lib/Analysis/Allocation.cpp`；`triton/lib/Analysis/Alias.cpp` |
| **Ch.13** Back-End Compilation | 后端编译：指令调度细节、代码发射、目标机器特性 | **第12章** 后端代码发射 — LLVM to PTX to CUBIN | LLVM NVPTX 后端 → PTX 汇编 → CUBIN（SASS）；多后端支持（NVIDIA CUDA、AMD ROCm/HIP、Ascend 昇腾） | `triton/python/triton/backends/`；`triton/lib/Conversion/TritonGPUToLLVM/` |

---

## A.2 EaC 章节详细映射

### Ch.1 Overview of Compilation -- 编译器概述

**EaC 核心概念**：
- 编译器的基本结构：前端（Front End）、优化器（Optimizer）、后端（Back End）
- IR（Intermediate Representation）作为各阶段间的桥梁
- 编译器的正确性要求：翻译必须保持源程序的语义

**Triton 中的体现**：
Triton 的编译管线清晰对应 EaC 的三段式结构：

```
前端: Triton DSL (Python) → TTIR (MLIR)
  ├── @triton.jit 捕获 Python AST
  ├── code_generator.py 遍历 AST, 生成 TTIR operations
  └── semantic.py 进行类型检查与推断

中间优化器: TTIR → TTGIR → [optimizations] → optimized TTGIR
  ├── TTIR → TTGIR lowering (layout 分配)
  ├── TTGIR transforms (coalescing, pipelining, warp spec 等)
  └── Analysis passes (alias, allocation, membar, axisinfo)

后端: TTGIR → LLVM IR → PTX → CUBIN
  ├── TritonGPUToLLVM conversion
  ├── LLVM NVPTX backend → PTX assembly
  └── PTX assembler → CUBIN (SASS)
```

**对应源文件**：`triton/python/triton/compiler/compiler.py`

---

### Ch.2 Scanners -- 词法分析

**EaC 核心概念**：
- 正则表达式、有限自动机（DFA/NFA）
- 从正则表达式到扫描器的自动生成

**Triton 中的体现**：
Triton 不使用传统的词法分析器。因为 Triton DSL 是嵌入在 Python 中的领域特定语言，Python 解释器自身承担了词法分析和语法分析的角色。`@triton.jit` 装饰器通过 Python 的 `inspect` 模块获取函数的源代码或字节码，然后由 `code_generator.py` 中的 AST 访问器（`visit` 方法）遍历 Python AST 节点。

**对应源文件**：`triton/python/triton/compiler/code_generator.py`

---

### Ch.3 Parsers -- 语法分析

**EaC 核心概念**：
- 上下文无关文法（CFG）、自上而下解析（LL）、自下而上解析（LR）
- 抽象语法树（AST）的构建

**Triton 中的体现**：
Python 的 `ast` 模块是 Triton DSL 的"解析器"。`code_generator.py` 中 `CodeGenerator` 类的 `visit_*` 方法（如 `visit_For`、`visit_If`、`visit_Assign`）实现 AST 节点的语义动作，将 Python 语法结构翻译为 TTIR operations。

**对应源文件**：`triton/python/triton/compiler/code_generator.py`

---

### Ch.4 Intermediate Representations -- 中间表示设计

**EaC 核心概念**：
- 图形 IR vs. 线性 IR vs. 混合 IR
- SSA（Static Single Assignment）形式：每个变量仅被赋值一次
- 基本块（Basic Block）与控制流图（CFG）
- 符号表的作用域管理

**Triton 中的体现**：
这是与 Triton 编译器设计最密切相关的 EaC 章节。Triton 采用 **两级 IR** 设计：
1. **TTIR**（Triton IR）：硬件无关的数据流 IR，定义在 `triton/include/triton/Dialect/Triton/IR/` 中，核心 Operations 包括 `tt.load`、`tt.store`、`tt.reduce`、`tt.dot`、`tt.broadcast`、`tt.trans`、`tt.make_range`、`tt.splat` 等。
2. **TTGIR**（TritonGPU IR）：GPU 硬件感知的并行 IR，定义在 `triton/include/triton/Dialect/TritonGPU/IR/` 中。其核心创新是 **Layout 系统**（Distributed Encoding），将数据在 GPU 线程、warp、block 上的分布信息编码在类型系统中。

两级 IR 均构建在 MLIR 基础设施之上，天然支持 SSA 形式、Region、Block 等结构。

**对应源文件**：
- `triton/include/triton/Dialect/Triton/IR/TritonOps.td`、`TritonTypes.td`
- `triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`、`TritonGPUAttrDefs.td`、`TritonGPUTypes.td`

---

### Ch.5 Syntax-Driven Translation / Semantic Analysis -- 语义分析

**EaC 核心概念**：
- 属性文法（Attribute Grammar）：将语义规则关联到语法产生式
- 类型检查（Type Checking）与类型推断（Type Inference）
- 语法制导翻译（Syntax-Directed Translation）

**Triton 中的体现**：
在 Triton 中有两个层面的体现：
1. **Python 层的语义分析**（第5章）：`triton/language/semantic.py` 实现类型检查——验证 `tl.load` 的指针类型、`tl.store` 的值与指针的匹配等。`constexpr` 机制在编译期进行常量推断与折叠。
2. **MLIR 层的方言转换**（第6章）：`TritonToTritonGPUPass` 基于 MLIR 的 DialectConversion 框架实现语法制导翻译——遍历 TTIR operations，为每个 Op 分配 Layout，必要时插入 `ConvertLayoutOp`。

**对应源文件**：
- `triton/python/triton/language/semantic.py`
- `triton/lib/Conversion/TritonToTritonGPU/`

---

### Ch.6 The Procedure Abstraction -- 过程抽象

**EaC 核心概念**：
- 调用约定（Calling Convention）：参数传递、返回值、寄存器保存
- 栈帧（Stack Frame）与活动记录（Activation Record）
- 作用域规则（Scope Rules）

**Triton 中的体现**：
TTIR 中定义了 `tt.func`、`tt.call`、`tt.return` 三个 Op 来支持过程抽象。`FuncOp` 实现 `FunctionOpInterface` 和 `CallableOpInterface`，定义函数的参数类型、返回类型和函数体（Body Region）。GPU kernel 的函数调用不使用传统栈帧，参数通过 GPU 寄存器或共享内存传递。

**对应源文件**：`triton/include/triton/Dialect/Triton/IR/TritonOps.td` (`FuncOp`, `CallOp`, `ReturnOp`)

---

### Ch.7 Code Shape / Memory Hierarchy -- 代码形态与内存层次

**EaC 核心概念**：
- 存储层次：寄存器、缓存、主存、外存
- 局部性原理：时间局部性（Temporal Locality）与空间局部性（Spatial Locality）
- 地址空间（Address Space）的概念

**Triton 中的体现**：
Triton 将 GPU 内存层次抽象为三个 Memory Space：
- **Global Memory**（`GlobalMemory` Resource）：HBM，大容量、高延迟
- **Shared Memory**（`SharedMemory` Resource）：on-chip SRAM，由 CTA 内所有线程共享
- **Register**：每个线程私有的最快存储

TTGIR 的 `ttg.async_copy_global_to_local` 实现从 global 到 shared 的异步数据搬运，`ttg.local_load`/`ttg.local_store` 操作 shared memory。`Coalesce` pass 优化全局内存访问的合并性以最大化带宽利用率。

**对应源文件**：
- `triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`
- `triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`

---

### Ch.8 Introduction to Optimization -- 优化概论

**EaC 核心概念**：
- 局部优化（Local Optimization）：基本块内的优化
- 全局优化（Global Optimization）：跨基本块的优化
- 数据流分析框架（Data-Flow Analysis Framework）

**Triton 中的体现**：
Triton 拥有丰富的优化 pass 管线：

**TTIR 层**：
- Loop Peeling：循环剥离优化

**TTGIR 层**：
- Coalescing：全局内存合并访问
- Prefetch：数据预取
- Pipelining：软件流水线
- Warp Specialization：warp 角色划分
- RemoveLayoutConversions：消除冗余的 layout 转换
- OptimizeDotOperands：点积操作数优化
- ReduceDataDuplication：减少数据重复
- ReorderInstructions：指令重排

**对应源文件**：
- `triton/include/triton/Dialect/Triton/Transforms/`
- `triton/include/triton/Dialect/TritonGPU/Transforms/`

---

### Ch.9 Data-Flow Analysis / Loop Optimizations -- 数据流分析与循环优化

**EaC 核心概念**：
- 支配树（Dominator Tree）与支配边界（Dominance Frontier）
- SSA 构建与销毁
- 循环优化：
  - 循环分块（Loop Tiling / Strip-Mining）
  - 循环剥离（Loop Peeling）
  - 循环展开（Loop Unrolling）
  - 循环合并（Loop Fusion）/循环分布（Loop Distribution）
- 依赖分析：循环携带依赖（Loop-Carried Dependence）vs. 循环无关依赖（Loop-Independent Dependence）

**Triton 中的体现**：
Triton 的 tile-based 编程模型天然将计算组织为块状迭代，与 tiling 优化完美对齐。`LoopPeeling` pass 实现循环剥离——将一个循环的前几次迭代或后几次迭代剥离为独立的基本块，以消除边界条件检查。`AxisInfo` 分析提供张量各维度的符号信息，支持依赖分析和优化决策。

**对应源文件**：
- `triton/lib/Dialect/Triton/Transforms/LoopPeeling.cpp`
- `triton/lib/Analysis/AxisInfo.cpp`

---

### Ch.10 Instruction Selection -- 指令选择

**EaC 核心概念**：
- 树模式匹配（Tree-Pattern Matching）：将 IR 子树匹配为机器指令
- DAG 覆盖（DAG Covering）：处理公共子表达式的指令选择
- Peephole 优化：局部指令序列替换

**Triton 中的体现**：
Triton 的指令选择是通过 MLIR 的 DialectConversion 框架实现的 `TritonGPUToLLVM` conversion，它不是传统的树模式匹配，而是 **Op-by-Op 转换**：每个 TTGIR Op 对应一个 C++ 的 `ConversionPattern`，将其翻译为 LLVM IR 指令序列。

关键转换模式：
- `ElementwiseOpToLLVM`：逐元素运算映射到 LLVM 算术指令
- `ReduceOpToLLVM`：归约操作映射到 warp shuffle + shared memory
- `DotOpToLLVM`：矩阵乘法映射到 Tensor Core MMA 指令
- `LoadOpToLLVM` / `StoreOpToLLVM`：内存访问映射到 LLVM load/store
- `ConvertLayoutOpToLLVM`：Layout 转换映射到 shared memory 中转 + warp shuffle
- `ControlFlowOpToLLVM`：控制流映射到 LLVM 分支指令

**对应源文件**：`triton/lib/Conversion/TritonGPUToLLVM/`

---

### Ch.11 Instruction Scheduling -- 指令调度

**EaC 核心概念**：
- 列表调度（List Scheduling）：基于优先级的指令调度
- 软件流水线（Software Pipelining）：重叠不同迭代的执行
- 模调度（Modulo Scheduling）：循环的软件流水线形式化方法
- 延迟隐藏（Latency Hiding）：通过调度减少流水线停顿

**Triton 中的体现**：
Triton 的软件流水线系统是第10章的核心内容：

- **`PipelineExpander`**：将异步拷贝+等待操作展开为流水线阶段
- **`PipeliningUtility`**：划分流水线阶段、插入 barrier 同步
- **乒乓 Buffer（Double Buffering）**：使用两块 shared memory buffer 交替使用，隐藏数据传输延迟
- **`WarpSpecialization` pass**：将 warp 划分为生产者（负责 global→shared 数据搬运）和消费者（负责计算），实现更激进的延迟隐藏

**对应源文件**：
- `triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/`
- `triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/`

---

### Ch.12 Register Allocation -- 寄存器分配

**EaC 核心概念**：
- 活跃范围分析（Liveness Analysis）：确定每个变量的活跃区间
- 干涉图（Interference Graph）：变量之间的冲突关系
- 图着色算法（Graph Coloring）：Chaitin-Briggs 算法
- 线性扫描分配（Linear Scan Allocation）：更快的近似算法
- 溢出（Spilling）：寄存器不足时将变量写入内存

**Triton 中的体现**：
Triton 的内存管理分为几个层次：

1. **Alias 分析**（`Alias.cpp`）：分析指针之间的别名关系，确定哪些 buffer 可以共享内存空间。
2. **Allocation 分析**（`Allocation.cpp`）：分析 buffer 的活跃范围（liveness），基于干涉图进行共享内存的复用分配。
3. **`AllocateSharedMemory` pass**：为 `ttg.local_alloc` 操作分配实际的 shared memory 地址。
4. **Shared Memory 上的寄存器分配**：Triton 将最终的寄存器分配委托给 LLVM 的 NVPTX 后端——LLVM 在生成 PTX 代码时进行图着色或线性扫描寄存器分配。

**对应源文件**：
- `triton/lib/Analysis/Allocation.cpp`
- `triton/lib/Analysis/Alias.cpp`

---

### Ch.13 Back-End Compilation -- 后端编译

**EaC 核心概念**：
- 指令调度与代码发射的结合
- 目标机器特性建模：指令延迟、功能单元约束
- 生成可重定位目标代码与可执行文件

**Triton 中的体现**：
Triton 的后端编译链条：

```
TTGIR → TritonGPUToLLVM → LLVM IR → NVPTX backend → PTX 汇编 → ptxas → CUBIN (SASS)
```

Triton 的多后端支持架构通过 Python 层的插件接口（`triton/python/triton/backends/`）实现：
- **NVIDIA 后端**（CUDA）：默认后端，生成 PTX/CUBIN
- **AMD 后端**（ROCm/HIP）：通过 `triton-ascend` 等插件支持
- **Ascend 后端**（昇腾 NPU）：华为达芬奇架构支持

每个后端需要实现 `compiler.py`（编译接口）和 `driver.py`（运行时驱动）两个模块。

**对应源文件**：`triton/python/triton/backends/`

---

## A.3 交叉索引：按 Triton 源码模块查 EaC 章节

| Triton 源码模块 | 主要涉及的 EaC 章节 | 本书章节 |
|----------------|-------------------|---------|
| `triton/python/triton/language/core.py` | Ch.2 (Scanners), Ch.3 (Parsers) | 第2章 |
| `triton/python/triton/language/semantic.py` | Ch.5 (Semantic Analysis) | 第5章 |
| `triton/python/triton/compiler/code_generator.py` | Ch.3 (Parsers), Ch.5 (Syntax-Directed Translation) | 第2章 |
| `triton/include/triton/Dialect/Triton/IR/TritonOps.td` | Ch.4 (IR Design) | 第3章 |
| `triton/include/triton/Dialect/Triton/IR/TritonTypes.td` | Ch.4 (IR Design), Ch.5 (Type Systems) | 第3、5章 |
| `triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td` | Ch.4 (IR Design) | 第4章 |
| `triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td` | Ch.4 (IR Design) | 第4章 |
| `triton/lib/Conversion/TritonToTritonGPU/` | Ch.5 (Syntax-Directed Translation) | 第6章 |
| `triton/lib/Analysis/AxisInfo.cpp` | Ch.9 (Data-Flow Analysis) | 第7章 |
| `triton/lib/Dialect/Triton/Transforms/LoopPeeling.cpp` | Ch.9 (Loop Optimization) | 第7章 |
| `triton/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` | Ch.7 (Memory Hierarchy), Ch.8 (Optimization) | 第8章 |
| `triton/lib/Conversion/TritonGPUToLLVM/` | Ch.10 (Instruction Selection) | 第9章 |
| `triton/lib/Dialect/TritonGPU/Transforms/Pipeliner/` | Ch.11 (Instruction Scheduling) | 第10章 |
| `triton/lib/Dialect/TritonGPU/Transforms/WarpSpecialization/` | Ch.11 (Instruction Scheduling) | 第10章 |
| `triton/lib/Analysis/Allocation.cpp` | Ch.12 (Register Allocation) | 第11章 |
| `triton/lib/Analysis/Alias.cpp` | Ch.12 (Register Allocation) | 第11章 |
| `triton/python/triton/backends/` | Ch.13 (Back-End Compilation) | 第12章 |
| `triton/python/triton/runtime/cache.py` | Ch.6 (Procedure Abstraction) | 第13章 |
| `triton/python/triton/runtime/autotuner.py` | (EaC 未直接覆盖，属于编译器扩展话题) | 第14章 |
