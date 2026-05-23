# 附录 B：MLIR 核心概念速查

本附录提供本书中频繁使用的 MLIR（Multi-Level Intermediate Representation）核心概念的快速参考。对于需要系统学习 MLIR 的读者，推荐先完成 MLIR 官方 Toy Tutorial（https://mlir.llvm.org/docs/Tutorials/Toy/）。本附录不替代 MLIR 官方文档，而是作为阅读本书时的概念速查手册。

---

## B.1 核心概念一览

| 概念 | 英文原文 | 一句话解释 |
|------|---------|----------|
| 方言 | Dialect | MLIR 的可扩展命名空间，定义一组相关的 Operations、Types、Attributes |
| 操作 | Operation / Op | MLIR 中最基本的计算单元，类似 LLVM IR 中的 Instruction，但更抽象 |
| 类型 | Type | 描述 SSA 值的数据类型，每个 Dialect 可定义自己的类型 |
| 属性 | Attribute | 编译期常量，附加在 Operation 上的元数据（如常量值、标识符） |
| 区域 | Region | 包含基本块列表的容器，附属于一个 Operation |
| 基本块 | Block | 一个有序的操作序列，以 Terminator Operation 结尾 |
| 通道 | Pass | IR 转换/优化的一次执行单元 |
| 模式重写 | Pattern Rewrite | 声明式的 IR 重写规则，用于定义 Op 的规范化（Canonicalization）和 Lowering |
| 方言转换 | DialectConversion | MLIR 的框架，支持将一种方言的 Op 系统性地转换为另一种方言的 Op |

---

## B.2 方言（Dialect）

### 定义

Dialect 是 MLIR 最核心的扩展机制。一个 Dialect 定义了一个命名空间，包含一组相关的 Operation、Type 和 Attribute。

### 作用

- 将不同抽象层次的概念分隔到不同的命名空间中
- 编译器可以在同一个 IR 中混合使用多种 Dialect 的 Operation
- 支持渐进式 Lowering：从高层 Dialect 逐步转换到底层 Dialect

### Triton 中的 Dialect

Triton 编译器使用以下核心 Dialect：

| Dialect 名称 | 命名空间 | 说明 | 定义文件 |
|-------------|---------|------|---------|
| Triton Dialect | `tt` | 硬件无关的数据流 IR | `triton/include/triton/Dialect/Triton/IR/TritonDialect.td` |
| TritonGPU Dialect | `ttg` | GPU 硬件感知的并行 IR | `triton/include/triton/Dialect/TritonGPU/IR/TritonGPUDialect.td` |
| TritonNvidiaGPU Dialect | `ttng` | NVIDIA GPU 专属扩展 | `third_party/nvidia/` |

此外，Triton 大量使用 MLIR 内置 Dialect：
- **`arith`**：整数和浮点数算术运算（`arith.addi`、`arith.mulf` 等）
- **`math`**：数学库函数（`math.sqrt`、`math.exp` 等）
- **`scf`**：结构化控制流（`scf.for`、`scf.if`、`scf.while`）
- **`cf`**：底层控制流（`cf.br`、`cf.cond_br`）
- **`func`**：函数定义与调用
- **`gpu`**：GPU 通用操作
- **`nvvm`**：NVIDIA 虚拟机中间表示操作
- **`llvm`**：LLVM IR 的 MLIR 表示

### 为什么 Triton 要定义自己的方言？

Triton 不直接使用 MLIR 的 `gpu`、`memref`、`linalg` 等标准方言，原因包括：

1. **语义精确性**：`tt.load`、`tt.store` 支持 Tensor of Pointer 和 predication（masking），语义比 `memref.load` 更契合 Triton 的 tile-based 模型。
2. **控制自由度**：自定义方言允许 Triton 完全控制 Lowering 路径和优化策略。
3. **Layout 系统集成**：TTGIR 的 Layout 编码（Encoding）需要嵌入到 Type 系统中，而标准方言的类型系统不具备这种能力。
4. **硬件抽象层次**：TTIR 是硬件无关的，TTGIR 是硬件感知的——标准方言无法表达这种两层抽象。

---

## B.3 操作（Operation / Op）

### 定义

Operation 是 MLIR IR 中的基本计算单元。在文本格式中，Operation 的一般形式为：

```mlir
%result = "dialect.opname"(%operand1, %operand2) {attr_name = attr_value} : (type1, type2) -> result_type
```

Triton 使用自定义的简洁格式（Custom Assembly Format），例如：

```mlir
%0 = tt.addptr %ptr, %offset : !tt.ptr<f32>, i32
%1 = tt.load %0 {cacheModifier = ca} : !tt.ptr<f32>
```

### Operation 的组成部分

| 组成部分 | 说明 |
|---------|------|
| **Op Name** | 方言名 + "." + 助记符，如 `tt.load`、`ttg.convert_layout` |
| **Operands** | 输入 SSA 值列表 |
| **Results** | 输出 SSA 值列表 |
| **Attributes** | 编译期常量元数据 |
| **Regions** | 嵌套的 Region 列表（如循环体、条件分支） |
| **Successors** | 控制流后继基本块列表 |
| **Location** | 源码位置信息（用于调试和诊断） |

### 本书中的常用 Op

**TTIR 核心 Op**：

| Op 名称 | 助记符 | 功能 |
|---------|-------|------|
| `tt.load` | `load` | 从指针加载数据 |
| `tt.store` | `store` | 向指针存储数据 |
| `tt.reduce` | `reduce` | 沿指定轴的归约操作（含 Region 定义归约函数） |
| `tt.dot` | `dot` | 矩阵乘法（$D = A \times B + C$） |
| `tt.broadcast` | `broadcast` | 广播张量维度 |
| `tt.trans` | `trans` | 转置/置换张量维度 |
| `tt.splat` | `splat` | 将标量扩展为张量 |
| `tt.make_range` | `make_range` | 生成连续整数序列 |
| `tt.reshape` | `reshape` | 改变张量形状 |
| `tt.cat` | `cat` | 拼接两个张量 |
| `tt.expand_dims` | `expand_dims` | 增加维度 |
| `tt.get_program_id` | `get_program_id` | 获取当前 block 的索引 |
| `tt.get_num_programs` | `get_num_programs` | 获取 block 总数 |
| `tt.scan` | `scan` | 前缀扫描操作（含关联扫描函数） |
| `tt.gather` | `gather` | 沿指定轴的 gather 操作 |

**TTGIR 核心 Op**：

| Op 名称 | 助记符 | 功能 |
|---------|-------|------|
| `ttg.convert_layout` | `convert_layout` | 改变张量的 Layout（Distribution） |
| `ttg.local_alloc` | `local_alloc` | 分配共享内存 buffer |
| `ttg.local_load` | `local_load` | 从共享内存加载为 Distributed Tensor |
| `ttg.local_store` | `local_store` | 将 Distributed Tensor 存入共享内存 |
| `ttg.local_dealloc` | `local_dealloc` | 释放共享内存 buffer |
| `ttg.async_copy_global_to_local` | `async_copy_global_to_local` | 异步拷贝：Global Memory → Shared Memory |
| `ttg.async_commit_group` | `async_commit_group` | 提交一组异步拷贝操作 |
| `ttg.async_wait` | `async_wait` | 等待异步拷贝完成 |
| `ttg.barrier` | `barrier` | CTA 内线程同步（含内存序） |
| `ttg.warp_specialize` | `warp_specialize` | Warp 分组执行不同代码 |
| `ttg.warp_id` | `warp_id` | 获取当前 warp ID |

---

## B.4 类型（Type）

### 定义

Type 描述 SSA 值的类型。每个 Dialect 可以定义自己的类型系统。Triton 的 Type 定义在 TableGen 文件中。

### Triton 的类型系统

**TTIR 类型**（定义于 `TritonTypes.td`）：

| TableGen 名称 | 助记符 | 说明 |
|---------------|-------|------|
| `TT_Float` | `f8E4M3FN`, `f16`, `bf16`, `f32`, `f64` 等 | 浮点数类型 |
| `TT_Int` | `i1`, `i4`, `i8`, `i16`, `i32`, `i64` | 整数类型 |
| `TT_PtrType` | `!tt.ptr<T>` | 指针类型，指向标量元素类型 T |
| `TT_Tensor` | `tensor<NxMx...xT>` | 张量类型 |
| `TT_PtrTensor` | `tensor<Nx...x!tt.ptr<T>>` | 指针张量（Pointer Tensor） |
| `TT_TensorDescType` | `!tt.tensordesc<...>` | TMA 张量描述符类型 |

**TTGIR 类型**（定义于 `TritonGPUTypes.td`）：

| TableGen 名称 | 助记符 | 说明 |
|---------------|-------|------|
| `TTG_MemDescType` | `!ttg.memdesc<Nx...xT, #encoding, #memorySpace>` | 内存描述符类型，包含 shape、elementType、encoding（Layout）、memorySpace |
| `TTG_AsyncToken` | `!ttg.async.token` | 异步操作令牌类型，建立 SSA 链路 |

### encoding（Layout）与 Type 的关系

在 TTGIR 中，`tensor` 类型通过 **encoding** 属性携带 Layout 信息。例如：

```mlir
// 一个 128x64 的 f32 张量，使用 blocked layout
tensor<128x64xf32, #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>>

// 一个共享内存描述符，使用 swizzled shared layout
!ttg.memdesc<128x64xf32, #ttg.swizzled_shared<{vec=4, perPhase=2, maxPhase=4, order=[1,0]}>>
```

这种设计是 Triton 编译器的一个重要特征：**类型系统直接编码了数据分布信息**。

---

## B.5 属性（Attribute）

### 定义

Attribute 是编译期已知的常量数据，附加在 Operation 上作为元数据。与 Operand（运行时值）不同，Attribute 在编译期就是确定的。

### Triton 中的 Attribute

**TTIR 属性**（定义于 `TritonAttrDefs.td`）：

| 属性名 | 类型 | 说明 |
|--------|------|------|
| `CacheModifier` | I32Enum | L1 缓存修饰：NONE, CA (cache at all levels), CG (cache at global), WB (write-back), CS (stream), WT (write-through), CV (volatile) |
| `EvictionPolicy` | I32Enum | 缓存驱逐策略：NORMAL, EVICT_FIRST, EVICT_LAST |
| `MemSemantic` | I32Enum | 原子操作内存语义：RELAXED, ACQUIRE, RELEASE, ACQUIRE_RELEASE |
| `MemSyncScope` | I32Enum | 同步范围：GPU, CTA, SYSTEM |
| `AtomicRMW` | I32Enum | 原子 RMW 操作类型：AND, OR, XOR, ADD, FADD, MAX, MIN, UMAX, UMIN, XCHG |
| `ProgramDim` | I32Enum | Program ID 维度：X, Y, Z |
| `RoundingMode` | I32Enum | 浮点舍入模式：RTZ (round toward zero), RTNE (round to nearest even) |
| `InputPrecision` | I32Enum | Dot 精度：TF32, TF32x3, IEEE, BF16x3, BF16x6 |
| `PaddingOption` | I32Enum | 越界填充选项：PAD_ZERO, PAD_NAN |

**TTGIR 属性**（定义于 `TritonGPUAttrDefs.td`）-- Layout 编码：

详见本附录 B.7 节。

---

## B.6 区域（Region）与基本块（Block）

### 定义

- **Region**：包含一个基本块列表的容器。Region 附加在 Operation 上，用于表示该 Op 的嵌套结构（如循环体、条件分支的两个路径、归约的组合函数）。
- **Block**：一个有序的操作序列，以 **Terminator Operation** 结尾。Block 可以有 Block Arguments（类似函数参数）。

### 在 Triton 中的使用

```mlir
// reduce 使用 Region 定义归约函数
%result = tt.reduce(%input) axis = 1 {
  ^bb0(%a: f32, %b: f32):
    %sum = arith.addf %a, %b : f32
    tt.reduce.return %sum : f32
} : tensor<128x64xf32> -> tensor<128xf32>
```

上例中：
- `tt.reduce` Operation 包含一个 Region（大括号 `{...}` 内的部分）
- Region 包含一个 Block（`^bb0`）
- Block 有两个 Block Arguments（`%a` 和 `%b`）
- Block 以 `tt.reduce.return` 作为 Terminator

### 嵌套关系

```
ModuleOp
  └── Region
       └── Block
            ├── tt.func @kernel
            │    └── Region
            │         └── Block
            │              ├── tt.get_program_id
            │              ├── scf.for %i = ...
            │              │    └── Region
            │              │         └── Block
            │              │              ├── tt.load ...
            │              │              ├── arith.addf ...
            │              │              └── scf.yield ...
            │              └── tt.return
            └── ...
```

---

## B.7 Pass 与 Pattern Rewrite

### Pass

Pass 是 MLIR 中对 IR 进行转换或分析的一次执行单元。Pass 可以：
- **分析**（Analysis Pass）：收集 IR 的属性，不修改 IR（如 Alias Analysis）
- **转换**（Transform Pass）：修改 IR（如 Lowering、优化）

Triton 的 Pass Pipeline 大致顺序为：

```
TTIR passes (LoopPeeling)
  ↓
TTIR → TTGIR lowering (TritonToTritonGPU)
  ↓
TTGIR passes (Coalescing, Prefetch, OptimizeDotOperands, RemoveLayoutConversions, ...)
  ↓
TTGIR → LLVM lowering (TritonGPUToLLVM)
  ↓
LLVM optimization passes
```

### Pattern Rewrite

Pattern Rewrite 是 MLIR 提供的一种声明式的 IR 重写机制。开发者定义 "匹配模式" 和 "重写模式"，MLIR 自动在 IR 上应用这些规则。

Pattern Rewrite 主要用于：
- **Canonicalization**（规范化）：将 IR 化简为标准形式（如 `x + 0 → x`、`reshape(reshape(x)) → reshape(x)`）
- **Lowering**：将高层 Op 替换为低层 Op 序列（如 `tt.dot → ttg.convert_layout + ttg.local_load + llvm.inline_asm`）

---

## B.8 Layout 编码（Encoding）速查

Layout Encoding 是 TTGIR 中最重要的属性，描述数据在 GPU 线程和内存上的分布方式。分为两大类：

### Distributed Encoding（分布式编码）

数据分布在多个线程/寄存器中，每个线程持有一部分。

| Encoding 名称 | 助记符 | 说明 | 适用场景 |
|--------------|-------|------|---------|
| `BlockedEncodingAttr` | `blocked` | 每个 warp 持有张量的一个连续子块 | 通用于 element-wise 操作，支持 coalesced 内存访问 |
| `NvidiaMmaEncodingAttr` | `nvidia_mma` | Tensor Core MMA 指令的输出布局 | NVIDIA GPU 上 `tt.dot` 的结果张量（Volta/Ampere/Hopper） |
| `AMDMfmaEncodingAttr` | `amd_mfma` | AMD Matrix Core MFMA 指令的输出布局 | AMD CDNA GPU 上 `tt.dot` 的结果张量 |
| `AMDWmmaEncodingAttr` | `amd_wmma` | AMD Wave Matrix Multiply-Accumulate 布局 | AMD RDNA GPU 上 `tt.dot` 的结果张量 |
| `DotOperandEncodingAttr` | `dot_op` | `tt.dot` 操作数的布局（A 矩阵或 B 矩阵） | MMA v1/v2 中 dot 操作数的输入要求 |
| `SliceEncodingAttr` | `slice` | 从父 layout 中挤出一个维度 | `expand_dims` 的逆操作优化 |
| `LinearEncodingAttr` | `linear` | 基于 LinearLayout 的新式编码（无 swizzle） | 统一的 LinearLayout 表示 |
| `GenericLinearEncodingAttr` | `generic_linear` | 支持 warp 级 swizzle 的 LinearLayout | 更灵活的新式编码 |

### Shared Encoding（共享内存编码）

数据存储在 shared memory 中，可被所有线程访问。

| Encoding 名称 | 助记符 | 说明 |
|--------------|-------|------|
| `SwizzledSharedEncodingAttr` | `swizzled_shared` | 带 swizzle 的共享内存布局，用于减少 bank conflict |
| `PaddedSharedEncodingAttr` | `padded_shared` | 带 padding 和线性重排的共享内存布局 |
| `NVMMASharedEncodingAttr` | `nvmma_shared` | MMAv3/MMAv5 的共享内存输入布局规范（blocked tiled 2D） |
| `SharedLinearEncodingAttr` | `shared_linear` | 基于 LinearLayout 的共享内存编码 |
| `AMDRotatingSharedEncodingAttr` | `amd_rotating_shared` | AMD 的轮转 swizzle 模式，减少 bank conflict |
| `PartitionedSharedEncodingAttr` | `partitioned_shared` | 跨多个独立物理分区的共享内存 |

### CGA Encoding（协作线程阵列编码）

| Encoding 名称 | 助记符 | 说明 |
|--------------|-------|------|
| `CGAEncodingAttr` | `cga` | 描述 CTA 在 Cooperative Grid Array 中的排列 |

---

## B.9 TableGen 快速入门

### 什么是 TableGen？

TableGen 是 LLVM/MLIR 的声明式领域特定语言，用于定义类型、操作、属性等。它是编译器的编译器——`*.td` 文件被 `mlir-tblgen` 工具处理，自动生成 C++ 代码。

### 基本语法

```tablegen
// 定义一个 Operation
def TT_LoadOp : TT_Op<"load", [Trait1, Trait2, ...]> {
  let summary = "简短描述";
  let description = [{长描述}];
  let arguments = (ins Type1:$operand1, Type2:$operand2);  // 输入操作数
  let results = (outs ResultType:$result);                 // 输出结果
  let assemblyFormat = "...";                              // 文本格式
}

// 定义一个 Type
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
  let parameters = (ins "Type":$pointeeType, "int":$addressSpace);
}

// 定义一个 Enum Attribute
def TT_CacheModifierAttr : I32EnumAttr<"CacheModifier", "", [...]> { ... }
```

### 关键关键字

| 关键字 | 含义 |
|--------|------|
| `def` | 定义一个 TableGen 记录 |
| `class` | 定义一个可复用的模板 |
| `let` | 设置记录的字段值 |
| `ins` / `outs` | 定义 Operation 的输入/输出 |
| `list<Trait> traits` | Operation 的特征列表（如 `Pure`、`Elementwise`） |
| `assemblyFormat` | 自定义 Operation 的打印/解析格式 |

---

## B.10 MLIR vs. LLVM IR 对比

| 特性 | MLIR | LLVM IR |
|------|------|---------|
| **抽象层次** | 多级抽象（从高层 DSL 到机器指令） | 单级抽象（接近汇编的低级表示） |
| **可扩展性** | Dialect 机制允许定义任意 Operation、Type、Attribute | 固定的 IR 指令集和类型系统 |
| **类型系统** | 每个 Dialect 可定义自定义类型 | 固定类型（整数、浮点、指针、结构体、数组等） |
| **SSA 形式** | 支持（每个值只赋值一次） | 支持（使用 phi 节点） |
| **控制流** | 通过 Region/Block/Successor 表达，支持结构化控制流 | 基本块 + 分支指令 + phi 节点 |
| **嵌套结构** | 原生支持（Region 内嵌 Block） | 无原生嵌套结构 |
| **元数据** | Attribute 系统（编译期常量元数据） | Metadata 系统 |
| **用途** | 编译器基础设施的通用框架 | 编译器后端的目标 IR |
| **典型用户** | TensorFlow、PyTorch、Triton、ONNX-MLIR、CIRCT | Clang、Rust、Swift |
| **代码生成** | 可渐进 Lowering 到 LLVM IR | 直接降级到目标机器码 |

在 Triton 的编译管线中，MLIR 覆盖从 TTIR 到 LLVM IR 的整个范围。到达 LLVM IR 后，剩余的寄存器分配、指令调度和机器码生成由 LLVM 的 NVPTX 后端完成。

---

## B.11 常用 MLIR 命令行工具

| 工具 | 用途 |
|------|------|
| `mlir-opt` | 加载 MLIR 模块、运行 pass 管线、输出转换后的 IR |
| `mlir-tblgen` | 从 `.td` 文件生成 C++ Op/Type/Attr 代码 |
| `mlir-translate` | 在不同 IR 格式间转换（如 MLIR ↔ LLVM IR） |

### Triton IR 调试命令

```bash
# 查看编译 IR
TRITON_DUMP_IR=1 python example.py

# 查看每个 pass 前后的 IR 变化
TRITON_DUMP_PASSES=1 python example.py

# 使用 Triton 解释器（不依赖 GPU）
TRITON_INTERPRET=1 python example.py
```
