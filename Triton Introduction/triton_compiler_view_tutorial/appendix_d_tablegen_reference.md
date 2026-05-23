# 附录 D：关键 TableGen 定义索引

本附录列出本书中涉及的关键 TableGen 定义，按文件路径和类型组织。每条记录包含：文件路径、`def` 名称、简短描述及讨论该定义的章节。

所有文件路径均相对于 workspace 根目录下的 `triton/` 源码树。

---

## D.1 TTIR -- Operation 定义

**文件**：`triton/include/triton/Dialect/Triton/IR/TritonOps.td`

| def 名称 | 助记符 | 描述 | 讨论章节 |
|----------|-------|------|---------|
| `TT_LoadOp` | `tt.load` | 从指针或指针张量加载数据；支持 mask（predication）、cache modifier、eviction policy、other 回退值 | 第2、3、9章 |
| `TT_StoreOp` | `tt.store` | 向指针或指针张量存储数据；支持 mask、cache modifier、eviction policy | 第2、3、9章 |
| `TT_ReduceOp` | `tt.reduce` | 沿指定轴进行归约；Region 定义归约组合函数；支持多输入/多输出 | 第3、9章 |
| `TT_DotOp` | `tt.dot` | 矩阵乘法 D = A x B + C；`inputPrecision` 控制 Tensor Core 精度模式（TF32, IEEE, BF16x3 等） | 第3、4、9章 |
| `TT_DotScaledOp` | `tt.dot_scaled` | 带逐块缩放的矩阵乘法（Microscaling spec）；支持 FP4/FP8 等低精度类型 | 第3、9章 |
| `TT_BroadcastOp` | `tt.broadcast` | 将大小为1的维度扩展到新大小 | 第3、7章 |
| `TT_TransOp` | `tt.trans` | 张量维度重排（同时实现 `tl.trans` 和 `tl.permute`） | 第3、4章 |
| `TT_SplatOp` | `tt.splat` | 将标量值扩展为张量（所有元素相同） | 第3章 |
| `TT_UnsplatOp` | `tt.unsplat` | 将单元素张量转换为标量 | 第3章 |
| `TT_MakeRangeOp` | `tt.make_range` | 生成 [start, end) 的连续整数序列张量 | 第2、3章 |
| `TT_ReshapeOp` | `tt.reshape` | 改变张量形状；可允许重排序（`allow_reorder`） | 第3、4章 |
| `TT_ExpandDimsOp` | `tt.expand_dims` | 在指定位置增加大小为1的维度 | 第3章 |
| `TT_CatOp` | `tt.cat` | 沿指定轴拼接两个张量 | 第3章 |
| `TT_JoinOp` | `tt.join` | 沿新的最内层维度拼接两个同型张量 | 第3章 |
| `TT_SplitOp` | `tt.split` | 沿最后一维拆分为两个张量 | 第3章 |
| `TT_ScanOp` | `tt.scan` | 关联扫描（prefix scan）；Region 定义扫描组合函数；`reverse` 控制方向 | 第3、9章 |
| `TT_GatherOp` | `tt.gather` | 沿指定轴、按索引张量 gather 元素 | 第3章 |
| `TT_HistogramOp` | `tt.histogram` | 计算输入张量的直方图 | 第3章 |
| `TT_AtomicRMWOp` | `tt.atomic_rmw` | 原子 read-modify-write 操作（AND, OR, XOR, ADD, FADD, MAX, MIN, XCHG 等） | 第3、8章 |
| `TT_AtomicCASOp` | `tt.atomic_cas` | 原子 compare-and-swap 操作 | 第3章 |
| `TT_AddPtrOp` | `tt.addptr` | 指针加法（ptr + offset） | 第3章 |
| `TT_IntToPtrOp` | `tt.int_to_ptr` | 将 int64 转换为指针 | 第3章 |
| `TT_PtrToIntOp` | `tt.ptr_to_int` | 将指针转换为 int64 | 第3章 |
| `TT_BitcastOp` | `tt.bitcast` | 等位宽类型间的位转换（支持指针） | 第3章 |
| `TT_FpToFpOp` | `tt.fp_to_fp` | 自定义浮点类型转换（F8 <-> FP16/BF16/FP32/FP64） | 第3、5章 |
| `TT_ClampFOp` | `tt.clampf` | 浮点 clamp 操作 | 第3章 |
| `TT_PreciseSqrtOp` | `tt.precise_sqrt` | 高精度平方根（对标 CUDA `__dsqrt_rn`） | 第3章 |
| `TT_PreciseDivFOp` | `tt.precise_divf` | 高精度浮点除法 | 第3章 |
| `TT_MulhiUIOp` | `tt.mulhiui` | 两整数乘积的高 N 位 | 第3章 |
| `TT_GetProgramIdOp` | `tt.get_program_id` | 获取当前 program（CTA）在指定维度上的 ID | 第2、3章 |
| `TT_GetNumProgramsOp` | `tt.get_num_programs` | 获取指定维度上的 program 总数 | 第2、3章 |
| `TT_MakeTensorDescOp` | `tt.make_tensor_descriptor` | 创建张量描述符（用于 TMA 操作） | 第3、8章 |
| `TT_DescriptorLoadOp` | `tt.descriptor_load` | 通过 TMA 描述符加载（NVIDIA Hopper TMA） | 第3、8章 |
| `TT_DescriptorStoreOp` | `tt.descriptor_store` | 通过 TMA 描述符存储 | 第3、8章 |
| `TT_DescriptorReduceOp` | `tt.descriptor_reduce` | 通过 TMA 描述符进行归约 store | 第3章 |
| `TT_DescriptorGatherOp` | `tt.descriptor_gather` | 通过 TMA 描述符 gather 多行 | 第3章 |
| `TT_DescriptorScatterOp` | `tt.descriptor_scatter` | 通过 TMA 描述符 scatter 多行 | 第3章 |
| `TT_MapElementwiseOp` | `tt.map_elementwise` | 标量子区域映射到张量（类似于 `tl.inline_asm_elementwise`） | 第3章 |
| `TT_ExternElementwiseOp` | `tt.extern_elementwise` | 调用外部函数实现逐元素操作（`libpath/libname:symbol`） | 第3章 |
| `TT_ElementwiseInlineAsmOp` | `tt.elementwise_inline_asm` | 使用内联 PTX 实现逐元素操作 | 第3章 |
| `TT_PrintOp` | `tt.print` | 设备端打印（调试用，等价于 CUDA `printf`） | 第3章 |
| `TT_AssertOp` | `tt.assert` | 设备端断言（调试用） | 第3章 |
| `FuncOp` | `tt.func` | 函数定义（SSACFG Region） | 第3、13章 |
| `CallOp` | `tt.call` | 函数调用 | 第3、13章 |
| `ReturnOp` | `tt.return` | 函数返回 | 第3、13章 |

---

## D.2 TTIR -- Type 定义

**文件**：`triton/include/triton/Dialect/Triton/IR/TritonTypes.td`

| def 名称 | 助记符 / 含义 | 描述 | 讨论章节 |
|----------|-------------|------|---------|
| `TT_Float` | 浮点类型集合 | F8E4M3FN, F8E5M2, F8E4M3FNUZ, F8E5M2FNUZ, F16, BF16, F32, F64 | 第3、5章 |
| `TT_Int` | 整数类型集合 | I1, I4, I8, I16, I32, I64 | 第3、5章 |
| `TT_PtrType` | `!tt.ptr<T>` | 指针类型，参数：pointeeType, addressSpace | 第3、5章 |
| `TT_Ptr` | `!tt.ptr<>` 约束 | 任意标量指针类型 | 第3章 |
| `TT_PtrTensor` | `tensor<...x!tt.ptr<T>>` | 指针张量 | 第3、5章 |
| `TT_FloatTensor` | `tensor<...xF>` | 浮点张量类型 | 第3章 |
| `TT_IntTensor` | `tensor<...xI>` | 整数张量类型 | 第3章 |
| `TT_Tensor` | `tensor<...>` | 通用张量类型（可含浮点、整数或指针） | 第3章 |
| `TT_TensorDescType` | `!tt.tensordesc<...>` | TMA 张量描述符类型，参数：shape, elementType, sharedLayout；实现 `TensorDescInterface` | 第3、8章 |

---

## D.3 TTIR -- Attribute 定义

**文件**：`triton/include/triton/Dialect/Triton/IR/TritonAttrDefs.td`

| def 名称 | 类型 | 取值 / 含义 | 讨论章节 |
|----------|------|------------|---------|
| `TT_CacheModifierAttr` | I32Enum | NONE, CA (cache at all levels), CG (cache global), WB (write-back), CS (streaming), WT (write-through), CV (volatile) | 第3、8章 |
| `TT_EvictionPolicyAttr` | I32Enum | NORMAL, EVICT_FIRST, EVICT_LAST | 第3、8章 |
| `TT_MemSemanticAttr` | I32Enum | RELAXED, ACQUIRE, RELEASE, ACQUIRE_RELEASE | 第3、8章 |
| `TT_MemSyncScopeAttr` | I32Enum | GPU, CTA, SYSTEM | 第3、8章 |
| `TT_AtomicRMWAttr` | I32Enum | AND, OR, XOR, ADD, FADD, MAX, MIN, UMAX, UMIN, XCHG | 第3、8章 |
| `TT_DescriptorReduceKindAttr` | I32Enum | ADD, MIN, MAX, INC, DEC, AND, OR, XOR | 第3章 |
| `TT_ProgramDim` | I32Enum | X, Y, Z | 第2、3章 |
| `TT_RoundingModeAttr` | I32Enum | RTZ (round toward zero), RTNE (round to nearest even) | 第3章 |
| `TT_PropagateNanAttr` | I32Enum | NONE, ALL | 第3章 |
| `TT_InputPrecisionAttr` | I32Enum | TF32, TF32x3, IEEE, BF16x3, BF16x6 | 第3、9章 |
| `TT_ScaleDotElemTypeAttr` | I32Enum | E4M3, E5M2, E2M3, E3M2, E2M1, BF16, FP16 | 第3章 |

---

## D.4 TTGIR -- Operation 定义

**文件**：`triton/include/triton/Dialect/TritonGPU/IR/TritonGPUOps.td`

| def 名称 | 助记符 | 描述 | 讨论章节 |
|----------|-------|------|---------|
| `TTG_ConvertLayoutOp` | `ttg.convert_layout` | 改变张量的 Layout/Encoding（纯元数据操作，在代码生成时转换为数据搬运指令） | 第4、6、9章 |
| `TTG_LocalAllocOp` | `ttg.local_alloc` | 分配共享内存 buffer；可接受初始化值 | 第4、8、11章 |
| `TTG_LocalDeallocOp` | `ttg.local_dealloc` | 释放共享内存 buffer | 第8、11章 |
| `TTG_LocalLoadOp` | `ttg.local_load` | 从共享内存加载为 Distributed Tensor | 第4、9章 |
| `TTG_LocalStoreOp` | `ttg.local_store` | 将 Distributed Tensor 存入共享内存 | 第4、9章 |
| `TTG_LocalGatherOp` | `ttg.local_gather` | 从共享内存中沿指定轴 gather | 第4章 |
| `TTG_LocalScatterOp` | `ttg.local_scatter` | 向共享内存中沿指定轴 scatter | 第4章 |
| `TTG_LocalAtomicScatterRMWOp` | `ttg.local_atomic_scatter_rmw` | 向共享内存中原子 scatter RMW | 第4章 |
| `TTG_AsyncCopyGlobalToLocalOp` | `ttg.async_copy_global_to_local` | 异步从 Global Memory 拷贝到 Shared Memory（通过 `cp.async` 指令） | 第4、8、10章 |
| `TTG_AsyncCommitGroupOp` | `ttg.async_commit_group` | 提交一组异步拷贝操作，使其可被 wait | 第8、10章 |
| `TTG_AsyncWaitOp` | `ttg.async_wait` | 等待异步拷贝完成（至多 `num` 组悬而未决） | 第8、10章 |
| `TTG_MemDescIndexOp` | `ttg.memdesc_index` | 对内存描述符取子视图（沿最外层维度取第 i 个） | 第4章 |
| `TTG_MemDescSubsliceOp` | `ttg.memdesc_subslice` | 对内存描述符取子视图（指定偏移量范围） | 第4章 |
| `TTG_MemDescTransOp` | `ttg.memdesc_trans` | 对内存描述符中的逻辑张量进行转置视图 | 第4章 |
| `TTG_MemDescReshapeOp` | `ttg.memdesc_reshape` | 对内存描述符中的逻辑张量进行 reshape 视图 | 第4章 |
| `TTG_MemDescReinterpretOp` | `ttg.memdesc_reinterpret` | 重解释内存描述符的类型和形状 | 第4章 |
| `TTG_BarrierOp` | `ttg.barrier` | CTA 内同步，addrSpace 位掩码指定同步的内存域（local/global_read/global_write/tensor_read/tensor_write） | 第8、10章 |
| `TTG_WarpSpecializeOp` | `ttg.warp_specialize` | 将 warp 划分为默认组和分区组，异步执行不同代码 | 第10章 |
| `TTG_WarpSpecializePartitionsOp` | `ttg.warp_specialize.partitions` | WarpSpec 中 IsolatedFromAbove 分区区域的容器 | 第10章 |
| `TTG_WarpYieldOp` | `ttg.warp_yield` | WarpSpec default region 的终止操作 | 第10章 |
| `TTG_WarpReturnOp` | `ttg.warp_return` | WarpSpec partition region 的隐式终止操作 | 第10章 |
| `TTG_WarpIdOp` | `ttg.warp_id` | 获取当前硬件 warp ID | 第10章 |
| `TTG_PredicateStageOp` | `ttg.predicate_stage` | 流水线阶段谓语操作 | 第10章 |
| `TTG_MaskOp` | `ttg.mask` | 流水线掩码操作（含 Region） | 第10章 |
| `TTG_Fp4ToFpOp` | `ttg.fp4_to_fp` | 将 packed FP4 (E2M1) 上转换为更高精度浮点 | 第3章 |
| `TTG_GlobalScratchAllocOp` | `ttg.global_scratch_alloc` | 分配全局内存临时 buffer | 第11章 |

---

## D.5 TTGIR -- Type 定义

**文件**：`triton/include/triton/Dialect/TritonGPU/IR/TritonGPUTypes.td`

| def 名称 | 助记符 | 描述 | 讨论章节 |
|----------|-------|------|---------|
| `TTG_MemDescType` | `!ttg.memdesc` | 内存描述符类型，参数：shape, elementType, encoding (Layout), memorySpace, mutableMemory, allocShape | 第4、8章 |
| `TTG_AsyncToken` | `!ttg.async.token` | 异步操作令牌，用于在异步拷贝和等待操作间建立 SSA 链接 | 第8、10章 |

---

## D.6 TTGIR -- Attribute 定义（Layout Encoding）

**文件**：`triton/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`

### Distributed Encoding（分布式编码）

| def 名称 | 助记符 | 关键参数 | 描述 | 讨论章节 |
|----------|-------|---------|------|---------|
| `BlockedEncodingAttr` | `blocked` | sizePerThread, threadsPerWarp, warpsPerCTA, order, CGALayout | 每个 warp 持有张量的连续子块；最常用的通用 Layout | 第4、6、8章 |
| `NvidiaMmaEncodingAttr` | `nvidia_mma` | versionMajor, versionMinor, warpsPerCTA, CGALayout, instrShape | NVIDIA Tensor Core MMA 输出布局。versionMajor=1 (Volta), =2 (Ampere/Hopper) | 第4、9章 |
| `AMDMfmaEncodingAttr` | `amd_mfma` | version, warpsPerCTA, instrShape, isTransposed, CGALayout, tilesPerWarp, elementBitWidth | AMD CDNA Matrix Core MFMA 输出布局。version=1 (CDNA1/gfx908), =2 (CDNA2/gfx90a), =3 (CDNA3/gfx942), =4 (CDNA4/gfx950) | 第4、9章 |
| `AMDWmmaEncodingAttr` | `amd_wmma` | version, ctaLayout (LinearLayout), isTransposed, CGALayout, instrShape | AMD RDNA Wave MMA 输出布局。version=1 (RDNA3), =2 (RDNA4), =3 (gfx1250) | 第4、9章 |
| `DotOperandEncodingAttr` | `dot_op` | opIdx (0=A, 1=B), parent (MmaEncoding), kWidth | `tt.dot` 的 A 或 B 操作数布局 | 第4、9章 |
| `SliceEncodingAttr` | `slice` | dim, parent (DistributedEncoding) | 从父 Layout 中挤出一个维度；`expand_dims` 的逆操作 | 第4章 |
| `LinearEncodingAttr` | `linear` | linearLayout (LinearLayout) | 新式 LinearLayout 编码，约束：寄存器/lane/block 基向量无 swizzle，移除广播后双射 | 第4章 |
| `GenericLinearEncodingAttr` | `generic_linear` | linearLayout (LinearLayout) | 新式 LinearLayout 编码（放宽约束），允许 warp 级 swizzle | 第4章 |

### Shared Encoding（共享内存编码）

| def 名称 | 助记符 | 关键参数 | 描述 | 讨论章节 |
|----------|-------|---------|------|---------|
| `SwizzledSharedEncodingAttr` | `swizzled_shared` | vec, perPhase, maxPhase, order, CGALayout | 带 swizzle 的共享内存布局；通过 XOR 模式减少 bank conflict | 第4、8章 |
| `PaddedSharedEncodingAttr` | `padded_shared` | intervals, paddings, linearComponent (LinearLayout) | 带 padding + 线性重排的共享内存布局；支持多级 padding 和通用线性基变换 | 第4、8章 |
| `NVMMASharedEncodingAttr` | `nvmma_shared` | swizzlingByteWidth, transposed, elementBitWidth, fp4Padded, CGALayout | MMAv3/MMAv5 的 2D blocked tiled 共享内存输入布局；符合 NVIDIA PTX wgmma 共享内存布局规范 | 第4、9章 |
| `SharedLinearEncodingAttr` | `shared_linear` | linearLayout, layoutAlignment | 基于 LinearLayout 的共享内存编码 | 第4章 |
| `AMDRotatingSharedEncodingAttr` | `amd_rotating_shared` | vec, perPhase, maxPhase, order, CGALayout | AMD 的轮转 swizzle 模式，swizzle offset 随 block 变化，减少 bank conflict | 第4、8章 |
| `PartitionedSharedEncodingAttr` | `partitioned_shared` | numPartitions, numGroups, partitionDim, partitionLayout | 跨多个独立物理分区的分片共享内存布局 | 第4、8章 |

### CGA Encoding

| def 名称 | 助记符 | 关键参数 | 描述 | 讨论章节 |
|----------|-------|---------|------|---------|
| `CGAEncodingAttr` | (嵌入在其他 Layout 中) | linearLayout (LinearLayout) | CTA 在 Cooperative Grid 中的排列映射；从 `block` 到 `dim0, dim1, ...` 的线性映射 | 第4章 |

---

## D.7 TTGIR -- Enum 定义

**文件**：`triton/include/triton/Dialect/TritonGPU/IR/TritonGPUEnums.td`

| def 名称 | 类型 | 取值 | 描述 | 讨论章节 |
|----------|------|------|------|---------|
| `TTG_AddrSpace` | I32BitEnum | None, Local, GlobalRead, GlobalWrite, TensorRead, TensorWrite, All | 地址空间位掩码枚举；用于 `ttg.barrier` 的内存同步域 | 第8章 |

---

## D.8 按章节索引

| 本书章节 | 涉及的 TableGen 定义 |
|---------|-------------------|
| 第1章 | （无直接 TableGen 对应，概述性内容） |
| 第2章 | `TT_GetProgramIdOp`, `TT_GetNumProgramsOp`, `TT_MakeRangeOp`, `TT_ProgramDim` |
| 第3章 | 所有 TTIR Ops (`TritonOps.td`)、TTIR Types (`TritonTypes.td`)、TTIR Attrs (`TritonAttrDefs.td`) |
| 第4章 | 所有 TTGIR Ops (`TritonGPUOps.td`)、TTGIR Types (`TritonGPUTypes.td`)、所有 Layout Encodings (`TritonGPUAttrDefs.td`) |
| 第5章 | TTIR Types（类型层次）、`TT_FpToFpOp`（类型转换规则） |
| 第6章 | `TTG_ConvertLayoutOp`、Layout Encoding 的兼容性检查 |
| 第7章 | （Loop Peeling 无直接 TableGen 定义，在 C++ 层实现） |
| 第8章 | `TTG_AsyncCopyGlobalToLocalOp`, `TTG_AsyncCommitGroupOp`, `TTG_AsyncWaitOp`, `TTG_BarrierOp`, `TTG_LocalAllocOp`, Shared Encoding attrs, `TT_CacheModifierAttr`, `TT_EvictionPolicyAttr` |
| 第9章 | `NvidiaMmaEncodingAttr`, `DotOperandEncodingAttr`, `AMDMfmaEncodingAttr`, TTGIR Ops（Load/Store/ConvertLayout） |
| 第10章 | `TTG_WarpSpecializeOp`, `TTG_WarpSpecializePartitionsOp`, `TTG_WarpYieldOp`, `TTG_WarpReturnOp`, `TTG_WarpIdOp`, `TTG_PredicateStageOp` |
| 第11章 | `TTG_LocalAllocOp`, `TTG_LocalDeallocOp` |
| 第12章 | （后端代码发射，无直接 TableGen 定义） |
| 第13章 | `FuncOp`, `CallOp`, `ReturnOp` |
| 第14章 | （Autotuning 在 Python 层，无 TableGen 定义） |
| 第15章 | （端到端回顾，覆盖全部） |
