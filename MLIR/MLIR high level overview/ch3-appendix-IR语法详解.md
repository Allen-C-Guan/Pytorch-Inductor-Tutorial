# ch3 附录：IR 语法详解

本附录对 ch3 正文中出现的五段关键 IR 代码进行逐 op 语法精讲。正文中只展示了代码，本附录补充每个操作的语法、行为与功能。五段代码对应 vectorization → unrolling → 清理单位维度 → hoisting → lowering 五个渐进 lowering 阶段。

## 纵览：五步 Lowering 的完整演进路线

### 1. 完整演进路线

以 `C = A @ B - bias`（4×4 tile 矩阵乘 + 残差减法）为例，从 `linalg.matmul` 到硬件可执行指令的完整 lowering 链：

```
linalg.matmul / linalg.generic        ← 正文的起点
         │
         │ Step 1: Vectorization（向量化）
         ▼
vector<4x4xf32> 上的 vector.contract  ← 附录一
         │
         │ Step 2: Unrolling（展开）
         ▼
vector<1x1> × vector<1x4> 的 contract ← 附录二
         │
         │ Step 3: Cast Away Unit Dims（清理单位维度）
         ▼
vector<4f32> 一维向量上的操作          ← 附录三
         │
         │ Step 4: Hoisting（提升）
         ▼
循环携带变量从张量变为向量            ← 附录四
         │
         │ Step 5: Lowering（递降）
         ▼
extract + splat + fma 硬件指令级      ← 附录五
```

每一步都是**渐进的、机械化的、正确性可保证的**变换——这是 MLIR 设计哲学的核心。

---
### 2. 每一步的设计思想、核心功能与实现方式

#### Step 1: Vectorization（向量化）

**设计思想**：从张量世界进入向量世界。张量是编程模型层的抽象（无限量、动态维度、无布局），向量是寄存器/SIMD 层的抽象（有限量、静态维度、对应硬件）。向量化是跨越这两个世界的**桥梁**。

**核心功能**：将 `linalg`/`tensor` 操作转换为 `vector` 操作。具体来说，用 `vector.transfer_read` 把张量块加载为向量，用 `vector.contract` 做向量级矩阵乘，用 `vector.transfer_write` 把结果写回张量。

**本质**：抽象层级的切换——从"数据之间的计算关系"（张量）变为"寄存器中的 SIMD 操作数"（向量）。

**如何实现**：通过统一的 `LinalgVectorizationPattern`，分析 `linalg.generic` 的 region 内容，识别出计算模式（如 `mulf + addf` = 乘加累加），然后生成对应的 `vector.contract`。`transfer_read`/`transfer_write` 作为两个世界之间的搬运操作。

---

#### Step 2: Unrolling（展开）

**设计思想**：高维虚拟向量是机器无关的抽象，硬件无法直接执行。展开的目的是把高维向量拆成低维的、接近硬件原生尺寸的向量块，为后续映射到具体硬件指令做准备。

**核心功能**：将 `vector<4x4xf32>` 上的操作拆成多个 `vector<1x4xf32>` 和 `vector<1x1xf32>` 上的操作。一个 `vector.contract` 被展开为 16 个更小的 `vector.contract`。

**本质**：问题分解（decomposition）——把一个大问题拆成多个同构的小问题，每个小问题的规模匹配硬件原生向量宽度。

**如何实现**：通过 `UnrollVectorOptions` 的 `setNativeShapeFn()` 指定硬件原生尺寸（如 contract 的 reduction 维度拆成 1、parallel 的最后一个维度保持 4）。对每个 vector op，按原生尺寸生成 `vector.extract_strided_slice` 提取子向量，执行操作后用 `vector.insert_strided_slice` 拼回。展开后 `extract_strided_slice` 的开销会被后续 canonicalization 消除。

---

#### Step 3: Cast Away Unit Dims（清理单位维度）

**设计思想**：展开后留下了大量尺寸为 1 的冗余维度（如 `vector<1x4xf32>` 中的第一维、`vector<1x1xf32>` 中的两个维度）。这些单位维度是展开的副产物，不携带有用信息，却使后续变换更复杂。

**核心功能**：消除向量类型中的前导单位维度，将 `vector<1x4xf32>` 变为 `vector<4xf32>`，`vector<1x1xf32>` 变为 `vector<1xf32>`。

**本质**：维度简化——从多维表示降为一维表示，消除展开留下的"维度碎片"，让后续变换面对更简单的 IR。

**如何实现**：通过 `populateCastAwayVectorLeadingUnitDimPatterns()`，对每个 vector op 的输入输出类型进行模式匹配——如果某个维度静态为 1，就在操作前后插入 `vector.extract`/`vector.broadcast` 来剥离或恢复该维度。由于下游 op 可能还没更新（如 `vector.contract` 仍期望二维输入），需要用 `vector.broadcast` 作为临时桥接（将 `vector<4xf32>` 提升回 `vector<1x4xf32>`）。

---

#### Step 4: Hoisting（提升）

**设计思想**：张量是不可变的 SSA 值，但通过张量传递数据意味着每次迭代都要 transfer_read（张量→向量）和 transfer_write（向量→张量），产生不必要的访存开销。提升的核心洞察是：如果循环携带的累加器可以全程保持在向量（寄存器）中，就不需要中间的张量往返。

**核心功能**：将 `scf.for` 的 `iter_args` 从 `tensor<4x4xf32>`（一个张量）变为 4 个 `vector<4xf32>`（多个向量），消除循环体内的 transfer_read/write 对。

**本质**：数据放置优化——把数据从内存（张量）提升到寄存器（向量），消除每次迭代的内存读写开销。类似于传统编译器中的标量替换（scalar replacement）。

**如何实现**：通过 `hoistRedundantVectorTransfersOnTensor()`，分析循环体内的 transfer_read/write 配对——如果 read 的索引和 write 的索引是同样的静态常量，就可以把 read 移到循环前、write 移到循环后，循环内部直接传递向量值。`scf.for` 的 `iter_args` 从一个张量拆分为多个向量，`scf.yield` 从 yield 张量变为 yield 多个向量。

---

#### Step 5: Lowering（递降）

**设计思想**：`vector.contract` 仍然是一个高层抽象——它编码了"乘法+归约+合并"这个模式。但硬件不知道什么是"contract"，硬件只知道 FMA（融合乘加）。递降的目的是把抽象操作分解为硬件可直接执行的基础指令。

**核心功能**：将 `vector.contract` 分解为 `vector.extract`（取标量）+ `vector.splat`（广播为向量）+ `vector.fma`（融合乘加）三个基础操作的序列。

**本质**：从抽象到具体——把"矩阵乘法"这个语义概念拆解为"逐元素取标量、广播、乘加"这个硬件执行序列。`fma` 是向量的最小计算单元，直接对应硬件 SIMD 指令。

**如何实现**：通过 `populateVectorContractLoweringPatterns()`，选择 lowering 策略（如 `ContractionOpToOuterProductOpLowering`）。对于 `vector<1> × vector<4> → vector<4>` 的 contract，等价于：取出 lhs 的标量元素，splat 成与 rhs 同形状的向量，然后做 fma。16 次 contract 被替换为 16 组 `extract + splat + fma`。每一步都是简单的一对一操作替换，正确性由 `vector.fma` 的语义等价性保证。

---

### 3. 五步 lowering 的宏观设计哲学

纵观五步 lowering，可以提炼出 MLIR vector 代码生成的核心设计原则：

**渐进递降（Progressive Lowering）**：不做一步到位的转换，而是分步进行，每一步只做一件事。好处是每步变换简单、可验证、可组合。代价是 IR 量膨胀，但这是可接受的——因为膨胀的 IR 只是中间产物，最终会被 canonicalization 和 dead code elimination 清理。

**关注点分离（Separation of Concerns）**：每一步解决不同层次的问题：
- Step 1 解决"进入哪个世界"（张量→向量）
- Step 2 解决"多大规模"（高维→低维）
- Step 3 解决"维度清理"（多维→一维）
- Step 4 解决"数据放在哪"（内存→寄存器）
- Step 5 解决"用什么指令"（抽象→硬件）


**机械化变换（Mechanical Transforms）**：每一步的 pattern 都是简单的、最小化的、一对一或一对少的操作替换。不需要复杂的分析或启发式算法。这使得变换易于实现、测试和维护。


---

## 附录一：向量化后（Vectorization）的 IR 语法详解

> 对应正文「向量化 (Vectorization)」一节中的 matmul 向量化后 IR。

### 本步 lowering 的设计哲学

**为什么一定要做这一步**：向量化之前，IR 处于张量层级——`linalg.matmul` 描述的是"哪个张量和哪个张量做什么运算"（计算关系），不涉及数据放在哪里、用什么指令执行。但最终代码要在硬件上跑，硬件只有寄存器和 SIMD 指令。张量和硬件之间存在一条鸿沟：张量是无限量、动态维度、无布局的抽象；硬件寄存器是有限量、固定宽度、有明确排布的物理资源。向量化就是跨越这条鸿沟的**桥梁**。它不是可选优化，而是必须的抽象层级切换——不做这一步，后续所有步骤都无从谈起，因为后面的 unrolling、hoisting、lowering 全部操作的是 `vector` 类型。

**本质**：抽象层级的切换——从"数据之间的计算关系"（张量）变为"寄存器中的 SIMD 操作数"（向量）。

**设计哲学**：用统一的 `LinalgVectorizationPattern` 处理所有 linalg op，而不是为每种 op 写独立的向量化 pattern。之所以能做到这一点，是因为所有 linalg named op（如 `linalg.matmul`、`linalg.conv_2d`）本质上都是 `linalg.generic` 的语法糖。`transfer_read`/`transfer_write` 作为两个世界之间的标准搬运操作，使得向量化本身变成一个机械化的过程：读张量→向量操作→写张量。

**功能**：将 `linalg`/`tensor` 操作转换为 `vector` 操作，生成 `vector.transfer_read`、`vector.contract`、`vector.transfer_write` 等向量级 IR。

---

### 整体结构概览

这段代码在做一件什么事（用 Python 伪代码）：

```python
# C[i,j] = A[i,:] @ B[:,j]   (4x4 结果块, A 是 4x256, B 是 256x4)
# C = C - bias
C_tile = zeros(4, 4)
for k in range(0, 256, 4):
    a_slice = A[0:4, k:k+4]       # 4x4 块
    b_slice = B[k:k+4, 0:4]       # 4x4 块
    C_tile += a_slice @ b_slice    # 矩阵乘累加
result = C_tile - bias_tile        # 残差减法
```

下面按**数据流顺序**逐个讲解每个 op。

---

### 1. `tensor.extract_slice` — 张量切片

#### 语法

```mlir
%result = tensor.extract_slice %source[<offsets>] [<sizes>] [<strides>]
    : tensor<...> to tensor<...>
```

#### 出现位置

```mlir
%14 = tensor.extract_slice ...                                    // bias 切片
%15 = tensor.extract_slice %arg5...                               // 输出切片
%17 = tensor.extract_slice ...                                    // A 矩阵切片
%18 = tensor.extract_slice ...                                    // B 矩阵切片
%25 = tensor.extract_slice %17[0, %arg6] [4, 4] [1, 1]           // 循环内 A 子块
       : tensor<4x256xf32> to tensor<4x4xf32>
%26 = tensor.extract_slice %18[%arg6, 0] [4, 4] [1, 1]           // 循环内 B 子块
       : tensor<256x4xf32> to tensor<4x4xf32>
```

#### 参数详解

| 参数 | 含义 |
|------|------|
| `%source` | 源张量 |
| `[offsets]` | 每个维度的起始偏移（支持动态值，如 `%arg6`） |
| `[sizes]` | 每个维度切出的尺寸 |
| `[strides]` | 每个维度的步幅（`[1,1]` = 连续访问） |
| `: tensor<...> to tensor<...>` | 源类型 → 结果类型 |

#### 行为

从源张量中提取一个**子张量 (sub-tensor)**。它是**纯函数式**的——不修改源张量，返回一个新的 SSA 值。

**具体例子**：

```mlir
%25 = tensor.extract_slice %17[0, %arg6] [4, 4] [1, 1]
    : tensor<4x256xf32> to tensor<4x4xf32>
```

- 源 `%17` 是 `4×256` 的张量
- 偏移 `[0, %arg6]`：第 0 维从 0 开始，第 1 维从 `%arg6` 开始
- 尺寸 `[4, 4]`：切出 `4×4` 的块
- 步幅 `[1, 1]`：连续访问
- 结果：`tensor<4x4xf32>`

等价 Python：`A[0:4, k:k+4]`

**本质**：这就是 tiling 后产生"tile"的核心操作。它**不搬运数据**——只是一个视图描述，类似于 numpy 的 slice view。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td`（`Tensor_ExtractSliceOp`）

---

### 2. `vector.transfer_write` — 向量写入张量/memref

#### 语法

```mlir
%result = vector.transfer_write %vector, %dest[%indices...]
    {in_bounds = [...]}
    : vector<...>, tensor<...>      // 或 memref<...>
```

#### 出现位置

```mlir
// 初始化输出为 0
%16 = vector.transfer_write %cst, %15[%c0, %c0]
    {in_bounds = [true, true]}
    : vector<4x4xf32>, tensor<4x4xf32>

// 循环内写回累加结果
%31 = vector.transfer_write %30, %arg7[%c0, %c0]
    {in_bounds = [true, true]}
    : vector<4x4xf32>, tensor<4x4xf32>

// 写回减法结果
%23 = vector.transfer_write %22, %19[%c0, %c0]
    {in_bounds = [true, true]}
    : vector<4x4xf32>, tensor<4x4xf32>
```

#### 参数详解

| 参数 | 含义 |
|------|------|
| `%vector` | 要写入的向量值 |
| `%dest` | 目标张量/memref |
| `[%indices]` | 写入起始位置的索引 |
| `in_bounds` | 每个向量维度是否保证不越界 |
| `: vector<...>, tensor<...>` | 向量类型, 目标类型 |

#### 行为

将一个向量写入张量/memref 中从 `indices` 开始的位置。当目标是 `tensor` 时，返回一个新的张量 SSA 值（函数式语义）；当目标是 `memref` 时，原地写入。

`in_bounds = [true, true]`：告诉编译器"这个写入一定不会越界"，编译器据此可以生成更高效的代码（不需要生成越界检查的分支）。

**具体例子**：

```mlir
%16 = vector.transfer_write %cst, %15[%c0, %c0]
    {in_bounds = [true, true]}
    : vector<4x4xf32>, tensor<4x4xf32>
```

- `%cst` 是一个 `vector<4x4xf32>` 的零向量
- 写入 `%15`（`tensor<4x4xf32>`）从 `[0, 0]` 开始的位置
- 因为向量大小 (4×4) = 张量大小 (4×4)，且 `in_bounds = true`，所以这是一次完整的覆盖写入
- 结果 `%16`：一个全零的 `tensor<4x4xf32>`

**本质**：这就是 `memset` / `fill` 的向量版本。这里用它来把输出 tile 初始化为 0。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`（`Vector_TransferWriteOp`）

---

### 3. `scf.for` + `iter_args` — 带循环携带变量的结构化循环

#### 语法

```mlir
%result = scf.for %iv = %lb to %ub step %step
    iter_args(%iter_name = %init_val) -> (result_type) {
  ... 循环体 ...
  scf.yield %next_val : type
}
```

#### 出现位置

```mlir
%19 = scf.for %arg6 = %c0 to %c256 step %c4
    iter_args(%arg7 = %16) -> (tensor<4x4xf32>) {
  ...
  scf.yield %31 : tensor<4x4xf32>
}
```

#### 参数详解

| 参数 | 含义 |
|------|------|
| `%arg6` | 循环变量（induction variable），从 `%c0` 到 `%c256`，步长 `%c4` |
| `%c0` | 下界（lower bound）= 0 |
| `%c256` | 上界（upper bound）= 256（不含） |
| `%c4` | 步长（step）= 4 |
| `iter_args(%arg7 = %16)` | 循环携带变量：名字 `%arg7`，初始值 `%16`（全零的 4×4 张量） |
| `-> (tensor<4x4xf32>)` | 循环结果类型 |

#### `iter_args` 机制——这是理解这段 IR 的关键

`iter_args` 是一个**循环携带变量 (loop-carried variable)** 机制，语义类似于函数式编程中的 `fold`：

```
%result = fold(%init, lambda(%iter, %prev): body(%iter, %prev))
```

**逐次迭代的值传递**：

```
迭代 0:  %arg6 = 0,   %arg7 = %16 (初始值，全零)
迭代 1:  %arg6 = 4,   %arg7 = %31 (上一次 yield 的值)
迭代 2:  %arg6 = 8,   %arg7 = %31 (上一次 yield 的值)
...
迭代 63: %arg6 = 252, %arg7 = %31 (上一次 yield 的值)
最终:    %19 = %31 (最后一次 yield 的值)
```

**在本段 IR 中的作用**：`%arg7` 跨迭代传递累加结果张量。每次迭代做一次 4×4 矩阵乘法并累加到 `%arg7` 上，循环结束后 `%19` 就是完整的矩阵乘法结果。

#### 行为

1. `%arg6` 从 0 开始，每次 +4，到 256 为止（共 64 次迭代：0, 4, 8, ..., 252）
2. 每次迭代中，`%arg7` 是上一次迭代 `scf.yield` 出来的值（第一次是 `%16`）
3. 循环体执行完后，`scf.yield` 把新值传给下一次迭代
4. 循环结束后，`%19` 的值是最后一次 yield 的值

**等价 Python**：

```python
result = zeros(4, 4)         # %16
for k in range(0, 256, 4):   # scf.for %arg6 = 0 to 256 step 4
    result = body(k, result)  # 每次迭代更新 result
# %19 = result (循环结束后的最终值)
```

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td`（`ForOp`）

---

### 4. `vector.transfer_read` — 从张量/memref 读取向量

#### 语法

```mlir
%vector = vector.transfer_read %source[%indices...], %padding
    {in_bounds = [...]}
    : tensor<...>, vector<...>      // 或 memref<...>, vector<...>
```

#### 出现位置

```mlir
// 读取 A 子块到向量
%27 = vector.transfer_read %25[%c0, %c0], %cst_0
    {in_bounds = [true, true]}
    : tensor<4x4xf32>, vector<4x4xf32>

// 读取 B 子块到向量
%28 = vector.transfer_read %26[%c0, %c0], %cst_0
    {in_bounds = [true, true]}
    : tensor<4x4xf32>, vector<4x4xf32>

// 读取当前累加结果到向量
%29 = vector.transfer_read %arg7[%c0, %c0], %cst_0
    {in_bounds = [true, true]}
    : tensor<4x4xf32>, vector<4x4xf32>

// 循环后读取 bias 和结果
%20 = vector.transfer_read %14[%c0, %c0], %cst_0 {in_bounds = [true, true]}
    : tensor<4x4xf32>, vector<4x4xf32>
%21 = vector.transfer_read %19[%c0, %c0], %cst_0 {in_bounds = [true, true]}
    : tensor<4x4xf32>, vector<4x4xf32>
```

#### 参数详解

| 参数 | 含义 |
|------|------|
| `%source` | 源张量/memref |
| `[%indices]` | 读取起始位置的索引 |
| `%padding` | 越界时使用的填充值（这里是 `%cst_0`，即 0.0） |
| `in_bounds` | 每个向量维度是否保证不越界 |
| `: tensor<...>, vector<...>` | 源类型 → 结果向量类型 |

#### 行为

从源张量/memref 中，从 `indices` 指定的位置开始，读取一个**与结果向量同形状的块**到向量中。

**具体例子**：

```mlir
%27 = vector.transfer_read %25[%c0, %c0], %cst_0
    {in_bounds = [true, true]}
    : tensor<4x4xf32>, vector<4x4xf32>
```

- `%25` 是一个 `tensor<4x4xf32>`（A 的一个子块）
- 从 `[0, 0]` 位置开始读取
- 结果是一个 `vector<4x4xf32>`——即把整个 4×4 张量读入 4×4 向量
- `in_bounds = [true, true]`：保证不越界，不需要边界检查
- `%cst_0`：如果越界就用 0.0 填充（但因为 in_bounds=true，这个不会用到）

**本质**：这就是把张量数据加载到 SIMD/向量寄存器的操作。等价于把 `tensor<4x4xf32>` 的全部 16 个浮点数加载到一个 `vector<4x4xf32>` 中。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`（`Vector_TransferReadOp`）

---

### 5. `vector.contract` — 向量收缩（矩阵乘法）

#### 语法

```mlir
%result = vector.contract {
    indexing_maps = [ <lhs_map>, <rhs_map>, <acc_map> ],
    iterator_types = [ "parallel" | "reduction", ... ],
    kind = #vector.kind<add>
} %lhs, %rhs, %acc
    : vector<...>, vector<...> into vector<...>
```

#### 出现位置

```mlir
%30 = vector.contract {
    indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,     // lhs (A) map
        affine_map<(d0, d1, d2) -> (d2, d1)>,     // rhs (B) map
        affine_map<(d0, d1, d2) -> (d0, d1)>      // acc (C) map
    ],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
} %27, %28, %29 : vector<4x4xf32>, vector<4x4xf32> into vector<4x4xf32>
```

#### 参数详解

| 参数 | 含义 |
|------|------|
| `%lhs` (`%27`) | 左操作数向量 `vector<4x4xf32>` |
| `%rhs` (`%28`) | 右操作数向量 `vector<4x4xf32>` |
| `%acc` (`%29`) | 累加器向量 `vector<4x4xf32>` |
| `indexing_maps` | 三个仿射映射，和 `linalg.generic` 的含义完全一致 |
| `iterator_types` | 循环维度类型，和 `linalg.generic` 的含义完全一致 |
| `kind = #vector.kind<add>` | 归约操作的类型：加法 |

#### `indexing_maps` 解读

和 `linalg.generic` 一样，`vector.contract` 也用 indexing map 描述访问模式：

```
迭代空间：(d0, d1, d2)

lhs (A) map: (d0, d1, d2) -> (d0, d2)
    → A 在维度 d0 和 d2 上变化

rhs (B) map: (d0, d1, d2) -> (d2, d1)
    → B 在维度 d2 和 d1 上变化

acc (C) map: (d0, d1, d2) -> (d0, d1)
    → C 在维度 d0 和 d1 上变化
```

结合 `iterator_types = ["parallel", "parallel", "reduction"]`：

- `d0` 是 parallel → 输出维度 0（矩阵行）
- `d1` 是 parallel → 输出维度 1（矩阵列）
- `d2` 是 reduction → 在这个维度上做内积累加

**数学含义**：

```
C[d0, d1] += Σ_{d2} A[d0, d2] * B[d2, d1]
```

这就是标准的矩阵乘法公式。

#### 行为

对于 `vector<4x4xf32>` 的输入，执行一个 **4×4 × 4×4 的矩阵乘法**，结果加上累加器 `%29`，返回 `vector<4x4xf32>`。

**等价 Python**：

```python
# %30 = %27 @ %28 + %29  (矩阵乘法 + 累加)
C = A @ B + acc
```

**注意**：`vector.contract` 是 `linalg.generic` 在向量层级的对应物——它们的 `indexing_maps` 和 `iterator_types` 语义完全一致，只是操作对象从张量变成了向量。这不是巧合，而是 MLIR 设计中**不同层级共享同一套结构化表示**的体现。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`（`Vector_ContractionOp`）

---

### 6. `scf.yield` — 循环体返回值

#### 语法

```mlir
scf.yield %values... : types...
```

#### 出现位置

```mlir
scf.yield %31 : tensor<4x4xf32>
```

#### 行为

`scf.yield` 是 `scf.for` 循环体的终止操作。它的含义取决于父操作：

- **在 `scf.for` 中**：yield 出来的值传递给**下一次迭代**的 `iter_args`；如果是最后一次迭代，yield 的值就是 `scf.for` 的返回值。
- 在本例中：`scf.yield %31` 把本次迭代计算出的累加结果 `%31` 传给下一次迭代的 `%arg7`。

**数据流**：

```
iter_args(%arg7 = %16)     ← 初始值
  ↓
迭代 0: body → scf.yield %31  → %arg7 = %31 (迭代 1 的输入)
  ↓
迭代 1: body → scf.yield %31  → %arg7 = %31 (迭代 2 的输入)
  ↓
  ...
  ↓
迭代 63: body → scf.yield %31 → %19 = %31 (循环的最终结果)
```

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/SCF/IR/SCFOps.td`（`YieldOp`）

---

### 7. `arith.subf` — 浮点减法

#### 语法

```mlir
%result = arith.subf %lhs, %rhs : type
```

#### 出现位置

```mlir
%22 = arith.subf %21, %20 : vector<4x4xf32>
```

#### 行为

浮点减法：`%21 - %20`。结果类型与操作数类型相同。

这里操作数类型是 `vector<4x4xf32>`，所以是**逐元素减法**——向量中的每个对应元素分别相减。

```python
# result = matmul_result - bias
%22 = %21 - %20   # 逐元素: result[i,j] = matmul_result[i,j] - bias[i,j]
```

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td`（`Arith_SubFOp`）

---

### 8. `tensor.insert_slice` — 将张量切片写回

#### 语法

```mlir
%result = tensor.insert_slice %source into %dest[<offsets>] [<sizes>] [<strides>]
    : tensor<...> into tensor<...>
```

#### 出现位置

```mlir
%24 = tensor.insert_slice %23 into %arg5...
```

#### 行为

`tensor.extract_slice` 的**逆操作**。将一个小张量写入大张量的指定位置，返回**新的大张量**（函数式语义，不修改原来的）。

```python
# 大张量 dest，把 source 填回去
dest_copy = dest.clone()
dest_copy[offset:offset+size] = source
return dest_copy
```

**在本段 IR 中的作用**：把计算完的 4×4 结果 tile 写回完整输出张量 `%arg5` 的对应位置。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td`（`Tensor_InsertSliceOp`）

---

### 9. `affine_map` — 仿射映射属性

#### 语法

```mlir
affine_map<(dim_vars)[symbol_vars] -> (expressions)>
```

#### 出现位置

```mlir
affine_map<(d0, d1, d2) -> (d0, d2)>    // lhs map
affine_map<(d0, d1, d2) -> (d2, d1)>    // rhs map
affine_map<(d0, d1, d2) -> (d0, d1)>    // acc map
```

#### 含义

`affine_map` 是一个从维度变量到仿射表达式的映射。在 `vector.contract` 中：

- 域 `(d0, d1, d2)` = 迭代空间的三个维度
- 结果 = 对应操作数的索引

这和 `linalg.generic` 的 `indexing_maps` **语义完全一致**——`vector.contract` 就是向量层级的 `linalg.generic`。

---

### 10. 常量引用

代码中出现了几个常量（省略了定义，但可以推断含义）：

| 常量 | 含义 |
|------|------|
| `%c0` | 整数常量 0（索引用） |
| `%c4` | 整数常量 4（步长） |
| `%c256` | 整数常量 256（上界） |
| `%cst` | 浮点向量常量，类型 `vector<4x4xf32>`，值为全零 |
| `%cst_0` | 浮点标量常量 0.0（transfer_read 的 padding 值） |

---

### 完整数据流图

把所有 op 串起来：

```
%14 = extract_slice(bias)                    // 取出 bias tile
%15 = extract_slice(output)                  // 取出输出位置
%16 = transfer_write(零向量 → %15)           // 初始化为 0
%17 = extract_slice(A_full)                  // 取出 A 的完整 slice
%18 = extract_slice(B_full)                  // 取出 B 的完整 slice

%19 = scf.for %k = 0 to 256 step 4, iter_args(累加器 = %16):
  │
  ├── %25 = extract_slice(%17[0, k][4,4])   // A[0:4, k:k+4]
  ├── %26 = extract_slice(%18[k, 0][4,4])   // B[k:k+4, 0:4]
  ├── %27 = transfer_read(%25)              // 加载 A 子块到向量
  ├── %28 = transfer_read(%26)              // 加载 B 子块到向量
  ├── %29 = transfer_read(累加器)           // 加载当前累加值
  ├── %30 = contract(%27, %28, %29)         // C += A_tile @ B_tile
  ├── %31 = transfer_write(%30 → 累加器)    // 写回累加结果
  └── scf.yield %31                         // 传递给下一次迭代

%20 = transfer_read(%14)                     // 加载 bias
%21 = transfer_read(%19)                     // 加载矩阵乘结果
%22 = arith.subf(%21, %20)                   // 残差减法
%23 = transfer_write(%22 → %19)              // 写回
%24 = insert_slice(%23 into 输出张量)        // 放回大张量
```

这段 IR 完美展示了 MLIR 代码生成的层次化策略：

| 层次 | 操作 | 对应 op |
|------|------|---------|
| 张量层级（tiling） | 切出子块 | `tensor.extract_slice` |
| 向量层级（vectorization） | 加载到向量寄存器 | `vector.transfer_read` |
| 向量层级（计算） | 矩阵乘累加 | `vector.contract` |
| 向量层级（写回） | 写回张量 | `vector.transfer_write` |
| 控制流 | k 维度循环 | `scf.for` + `iter_args` |
| 标量运算 | 减法 | `arith.subf` |
| 张量层级（写回） | 拼回大张量 | `tensor.insert_slice` |

---

## 附录二：展开后（Unrolling）的 IR 语法详解

> 对应正文「展开 (Unrolling)」一节中的 IR。这是附录一中代码的继续 lowering。

### 本步 lowering 的设计哲学

**为什么一定要做这一步**：向量化生成的 `vector<4x4xf32>` 是一个"虚拟向量"——机器无关的抽象表示。**没有任何硬件有 4×4 的二维向量寄存器**。真实硬件的向量寄存器是一维的、固定宽度的（如 x86 XMM 是 128 位 = 4 个 f32）。不展开，后续 lowering 就无法进行——`vector<4x4>` 上的 `contract` 无法翻译成任何硬件指令，因为硬件根本不存在这种维度的操作。

**为什么不在向量化时就展开**：向量化阶段不知道目标硬件的向量宽度。`vector<4x4xf32>` 是平台无关的中间表示，展开阶段才通过 `setNativeShapeFn()` 引入硬件参数。同一个向量化结果只需更换展开参数就能适配不同硬件，不需要重新向量化。

**本质**：问题分解——把一个大问题（`vector<4x4>` 上的 contract）拆成多个同构的小问题（`vector<1x1> × vector<1x4>` 上的 contract），每个小问题的规模匹配硬件原生向量宽度。

**设计哲学**：展开让我们能够把大的高维向量存入小的低维寄存器中，并产生足够的线性的向量指令来高效利用 SIMD/SIMT 流水线。展开产生的 `vector.extract_strided_slice`/`vector.insert_strided_slice` 是临时产物，会被后续 canonicalization 消除。

**功能**：将一个 `vector<4x4xf32>` 上的 `vector.contract` 展开为 16 个 `vector<1x1xf32> × vector<1x4xf32>` 上的更小 `vector.contract`。

---

### 新出现的 op：`vector.extract_strided_slice`

#### 语法

```mlir
%result = vector.extract_strided_slice %source
    { offsets = [...], sizes = [...], strides = [...] }
    : vector<...> to vector<...>
```

#### 定义位置

`llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`（`Vector_ExtractStridedSliceOp`）

#### 参数

| 参数 | 含义 |
|------|------|
| `%source` | 源向量 |
| `offsets` | 每个维度的起始偏移 |
| `sizes` | 每个维度切出的尺寸 |
| `strides` | 每个维度的步幅 |
| `: vector<...> to vector<...>` | 源向量类型 → 结果向量类型 |

#### 行为

从源向量中提取一个**子向量切片**。和 `tensor.extract_slice` 的概念完全一致，只是操作对象从张量变成了向量。

#### 本段 IR 中的两种用法

**用法 1：从 `vector<4x4xf32>` 切出 `vector<1x4xf32>`（取一行）**

```mlir
%16 = vector.extract_strided_slice %cst
    {offsets = [0, 0], sizes = [1, 4], strides = [1, 1]}
    : vector<4x4xf32> to vector<1x4xf32>
```

等价 Python：`row0 = zero_matrix[0:1, 0:4]`，即从 4×4 全零矩阵中切出第 0 行。

```mlir
%18 = vector.extract_strided_slice %cst
    {offsets = [1, 0], sizes = [1, 4], strides = [1, 1]}
    : vector<4x4xf32> to vector<1x4xf32>
```

等价 Python：`row1 = zero_matrix[1:2, 0:4]`，切出第 1 行。

**用法 2：从 `vector<1x4xf32>` 切出 `vector<1x1xf32>`（取一个元素）**

```mlir
%58 = vector.extract_strided_slice %46
    {offsets = [0, 0], sizes = [1, 1], strides = [1, 1]}
    : vector<1x4xf32> to vector<1x1xf32>
```

等价 Python：`elem = row[0:1, 0:1]`，即从 1×4 向量中取出位置 `[0,0]` 的单个元素。

```mlir
%60 = vector.extract_strided_slice %46
    {offsets = [0, 1], sizes = [1, 1], strides = [1, 1]}
    : vector<1x4xf32> to vector<1x1xf32>
```

取出位置 `[0,1]` 的元素。

#### 本质

`vector.extract_strided_slice` 是向量层级的"切片"，对应张量层级的 `tensor.extract_slice`。在 lowering 过程中用于**把高维向量拆成低维向量**，即 unrolling。

---

### 整体讲解：这段代码在做什么

#### 和上一版对比：发生了什么 lowering

上一版是这样的：

```
一次 transfer_read 加载整个 4×4 tensor → vector<4x4xf32>
一次 vector.contract: vector<4x4xf32> × vector<4x4xf32> → vector<4x4xf32>
一次 transfer_write 写回整个 4×4 结果
```

这一版变成了：

```
逐行 transfer_read 加载 tensor<4x4xf32> 的每一行 → vector<1x4xf32>（共 4 行）
逐个 extract_strided_slice 从 A 的行中取出单个元素 → vector<1x1xf32>
逐个 vector.contract: vector<1x1xf32> × vector<1x4xf32> → vector<1x4xf32>（共 16 次）
逐行 transfer_write 写回结果（共 4 行）
```

这就是 MLIR 中 **vector unrolling**（向量展开）——把 `vector<4x4xf32>` 上的操作拆成多个 `vector<1x4xf32>` 或 `vector<1x1xf32>` 上的操作，逐步靠近硬件能直接执行的粒度。

#### 逐段拆解

##### 第 1 段：逐行初始化输出为 0

```mlir
// 从 4×4 全零常量向量中，逐行切出 1×4 的行向量
%16 = extract_strided_slice %cst {offsets=[0,0], sizes=[1,4]} : vector<4x4xf32> → vector<1x4xf32>
%18 = extract_strided_slice %cst {offsets=[1,0], sizes=[1,4]} : vector<4x4xf32> → vector<1x4xf32>
%20 = extract_strided_slice %cst {offsets=[2,0], sizes=[1,4]} : vector<4x4xf32> → vector<1x4xf32>
%22 = extract_strided_slice %cst {offsets=[3,0], sizes=[1,4]} : vector<4x4xf32> → vector<1x4xf32>

// 逐行写入输出 tensor，用 iter_args 串联
%17 = transfer_write %16 → %15[%c0, %c0]    // 写第 0 行
%19 = transfer_write %18 → %17[%c1, %c0]    // 写第 1 行（基于 %17）
%21 = transfer_write %20 → %19[%c2, %c0]    // 写第 2 行（基于 %19）
%23 = transfer_write %22 → %21[%c3, %c0]    // 写第 3 行（基于 %21）
```

**等价 Python**：`output_tile = zeros(4, 4)`

**对比上一版**：上一版用一次 `transfer_write` 写整个 `vector<4x4xf32>`。现在拆成 4 次逐行写入。

##### 第 2 段：scf.for 循环内的矩阵乘累加

循环结构不变：`%arg6` 从 0 到 256，步长 4。`%arg7` 传递累加结果（初始值 `%23`）。

**2a：逐行加载 A、B、累加器的当前行**

```mlir
// 从 A 的 4×4 tile 中逐行读取
%46 = transfer_read %44[%c0, %c0] → vector<1x4xf32>    // A 的第 0 行
%47 = transfer_read %44[%c1, %c0] → vector<1x4xf32>    // A 的第 1 行
%48 = transfer_read %44[%c2, %c0] → vector<1x4xf32>    // A 的第 2 行
%49 = transfer_read %44[%c3, %c0] → vector<1x4xf32>    // A 的第 3 行

// 从 B 的 4×4 tile 中逐行读取
%50 = transfer_read %45[%c0, %c0] → vector<1x4xf32>    // B 的第 0 行
%51 = transfer_read %45[%c1, %c0] → vector<1x4xf32>    // B 的第 1 行
%52 = transfer_read %45[%c2, %c0] → vector<1x4xf32>    // B 的第 2 行
%53 = transfer_read %45[%c3, %c0] → vector<1x4xf32>    // B 的第 3 行

// 从累加器中逐行读取
%54 = transfer_read %arg7[%c0, %c0] → vector<1x4xf32>  // C 的第 0 行
%55 = transfer_read %arg7[%c1, %c0] → vector<1x4xf32>  // C 的第 1 行
%56 = transfer_read %arg7[%c2, %c0] → vector<1x4xf32>  // C 的第 2 行
%57 = transfer_read %arg7[%c3, %c0] → vector<1x4xf32>  // C 的第 3 行
```

**2b：展开的矩阵乘法——4 组 × 4 步 = 16 次 contract**

以**输出第 0 行**（`%54` → `%65`）为例，追踪 4 步 contract：

```mlir
// 步骤 1: A[0,0] × B[0,:] + C[0,:]
%58 = extract_strided_slice %46 {offsets=[0,0], sizes=[1,1]}  // A[0,0] → vector<1x1xf32>
%59 = contract(%58, %50, %54)  // A[0,0] × B的整行0 + C的第0行

// 步骤 2: A[0,1] × B[1,:] + 上一步结果
%60 = extract_strided_slice %46 {offsets=[0,1], sizes=[1,1]}  // A[0,1]
%61 = contract(%60, %51, %59)  // A[0,1] × B的整行1 + 上一步结果

// 步骤 3: A[0,2] × B[2,:] + 上一步结果
%62 = extract_strided_slice %46 {offsets=[0,2], sizes=[1,1]}  // A[0,2]
%63 = contract(%62, %52, %61)  // A[0,2] × B的整行2 + 上一步结果

// 步骤 4: A[0,3] × B[3,:] + 上一步结果
%64 = extract_strided_slice %46 {offsets=[0,3], sizes=[1,1]}  // A[0,3]
%65 = contract(%64, %53, %63)  // A[0,3] × B的整行3 + 上一步结果
```

展开为数学公式：

```
C[0,:] = C[0,:] + A[0,0]×B[0,:] + A[0,1]×B[1,:] + A[0,2]×B[2,:] + A[0,3]×B[3,:]
```

这就是 `C[0,:] = C[0,:] + A[0,:] @ B` ——**矩阵乘法第 0 行的完整计算**。

同样的模式再重复 3 遍（`%66-%73` 算第 1 行，`%74-%81` 算第 2 行，`%82-%89` 算第 3 行）。

**每个 contract 的具体语义**：

```mlir
%59 = vector.contract {...} %58, %50, %54
    : vector<1x1xf32>, vector<1x4xf32> into vector<1x4xf32>
```

- lhs = `vector<1x1xf32>`：A 的一个标量（标量用 1×1 向量表示）
- rhs = `vector<1x4xf32>`：B 的一整行
- acc = `vector<1x4xf32>`：C 的一整行
- 结果 = `vector<1x4xf32>`：`acc + scalar × row`

这本质上是一个 **标量-向量乘加**（scalar-vector multiply-accumulate）。

**2c：逐行写回结果**

```mlir
%90 = transfer_write %65 → %arg7[%c0, %c0]    // 写第 0 行结果
%91 = transfer_write %73 → %90[%c1, %c0]      // 写第 1 行结果
%92 = transfer_write %81 → %91[%c2, %c0]      // 写第 2 行结果
%93 = transfer_write %89 → %92[%c3, %c0]      // 写第 3 行结果
scf.yield %93                                 // 传给下一次迭代
```

##### 第 3 段：循环后的残差减法（同样逐行展开）

```mlir
// 逐行读取 bias 和 matmul 结果
%27-%30 = transfer_read %14 的 4 行     // bias
%31-%34 = transfer_read %26 的 4 行     // matmul 结果

// 逐行做减法
%35 = arith.subf %31, %27 : vector<1x4xf32>    // result_row0 = matmul_row0 - bias_row0
%36 = arith.subf %32, %28 : vector<1x4xf32>    // result_row1 = matmul_row1 - bias_row1
%37 = arith.subf %33, %29 : vector<1x4xf32>
%38 = arith.subf %34, %30 : vector<1x4xf32>

// 逐行写回
%39-%42 = transfer_write 逐行写回
%43 = tensor.insert_slice 写回大张量
```

---

### 两次 lowering 的对比总结

```
上一版 (vectorization 后)                 这一版 (unrolling 后)
━━━━━━━━━━━━━━━━━━━━━                    ━━━━━━━━━━━━━━━━━━━━━
transfer_read 整个 4×4 tile              transfer_read 逐行 (1×4)
    → vector<4x4xf32>                       → vector<1x4xf32> × 4

一次 vector.contract                     extract_strided_slice 取标量
  vector<4x4> × vector<4x4>              → vector<1x1xf32>
  → vector<4x4>                          + 16 次 vector.contract
                                             vector<1x1> × vector<1x4>
                                             → vector<1x4>

transfer_write 整个 4×4                  transfer_write 逐行 (1×4)
```

**本质**：一个 `vector<4x4xf32>` 上的 `contract` 被 **unrolling** 成 16 个 `vector<1x1xf32> × vector<1x4xf32>` 上的 `contract`。这是 MLIR vector dialect 的标准渐进 lowering 路径：高维虚拟向量 → 拆成低维操作 → 最终映射到硬件 SIMD 指令。

每次 lowering 都变得更"啰嗦"（IR 量膨胀），但也**更接近硬件**——硬件不会一次做 4×4 的矩阵乘法，而是做标量×向量的乘加。这个过程就是编译器把高层抽象逐步"摊开"成硬件能执行的操作。

---

## 附录三：清理高维向量后的 IR 语法详解

> 对应正文「清理高维向量」一节中的 IR。这是附录二中代码的继续 lowering。

### 本步 lowering 的设计哲学

**为什么一定要做这一步**：展开是按维度机械切割的，会忠实地保留原向量的维度结构。`vector<4x4xf32>` 沿第一维切 4 份得到 `vector<1x4xf32>`，沿第二维再切得到 `vector<1x1xf32>`。这些 "1" 是切割的副产物，不携带任何有意义的维度信息。留着它们会带来两个实际问题：(1) 下一步 hoisting 需要把循环携带变量从张量变成向量，而多维向量在 `iter_args` 中传递需要更多的拆装操作；(2) 最终的 `fma` 只能在一维向量上执行，`vector<1x4xf32>` 无法直接 `fma`。

**为什么必须在 hoisting 之前做**：正文说得很清楚——hoisting 会把向量变成循环携带的 (loop carried)，一旦完成，循环本身就成为后续变换的"边界" (barrier)。之后再想清理单位维度就需要写跨循环的 pattern，这种 pattern 难写、难读、难维护。简言之：先打扫干净，再搬家具进来。

**本质**：维度简化——从多维表示降为一维表示，消除展开留下的"维度碎片"，让后续变换面对更简单的 IR。

**设计哲学**：这一步是展开的"善后"工作。展开负责拆分，清理负责消除拆分留下的维度噪声。两步分工明确，各自保持简单。

**功能**：消除向量类型中的前导单位维度，将 `vector<1x4xf32>` 变为 `vector<4xf32>`，`vector<1x1xf32>` 变为 `vector<1xf32>`。由于下游 `vector.contract` 可能仍期望二维输入，需要用 `vector.broadcast` 作为临时桥接。

---

这段 IR 是上一版（unrolling 后）的**继续 lowering**——清理高维向量中的单位维度。和上一版对比，变化集中在一个新模式。

### 新出现的 op：`vector.broadcast`

#### 语法

```mlir
%result = vector.broadcast %source : vector<...> to vector<...>
```

#### 参数

| 参数 | 含义 |
|------|------|
| `%source` | 源向量 |
| `: vector<...> to vector<...>` | 源类型 → 结果类型 |

#### 行为

将源向量的值**广播 (broadcast)** 到更高维的向量中。源中缺失的维度通过复制来填充。

**具体例子**：

```mlir
%47 = vector.broadcast %46 : vector<4xf32> to vector<1x4xf32>
```

- `%46` 是 `vector<4xf32>`，即一个 4 元素的一维向量，如 `[b0, b1, b2, b3]`
- 结果是 `vector<1x4xf32>`，即一个 1×4 的二维向量
- 广播：在前面加一个维度 → `[[b0, b1, b2, b3]]`

等价 Python：`np.broadcast_to(row, (1, 4))` 或 `row[np.newaxis, :]`

#### 为什么需要这个 broadcast？

这是这一版 lowering 的关键所在。上一版中，B 矩阵的行被读取为 `vector<1x4xf32>`：

```
上一版: transfer_read → vector<1x4xf32>    // 直接得到二维向量
```

这一版中，transfer_read 去掉了多余的单位维度，读取为一维向量 `vector<4xf32>`：

```
这一版: transfer_read → vector<4xf32>       // 一维向量
```

但下游的 `vector.contract` 仍然要求 rhs 的类型是 `vector<1x4xf32>`（因为 contract 的 indexing map 还没更新），所以需要用 `broadcast` 把 `vector<4xf32>` 提升回 `vector<1x4xf32>`。

**这是一个过渡态**——后面的 lowering 步骤会进一步消除 `vector<1x4xf32>` 中的单位维度，最终所有向量都变成一维的。

---

### 整体讲解：这段代码在做什么

#### 和上一版对比：发生了什么 lowering

上一版（unrolling 后）是这样的：

```
transfer_read  → vector<1x4xf32>         // 二维向量，第一维是 1
extract_strided_slice → vector<1x1xf32>  // 二维标量
contract: vector<1x1> × vector<1x4> → vector<1x4>
```

这一版变成了：

```
transfer_read  → vector<4xf32>                    // 一维！去掉了单位维度
broadcast      → vector<1x4xf32>                   // 暂时升回二维给 contract 用
extract_strided_slice → vector<1xf32>             // 一维标量
contract: vector<1> × vector<1x4> → vector<4>     // 混合维度
```

这就是正文中提到的**「清理高维向量」**——用 `populateCastAwayVectorLeadingUnitDimPatterns()` 把 `vector<1x4xf32>` 消除前面的单位维度，变成 `vector<4xf32>`。

#### 逐段拆解

##### 第 1 段：初始化（全部变成一维）

```mlir
// 上一版: transfer_write %cst (vector<4x4xf32> 的切片)
// 这一版: 直接写入一维 vector<4xf32>
%16 = transfer_write %cst → %15[%c0, %c0]    : vector<4xf32>, tensor<4x4xf32>
%17 = transfer_write %cst → %16[%c1, %c0]    : vector<4xf32>, tensor<4x4xf32>
%18 = transfer_write %cst → %17[%c2, %c0]    : vector<4xf32>, tensor<4x4xf32>
%19 = transfer_write %cst → %18[%c3, %c0]    : vector<4xf32>, tensor<4x4xf32>
```

`%cst` 现在是 `vector<4xf32>` 的零向量，不再是 `vector<4x4xf32>` 的切片。`in_bounds` 从 `[true, true]` 变为 `[true]`。

##### 第 2 段：循环内计算

**2a：逐行加载——全部变成一维**

```mlir
// A 的 4 行：一维 vector<4xf32>
%42 = transfer_read %40[%c0, %c0] → vector<4xf32>
%43 = transfer_read %40[%c1, %c0] → vector<4xf32>
%44 = transfer_read %40[%c2, %c0] → vector<4xf32>
%45 = transfer_read %40[%c3, %c0] → vector<4xf32>

// B 的 4 行：一维 vector<4xf32>，但 broadcast 回二维 vector<1x4xf32>
%46 = transfer_read %41[%c0, %c0] → vector<4xf32>
%47 = broadcast %46 → vector<1x4xf32>    // ← 新！暂时升维
%48 = transfer_read %41[%c1, %c0] → vector<4xf32>
%49 = broadcast %48 → vector<1x4xf32>
...

// 累加器的 4 行：一维 vector<4xf32>
%54 = transfer_read %arg7[%c0, %c0] → vector<4xf32>
...
```

**2b：16 次 contract——维度变了**

以输出第 0 行为例：

```mlir
// 从 A 的第 0 行中逐个取元素，现在是一维 extract
%58 = extract_strided_slice %42 {offsets=[0], sizes=[1]}
    : vector<4xf32> to vector<1xf32>     // 一维标量，不再是 vector<1x1xf32>

// contract 的参数类型变了
%59 = contract %58, %47, %54
    : vector<1xf32>, vector<1x4xf32> into vector<4xf32>
    //  ↑ lhs: 1-D标量  ↑ rhs: 2-D行     ↑ result: 1-D行
```

**数学语义不变**：`C[0,:] = C[0,:] + A[0,0] × B[0,:]`。只是向量的维度表示从二维降到了一维。

**2c：逐行写回——一维**

```mlir
%90 = transfer_write %65 → %arg7[%c0, %c0]  : vector<4xf32>, tensor<4x4xf32>
%91 = transfer_write %73 → %90[%c1, %c0]
%92 = transfer_write %81 → %91[%c2, %c0]
%93 = transfer_write %89 → %92[%c3, %c0]
```

##### 第 3 段：残差减法——全部一维

```mlir
// 逐行读取（一维）
%23-%26 = transfer_read %14 的 4 行    // bias → vector<4xf32>
%27-%30 = transfer_read %22 的 4 行    // 结果 → vector<4xf32>

// 逐行减法（一维）
%31 = arith.subf %27, %23 : vector<4xf32>
...

// 逐行写回（一维）
%35-%38 = transfer_write 逐行
%39 = tensor.insert_slice 写回大张量
```

---

### 三版 lowering 的演进总结

```
第一版 (vectorization)        第二版 (unrolling)           第三版 (清理单位维度)
━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━
vector<4x4xf32>               vector<1x4xf32>              vector<4xf32>
  一次 contract                 16 次 contract                16 次 contract
  vector<4x4>×<4x4>             vector<1x1>×<1x4>            vector<1>×<1x4>
  → vector<4x4>                 → vector<1x4>                → vector<4>

extract_strided_slice 不需要   从 vector<1x4> 切出           从 vector<4> 切出
                               vector<1x1>                   vector<1>

transfer_read/write            逐行 vector<1x4>              逐行 vector<4>
  整块 4×4                                                  + broadcast 临时升维

维度层级                       2-D (单位维度还在)            1-D (单位维度已清理)
                               [1,4], [1,1]                 [4], [1]
```

**这一版的核心变化**：所有向量从二维（`vector<1x4xf32>`、`vector<1x1xf32>`）降为一维（`vector<4xf32>`、`vector<1xf32>`）。`vector.broadcast` 是清理过程中的临时桥梁——transfer_read 已经产出一维向量，但 contract 还期望二维输入，所以用 broadcast 桥接。后续 lowering 会彻底消除这些 broadcast，让 contract 也工作在纯一维向量上。

---

## 附录四：Hoisting 后的 IR 语法详解

> 对应正文「Hoisting」一节中的 IR。这是附录三中代码的继续 lowering。

### 本步 lowering 的设计哲学

**为什么一定要做这一步**：前三步之后，循环体的执行模式是：每次迭代先 `transfer_read`（从张量读取向量），再计算，最后 `transfer_write`（向量写回张量）。64 次迭代 = 64 次张量读取 + 64 次张量写入。这些内存访问完全是多余的——累加器的值每次迭代都是上一次算出来的向量，完全可以在向量（寄存器）中直接传递，根本不需要写回张量再读出来。这是五步中对运行时性能影响最大的一步——一次 transfer 对应一次实际内存访问（cache miss 延迟数百个周期），而向量计算只需 1-4 个周期。

**为什么会产生这个问题**：这是张量 SSA 语义的必然代价。张量是不可变的，要"更新"累加器就必须走 `write → 新张量 → read` 的路径。Hoisting 打破了这个循环，让数据全程停留在向量（寄存器）中。

**为什么不是所有 transfer 都能 hoist**：前提条件是索引必须是静态常量。如果索引依赖循环变量（如 `transfer_read %src[%i, 0]`），就不能 hoist——因为每次迭代读取的数据位置不同。

**本质**：数据放置优化——把数据从内存（张量）提升到寄存器（向量），消除每次迭代的内存读写开销。类似于传统编译器中的标量替换（scalar replacement）。

**设计哲学**：分析循环体内的 transfer_read/write 配对，如果配对的索引是同样的静态常量，就把 read 移到循环前、write 移到循环后，循环内部只传递向量值。

**功能**：将 `scf.for` 的 `iter_args` 从 `tensor<4x4xf32>`（一个张量）变为 4 个 `vector<4xf32>`（多个向量），消除循环体内的 transfer_read/write 对。

---

这段 IR 是上一版（清理单位维度后）的**继续 lowering**——hoisting（提升）。关键变化是循环携带变量从**张量**变成了**多个向量**。

### 新语法：多返回值的 `scf.for`

#### 语法变化

之前的 `scf.for` 只有一个 `iter_args`、返回一个值：

```mlir
%19 = scf.for ... iter_args(%arg7 = %19_init) -> (tensor<4x4xf32>) {
  ...
  scf.yield %93 : tensor<4x4xf32>
}
```

这一版有 **4 个 `iter_args`**、返回 **4 个值**：

```mlir
%22:4 = scf.for %arg6 = %c0 to %c256 step %c4
          iter_args(%arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst)
        -> (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>) {
  ...
  scf.yield %85, %77, %69, %61 : vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xf32>
}
```

#### 逐部分讲解

##### `%22:4` — 多返回值 SSA

`%22:4` 表示 `%22` 是一个有 **4 个结果的 SSA 值**。后续用 `%22#0`、`%22#1`、`%22#2`、`%22#3` 分别访问第 0、1、2、3 个结果。

```mlir
%23 = vector.transfer_write %22#3, %20[%c0, %c0]    // 第 3 个结果 → 写到第 0 行
%24 = vector.transfer_write %22#2, %23[%c1, %c0]    // 第 2 个结果 → 写到第 1 行
%25 = vector.transfer_write %22#1, %24[%c2, %c0]    // 第 1 个结果 → 写到第 2 行
%26 = vector.transfer_write %22#0, %25[%c3, %c0]    // 第 0 个结果 → 写到第 3 行
```

注意结果编号和行号的对应是**反的**：`%22#3` 对应第 0 行，`%22#0` 对应第 3 行。这是因为循环体内累加器的引用顺序导致的。

##### `iter_args(%arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst)` — 4 个循环携带变量

4 个初始值都是 `%cst`（`vector<4xf32>` 的零向量）。这意味着：
- `%arg7` = 第 3 行的累加器（初始为 0 向量）
- `%arg8` = 第 2 行的累加器
- `%arg9` = 第 1 行的累加器
- `%arg10` = 第 0 行的累加器

##### `scf.yield %85, %77, %69, %61` — yield 4 个向量

yield 的值传给下一次迭代的 `(%arg7, %arg8, %arg9, %arg10)`：

```
迭代 N 的 yield: %85 → %arg7, %77 → %arg8, %69 → %arg9, %61 → %arg10
```

---

### 整体讲解：Hoisting 做了什么

#### 核心变化：循环携带变量从张量变成向量

**上一版（清理单位维度后）**

```
iter_args(%arg7 = tensor_init)   ← 一个 tensor<4x4xf32>

循环内:
  %54 = transfer_read %arg7[%c0, %c0] → vector<4xf32>    // 从张量读取向量
  %55 = transfer_read %arg7[%c1, %c0] → vector<4xf32>
  ...
  %90 = transfer_write %65 → %arg7[%c0, %c0]              // 向量写回张量
  ...
  scf.yield %93 : tensor<4x4xf32>                          // yield 整个张量
```

数据在每次迭代中经过：**张量 → transfer_read → 向量 → 计算 → transfer_write → 张量**。每次迭代都有额外的读写开销。

**这一版（Hoisting 后）**

```
iter_args(%arg7 = %cst, %arg8 = %cst, %arg9 = %cst, %arg10 = %cst)
          ↑ 4 个 vector<4xf32>，直接作为向量传递

循环内:
  // 不再需要 transfer_read 从 %arg7 读取向量！%arg7 本身就是向量
  %55 = contract %54, %47, %arg10    // 直接用向量作为累加器
  ...
  // 不再需要 transfer_write！直接 yield 向量
  scf.yield %85, %77, %69, %61       // 直接 yield 向量
```

数据路径变成：**向量 → 计算 → 向量**。消除了每次迭代中的 transfer_read/transfer_write 往返。

#### 对比：循环体内部的变化

以输出第 0 行（`%arg10`）为例：

```mlir
// 上一版：先从张量中读取向量
%54 = transfer_read %arg7[%c0, %c0] → vector<4xf32>   // 从张量读
%59 = contract %58, %47, %54                           // 用读出的向量作为 acc

// 这一版：直接使用 iter_args 中的向量
%55 = contract %54, %47, %arg10                        // %arg10 直接就是向量！
```

循环体末尾：

```mlir
// 上一版：写回张量再 yield
%90 = transfer_write %65 → %arg7[%c0, %c0]
%91 = transfer_write %73 → %90[%c1, %c0]
%92 = transfer_write %81 → %91[%c2, %c0]
%93 = transfer_write %89 → %92[%c3, %c0]
scf.yield %93 : tensor<4x4xf32>

// 这一版：直接 yield 向量
scf.yield %85, %77, %69, %61 : vector<4xf32> × 4
```

#### Hoisting 的条件

正文中提到，hoisting 需要：
1. **配对的 transfer ops**：循环开始 `transfer_read`，循环结束 `transfer_write`
2. **索引是静态常量**：`%c0, %c1, %c2, %c3` 都是常量

满足这两个条件，就可以把 transfer_read 移到循环前、transfer_write 移到循环后，循环内部只传递向量值。

#### 循环后的写回

循环结束后，4 个向量结果需要写回张量：

```mlir
// 从多返回值中取出 4 个向量，逐行写回张量
%23 = transfer_write %22#3 → %20[%c0, %c0]    // 结果 row0 → 张量第 0 行
%24 = transfer_write %22#2 → %23[%c1, %c0]    // 结果 row1 → 张量第 1 行
%25 = transfer_write %22#1 → %24[%c2, %c0]    // 结果 row2 → 张量第 2 行
%26 = transfer_write %22#0 → %25[%c3, %c0]    // 结果 row3 → 张量第 3 行
```

残差减法：

```mlir
// 读取 bias 和 matmul 结果（现在是向量，不再是张量）
%27-%30 = transfer_read %15 的 4 行     // bias
// 直接用 %22#3 等作为 matmul 结果（向量），无需再从张量读取
%31 = arith.subf %22#3, %27             // 向量减法
%32 = arith.subf %22#2, %28
%33 = arith.subf %22#1, %29
%34 = arith.subf %22#0, %30
```

注意这里 `%22#3` 直接作为向量参与减法，不再需要先 transfer_read 再减法。

---

### 四版 lowering 的演进总结

```
第一版 (vectorization)     第二版 (unrolling)       第三版 (清理单位维度)      第四版 (hoisting)
━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━━━      ━━━━━━━━━━━━━━━━━━━━
iter_args:                 iter_args:               iter_args:                iter_args:
  tensor<4x4xf32> × 1       tensor<4x4xf32> × 1      tensor<4x4xf32> × 1      vector<4xf32> × 4

循环内累加器:               循环内累加器:             循环内累加器:              循环内累加器:
  transfer_read               transfer_read            transfer_read             直接 iter_args
  从张量→向量                  从张量→向量              从张量→向量               （已是向量）

循环内写回:                 循环内写回:               循环内写回:                循环内写回:
  transfer_write              transfer_write           transfer_write            scf.yield 向量
  向量→张量                   向量→张量                向量→张量                  （无需写回张量）

scf.yield:                 scf.yield:               scf.yield:                scf.yield:
  tensor<4x4>                 tensor<4x4>              tensor<4x4>               4 × vector<4>

每次迭代的额外开销:          额外开销:                 额外开销:                  额外开销:
  3次 read + 1次 write       12次 read + 4次 write    12次 read + 4次 write     0次 read + 0次 write
  (含初始化)                  (逐行展开)                (逐行，一维)               (全在寄存器中)
```

**Hoisting 的本质**：把循环携带变量从张量世界提升到向量世界，消除了每次迭代中的张量→向量（read）和向量→张量（write）转换。累加完全在向量（寄存器）中完成，只在循环结束后才写回张量。这对应正文中说的"取消配对的 transfer write-read ops，来避免通过张量传递数据，以避免额外访存"。

---

## 附录五：最终递降（Lowering）后的 IR 语法详解

> 对应正文「递降 (Lowering)」一节中的最终形态 IR。这是附录四中代码的继续 lowering——`vector.contract` 被彻底展开为最基础的向量操作。

### 本步 lowering 的设计哲学

**为什么一定要做这一步**：经过前四步，循环体内的计算核心仍然是 `vector.contract`——一个编码了"乘法+归约+合并"模式的抽象操作。但硬件不知道什么是 "contract"。x86 有 `VFMADD` 指令，ARM 有 `VFMA` 指令，GPU 有 `FFMA` 指令，但没有哪个硬件有 "contract" 指令。`vector.contract` 是 MLIR 虚拟向量层级的操作，没有直接对应的硬件指令，必须分解为硬件能理解的基本操作。

**为什么不在 unrolling 阶段就分解**：unrolling 解决"多大规模"，lowering 解决"用什么指令"，这是两个不同的关注点。而且 `vector.contract` 有多种 lowering 策略（outer product 序列、dot product 序列、直接映射 Tensor Core），这些策略应在 lowering 阶段根据目标硬件决定，不应在 unrolling 阶段就绑死。

**本质**：从抽象到具体——把"矩阵乘法"这个语义概念拆解为"逐元素取标量、广播、乘加"这个硬件执行序列。`fma` 是向量计算的最小不可分割单元，直接对应硬件 SIMD 指令。

**设计哲学**：每一步都是简单的一对一操作替换。`vector<1> × vector<4> → vector<4>` 的 contract 等价于：取出 lhs 标量 → splat 成同形向量 → fma。正确性由 `fma` 的语义等价性保证，不需要复杂的分析。

**功能**：将 `vector.contract` 分解为 `vector.extract`（取标量）+ `vector.splat`（广播为向量）+ `vector.fma`（融合乘加）三个基础操作的序列。

---

### 新出现的 op

#### 1. `vector.extract` — 从向量中取出一个元素

##### 语法

```mlir
%result = vector.extract %source[<index>] : vector<...>
```

##### 出现位置

```mlir
%42 = vector.extract %34[0] : vector<4xf32>
%45 = vector.extract %34[1] : vector<4xf32>
%48 = vector.extract %34[2] : vector<4xf32>
%51 = vector.extract %34[3] : vector<4xf32>
```

##### 行为

从源向量中取出指定位置的**单个元素**，结果类型是标量。这里从 `vector<4xf32>` 中逐个取出 4 个浮点数。

等价 Python：`x = vec[0]`、`x = vec[1]`。

与 `vector.extract_strided_slice` 的区别：`extract` 取出的是**标量值**（f32），`extract_strided_slice` 取出的是**子向量**（vector<1xf32>）。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`（`Vector_ExtractOp`）

---

#### 2. `vector.splat` — 将标量广播为向量

##### 语法

```mlir
%result = vector.splat %source : vector<...>
```

##### 出现位置

```mlir
%43 = vector.splat %42 : vector<4xf32>
%46 = vector.splat %45 : vector<4xf32>
```

##### 行为

将一个**标量值**复制填充到向量的每个位置，生成一个所有元素相同的向量。

**具体例子**：

```mlir
%42 = vector.extract %34[0] : vector<4xf32>   // 取出标量，比如 3.14
%43 = vector.splat %42 : vector<4xf32>         // → [3.14, 3.14, 3.14, 3.14]
```

等价 Python：`np.full(4, scalar)` 或 `[scalar] * 4`。

与 `vector.broadcast` 的区别：`splat` 从**标量到向量**，`broadcast` 从**低维向量到高维向量**。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`（`Vector_SplatOp`）

---

#### 3. `vector.fma` — 融合乘加（Fused Multiply-Add）

##### 语法

```mlir
%result = vector.fma %a, %b, %c : vector<...>
```

##### 出现位置

```mlir
%44 = vector.fma %43, %38, %arg10 : vector<4xf32>
%47 = vector.fma %46, %39, %44 : vector<4xf32>
```

##### 行为

融合乘加：`result = a * b + c`。三个操作数类型相同，结果类型相同。对向量的每个元素逐个执行。

**具体例子**：

```mlir
%44 = vector.fma %43, %38, %arg10 : vector<4xf32>
// 语义: result[i] = %43[i] * %38[i] + %arg10[i]
```

等价 Python：`result = a * b + c`（逐元素）。

**与 `vector.contract` 的关系**：`vector.fma` 是 `vector.contract` 最终 lowering 的产物。`contract` 被展开为多次 `extract + splat + fma` 的序列。`fma` 是最基础的向量计算指令，直接映射到硬件的 FMA 指令（如 x86 的 `VFMADD`、ARM 的 `VFMA`、GPU 的 `FFMA`）。

> 源码定义：`llvm-project/mlir/include/mlir/Dialect/Vector/IR/VectorOps.td`（`Vector_FMAOp`）

---

### 整体讲解：这段代码在做什么

#### 和上一版对比：`vector.contract` → `extract + splat + fma`

上一版中，矩阵乘法的核心是 `vector.contract`：

```mlir
// 上一版
%58 = extract_strided_slice %42 {offsets=[0], sizes=[1]} : vector<4xf32> → vector<1xf32>
%59 = contract %58, %47, %arg10 : vector<1xf32>, vector<1x4xf32> → vector<4xf32>
```

这一版中，`contract` 被完全展开为三步基础操作：

```mlir
// 这一版
%42 = vector.extract %34[0] : vector<4xf32>          // 取出 A[i,k] 标量
%43 = vector.splat %42 : vector<4xf32>                // 广播为向量 [a, a, a, a]
%44 = vector.fma %43, %38, %arg10 : vector<4xf32>     // a * B[k,:] + C[i,:]
```

#### 逐段拆解

##### 循环内的矩阵乘法：4 行 × 4 步 = 16 次 fma

以**输出第 0 行**（`%arg10`）为例，追踪 4 步：

```mlir
// 步骤 1: A[0,0] × B[0,:] + C[0,:]
%42 = extract %34[0]      // A 的第 0 行的第 0 个元素 → 标量 A[0,0]
%43 = splat %42            // → [A[0,0], A[0,0], A[0,0], A[0,0]]
%44 = fma %43, %38, %arg10 // result = A[0,0]×B[0,:] + C_prev[0,:]

// 步骤 2: A[0,1] × B[1,:] + 上一步结果
%45 = extract %34[1]      // A[0,1]
%46 = splat %45            // → [A[0,1], A[0,1], A[0,1], A[0,1]]
%47 = fma %46, %39, %44   // result = A[0,1]×B[1,:] + 上一步结果

// 步骤 3: A[0,2] × B[2,:] + 上一步结果
%48 = extract %34[2]      // A[0,2]
%49 = splat %48
%50 = fma %49, %40, %47

// 步骤 4: A[0,3] × B[3,:] + 上一步结果
%51 = extract %34[3]      // A[0,3]
%52 = splat %51
%53 = fma %52, %41, %50
```

展开为数学公式：

```
C[0,:] = A[0,0]×B[0,:] + A[0,1]×B[1,:] + A[0,2]×B[2,:] + A[0,3]×B[3,:] + C_init[0,:]
```

同样的模式重复 3 遍处理第 1-3 行（`%arg9`、`%arg8`、`%arg7`）。

##### 循环后的残差减法

与上一版完全一致（hoisting 后结构不变）：

```mlir
%19-%22 = transfer_read %15 的 4 行     // bias
%23-%26 = arith.subf %18#3-%18#0 - bias  // 残差减法
%27-%30 = transfer_write 逐行写回
%31 = tensor.insert_slice 写回大张量
```

---

### 五版 lowering 的完整演进

```
第一版              第二版              第三版              第四版              第五版 (最终)
(vectorization)     (unrolling)        (清理单位维度)      (hoisting)         (lowering)
━━━━━━━━━━━━       ━━━━━━━━━━━━       ━━━━━━━━━━━━━━      ━━━━━━━━━━━━       ━━━━━━━━━━━━
一次                16 次               16 次               16 次              16 组
 vector.contract     contract            contract            contract           extract+splat+fma
 4x4 × 4x4          1x1 × 1x4           1 × 1x4             1 × 1x4            scalar × 4f32
 → 4x4              → 1x4               → 4f32              → 4f32             → 4f32

维度: 2-D           维度: 2-D           维度: 混合           维度: 1-D          维度: 1-D
 vector<4x4>        vector<1x4>         broadcast 桥接       vector<4>          vector<4>
                    vector<1x1>

循环携带:            循环携带:           循环携带:            循环携带:          循环携带:
 tensor<4x4>         tensor<4x4>        tensor<4x4>          4 × vector<4>      4 × vector<4>

循环内 transfer:     循环内 transfer:    循环内 transfer:     循环内 transfer:   循环内 transfer:
 read+write          read+write          read+write           无                 无
 (3+1次)             (12+4次)            (12+4次)             (寄存器中)         (寄存器中)

计算核心:           计算核心:           计算核心:            计算核心:          计算核心:
 vector.contract     vector.contract     vector.contract      vector.contract    vector.fma
 (高维抽象)          (展开但仍抽象)       (混合维度)           (纯1-D向量)        (硬件指令级)
```

**最终版的本质**：`vector.contract` 被彻底分解为 `extract`（取标量）+ `splat`（广播为向量）+ `fma`（融合乘加）三个最基础的操作。这三个操作都可以**直接映射到硬件 SIMD 指令**——`extract` → 寄存器元素选取，`splat` → 广播指令，`fma` → FMA 指令。至此，从高层 `linalg.matmul` 到硬件可执行指令的完整 lowering 链完成。
