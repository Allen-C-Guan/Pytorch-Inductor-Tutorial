# linalg Op 为什么能简化融合：从设计哲学到链式映射

> 源码版本：MLIR (llvm-project)，文件路径均为 workspace 根目录下的相对路径。

---

## 0. 设计思想与哲学：linalg op 是什么

### 0.1 一句话定义

`linalg` op 的本质是**完美嵌套循环 (perfect loop nest)** 的声明式表示。它把"循环结构"和"循环体计算"彻底分离——循环结构编码为属性（`indexing_maps` 和 `iterator_types`），循环体编码为 region 内的标量操作。

### 0.2 设计哲学：声明式 vs 命令式

理解 `linalg` 的关键在于对比两种表示循环的哲学：

| 维度 | 命令式（`scf.for`） | 声明式（`linalg.generic`） |
|------|---------------------|---------------------------|
| 循环结构 | 写在代码中，是 IR 的一部分 | 编码为属性，是**数据** |
| 循环变量与数组索引的关系 | 隐含在 load/store 操作中 | 显式编码为 AffineMap |
| 循环语义（parallel/reduction） | 看不出来，需要分析 | 显式标注为 `iterator_types` |
| 转换时需要做什么 | 遍历 IR、分析语句、改写代码 | 操作 AffineMap，做数学变换 |

`linalg` 的核心洞察是：**如果你把循环的"结构"从 IR 中抽出来变成数据（AffineMap 属性），那么对循环结构的变换就不再是 IR 改写，而是纯粹的数学运算。**

这就好比：用经纬度描述一座城市的位置（数据），和用"往东走 3 公里再往北走 2 公里"描述（指令）。前者可以直接做坐标变换，后者必须模拟走一遍。

### 0.3 linalg 在 MLIR 编译栈中的角色

`linalg` 位于 MLIR 编译栈的**中间层**，是代码生成 pipeline 的枢纽：

```
高层（模型描述）           mhlo / tosa
                            │
                     ┌──────▼──────┐
中间层（结构化代码生成） │  linalg      │  ← 本文的主角
                     │  + tensor    │
                     └──────┬──────┘
                            │ tiling / fusion / vectorization
                     ┌──────▼──────┐
                     │  vector      │
                     │  + scf       │
                     └──────┬──────┘
                            │ bufferization + lowering
底层（目标描述）            llvm / spv
```

`linalg` 承担的核心职责是：**在张量层级上提供一套统一的、便于分析和变换的循环表示**，使得 tiling、fusion、vectorization 等关键变换可以高效实现。

### 0.4 "完美嵌套循环"是什么

**完美嵌套循环 (perfect loop nest)** 是指所有循环语句都严格嵌套、不存在循环外的语句、也不存在非循环的控制流。形式化地：

```
for i in range(I):           ← 最外层
    for j in range(J):       ← 严格嵌套
        for k in range(K):   ← 最内层
            ...              ← 所有计算都在最内层
```

**非完美嵌套的例子**（linalg 不表示这种）：

```
for i in range(I):
    A[i] = ...               ← 在外层循环体内，不在最内层
    for j in range(J):
        B[i][j] = ...
```

完美嵌套循环的关键性质是：**循环维度形成一个单纯形的迭代空间 (iteration domain)**，可以用一个仿射映射精确描述每个操作数在这个空间中的访问模式。这正是 `indexing_maps` 能够存在的前提。

---

## 1. `linalg.generic` 语法精讲

### 1.1 完整语法

`linalg.generic` 的定义位于 `llvm-project/mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td`（第 55-224 行）。

```mlir
linalg.generic {
  indexing_maps = [ <affine_map_0>, <affine_map_1>, ..., <affine_map_N> ],
  iterator_types = [ "parallel" | "reduction" | "window", ... ]
}
ins(<input_0>, <input_1>, ... : <type_0>, <type_1>, ...)
outs(<output_0>, <output_1>, ... : <type_0>, <type_1>, ...)
{
  ^bb0(<elem_0>: <scalar_type_0>, <elem_1>: <scalar_type_1>, ...):
    <标量计算>
    linalg.yield <result_0>, <result_1>, ... : <type_0>, <type_1>, ...
}
-> (<result_tensor_types>)
```

### 1.2 逐字段详解

#### `indexing_maps`：循环变量 → 操作数索引的映射

- **类型**：`ArrayAttr<AffineMapAttr>`，每个 input 和 output 操作数各一个
- **语义**：每个 AffineMap 的**域 (domain)** 是循环变量空间，**值域 (codomain)** 是对应操作数的索引空间
- **不变量**：每个 AffineMap 的维度数 = `iterator_types` 的长度（循环层数），结果数 = 对应操作数的 rank

**示例**：矩阵乘法 `C[m,n] += A[m,k] * B[k,n]`

```
循环变量空间：(m, n, k)

操作数 A 的 map：(m, n, k) -> (m, k)    ← 在循环 (m,n,k) 下，A 用索引 (m,k) 访问
操作数 B 的 map：(m, n, k) -> (k, n)    ← 在循环 (m,n,k) 下，B 用索引 (k,n) 访问
操作数 C 的 map：(m, n, k) -> (m, n)    ← 在循环 (m,n,k) 下，C 用索引 (m,n) 访问
```

这个 map 本质上回答了一个问题：**当循环变量取某一组值时，从每个操作数中取哪个位置的元素？**

#### `iterator_types`：循环维度的数学语义

- **类型**：`ArrayAttr<enum>`，每个循环维度各一个
- **取值**：`"parallel"` / `"reduction"` / `"window"`

这三个值的区别不在于循环的"语法形式"（都是 `for i in range(...)`），而在于该循环维度与**输出张量的数学关系**：

| 类型 | 含义 | 判断方法 |
|------|------|---------|
| `"parallel"` | 该维度的不同迭代值写入输出的**不同位置** | 该维度**出现在** output 的 indexing map 中 |
| `"reduction"` | 该维度的不同迭代值写入输出的**同一位置**（做累加） | 该维度**未出现在** output 的 indexing map 中 |
| `"window"` | 滑动窗口式访问（用于卷积） | 输入索引是多个循环变量的线性组合 |

**直觉理解**：

```
C[m,n] += A[m,k] * B[k,n]

看 output C 的 indexing map：(m, n, k) -> (m, n)

m 出现在输出 map 中 → parallel   （m=0 写 C[0,:], m=1 写 C[1,:]，各写各的）
n 出现在输出 map 中 → parallel   （n=0 写 C[:,0], n=1 写 C[:,1]，各写各的）
k 不在输出 map 中   → reduction  （k=0 和 k=1 都写同一个 C[m,n]，做累加）
```

所以 `iterator_types = ["parallel", "parallel", "reduction"]`。

#### `ins(...)` / `outs(...)`：操作数

- `ins`：输入操作数（只读），类型为 `tensor` 或 `memref`
- `outs`：输出操作数（可写），类型为 `tensor` 或 `memref`
- **不变量**：`indexing_maps` 的数量 = `ins` 的数量 + `outs` 的数量

#### Region（`^bb0`）：标量计算体

- **Block 参数**：每个操作数（input + output）各提供一个**标量参数**，代表该操作数在当前循环迭代位置的元素值
- **Terminator**：`linalg.yield`，返回每个 output 位置的计算结果
- **关键设计**：region 内只有标量操作，不包含任何循环或索引逻辑

#### `doc`（可选）：文档字符串

- **类型**：`OptionalAttr<StrAttr>`
- 用于标注这个 op 的数学语义（如 `"C(m,n) += A(m,k) * B(k,n)"`）

#### `library_call`（可选）：外部库名

- **类型**：`OptionalAttr<StrAttr>`
- 指定一个外部库函数名，`linalg.generic` 可以在 lowering 时选择调用该外部函数而非展开为循环

### 1.3 不变量总结

`linalg.generic` 有严格的不变量约束（由 verifier 检查）：

1. `len(indexing_maps)` = `len(ins)` + `len(outs)`
2. 每个 AffineMap 的维度数 = `len(iterator_types)`
3. 第 i 个 AffineMap 的结果数 = 第 i 个操作数的 rank
4. Region 的 block 参数数量 = `len(ins)` + `len(outs)`
5. `linalg.yield` 的操作数数量 = `len(outs)`

### 1.4 完整示例

#### 例 1：矩阵乘法

```mlir
#matmul_accesses = [
  (m, n, k) -> (m, k),     // A 的 indexing map
  (m, n, k) -> (k, n),     // B 的 indexing map
  (m, n, k) -> (m, n)      // C 的 indexing map (output)
]
#matmul_trait = {
  indexing_maps = #matmul_accesses,
  iterator_types = ["parallel", "parallel", "reduction"]
}

linalg.generic #matmul_trait
  ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
  outs(%C: tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %mul = arith.mulf %a, %b: f32
    %sum = arith.addf %c, %mul: f32
    linalg.yield %sum: f32
} -> tensor<?x?xf32>
```

等价的显式循环：

```python
for m in range(M):         # parallel
    for n in range(N):     # parallel
        for k in range(K): # reduction
            C[m][n] += A[m][k] * B[k][n]
```

#### 例 2：elementwise add

```mlir
#add_accesses = [
  (i, j) -> (i, j),    // A
  (i, j) -> (i, j),    // B
  (i, j) -> (i, j)     // C (output)
]

linalg.generic {
  indexing_maps = #add_accesses,
  iterator_types = ["parallel", "parallel"]
} ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>)
  outs(%C: tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %sum = arith.addf %a, %b: f32
    linalg.yield %sum: f32
} -> tensor<?x?xf32>
```

#### 例 3：矩阵转置

```mlir
#transpose_accesses = [
  (m, n) -> (n, m),    // A: 注意 map 交换了维度
  (m, n) -> (m, n)     // B (output)
]

linalg.generic {
  indexing_maps = #transpose_accesses,
  iterator_types = ["parallel", "parallel"]
} ins(%A: tensor<?x?xf32>)
  outs(%B: tensor<?x?xf32>) {
  ^bb0(%a: f32, %b: f32):
    linalg.yield %a: f32
} -> tensor<?x?xf32>
```

---

## 2. 为什么能简化？—— linalg vs scf 的本质差异

### 2.1 问题场景：融合两个循环

假设我们有 producer-consumer 关系：

```
Producer:  D = transpose(A)      即 D[m,n] = A[n,m]
Consumer:  E = D + C             即 E[m,n] = D[m,n] + C[m,n]
```

融合目标：消除中间张量 D，把两个操作合并成一个循环。

### 2.2 用 `scf.for` 表示——融合需要什么

```mlir
// ---- Producer ----
scf.for %m = %c0 to %M step %c1 {
  scf.for %n = %c0 to %N step %c1 {
    %a = memref.load %A[%n, %m] : memref<?x?xf32>   // ← 注意索引
    memref.store %a, %D[%m, %n] : memref<?x?xf32>
  }
}

// ---- Consumer ----
scf.for %m = %c0 to %M step %c1 {
  scf.for %n = %c0 to %N step %c1 {
    %d = memref.load %D[%m, %n] : memref<?x?xf32>
    %c = memref.load %C[%m, %n] : memref<?x?xf32>
    %sum = arith.addf %d, %c : f32
    memref.store %sum, %E[%m, %n] : memref<?x?xf32>
  }
}
```

要融合它们，编译器需要回答以下问题：

**问题 1：循环边界是否兼容？**
- Producer 的外层循环范围是 `[0, M)`，内层是 `[0, N)`
- Consumer 的外层循环范围是 `[0, M)`，内层是 `[0, N)`
- 需要逐层比较上下界，确认匹配。这需要追踪 `%M`、`%N` 的定义链，可能涉及复杂的 SSA 分析。

**问题 2：中间张量 D 在 consumer 中如何被访问？**
- Consumer 用 `D[m, n]`，Producer 写 `D[m, n]`。需要分析 load/store 的索引表达式，确认它们访问同一位置。
- 如果索引表达式不是简单的变量（如 `D[i+1, j*2]`），分析会更加复杂。

**问题 3：融合后，producer 的操作数 A 在 consumer 的循环中如何索引？**
- Producer 用 `A[n, m]`，但融合后在 consumer 的 `(m, n)` 循环空间中，需要知道用什么索引访问 A。
- 这需要**人工推演**索引变换：在 consumer 的循环 `(m, n)` 中，原来 producer 用 `A[n, m]`，所以融合后也用 `A[n, m]`——但这个推演过程需要理解 producer 的 load 语句中的索引表达式。

**问题 4：有没有其他依赖？**
- 如果 D 被其他地方使用，不能简单融合——需要做 live range 分析。

这些问题在 `scf.for` 层面都需要**过程式代码分析**：遍历 IR、理解 load/store 的语义、匹配索引表达式、追踪 SSA 定义链。每一个步骤都是 ad-hoc 的实现，难以泛化。

### 2.3 用 `linalg.generic` 表示——融合需要什么

```mlir
// ---- Producer: D = transpose(A) ----
linalg.generic {
  indexing_maps = [
    (m, n) -> (n, m),     // A
    (m, n) -> (m, n)      // D (output)
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%A: tensor<?x?xf32>) outs(%D: tensor<?x?xf32>) { ... }

// ---- Consumer: E = D + C ----
linalg.generic {
  indexing_maps = [
    (m, n) -> (m, n),     // D (input)
    (m, n) -> (m, n),     // C
    (m, n) -> (m, n)      // E (output)
  ],
  iterator_types = ["parallel", "parallel"]
} ins(%D, %C: tensor<?x?xf32>, tensor<?x?xf32>) outs(%E: tensor<?x?xf32>) { ... }
```

回答同样的问题：

**问题 1：循环边界是否兼容？** → 不需要分析。
- `linalg.generic` 没有显式的循环边界。循环范围由操作数的 shape 和 indexing map 隐式推导（`getShapesToLoopsMap()`）。只要 indexing map 可以计算出合法的迭代范围，就自动兼容。

**问题 2：中间张量 D 如何被访问？** → 直接看 indexing map。
- Consumer 中 D 的 indexing map 是 `(m,n) -> (m,n)`，Producer 输出 D 的 map 也是 `(m,n) -> (m,n)`。信息就在属性里，不需要分析 load/store 语句。

**问题 3：融合后 A 如何索引？** → 一次 `inverse + compose` 搞定。
- 这是核心，下一节详细展开。

**问题 4：有没有其他依赖？** → SSA def-use chain 直接回答。
- 张量是不可变的 SSA 值，def-use chain 就是天然的依赖图。

### 2.4 本质差异总结

```
                scf.for                         linalg.generic
            ┌──────────────┐               ┌──────────────────┐
循环结构    │  IR 的一部分   │               │  属性 (数据)       │
            │  (需要遍历分析) │               │  (直接读取)        │
            └──────────────┘               └──────────────────┘
索引关系    │  load/store    │               │  AffineMap        │
            │  (隐含在语句中) │               │  (显式编码)        │
            └──────────────┘               └──────────────────┘
融合变换    │  遍历 IR       │               │  数学运算          │
            │  改写语句      │               │  inverse + compose │
            └──────────────┘               └──────────────────┘
```

**一句话总结**：`linalg.generic` 把循环融合从一个"IR 分析与改写问题"变成了一个"仿射映射的函数复合问题"。前者需要过程式代码分析，后者只需要初等代数运算。

---

## 3. 链式映射详解：`inverse + compose` 的数学原理

### 3.1 从一个具体例子出发

沿用前面的例子：

```
Producer:  D = transpose(A)      即 D[m,n] = A[n,m]
Consumer:  E = D + C             即 E[m,n] = D[m,n] + C[m,n]
```

融合后，我们要生成一个新的 `linalg.generic`，其循环变量空间沿用 consumer 的 `(m, n)`。关键问题是：**在这个新的循环空间中，A 的 indexing map 是什么？**

### 3.2 建立映射链条

我们有以下已知的映射关系：

```
Producer 的 result indexing map:
    prod_result_map: (m, n) → (m, n)
    语义：producer 循环变量 (m,n) → D 的索引 (m,n)

Producer 的 A 操作数 indexing map:
    prod_A_map: (m, n) → (n, m)
    语义：producer 循环变量 (m,n) → A 的索引 (n,m)

Consumer 的 D 操作数 indexing map:
    cons_D_map: (m, n) → (m, n)
    语义：consumer 循环变量 (m,n) → D 的索引 (m,n)
```

我们需要求的：

```
fused_A_map: (m, n) → ?
语义：融合后的循环变量 (m,n) → A 的索引 (?)
```

### 3.3 推导过程：三步链式映射

思路是沿着"融合后的循环 → 张量索引 → producer 循环 → A 的索引"这条链走：

```
consumer_loop ──①──> tensor_index ──②──> producer_loop ──③──> A_index
    (m, n)           (m, n)            (m, n)            (n, m)
```

#### 第 ① 步：consumer_loop → tensor_index

直接用 consumer 的 D 操作数 indexing map：

```
cons_D_map: (m, n) → (m, n)
```

含义：在 consumer 的循环变量 `(m, n)` 下，D 被访问的索引是 `(m, n)`。

#### 第 ② 步：tensor_index → producer_loop

这是关键一步——**求逆**。producer 的 result indexing map 是 `(m,n) → (m,n)`，它的逆是：

```
inv(prod_result_map): (m, n) → (m, n)
```

含义：如果知道 D 的索引是 `(m, n)`，那么 producer 的循环变量就是 `(m, n)`。

为什么要求逆？因为 producer 的 result map 是 "producer_loop → D_index"，而我们已知 D_index，要求 producer_loop，所以需要反过来。

#### 第 ③ 步：producer_loop → A_index

直接用 producer 的 A 操作数 indexing map：

```
prod_A_map: (m, n) → (n, m)
```

含义：在 producer 的循环变量 `(m, n)` 下，A 被访问的索引是 `(n, m)`。

#### 合成

把三步串联起来：

```
fused_A_map = prod_A_map ∘ inv(prod_result_map) ∘ cons_D_map
```

带入具体值：

```
cons_D_map:    (d0, d1) → (d0, d1)
inv(prod_res): (d0, d1) → (d0, d1)       ← 恒等映射的逆还是恒等
prod_A_map:    (d0, d1) → (d1, d0)

合成结果：(d0, d1) → (d1, d0)
```

验证：融合后用 `(d0, d1) → (d1, d0)` 来访问 A，即 `A[n, m]`，这和原来 producer 中的行为一致。 ✓

### 3.4 更有趣的例子：带非平凡 permutation

考虑：

```
Producer:  D = transpose(A)  其中 D[i,j] = A[j,i]
            indexing_maps = [
              (p, q) -> (q, p),    // A
              (p, q) -> (p, q)     // D (output)
            ]

Consumer:  E[i,j] = D[j,i] + C[i,j]
            indexing_maps = [
              (i, j) -> (j, i),    // D (input)  ← 注意：consumer 对 D 做了转置访问
              (i, j) -> (i, j),    // C
              (i, j) -> (i, j)     // E (output)
            ]
```

#### 求融合后 A 的 indexing map

```
① cons_D_map:      (i, j) → (j, i)

② inv(prod_res):   inv((p, q) → (p, q)) = (d0, d1) → (d0, d1)

③ prod_A_map:      (p, q) → (q, p)

合成：
  先做 ② ∘ ①: (i, j) → (j, i)   （tensor_index → producer_loop）
  再做 ③ ∘ 上一步: (j, i) → (i, j)  （producer_loop → A_index）

最终结果：(i, j) → (i, j)
```

含义：融合后，A 的访问模式变成了直接索引 `(i, j)`。两次转置互相抵消了！这就是代数化简的威力——不需要任何"模式匹配"或"特判"，纯粹的数学运算自动给出了最优结果。

### 3.5 源码对应

上述推导过程直接对应 MLIR 源码中的实现。

#### 核心：`getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp`

文件：`llvm-project/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`（第 48-75 行）

```cpp
static AffineMap getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
    OpOperand *producerOpOperand, AffineMap producerResultIndexMap,
    AffineMap fusedConsumerArgIndexMap) {

  // ② tensor_index -> producer_loop
  AffineMap invProducerResultIndexMap =
      inversePermutation(producerResultIndexMap);

  // ③ producer_loop -> producer_arg_tensor_index
  AffineMap argMap = producer.getMatchingIndexingMap(producerOpOperand);

  // ③ ∘ ②: tensor_index -> producer_arg_index
  AffineMap t1 = argMap.compose(invProducerResultIndexMap);

  // (③ ∘ ②) ∘ ①: consumer_loop -> producer_arg_index
  return t1.compose(fusedConsumerArgIndexMap);
}
```

#### `inversePermutation` 的实现

文件：`llvm-project/mlir/lib/IR/AffineMap.cpp`（第 788-810 行）

```cpp
AffineMap mlir::inversePermutation(AffineMap map) {
  SmallVector<AffineExpr, 4> exprs(map.getNumDims());
  for (const auto &en : llvm::enumerate(map.getResults())) {
    auto expr = en.value();
    if (auto d = dyn_cast<AffineDimExpr>(expr)) {
      if (exprs[d.getPosition()]) continue;
      exprs[d.getPosition()] = getAffineDimExpr(en.index(), d.getContext());
    }
  }
  // ... 如果不是满射排列，返回空 map
}
```

它的工作原理：对于排列映射 `(d0, d1) -> (d1, d0)`，把结果中的每个维度编号映射回它的位置，得到逆映射 `(d0, d1) -> (d1, d0)`。

**关键约束**：只对 permutation map 有效（即每个维度恰好出现一次）。这就是为什么融合前置条件要求 `producerResultIndexMap.isPermutation()`（[ElementwiseOpFusion.cpp:179](llvm-project/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp#L179)）。

#### `AffineMap::compose` 的实现

文件：`llvm-project/mlir/lib/IR/AffineMap.cpp`（第 554-576 行）

```cpp
AffineMap AffineMap::compose(AffineMap map) const {
  // `this` 的维度数 = `map` 的结果数
  // 将 `this` 的维度替换为 `map` 的结果表达式
  auto newMap = map.replaceDimsAndSymbols(newDims, newSymbols, ...);
  for (auto expr : getResults())
    exprs.push_back(expr.compose(newMap));
  return AffineMap::get(numDims, numSymbols, exprs, ...);
}
```

`map1.compose(map2)` 的语义是**函数复合**：先过 `map2`，再过 `map1`。即 `map1 ∘ map2`。

### 3.6 consumer_to_producer 循环映射

除了计算每个操作数的新 indexing map，融合还需要一个额外的映射：**从 consumer 循环变量到 producer 循环变量的映射**。这个映射用于在融合后的 region 中重写 `linalg.index` 操作。

文件：`llvm-project/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp`（第 436-447 行）

```cpp
// consumer loop -> tensor index
AffineMap consumerResultIndexMap = consumer.getMatchingIndexingMap(fusedOperand);
// tensor index -> producer loop
AffineMap invProducerResultIndexMap = inversePermutation(producerResultIndexMap);
// consumer loop -> producer loop
AffineMap consumerToProducerLoopsMap =
    invProducerResultIndexMap.compose(consumerResultIndexMap);
```

这就是 `inverse(producerIndexMap).compose(consumerIndexMap)` 的精确定义。

### 3.7 融合前置条件

不是所有 producer-consumer 对都可以融合。`areElementwiseOpsFusable`（[ElementwiseOpFusion.cpp:140-215](llvm-project/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp#L140-L215)）检查以下条件：

1. 两者都是 `GenericOp`
2. Producer 使用纯张量语义（无 memref aliasing 风险）
3. **Producer 的所有 iterator 都是 parallel**（不允许有 reduction）
4. 被融合的操作数是 consumer 的 input（而非 output）
5. Consumer 的对应 indexing map 的结果数 = producer 的循环层数
6. **Producer 的 result indexing map 是一个排列**（保证 `inversePermutation` 可行）
7. 如果 consumer 有 reduction 维度，融合后所有维度仍有对应的操作数定义范围

条件 3 和 6 是核心数学约束：elementwise fusion 只适用于"每个循环维度一对一映射到输出维度"的情况，这样才能通过求逆+复合来计算新的 indexing map。

---

## 4. 总结：底层原理与作用

### 4.1 简化的底层原理

`linalg` op 能简化变换的根本原因可以归纳为一个核心公式：

> **结构即数据 → 变换即运算**

1. **结构即数据**：循环的迭代空间、维度语义、操作数访问模式这些"结构信息"不再是散落在 IR 各处的隐含语义，而是被编码为 `indexing_maps`（AffineMap 数组）和 `iterator_types` 这两个**属性**。属性是数据，可以直接读取和操作。

2. **变换即运算**：对循环结构的变换不再是"分析 IR → 理解语义 → 改写语句"的过程式操作，而是"对 AffineMap 做逆运算和函数复合"的纯数学运算。`inverse + compose` 就是融合，不需要任何特判或模式匹配。

3. **正确性由数学保证**：AffineMap 的 `inverse` 和 `compose` 是良定义的数学运算（逆函数和函数复合）。只要前置条件满足（permutation map 可逆），变换的正确性由数学定理保证，无需逐一验证。

### 4.2 对比：scf.for 为什么做不到

`scf.for` 中，循环结构是 IR 的一部分：

```
scf.for 内的 load/store 语句 → 索引关系隐含在表达式中
scf.for 的上下界          → 循环范围散落在操作数中
是否有数据依赖            → 需要活跃变量分析
```

要把这些隐含信息提取出来做变换，等于在做"逆向工程"——从过程式代码中恢复出原本的数学结构。这就是为什么传统编译器的循环融合实现往往极其复杂且容易出错。

`linalg.generic` 的做法是**一开始就把数学结构保留为数据**，不让它在 IR lowering 过程中丢失。

### 4.3 更广义的启示

`linalg` 的设计哲学在编译器领域有更深层的启示：

| 传统编译器思路 | linalg 的思路 |
|---|---|
| 先把高结构信息 lowering 成低级 IR，再在低级 IR 上做分析和变换 | **尽可能久地保留高结构信息**，在高结构层级做变换，最后才 lowering |
| 变换 = 遍历 IR + 模式匹配 + 改写 | 变换 = 对结构化数据做数学运算 |
| 每种变换需要独立的实现逻辑 | 不同的变换共享同一套数学基础设施（AffineMap 运算） |

这一思想与 Halide 的 "schedule decoupled from algorithm"、TVM 的 tensor expression 有异曲同工之妙——**将"计算什么"和"如何执行"分离**，使得变换可以在干净的数学基础上进行。

`linalg` 用 MLIR 的 attribute + region 机制实现了这一点：attribute 编码"如何执行"（循环结构），region 编码"计算什么"（标量操作）。变换只触碰 attribute，不动 region。

---

## 源码引用索引

| 内容 | 文件路径 | 行号 |
|------|---------|------|
| `GenericOp` 定义 | `llvm-project/mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td` | 55-224 |
| `indexing_maps` / `iterator_types` 属性定义 | 同上 | 143-146 |
| Matmul 示例 | 同上 | 87-140 |
| Elementwise fusion 入口 | `llvm-project/mlir/lib/Dialect/Linalg/Transforms/ElementwiseOpFusion.cpp` | 339-461 |
| `getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp` | 同上 | 48-75 |
| `areElementwiseOpsFusable` 前置条件 | 同上 | 140-215 |
| `consumerToProducerLoopsMap` 计算 | 同上 | 436-447 |
| `inversePermutation` 实现 | `llvm-project/mlir/lib/IR/AffineMap.cpp` | 788-810 |
| `AffineMap::compose` 实现 | 同上 | 554-576 |
