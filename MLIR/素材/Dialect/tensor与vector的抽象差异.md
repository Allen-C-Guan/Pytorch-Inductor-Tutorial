# Tensor 与 Vector 的抽象差异：为什么 Tensor 需要 read/write 而 Vector 不需要

> 本文档回答一个核心问题：为什么在 MLIR 的 lowering 过程中，tensor 必须不停地 `transfer_read` / `transfer_write`，而 vector 却可以在 SSA 链中直接传递？

---

## 两种抽象的本质

### Tensor：容器抽象（Container Abstraction）

tensor 的设计意图是**数据容器**——它描述"一块多维数据存在那里"，但你**不能直接对它做计算**。

```
tensor<4x4xf32>    // "一块 4×4 的浮点数据"
```

在 MLIR 中，你不能写 `%c = arith.addf %A, %B : tensor<4x4xf32>`。张量没有 element-wise 的算术操作。要对张量做计算，你必须：

1. 先从张量中**提取**数据到可以计算的形式（`transfer_read` → 向量）
2. 在向量上做计算（`fma`、`addf` 等）
3. 把计算结果**放回**张量（`transfer_write` → 张量）

这就是 read/write 循环的**根源**——tensor 是一个只能"存取"的容器，不是计算单元。

### Vector：计算单元抽象（Compute Unit Abstraction）

vector 的设计意图是**计算单元**——它描述"一组可以直接运算的数据"。

```
vector<4xf32>      // "4 个浮点数，可以直接做 SIMD 运算"
```

在 MLIR 中，你可以直接写 `%c = arith.addf %a, %b : vector<4xf32>`。向量有完整的 element-wise 算术操作：`arith.addf`、`arith.mulf`、`vector.fma` 等。计算结果**直接就是向量**，不需要经过任何中间容器。

## 是什么特性造成了 read/write 的差异？

差异的根源在于：**tensor 上没有计算操作，只有结构化操作（linalg ops）和存取操作（transfer ops）。**

```
                tensor                         vector
            ┌──────────────┐               ┌──────────────┐
可用的操作    │ linalg.*     │               │ arith.*      │
            │ transfer_read│               │ vector.*     │
            │ transfer_write│              │ math.*       │
            │ extract_slice│               │              │
            │ insert_slice │               │ 直接计算 ✓    │
            ├──────────────┤               ├──────────────┤
计算方式     │ 不能直接算 ✗  │               │ 直接计算 ✓    │
            │ 必须先取出    │               │              │
            │ → 计算       │               │              │
            │ → 放回去     │               │              │
            └──────────────┘               └──────────────┘
```

用 Python 类比：

```python
# tensor 像 numpy 数组，但不能直接做算术
A = Tensor([4, 4], dtype=f32)          # 一个容器
B = Tensor([4, 4], dtype=f32)
# C = A + B   ← 不允许！tensor 没有算术操作

# 必须走"取出 → 计算 → 放回"的流程
a_vec = transfer_read(A, [0, 0])       # 取出数据
b_vec = transfer_read(B, [0, 0])
c_vec = fma(a_vec, b_vec, acc)          # 在 vector 上计算
C = transfer_write(c_vec, C, [0, 0])   # 放回容器

# vector 像直接可运算的变量
a = Vector([1.0, 2.0, 3.0, 4.0])       # 一个计算单元
b = Vector([5.0, 6.0, 7.0, 8.0])
c = a + b                               # 直接计算，结果就是 vector
d = c * 2.0                             # 继续计算，不需要任何中间步骤
```

## 为什么 SSA 语义下 tensor 必须走 read/write

两者其实都是 **SSA 值语义（value semantics）**——不可变，每次"修改"都产生新值。但 tensor 的 SSA 链条天然更长：

```
Tensor 的 SSA 链（每次迭代）:

%tensor_v0                                      ← 初始张量
  │
  ├─ transfer_read ──→ %vec0                    ← 取出
  │                      │
  │                    fma ──→ %vec_result      ← 计算（在 vector 上）
  │                                │
  ├─ transfer_write ──→ %tensor_v1              ← 放回（新张量）
  │
  ├─ transfer_read ──→ %vec1                    ← 下次迭代又得取出
  │                      │
  │                    fma ──→ %vec_result2
  │                                │
  ├─ transfer_write ──→ %tensor_v2              ← 又放回
  ...

Vector 的 SSA 链（hoisting 后）:

%vec_init                                       ← 初始向量
  │
  ├─ fma ──→ %vec_result                        ← 直接计算
  │             │
  │           fma ──→ %vec_result2              ← 继续计算
  │                      │
  │                    fma ──→ %vec_result3     ← 继续
  ...
```

关键区别：**vector 的 SSA 链可以直接传递计算结果，tensor 的 SSA 链每传递一次都要经过"取出→放回"。**

这是 tensor 作为"容器抽象"的直接后果——容器的设计意图是"存放数据"，所以你要用数据就得取出来，要存结果就得放回去。vector 作为"计算单元抽象"，它的设计意图就是"直接参与运算"，所以计算结果本身就是 vector，无需中间步骤。

## 为什么这样设计？意义是什么？

### Tensor 的"纯容器"设计带来的好处

1. **高层变换的自由度**：因为 tensor 不绑定具体的存储位置和布局，编译器可以自由地做 fusion、tiling 等变换，而不需要考虑内存分配。两个 `linalg` op 如果操作同一个 tensor，编译器可以轻松地把它们融合成一个 op——因为 tensor 是抽象的，融合只是改变 indexing map，不涉及物理数据搬运。

2. **正确性易于保证**：tensor 是不可变的 SSA 值，没有 aliasing 问题。两个不同的 tensor 值一定指向不同的数据。这让数据流分析和依赖分析极其简单——只要看 SSA def-use chain 就够了。

3. **与内存解耦**：tensor 刻意不定义布局（layout）、步幅（stride）、对齐（alignment）。这使得 bufferization（tensor → memref 转换）可以被推迟到最后，在此之前所有变换都在"纯粹的数学世界"中进行。

### Vector 的"计算单元"设计带来的好处

1. **直接映射到硬件**：vector 的大小是静态的，直接对应硬件寄存器宽度。`vector.fma` 直接映射到硬件 FMA 指令。没有中间抽象层。

2. **SSA 链直接传递**：vector 值可以在 SSA 链中直接传递（包括作为 `scf.for` 的 `iter_args`），不需要经过任何"存取"步骤。这就是 hoisting 能消除 read/write 的前提条件——vector 本身就是计算结果，不需要容器中转。

### 两者配合的设计意义

```
                高层变换的自由度            低层映射的精确性
                ◄──────────────────────────────────────────►
            ┌─────────────┐                           ┌─────────────┐
            │   tensor     │                           │   vector     │
            │              │                           │              │
            │  纯容器抽象   │    transfer_read/write    │  纯计算抽象   │
            │  不关心布局   │ ◄──────────────────────► │  直接可计算   │
            │  无限量      │    （两个世界的桥梁）       │  对应寄存器   │
            │  动态维度    │                           │  静态维度    │
            └─────────────┘                           └─────────────┘
```

如果只有 tensor，编译器可以自由做高层变换，但无法生成可执行代码。

如果只有 vector，编译器可以直接生成硬件指令，但缺乏做 fusion、tiling 等高层变换所需的信息。

**两者配合使得 MLIR 可以在高层享受变换自由度，在低层享受映射精确性，通过 transfer_read/write 作为桥梁连接两个世界。而 hoisting 的本质就是：一旦数据进入了 vector 世界，就不再需要退回 tensor 世界。**
