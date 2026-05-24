# Triton：面向分块神经网络计算的中间语言与编译器

**作者：** Philippe Tillet（哈佛大学）, H. T. Kung（哈佛大学）, David Cox（哈佛大学、IBM）

**发表：** MAPL '19 (Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages), 2019年6月22日，美国亚利桑那州凤凰城

---

## 摘要

深度学习领域中创新研究思想的验证和部署，往往受限于某些基本原语是否有高效的计算内核。特别是，那些无法利用现有厂商库（如 cuBLAS、cuDNN）的操作，除非由专家编写自定义实现，否则面临着设备利用率低下的风险——而专家级实现通常以牺牲可移植性为代价。因此，开发新的编程抽象，以最小的性能代价来指定自定义的深度学习工作负载，已变得至关重要。

我们提出了 Triton，一种以**分块（tile）**——即静态形状的多维子数组——为核心概念的语言和编译器。我们的方法围绕以下两点展开：(1) 一种基于 C 的语言和基于 LLVM 的中间表示（IR），用于以参数化分块变量上的操作来表达张量程序；(2) 一组新颖的分块级优化 pass，用于将这些程序编译为高效的 GPU 代码。我们展示了 Triton 如何用于构建矩阵乘法和卷积内核的可移植实现，其性能可与手工调优的厂商库（cuBLAS / cuDNN）相媲美，或用于高效实现诸如移位卷积（shift convolutions）等最新研究思想。

**CCS 概念：** 计算方法学 → 并行计算方法学

**关键词：** 编译器；神经网络；GPU

**ACM 引用格式：** Philippe Tillet, H. T. Kung, and David Cox. 2019. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. In *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL '19)*, June 22, 2019, Phoenix, AZ, USA. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3315508.3329973

> **版权声明：** 允许个人或课堂教学使用本作品的全部或部分内容制作数字或纸质副本，无需付费，前提是副本不得为盈利或商业目的而制作或分发，且副本须载有本声明及首页的完整引用。属于作者以外其他人所有的本作品组成部分的版权必须得到尊重。允许在注明出处的情况下进行摘要。如需以其他方式复制、重新发布、发布到服务器或分发给列表，需事先获得特定许可和/或付费。许可请求请发送至 permissions@acm.org。
>
> MAPL '19, 2019年6月22日，美国亚利桑那州凤凰城
>
> © 2019 版权归属作者。出版权授权给 ACM。
>
> ACM ISBN 978-1-4503-6719-6/19/06...$15.00
>
> https://doi.org/10.1145/3315508.3329973

---

## 1. 引言

深度神经网络（DNN）近年来的复兴，在很大程度上得益于可编程并行计算设备的广泛普及[24]。特别是，众核架构（如 GPU）性能的持续提升发挥了基础性作用，使研究人员和工程师能够利用越来越多的数据探索日益增长的、规模越来越大的模型。这一努力得到了厂商库（cuBLAS、cuDNN）集合的支持，这些库旨在尽可能快地将最新的硬件创新带给实践者。然而不幸的是，这些库仅支持一组受限的张量操作，将新原语的实现留给了领域专家[13, 17, 25]。

这一观察推动了各种面向 DNN 的领域特定语言（DSL）的发展，它们基于多面体机制（如 Tensor Comprehensions [43]）和/或循环综合技术（如 Halide [37]、TVM [10] 和 PlaidML [22]）。但是，尽管这些系统在某些问题类别（如深度可分离卷积，例如 MobileNet [20]）上通常表现良好，它们在实践中往往比厂商库慢得多（例如参见图 1），并且缺乏足够的表达能力来实现那些无法在嵌套循环中用仿射数组索引直接指定的结构化稀疏模式[28, 31, 47]。

![图1](图1说明：各种 C = AB^T 实现与 Roofline [46] 模型的性能对比（NVIDIA GeForce GTX1070），其中 A ∈ R^{1760×1760}，B ∈ R^{N×1760}，N 调节算术强度。)

这些问题常常通过使用微内核[11, 21]——即手写的分块级内联函数——来解决，但这种解决方案需要大量手工劳动且缺乏可移植性。虽然最近提出了一些用于分块的高级编程抽象[23, 41]，但底层的编译器后端仍然缺乏对分块级操作和优化的支持。为此，我们提出了 Triton（图 2），一个开源的^1^ 中间语言和编译器，用于指定分块程序并将其编译为高效的 GPU 代码。

> ^1^ http://triton-lang.org

![图2](图2说明：Triton 整体概览。包含 Triton-C → Triton-IR → Triton-JIT → 机器码的流程，以及自动调优器和基准测试模块。与现有 DSL 的接口不在本文讨论范围内。)

本文的主要贡献总结如下：

- **Triton-C（第 3 节）：** 一种类似 C 的语言，用于以参数化分块变量表达张量程序。该语言的目的是为现有的 DNN 转译器（如 PlaidML、Tensor Comprehensions）和熟悉 CUDA 的程序员提供一个稳定的接口。代码清单 1 展示了一个简单矩阵乘法任务的 Triton-C 源代码。
- **Triton-IR（第 4 节）：** 一个基于 LLVM 的中间表示（IR），提供适合分块级程序分析、变换和优化的环境。代码清单 5 展示了一个修正线性单元（ReLU）函数的 Triton-IR 代码。在此工作中，Triton-IR 程序在解析过程中直接由 Triton-C 构建，但将来也可以探索从嵌入式 DSL 或更高级别的 DNN 编译器（如 TVM）自动生成。
- **Triton-JIT（第 5 节）：** 一个即时（JIT）编译器和代码生成后端，用于将 Triton-IR 程序编译为高效的 LLVM 位码。包括：(1) 一组分块级的、与机器无关的 pass，旨在独立于任何编译目标来简化输入计算内核；(2) 一组分块级的与机器相关的 pass，用于生成可在 GPU 上高效运行的 LLVM-IR；(3) 一个自动调优器，用于优化与上述 pass 相关的任何元参数。
- **数值实验（第 6 节）：** 对 Triton 的数值评估，证明其能够：(1) 生成与 cuBLAS 性能相当的矩阵乘法实现，并且在循环和 transformer 神经网络上比替代 DSL 快最多 3 倍；(2) 在不损失性能的情况下重新实现 cuDNN 用于密集卷积的 IMPLICIT_GEMM 算法；(3) 为诸如 shift-conv [47] 模块等新颖的研究思想创建高效实现。

本文将以对现有相关文献的简要分析作为开篇（第 2 节），并以总结和未来工作方向作为结尾（第 7 节）。

**代码清单 1：** C = A × B^T 在 Triton-C 中的实现。Triton 特有的关键字以紫色显示。
```c
// 分块形状是参数化的，
// 可由编译后端优化
const tunable int TM = {16, 32, 64, 128};
const tunable int TN = {16, 32, 64, 128};
const tunable int TK = {8, 16};
// C = A * B.T
kernel void matmul_nt(float *a, float *b, float *c,
                       int M, int N, int K) {
  // 索引的 1D 分块
  int rm[TM] = get_global_range(0);
  int rn[TN] = get_global_range(1);
  int rk[TK] = 0 ... TK;
  // 累加器的 2D 分块
  float C[TM, TN] = 0;
  // 指针的 2D 分块
  float *pa[TM, TK] = a + rm[:, newaxis] + rk * M;
  float *pb[TN, TK] = b + rn[:, newaxis] + rk * K;
  for (int k = K; k >= 0; k -= TK) {
    bool check_k[TK] = rk < k;
    bool check_a[TM, TK] = (rm < M)[:, newaxis] && check_k;
    bool check_b[TN, TK] = (rn < N)[:, newaxis] && check_k;
    // 加载分块操作数
    float A[TM, TK] = check_a ? *pa : 0;
    float B[TN, TK] = check_b ? *pb : 0;
    // 累加
    C += dot(A, trans(B));
    // 更新指针
    pa = pa + TK * M;
    pb = pb + TK * N;
  }
  // 写回累加器
  float *pc[TM, TN] = c + rm[:, newaxis] + rn * M;
  bool check_c[TM, TN] = (rm < M)[:, newaxis] && (rn < N);
  @check_c *pc = C;
}
```

---

## 2. 相关工作

框架[1, 9, 36]和深度学习库的存在，对新型神经网络架构和算法的涌现起到了至关重要的作用。然而，尽管用于线性代数编译器的分析性[5, 48]和经验性[6, 30]启发式方法有所进展，这些软件仍然不可避免地依赖手工优化的子程序（如 cuBLAS 和 cuDNN）。这导致了各种用于 DNN 的 DSL 和编译器的发展，它们通常基于以下三种不同方法之一：

- **张量级 IR** 已被 XLA [16] 和 Glow [38] 用于通过模式匹配将张量程序转换为预定义的 LLVM-IR 和 CUDA-C 操作模板（例如张量收缩、逐元素操作等）。
- **多面体模型** [18] 已被 Tensor Comprehensions (TC) [43] 和 Diesel [14] 用于参数化并自动化一个或多个 DNN 层到 LLVM-IR 和 CUDA-C 程序的编译。
- **循环综合器** 已被 Halide [37] 和 TVM [10] 用于将张量计算转换为循环嵌套，这些循环嵌套可以使用用户定义的（尽管可能参数化的[11]）调度进行手动优化。

相比之下，Triton 依赖于在传统编译管线中添加分块级操作和优化。这种方法提供了：(1) 比 XLA 和 Glow 更大的灵活性；(2) 与 TC 和 Diesel 相反，支持非仿射张量索引；(3) 自动推断可能的执行调度，而无需像 Halide 或 TVM 那样手动指定。Triton 的优势是以增加编程工作量作为代价的——参见代码清单 2，了解这些 DSL 中矩阵乘法的实现。

**代码清单 2：** C = A × B^T 在 TF、PlaidML、TC 和 TVM 中的实现
```c
C = tf.matmul(A, tf.transpose(B))                    // TF
C[i, j: I, J] = +(A[i, k] * B[j, k]);               // PlaidML
C(i, j) +=! A(i, k) * B(j, k)                        // TC
tvm.sum(A[i, k] * B[j, k], axis=k)                   // TVM
```

---

## 3. Triton-C 语言

Triton-C 的目的是为现有的（以及未来的）DNN 转译器以及熟悉底层 GPU 编程的程序员提供一个稳定的前端。在本节中，我们将描述 Triton-C 的类 CUDA 语法（第 3.1 节）、类 NumPy [35] 语义（第 3.2 节）及其"单程序多数据"（SPMD）编程模型（第 3.3 节）。

### 3.1 语法

Triton-C 的语法基于 ANSI C（更具体地说是 CUDA-C），但进行了修改和扩展（见代码清单 3）以适应接下来两个小节中描述的语义和编程模型。这些变化分为以下几类：

**分块声明：** 我们添加了用于声明多维数组的特殊语法（如 `int tile[16, 16]`），以强调其与 ANSI C 中嵌套数组（如 `int tile[16][16]`）在语义上的区别。分块形状必须是常量，但也可以使用 `tunable` 关键字使其参数化。一维整数分块可以使用省略号初始化（如 `int range[8] = 0 ... 8`）。

**代码清单 3：** Triton-C 的语法扩展（假定某些 C 结构的存在以蓝色显示）
```
// 广播语义
slice       : ':' | 'newaxis'
slice_list   : slice | slice_list ',' slice
slice_expr   : postfix_expr | expr '[' slice_list ']'
// 范围初始化
constant_range : expr '...' expr
// 内建函数
global_range  : 'get_global_range' '(' constant ')'
dot          : 'dot' '(' expr ',' expr ')'
trans        : 'trans' '(' expr ',' expr ')'
intrinsic_expr : global_range | dot | trans
// 谓词化
predicate_expr : '@' expr
// 抽象声明符的分块扩展
abstract_decl : abstract_decl | '[' constant_list ']'
// C 表达式的扩展
expr         : expr | constant_range | slice_expr | intrinsic_expr
// C 说明符的扩展
storage_spec : storage_spec | 'kernel'
type_spec    : type_spec | 'tunable'
// C 语句的扩展
statement    : statement | predicate_expr statement
```

**内建函数：** 虽然为逐元素数组操作（+、-、&&、* 等）保留了通用的 C 语法，但添加了各种内建函数（`dot`、`trans`、`get_global_range`）以支持分块语义（第 3.2.1 节）和 SPMD 编程模型。

**广播：** N 维分块可以使用 `newaxis` 关键字和通常的切片语法沿任意轴广播（例如 `int broadcast [8, 8] = range[:, newaxis]` 用于堆叠列）。注意，切片分块以获取标量或子数组在其他情况下是被禁止的。

**谓词化：** 分块操作内的基本控制流（第 4.3 节）通过使用带 `@` 前缀的谓词化语句来实现。

### 3.2 语义

#### 3.2.1 分块语义

Triton-C 中内建的分块类型和操作（即分块语义）的存在提供了两个主要好处。第一，它通过隐藏与分块内部内存合并[12]、缓存管理[32]和专用硬件利用[27]相关的重要性能细节，简化了张量程序的结构。第二，它为编译器自动执行这些优化打开了大门，如第 5 节所述。

#### 3.2.2 广播语义

Triton-C 中的分块是强类型的，在某种意义上，某些指令静态地要求其操作数遵守严格的形状约束。例如，不能将标量加到数组上，除非先对其进行适当的广播。广播语义[35]提供了一组规则来隐式执行这些转换（示例见代码清单 4）：

1. **填充（Padding）：** 将较短操作数的形状用 1 进行左填充，直到两个操作数具有相同的维度。
2. **广播（Broadcasting）：** 将两个操作数的内容按需复制多次，直到它们的形状相同；如果无法做到这一点，则发出错误。

**代码清单 4：** 广播语义实践
```c
int a[16], b[32, 16], c[16, 1];
// a 先被重塑为 [1, 16]，然后被广播到 [32, 16]
int x_1[32, 16] = a[newaxis, :] + b;
// 同上，但隐式完成
int x_2[32, 16] = a + b;
// a 先被重塑为 [1, 16]
// a 被广播到 [16, 16]
// c 被广播到 [16, 16]
int y[16, 16] = a + c;
```

### 3.3 编程模型

GPU 上 CUDA [33] 代码的执行由一个 SPMD [4] 编程模型支持，在该模型中，每个内核与一个所谓的启动网格中可识别的线程块相关联。Triton 的编程模型与之类似，但每个内核是单线程的——尽管自动并行化——并与一组因实例而异的全局范围（global ranges）相关联（见图 3）。这种方法产生了更简单的内核，其中不存在类似 CUDA 的并发原语（共享内存同步、线程间通信等）。

与内核关联的全局范围可以使用 `get_global_range(axis)` 内建函数查询，以创建例如指针分块，如代码清单 1 所示。

![图3](图3说明：CUDA 编程模型与 Triton 编程模型之间的区别)

---

## 4. Triton IR

Triton-IR 是一个基于 LLVM 的中间表示（IR），其目的是为分块级程序分析、变换和优化提供合适的环境。在本工作中，Triton-IR 程序在解析过程中直接由 Triton-C 构建，尽管将来它们也可以直接从更高级的 DSL 生成。

**代码清单 5：** A = max(A, 0) 在 Triton-IR 中的实现。注意此处分块形状是非参数化的。在本文中，它们的值由 Triton-JIT 实例化。
```
define kernel void @relu(float* %A, i32 %M, i32 %N) {
prologue:
  %rm = call i32 <8> get_global_range(0);
  %rn = call i32 <8> get_global_range(1);
  ; 广播形状
  %1 = reshape i32 <8, 8> %M;
  %M0 = broadcast i32 <8, 8> %1;
  %2 = reshape i32 <8, 8> %N;
  %N0 = broadcast i32 <8, 8> %2;
  ; 广播全局范围
  %3 = reshape i32 <8, 1> %rm;
  %rm_bc = broadcast i32 <8, 8> %3;
  %4 = reshape i32 <1, 8> %rn;
  %rn_bc = broadcast i32 <8, 8> %4;
  ; 计算掩码
  %pm = icmp slt %rm_bc, %M0;
  %pn = icmp slt %rn_bc, %N0;
  %msk = and %pm, %pn;
  ; 计算指针
  %A0 = splat float*<8, 8> %A;
  %5 = getelementptr %A0, %rm_bc;
  %6 = mul %rn_bc, %M0;
  %pa = getelementptr %5, %6;
  ; 计算结果
  %a = load %pa;
  %_0 = splat float <8, 8> 0;
  %result = max %float %a, %_0;
  ; 写回
  store fp32 <8, 8> %pa, %result
}
```

Triton-IR 与 LLVM-IR 程序共享相同的高级结构（第 4.1 节回顾），但前者还包括了若干对分块级数据流（第 4.2 节）和控制流（第 4.3 节）分析必需的扩展。这些新颖的扩展对于执行第 5 节中概述的优化以及安全地访问任意形状的张量（如第 6 节所示）至关重要。

### 4.1 结构

#### 4.1.1 模块

在最高层次上，Triton-IR 程序由一个或多个称为模块的基本编译单元组成。这些模块彼此独立编译，最终由链接器聚合起来，链接器的作用是解析前向声明并适当地合并全局定义。

每个模块本身由函数、全局变量、常量和其他杂项符号（如元数据、函数属性）组成。

#### 4.1.2 函数

Triton-IR 函数定义由返回类型、名称以及可能为空的参数列表组成。可以根据需要添加额外的可见性、对齐和链接说明符。函数属性（如内联提示）和参数属性（如只读、别名提示）也可以指定，以允许编译器后端通过例如更好地利用只读内存缓存来执行更激进的优化。

此头部之后是一个主体，由一组基本块组成，其相互依赖关系形成函数的控制流图（CFG）。

#### 4.1.3 基本块

基本块定义为只能在其末尾包含所谓终止指令（即分支、返回）的直线代码序列。

Triton-IR 使用静态单赋值（SSA）形式，这意味着每个基本块中的每个变量必须 (1) 只被赋值一次且 (2) 在使用前已被定义。由此，每个基本块隐式地定义了一个数据流图（DFG），其不同路径对应于程序 SSA 表示中的 use-def 链。这种形式可以直接从抽象语法树（AST）创建，如[7]所示。

### 4.2 对分块级数据流分析的支持

#### 4.2.1 类型

多维分块是 Triton-IR 中数据流分析的核心，可以使用类似于 LLVM-IR 中向量声明的语法来声明。例如，`i32<8, 8>` 是对应于 8×8 的 32 位整数分块的类型。注意，Triton-IR 中没有 `tunable` 关键字，因此在生成程序之前必须解析参数化的形状值。在我们的例子中，这由 Triton-JIT 的自动调优器完成（第 5.3 节）。

#### 4.2.2 指令

Triton-IR 引入了一组重分块（retiling）指令，其目的是支持第 3.2.2 节中描述的广播语义：

- **reshape 指令**使用输入参数中的数据创建指定形状的分块。这特别适用于通过用 1 填充输入形状以准备隐式或显式广播，从而将变量重新解释为更高维数组。
- **broadcast 指令**通过沿大小为 1 的维度将其输入参数复制所需的次数来创建指定形状的分块——如图 4 所示。

![图4](图4说明：broadcast <3,3> 指令。分别展示了 [3×1] 输入和 [1×3] 输入时的广播效果。)

常规的标量指令（`cmp`、`getelementptr`、`add`、`load`……）被保留并扩展，以表示分块操作数上的逐元素操作。最后，Triton-IR 还公开了用于转置（`trans`）和矩阵乘法（`dot`）的专用算术指令。

### 4.3 对分块级控制流分析的支持

Triton-IR 中存在分块级操作所产生的一个问题，是无法在分块内部表达发散控制流。例如，程序可能需要部分保护分块级加载以免发生内存访问违规，但这无法通过分支来实现，因为分块元素不能单独访问。

**代码清单 6：** Triton-IR 中的分块级谓词化
```
;pt[i,j], pf[i,j] = (true, false) 如果 x[i,j] < 5
;pt[i,j], pf[i,j] = (false, true) 如果 x[i,j] >= 5
%pt, %pf = icmpp slt %x, 5
@%pt %x1 = add %y, 1
@%pf %x2 = sub %y, 1
; 合并来自不同谓词的值
%x = psi i32 <8,8> [%pt, %x1], [%pf, %x2]
%z = mul i32 <8,8> %x, 2
```

我们建议通过使用谓词化 SSA（PSSA）形式[8]和 ψ 函数[39]来解决这个问题。这需要在 Triton-IR 中添加两类指令（见代码清单 6）：

- **cmpp 指令** [8] 类似于通常的比较（`cmp`）指令，区别在于它们返回两个相反的谓词而非一个。
- **psi 指令**合并来自不同谓词化指令流的指令。

---

## 5. Triton-JIT 编译器

Triton-JIT 的目标是通过一组与机器无关（第 5.1 节）和与机器相关（第 5.2 节）的 pass（由自动调优引擎支持，第 5.3 节），简化 Triton-IR 程序并将其编译为高效的机器代码。

### 5.1 与机器无关的 Pass

#### 5.1.1 预取（Pre-Fetching）

循环内的分块级内存操作可能是有问题的，因为在没有足够独立指令的情况下，它们可能引发无法隐藏的严重延迟。然而，可以通过在 Triton-IR 中直接检测循环并在需要之处添加适当的预取代码来缓解此问题（见代码清单 7）。

**代码清单 7：** 自动预取
```
// 预取前
B0:
  %p0 = getelementptr %1, %2
B1:
  %p = phi [%p0, B0], [%p1, B1]
  %x = load %p
  ; 递增指针
  %p1 = getelementptr %p, %3

// 预取后
B0:
  %p0 = getelementptr %1, %2
  %x0 = load %p0
B1:
  %p = phi [%p0, B0], [%p1, B1]
  %x = phi [%x0, B0], [%x1, B1]
  ; 递增指针
  %p1 = getelementptr %p, %3
  ; 预取
  %x1 = load %p
```

#### 5.1.2 分块级窥孔优化

Triton-IR 中分块级操作的存在为窥孔优化器[29]提供了新的机会。例如，可以使用恒等式 X = (X^T)^T 对任意分块 X 简化转置链。我们相信，未来还可以利用与对角分块等相关的其他代数性质。

### 5.2 与机器相关的 Pass

我们现在展示一组针对遵循图 5 所示高级模型的机器的优化 pass。具体来说，Triton-JIT 执行的优化包括 (1) 层次化分块、(2) 内存合并、(3) 共享内存分配和 (4) 共享内存同步。

![图5](图5说明：Triton-IR 机器模型中的层次化分块。展示了从 Tile → Micro-Tile → Nano-Tile 的分解过程，以及从 Device → Core → SIMD Unit → ALU 的硬件层次结构映射。)

#### 5.2.1 层次化分块（Hierarchical Tiling）

嵌套分块策略（见图 5）旨在将分块分解为微分块（micro-tiles），并最终分解为纳米分块（nano-tiles），以尽可能紧密地适配机器的计算能力和内存层次结构。虽然这种技术在自动调优框架[34, 40]中常被使用，但 Triton-IR 的结构使得可以为任何可表达的程序自动枚举和优化有效的嵌套分块配置（且无需多面体机制）。

#### 5.2.2 内存合并（Memory Coalescing）

当相邻线程同时访问相近的内存位置时，内存访问被称为合并的。这一点很重要，因为内存通常以较大的块为单位从 DRAM 中获取。

由于 Triton-IR 程序是单线程且自动并行化的，我们的编译器后端能够在每个微分块内部排序线程，从而在可能的情况下避免非合并的内存访问。这种策略减少了加载分块列所需的内存事务数量（见图 6）。

![图6](图6说明：非合并 (a) 和合并 (b) 的 DRAM 访问。不同线程以不同颜色表示。)

#### 5.2.3 共享内存分配（Shared Memory Allocation）

具有高算术强度的分块级操作（如 `dot`）可以通过将其操作数临时存储在快速的共享内存中获益。共享内存分配 pass 的目的是确定何时以及何处的分块应被暂存到这一空间中。如图 7 所示，可以通过首先计算每个感兴趣变量的活跃范围，然后使用[15]中提出的线性时间存储分配算法来实现。

![图7](图7说明：共享内存分配。展示了活跃区间如何在时间轴上重叠，以及如何在有限容量内进行分配，例如 4kB、4kB、4kB、8kB。)

#### 5.2.4 共享内存同步（Shared Memory Synchronization）

在我们的机器模型中，对共享内存的读写是异步的。共享内存同步 pass 的目标是自动在生成的 GPU 源代码中插入屏障（barriers），以保持程序正确性。这通过使用以下数据流方程进行前向数据流分析，检测读后写（RAW）和写后读（WAR）冲突来实现：

- **in(RAW)_s** = ⋃_{p ∈ pred(s)} out(RAW)_p
- **in(WAR)_s** = ⋃_{p ∈ pred(s)} out(WAR)_p
- **out(RAW)_s** = { ∅ 如果 in(RAW)_s ∩ read(s) ≠ ∅（触发 barrier）; 否则 in(RAW)_s ∪ write(s) }
- **out(WAR)_s** = { ∅ 如果 in(WAR)_s ∩ write(s) ≠ ∅（触发 barrier）; 否则 in(WAR)_s ∪ read(s) }

### 5.3 自动调优器（Auto-tuner）

传统的自动调优器[42, 45]通常依赖手写的参数化代码模板，以在预定义的工作负载上实现良好性能。相比之下，Triton-JIT 可以直接从 Triton-IR 程序中提取优化空间，只需将上述每个优化 pass 相关联的元参数进行拼接即可。

在本工作中，仅考虑层次化分块 pass，每个分块每维度最多 3 个分块参数。这些参数然后通过在以下范围内对 2 的幂进行穷举搜索来优化：(a) 分块大小在 32 和 128 之间；(b) 微分块大小在 8 和 32 之间；(c) 纳米分块大小在 1 和 4 之间。未来可以使用更好的自动调优方法。

---

## 6. 数值实验

在本节中，我们在来自深度学习文献的各种工作负载上评估 Triton 的性能。我们使用 NVIDIA GeForce GTX1070 并将我们的系统与最新的厂商库（cuBLAS 10.0、cuDNN 7.0）以及相关的编译器技术（Auto-TVM、TC、PlaidML）进行比较。在适用的情况下，我们按照官方文档指南为每个单独的问题大小对这些 DSL 进行了自动调优。

### 6.1 矩阵乘法

形如 A = D × W^T（D ∈ R^{M×K}，W ∈ R^{N×K}）的矩阵乘法任务是神经网络计算的核心。在此我们考虑来自循环（DeepSpeech2 [3]）和 transformer [44] 神经网络的各种任务；我们在图 8 中报告其性能。

![图8](图8说明：矩阵乘法性能。对比了 Triton、cuBLAS 10.0、AutoTVM、Tensor Comprehensions 和 PlaidML 在 Square、DeepSpeech2 和 Transformer 任务上的 TFLOPS。Triton 与 cuBLAS 基本持平，比替代 DSL 快 2-3 倍。)

Triton 和 cuBLAS 通常彼此不相上下，在某些任务上达到设备峰值性能的 90% 以上。然而，对于较浅的 transformer 神经网络，cuBLAS 仍然比 Triton 更快，这得益于 3D 算法[2]的使用，该算法将深层归约拆分为独立的块，以在 M 和 N 太小时提供更多并行性。在其他情况下，现有 DSL 比我们的解决方案慢 2-3 倍——除了 TVM（慢不到 2 倍），当输入形状是 32 的倍数时。

### 6.2 卷积

卷积神经网络（CNN）是一类重要的机器学习模型，应当得到 DSL 和编译器的良好支持。它们基于卷积层（图 9a），其作为矩阵乘法的实现（图 9b）对于利用专用张量处理硬件是必要的——然而现有 DSL 尚不支持。在此，我们对 cuDNN 的 "IMPLICIT_GEMM" 算法的 Triton 重新实现进行基准测试（第 6.2.1 节），并为移位卷积提供首个融合内核（第 6.2.2 节）。我们使用指针增量的查找表来实现这些例程，如代码清单 8 所示。

![图9](图9说明：密集和移位卷积层 (a) 视为矩阵乘法 (b)。展示了密集卷积中数据、滤波器和激活之间的映射关系，以及移位卷积中移位数据和逐点滤波器的结构。)

#### 6.2.1 密集卷积

本小节考虑的卷积层来自深度学习文献，如表 1 所示。

**表 1：** 本文考虑的卷积任务

| 任务 | H | W | C | B | K | R | S | 应用 |
|------|---|---|---|---|---|---|---|---|------|
| Task 1 | 112 | 112 | 64 | 4 | 128 | 3 | 3 | ResNet [19] |
| Task 2 | 56 | 56 | 128 | 4 | 256 | 3 | 3 | ResNet |
| Task 3 | 28 | 28 | 256 | 4 | 512 | 3 | 3 | ResNet |
| Task 4 | 14 | 14 | 512 | 4 | 512 | 3 | 3 | ResNet |
| Task 5 | 7 | 7 | 512 | 4 | 512 | 3 | 3 | ResNet |
| Task 6 | 161 | 700 | 1 | 8 | 64 | 5 | 5 | DeepSpeech2 [3] |
| Task 7 | 79 | 341 | 32 | 8 | 32 | 5 | 10 | DeepSpeech2 |

如图 10 所示，对于 ResNet，Triton 优于 cuDNN 的 IMPLICIT_GEMM 实现。这可能是因为 cuDNN 还为 3×3 卷积维护了更好的算法（即 Winograd [25]），从而为优化次要内核留下了较少的工程资源。当快速算法不可用时（例如 DeepSpeech2），cuDNN 和 Triton 不相上下。

![图10](图10说明：隐式矩阵乘法性能。对比了 Triton 和 cuDNN 7.0 的 IMPLICIT_GEMM 在 ResNet 和 DeepSpeech2 的 7 个任务上的 TFLOPS。)

#### 6.2.2 移位卷积

最后，我们考虑将表 1 中的 Task 1-5 实现为移位卷积——一种新颖的 CNN 方法（见图 9a）。我们将 Triton 中融合 shift-conv 模块的实现（代码清单 8）与依赖手写 shift 内核和单独调用 cuBLAS 的朴素实现进行性能比较。我们还报告了不执行 shift 时（即 1×1 卷积）可获得的最大性能。如图 11 所示，我们的 Triton 实现能够几乎完全隐藏 shift 的代价。

**代码清单 8：** shift-convolutions 在 Triton-C 中的实现
```c
const tunable int TM = {16, 32, 64, 128};
const tunable int TN = {16, 32, 64, 128};
const tunable int TK = {8};
__constant__ int* delta = alloc_const int [512];
for(int c = 0; c < C; c++)
  delta[c] = c*H*W + shift_h[c]*W + shift_w[c]
void shift_conv(restrict read_only float *a,
                restrict read_only float *b, float *c,
                int M, int N, int K){
  int rxa[TM] = get_global_range[TM](0);
  int ryb[TN] = get_global_range[TN](1);
  int rka[TK] = 0 ... TK;
  int rkb[TK] = 0 ... TK;
  float C[TM, TN] = 0;
  float* pxa[TM, TK] = a + rxa[:, newaxis];
  float* pb[TN, TK] = b
      + ryb[:, newaxis] + rkb*N;
  __constant__ int* pd[TK] = delta + rka;
  for(int k = K; k > 0; k = k - TK){
    int delta[TK] = *pd;
    float *pa[TM, TK] = pxa + delta[newaxis, :];
    float a[TM, TK] = *pa;
    float b[TN, TK] = *pb;
    C = dot(a, trans(b), C);
    pb = pb + TK*N;
    pd = pd + TK;
  }
  int rxc[TM] = get_global_range[TM](0);
  int ryc[TN] = get_global_range[TN](1);
  float* pc[TM, TN] = c + rxc[:, newaxis] + ryc*M;
  bool checkc0[TM] = rxc < M;
  bool checkc1[TN] = ryc < N;
  bool checkc[TM, TN] = checkc0[:, newaxis] && checkc1;
  @checkc *pc = C;
}
```

![图11](图11说明：Triton 中移位卷积的性能。对比了朴素 shift、融合 shift 和最大可获得性能在 ResNet Task 1-5 上的 TFLOPS。Triton 融合实现几乎完全隐藏了移位的开销。)

---

## 7. 结论

在本文中，我们提出了 Triton，一种开源的语言和编译器，用于将分块神经网络计算表达并编译为高效的机器代码。我们展示了仅向 LLVM-IR 添加少量数据流和控制流扩展，就可以实现各种分块级优化 pass，这些 pass 共同带来了与厂商库相当的性能。我们还提出了 Triton-C，一种更高级的语言，在这种语言中，我们能够简洁地实现面向 CNN 的新型神经网络架构的高效内核。

未来的工作方向包括对 tensor core 的支持、量化内核[26]的实现以及与更高级别 DSL 的集成。

---

## 8. 致谢

本工作得到了 NVIDIA Graduate Fellowship 的支持。

---

## 参考文献

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G. Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, and Xiaoqiang Zheng. 2016. TensorFlow: A System for Large-scale Machine Learning. In *Proceedings of the 12th USENIX Conference on Operating Systems Design and Implementation (OSDI'16)*.

[2] R. C. Agarwal, S. M. Balle, F. G. Gustavson, M. Joshi, and P. Palkar. 1995. A three-dimensional approach to parallel matrix multiplication. *IBM Journal of Research and Development* 39, 5 (Sep. 1995), 575–582.

[3] Dario Amodei, et al. 2015. Deep Speech 2: End-to-End Speech Recognition in English and Mandarin. *CoRR* abs/1512.02595 (2015). arXiv:1512.02595

[4] M. Auguin and F. Larbey. [n. d.]. Opsila: an advanced SIMD for numerical analysis and signal processing. ([n. d.]), 311–318.

[5] Bin Bao and Chen Ding. 2013. Defensive Loop Tiling for Shared Cache. In *Proceedings of CGO '13*. 1–11.

[6] Muthu Manikandan Baskaran, et al. 2010. Parameterized Tiling Revisited. In *Proceedings of CGO '10*. 200–209.

[7] Matthias Braun, et al. 2013. Simple and Efficient Construction of Static Single Assignment Form. In *Proceedings of CC'13*. 102–122.

[8] Lori Carter, Beth Simon, Brad Calder, Larry Carter, and Ferrante. [n. d.]. Predicated Static Single Assignment. In *Proceedings of the PACT 1999 Conference on Parallel Architectures and Compilation Techniques*.

[9] Tianqi Chen, et al. 2015. MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems. *CoRR* abs/1512.01274 (2015).

[10] Tianqi Chen, et al. 2018. TVM: End-to-End Optimization Stack for Deep Learning. *CoRR* abs/1802.04799 (2018).

[11] Tianqi Chen, et al. 2018. TVM: End-to-End Optimization Stack for Deep Learning. *CoRR* abs/1802.04799 (2018).

[12] Jack W. Davidson and Sanjay Jinturkar. 1994. Memory Access Coalescing: A Technique for Eliminating Redundant Memory Accesses. In *Proceedings of PLDI '94*. 186–195.

[13] Greg Diamos, et al. 2016. Persistent RNNs: Stashing Recurrent Weights On-Chip. In *Proceedings of ICML 2016*.

[14] Venmugil Elango, et al. 2018. Diesel: DSL for Linear Algebra and Neural Net Computations on GPUs. In *Proceedings of MAPL 2018*. 42–51.

[15] Jordan Gergov. 1999. Algorithms for Compile-time Memory Optimization. In *Proceedings of SODA '99*.

[16] Google Inc. 2017. Tensorflow XLA. https://www.tensorflow.org/performance/xla/

[17] Scott Gray and Alex Radford ans Diederik P. Kingma. 2017. GPU kernels for block-sparse weights. *CoRR* abs/1711.09224 (2017).

[18] M. Griebl, C. Lengauer, and S. Wetzel. 1998. Code generation in the polytope model. In *Proceedings of PACT 1998*. 106–111.

[19] K. He, X. Zhang, S. Ren, and J. Sun. 2016. Deep Residual Learning for Image Recognition. In *CVPR 2016*. 770–778.

[20] Andrew G. Howard, et al. 2017. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *CoRR* abs/1704.04861 (2017). arXiv:1704.04861

[21] Jianyu Huang and Robert A. van de Geijn. 2016. BLISlab: A Sandbox for Optimizing GEMM. FLAME Working Note #80, TR-16-13. The University of Texas at Austin.

[22] Intel AI. 2018. PlaidML. https://www.intel.ai/reintroducing-plaidml

[23] Andrew Kerr, Duane Merrill, Julien Demouth, and John Tran. [n. d.]. CUTLASS: Fast Linear Algebra in CUDA C++. ([n. d.]). https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

[24] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2012. ImageNet Classification with Deep Convolutional Neural Networks. In *Proceedings of NIPS'12*.

[25] Andrew Lavin and Scott Gray. 2016. Fast Algorithms for Convolutional Neural Networks. *CVPR'16*.

[26] Darryl D. Lin, Sachin S. Talathi, and V. Sreekanth Annapureddy. 2016. Fixed Point Quantization of Deep Convolutional Networks. In *Proceedings of ICML'16*. 2849–2858.

[27] Matt Martineau, Patrick Atkinson, and Simon McIntosh-Smith. 2017. Benchmarking the NVIDIA V100 GPU and Tensor Cores. Springer.

[28] Bradley McDanel, Surat Teerapittayanon, and H.T. Kung. 2017. Embedded Binarized Neural Networks. In *Proceedings of EWSN '17*. 168–173.

[29] W. M. McKeeman. 1965. Peephole Optimization. *Commun. ACM* 8, 7 (July 1965), 443–444.

[30] Sanyam Mehta, et al. 2016. TurboTiling: Leveraging Prefetching to Boost Performance of Tiled Codes. In *Proceedings of ICS '16*. Article 38, 12 pages.

[31] Sharan Narang, Eric Undersander, and Gregory F. Diamos. 2017. Block-Sparse Recurrent Neural Networks. *CoRR* abs/1711.02782 (2017).

[32] Rajib Nath, Stanimire Tomov, and Jack Dongarra. 2011. Accelerating GPU Kernels for Dense Linear Algebra. In *Proceedings of VECPAR'10*. 83–92.

[33] John Nickolls, Ian Buck, Michael Garland, and Kevin Skadron. 2008. Scalable Parallel Programming with CUDA. *Queue* 6, 2 (March 2008), 40–53.

[34] Cedric Nugteren. 2017. CLBlast: A Tuned OpenCL BLAS Library. *CoRR* abs/1705.05249 (2017). arXiv:1705.05249

[35] Travis Oliphant. 2006–. NumPy: A guide to NumPy. USA: Trelgol Publishing.

[36] PyTorch. 2016. https://github.com/pytorch/pytorch

[37] Jonathan Ragan-Kelley, et al. 2013. Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomposition in Image Processing Pipelines. In *Proceedings of PLDI '13*.

[38] Nadav Rotem, et al. 2018. Glow: Graph Lowering Compiler Techniques for Neural Networks. *CoRR* abs/1805.00907 (2018).

[39] Arthur Stoutchinin and Francois de Ferriere. 2001. Efficient Static Single Assignment Form for Predication. In *Proceedings of MICRO 34*. 172–181.

[40] Philippe Tillet and David Cox. 2017. Input-aware Auto-tuning of Compute-bound HPC Kernels. In *Proceedings of SC '17*. ACM.

[41] Didem Unat, et al. 2016. TiDA: High-Level Programming Abstractions for Data Locality Management.

[42] Field G. Van Zee and Robert A. van de Geijn. 2015. BLIS: A Framework for Rapidly Instantiating BLAS Functionality. *ACM Trans. Math. Softw.* 41, 3 (June 2015).

[43] Nicolas Vasilache, et al. 2018. Tensor Comprehensions: Framework-Agnostic High-Performance Machine Learning Abstractions. *CoRR* abs/1802.04730 (2018).

[44] Ashish Vaswani, et al. 2017. Attention Is All You Need. *CoRR* abs/1706.03762 (2017). arXiv:1706.03762

[45] R. Clint Whaley, Antoine Petitet, and Jack J. Dongarra. 2000. Automated Empirical Optimization of Software and the ATLAS Project. *PARALLEL COMPUTING* 27 (2000), 2001.

[46] Samuel Williams, Andrew Waterman, and David Patterson. 2009. Roofline: An Insightful Visual Performance Model for Multicore Architectures. *Commun. ACM* 52, 4 (April 2009), 65–76.

[47] Bichen Wu, et al. 2017. Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions. *CoRR* abs/1711.08141 (2017).

[48] Jiacheng Zhao, et al. 2018. Revisiting Loop Tiling for Datacenters: Live and Let Live. In *Proceedings of ICS '18*. 328–340.

---

> **翻译说明：** 本文为 Triton 2019 年原始论文的完整中文翻译。翻译严格遵守原文，力求"原汁原味"地传达原文的学术内容、技术细节和结构安排。所有代码清单、图注、表格和参考文献均完整保留。原始英文 PDF 见同目录下的 `triton_paper_2019_original.pdf`。
