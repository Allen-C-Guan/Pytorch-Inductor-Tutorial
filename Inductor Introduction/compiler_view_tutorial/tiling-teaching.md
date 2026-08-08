# Inductor Tiling 系统：从数学本质到代码生成

> 一篇面向社区开发者的教学文章。从 tiling 的数学本质讲起，到 Inductor 如何用数据结构编码这些数学概念，再拿一个真实 reduction case 从头走到尾，把"要素 → 数据结构 → 代码"这条链完整串起来。文中所有数据结构值均为真实 dump，非杜撰。
>
> 环境：PyTorch 2.13.0 + torch_npu 2.13.0 triton_experimental，Ascend NPU。源码引用路径：`torch/_inductor/` = 上游 inductor（GPU 参考实现，NPU 继承其中大部分）；`torch_npu/_inductor/triton_experimental/codegen/` = NPU override（线性化 / group-dispatch 路径，会绕过上游部分 codegen 函数——凡被绕过的上游行号均标注 ★）。

---

## 1. Tiling 是什么——问题与目标

一个 fusion（被融合成一个 kernel 的若干算子）的输出，逻辑上是"对每个元素算一次"的循环。理想世界里 kernel 可以写成 `for i in range(numel): out[i] = …`。但硬件不允许：

- **并行执行单元有限**（GPU 的 SM、NPU 的 Vector Core，几十个量级）——不能为 `numel` 个元素各开一个执行单元，必须把它们分摊到有限个 program 上；
- **每个 program 的寄存器 / Unified Buffer 有限**——一个 program 一次只能"看见"有限个元素（即 block size），放不下整个迭代域；
- **reduction 还要在 program 内或跨 program 累加**。

**Tiling 就是把迭代域划分成若干 block（tile），每个 block 由一个 program 处理：**

```
   迭代域                                          例 [0, 8192)
   ┌─────────────────────────────────────┐
   │ 0   1   2   3 ...        numel-1    │      理想：一次遍历全部
   └─────────────────────────────────────┘
            │  tiling: 按 block_size 切
            ▼
   ┌────────┬────────┬────────┬────────┐
   │ block0 │ block1 │ block2 │  ...   │     每个 block = 一个 program
   │ 0..B-1 │ B..2B-1│2B..3B-1│        │     program 内：tl.arange(0,B) 看见本块
   └────────┴────────┴────────┴────────┘
      pid=0    pid=1    pid=2    ...          跨 block：tl.program_id(pid) 并行
```

所以 tiling 回答两件物理事实：**"切成多大块"**（block size，受寄存器/UB 约束）和 **"用多少个 program、怎么分工"**（grid，受并行单元约束）。

但这里有一个被轻描淡写带过的关键：上图把迭代域画成一条扁平的 `[0, numel)`。**迭代域在数学上其实不是一条线，而是一个有结构的空间**——"切块"到底切的是什么、切完之后 program 内怎么定位到一个具体的内存位置，都要回到这个结构才能讲清。

---

## 2. 迭代域的数学结构

`[0, numel)` 是迭代域的**扁平视图**。但要真正理解 tiling——理解一个 kernel 为什么有这个 rank、为什么有些维塌了有些没塌、为什么地址表达式长成那个样子——必须退一步，先看清一个 kernel 里到底涉及**哪些坐标系**，它们之间如何用线性映射连接。

### 2.1 三个线性空间

一个 triton kernel 在跑的时候，有三个不同性质的"坐标系"在同时工作。它们各自是线性的（有限维，每个维度是一段半开区间 `[0, size)`，索引靠整数加法），两两之间用线性映射连接：

```
  迭代空间（kernel 的格子）              张量空间（每个 tensor 的坐标系）
  ┌──────────────────┐                 ┌──────────────────────────────┐
  │ program_id 坐标    │                 │  T_A: 形状(d₀_A,…,dₙ_A)      │
  │ + tl.arange 坐标   │    ──┬──►       │  T_B: 形状(d₀_B,…,dₙ_B)      │
  │ = 格点 坐标        │      │           │  各自有各自的维数、各自 strides │
  └──────────────────┘      │           └────────────┬───────────────┘
                            │                        │
                            │  entry 分解              │ strides 仿射
                            │  (混合进制)               │ Σ coordₖ · strideₖ
                            ▼                        ▼
                         ┌──────────────────────────────────────────┐
                         │  内存空间：一条 1D 长线                    │
                         │  offset ∈ [0, byte_len)                   │
                         │  没有轴概念——所有 tensor 都落脚到这里      │
                         │  同一个字节可能被不同 tensor 的不同坐标命名 │
                         └──────────────────────────────────────────┘
```

**空间 1：迭代空间（kernel 的"格子"，记作 `I_kern`）。** 这是 kernel 自己选定的覆盖方案——kernel 启动一个多维 grid（`program_id`），每个 program 内再用 `tl.arange` 铺开。把 `program_id` 坐标和所有 `tl.arange` 坐标合在一起，得到一个有限笛卡尔积 `I_kern = [0,d₀)×…×[0,d_{r-1})`——这就是 kernel 用来安排工作的**迭代坐标格子**，r = kernel 的 rank，`|I_kern|` 是格点总数。**I_kern 的维数 r 正是 tiling 要决定的。**

> **注意 `I_kern` ≠ `I_T`**——即使碰巧形状相同（如某些 pointwise case 中 `I_kern` 恰好和张量的形状一致），它们是**两个不同的数学对象**：`I_kern` 是 kernel 自己选的格子，`I_T` 是张量的本征坐标系。kernel 为什么需要自己的格子、这个格子怎么从各张量的 `I_T` 中定出来，在 §2.2 讲。

**"工位"**：就是 `I_kern` 这格子里的**一个格点**（lattice point）——一个具体坐标 `p ∈ I_kern`，kernel 在这个坐标位置上做**一个元素的 compute / load / store**。不是每个工位都在做 compute（某些工位可能被 mask 挡掉、超出 numel 边界而无效），但 kernel 能访问的每一个元素位置都一定落在某个工位上。所有的工位合起来覆盖了这个 kernel 要处理的全部元素位置。

有了 kernel 自己的格子 `I_kern`，再来看各个**张量自己的坐标空间**。张量 `T` 的形状为 `(d₀, d₁, …, d_{n-1})`。它的**本征坐标空间**——记作 `I_T`，区别于 `I_kern`——同样是有限笛卡尔积：

```
I_T   = [0,d₀) × [0,d₁) × … × [0,d_{n-1})          （张量 T 自己的坐标空间）
|I_T| = numel = d₀·d₁·…·d_{n-1}
naive rank(I_T) = n                   ← n 个自由坐标，朴素地就是 n 层循环
```

`I_T` 是张量 T 的"本征视角"。它**不**直接决定 kernel 怎么写循环、怎么分 program——那是 `I_kern` 的事。`I_kern` 可能跟 `I_T` 形状恰好一致（连续 pointwise 时），可能比 `I_T` 多一维（reduction 时，输出 `I_out=[0,256)` 但 `I_kern=[0,256)×[0,4096)` 含归约轴），也可能 `I_kern` 的同一维上跨着 `I_T` 的两条轴（非连续时 `I_T` 的 `(16384,8)` 两轴被压进 `I_kern` 的一条 x-root，由两个 entry 分拆）。两个空间之间的翻译由 entry 完成（§2.4、§2.5）。

> 例：`(256,4096).sum(-1)` 的输入张量迭代域 `I_T = [0,256) × [0,4096)`，naive rank 2，numel = 256·4096。

**空间 2：张量空间（每个 tensor 的"本征坐标系"，`S_T`）。** 张量空间是"T 怎么看自己"——横轴纵轴各多少格、沿哪条走一格跳几个字节。张量互不共享坐标系：两个形状不同的张量 `T_A` 和 `T_B`，它们的逻辑轴 `(i₀,i₁,…)` 是各自独立的。**张量空间不需要和迭代空间对齐**——同一个 kernel 里可以跑形状完全不同的张量（如 pointwise 里 output `(16384,)`、weight `(16384,1)`、input `(16384,8)`），这正是 entry 要做的事：**把迭代格点坐标翻译成各张量自己的坐标系坐标**。

**空间 3：内存空间（"地基"，1D 仿射直线）。** 1D 的，没有坐标轴，只有地址偏移。所有张量的所有元素最终都落脚到这里——张量是"内存的一块区域 + 一个坐标命名系统"。不同张量、不同 strides，只是对同一块 1D 内存的**不同线性参数化**。同一个物理元素可能同时出现在两个张量的参数化里（in-place / alias / view）——这不是两个元素，是**同一点被两套坐标系命名**。

> 几何直觉：把内存想象成一条无限长的**直线**。每个张量是这条线上的一个**窗口**——一个有 sample-rate（strides）的采样格子。两个有重叠内存的张量，就等于两个窗口在直线上**交叠**：同一段内存区间，左窗口的坐标 (3,0) 和右窗口的坐标 (0,1) 可能**指向同一个地址**——因为两个窗口的 strides 不同，坐标系不一样。kernel 的迭代空间是"今天我干活要用的格子"，它按 entry 翻译成每个张量窗口里的坐标，再由 strides 映射到直线上的地址。

### 2.2 融合核的共享几何：kernel rank 是怎么来的

§2.1 说清楚了三个空间。现在回答一个关键问题：**当 N 个张量（形状可能不同、strides 可能不同）被 fusion 进同一个 kernel 时，kernel 的迭代空间 `I_kern` 到底是什么？它怎么从这些张量的 `{I_T, S_T}` 决定出来？kernel 的 rank 在几何上意味着什么？**

**(1) 每个张量定义了迭代空间上的一个坐标系**

kernel 不是对每个张量单独遍历，而是**走同一个格子**。在这个格子 `I_kern` 上，kernel 赋予每个迭代坐标 `p ∈ I_kern` 一个地址公式：

```
addr_T(p) = base_T + Σₖ coords_T_ₖ(p) · stride_T_ₖ
```

其中 `coords_T(p)` 是坐标转换——entry——把迭代格点 `p` 翻译成张量 `T` 自己的坐标。也就是：

```
I_kern  ──entry分解──►  S_T  ──strides仿射──► 内存（1D 直线）
```

同一个迭代格点 `p`，通过不同张量的 entry 翻译 + strides 仿射，映射到 1D 内存的不同地址（或同一地址，如果两组仿射碰巧相等）。

**(2) 调度器求"覆盖多面体"**

融合核里有多个张量。调度器（`simplify_and_reorder`，ir.py + sizevars.py）的工作是**找到最小的超矩形 `I_kern`**，使得对于每个参与融合的张量 `T`，`I_kern` 的坐标能通过某种重新排列/拆开，一对一遍历 `T` 需要的每一个元素坐标。每个张量 `T` 的 `_sizes` 列表（`simplify_and_reorder` 产出的 per-loop-dim 长度）定义了该张量在该 kernel 的迭代格子上**需要几条独立轴**（等于自己未被合并掉的张量轴数），以及每条轴的**宽度**（size）。所有张量的 `_sizes` 按 prefix（X / R / Y 等）对齐后**取并**——同 prefix 下共享一套长度序列（不同张量可能该 prefix 的 rank 不同，但 codegen 期间 kernel 的 `set_ranges` 用的是同一个序列，调度器保证一致性）——就形成 kernel 的覆盖多面体 `I_kern`。

**(3) kernel 的 rank：几何 / 代数含义**

> **几何含义：** `dim(I_kern)` —— 这个覆盖超矩形有几条正交边。你需要多少个**独立的、互不化简的坐标**才能唯一标定 `I_kern` 里的一个格点（即一个"工位"，§2.1）。它就是 kernel 迭代格子真实的独立方向数。
>
> **代数含义：** kernel 的 rank = 所有分到该 prefix 的、size>1 的、且无法被连续合并掉的张量轴数（§2.4 rank 公式）。它是 `I_kern` 作为自由 `Z`-模（整数加法群上的自由模）的秩——即 `I_kern` 里线性无关（按整数线性组合）的迭代方向的条数。
>
> **直观对应：** 生成的 triton kernel 里**有几层独立循环变量**（`for` 层数 + `tl.arange` 覆盖的独立方向的条数）。case-3 reduction kernel 有 2 层独立变量（x0 被 grid 并行覆盖、r0_1 被 `for` 循环覆盖），所以 rank=2。一条 rank=1 的单纯 pointwise kernel（连续情况下所有轴全合并了），生成代码里就一层 `xoffset + tl.arange`。

rank 的几何意义——即"坍缩"的意义——将在 §2.4 中探讨。在此之前，先建立 1D 内存这条直线上地址如何从任何坐标算出。

### 2.3 物理内存是 1D：张量 = buffer + 仿射映射

不管张量几维，**DRAM / buffer 永远是一段 1D 字节数组**。一个 N 维张量本质上是

```
张量  =  一段 1D 连续内存（base 起）
       + 一个仿射映射  (i₀,…,i_{n-1})  ↦  offset = Σₖ iₖ · strideₖ
```

`stride` 就是这个**仿射映射的系数**。所以 N 维的逻辑坐标，落到内存永远是**一个 1D 整数偏移**。这条事实决定了 tiling 的最后一步：不管迭代域几维、tile 切成什么形状，program 内算出的每个元素地址，最终都得收敛到一个 1D offset。

### 2.4 坍缩：rank 的代数化简与几何条件

§2.2 解释了 kernel 的迭代空间是一个覆盖多面体，它的 naive 维数 = 所有参与张量的所有轴的初始维数（去除 size-1 后）。Inductor 不会朴素地按这个 naive 维数枚举 `I_kern` 的全部格点。它对 `I_kern` 做一串**代数化简**，把独立迭代方向的数目（rank）尽量降下来。**坍缩后每条幸存的迭代方向，就是一个 entry（§2.5）**——所以 entry 的数量 = rank。降秩只有三种操作，其中唯一需要"几何条件"的是第三种：

**(a) reduction 划分 = 直积分解（把轴分到不同子空间）**

对 `T.sum(dim=k)`，索引集按"是否被消去"正交分裂：

```
I = I_iter × I_reduce,    I_iter = ∏_{j≠k}[0,d_j)（存活/输出轴 → X-root）
                          I_reduce = [0,d_k)      （归约轴 → R-root）
```

reduction 轴和非 reduction 轴不相交、且并起来覆盖全部轴，所以这是集合的**直积分解**：`rank(I) = rank(I_iter) + rank(I_reduce)`。它**不降总秩**，只把秩**切分到不同 root**。几何上就是积空间的一个直和分解——不需要任何条件，纯由算子语义（哪个轴被 `sum` 消掉）决定。

**(b) 丢 size-1 = 商掉单点空间**

`dₖ = 1` 时 `[0,1) = {0}` 是单点空间（积空间的终端元），`A × {0} ≅ A`——乘上它不增加任何自由度。所以：

```
rank_effective = #{ k : dₖ > 1 }        ← size-1 轴直接删除，不计入 rank
```

broadcast 维就是这样消失的：多一个 broadcast 轴不会多一个 entry。**几何上就是点退化——一个单点集合不贡献维数，也不需要"条件"，永远可丢。**

**(c) 连续合并 = 平坦同构塌缩（唯一的"两维合一"机会，需要几何条件）**

这是整个坍缩理论里唯一需要"几何条件"的操作。两个相邻维 `(a 外, b 内)`，size `(s_a, s_b)`，stride `(δ_a, δ_b)`。合并成立当且仅当

```
δ_a = δ_b · s_b          （外维 stride = 内维 stride × 内维 size，即存储上连续）
且对核内所有共享这两维的 buffer 都成立（sound test）
```

时，偏移 `i_a·δ_a + i_b·δ_b = δ_b·(i_a·s_b + i_b)`，括号里 `(i_a·s_b + i_b) ∈ [0, s_a·s_b)` 是个**单一平坦坐标**，于是

```
[0,s_a) × [0,s_b)  ≅  [0, s_a·s_b)     ← 两维塌成一维，rank −1
```

**几何解读——为什么 stride 条件 = 连续是关键：** 想象迭代空间里一个 2D 矩形区域 `(a,b) ∈ [0,s_a)×[0,s_b)`。这个矩形是一张格子网。每个张量定义了一个**线性映射**（strides 仿射）把这个 2D 格子网投到 1D 内存直线上。当 `δ_a = δ_b·s_b` 时，沿 a 走一格相当于沿 b 走 `s_b` 格——a 方向上相邻的两个格点，在内存直线上**中间没有空洞**：整个矩形的像是一段**连通的区间** `[0, s_a·s_b·δ_b)`。此时这个 2D 格子网在 1D 直线上构成了一个**密集的均匀采样**——a 轴和 b 轴在直线上"分不开"，可以用单一均匀步长 `δ_b` 覆盖全部 `s_a·s_b` 个点。

当 `δ_a ≠ δ_b·s_b` 时，a 方向的步长不等于"b 方向走完一行再回到 a 下一行"所对应的内存偏移——这意味着 a 方向的相邻格点在内存直线上中间**有空洞**。整个 2D 矩形的像在 1D 直线上是**有间断的**。a 轴和 b 轴在内存上"分得开"——a 的步子 ≠ b 步子 × s_b，它们定义了不同的采样方向。因此不能合成一个坐标 + 一个 stride，必须保留两个独立的坐标（两个 entry）。

拿一个具体例子来验证。`big = randn(32768,8)[::2,:]` 的 shape 是 `(16384,8)`，strides 是 `(16,1)`。外维 stride=16，内维 stride=1，内维 size=8：`δ_外=16 ≠ 1·8=8`，非连续 → 合并被 stride 数学挡掉 → rank 保持 2 → 需要 2 个 entry。

> 直观类比：把 2D 格子网想象成一张**渔网**，1D 内存直线是**一根晾衣绳**。连续情况 = 渔网是整整齐齐叠好再挂上去的——挂上去就是一串紧密的结点，你看不出原来有几层线。非连续 `[::2,:]` = 渔网被**撑开**了——每隔一行才有一个结点落在绳上，中间是空的。绳上的结点不再是一串等距紧密排列，而是带洞的。你必须记住"哪一行、哪一列"（2 个坐标），才能找到该结点在原始渔网里的位置。

**为什么非连续就一定不能合并？** 合并两维 = 让它们**共享一个坐标、一个 stride**。一个 entry 只能携带**一个均匀 stride**。两维只有在连续（能合成一个平坦下标、共用一个 stride）时才能共享一个 entry；非连续的两维**需要不同的 stride**，一个 entry 装不下两个 stride，**必须拆成两个 entry**。这就是"只要不满足 size-1 与连续合并，两维就无法坍缩"的根本原因——它直接来自 §2.3 中"地址是仿射映射、entry 是其中的一个系数项"的几何事实。

> 注意：上述 case-4 `big = randn(32768,8)[::2,:]` 的 `δ_外=16 ≠ 1·8`，非连续 → 合并不了，这是**数学事实**，与后端无关。NPU 上 `should_merge_loops=False` 还会**额外**抑制合并（连连续的也不合）——那是后端策略，不是数学。

**rank 公式**：

```
rank(root) = (#分到该 root 且 size>1 的轴数)  −  (#该 root 内完成的连续合并数)
```

> 一棵 root 的 rank = 它的 entry 数。公式里的 "root" 概念将在 §3.2 中正式定义——这里只需知道：经过 reduction 划分后，迭代域被切分成若干子空间（X-root、R-root 等），每个子空间独立坍缩。

### 2.5 混合进制：rank-r 子空间的规范型

rank 为 r 的一棵 root，其平坦索引 `idx ∈ [0, numel)` 被**唯一**分解成 r 个数位（mixed-radix 数制）：

```
idx  ↔  (c₀, c₁, …, c_{r-1}),    cₖ ∈ [0, ℓₖ),    ∏ ℓₖ = numel

c₀      = idx mod ℓ₀                                   ← 最内位（快维）
c₁      = (idx // ℓ₀)            mod ℓ₁
c₂      = (idx // ℓ₀ℓ₁)          mod ℓ₂
…
c_{r-1} = idx // (ℓ₀·…·ℓ_{r-2})                       ← 最外位（慢维）
```

这是 `[0,numel)` 与积空间 `∏[0,ℓₖ)` 之间的**标准同构**。而每个**数位**恰好对应一个 entry：

| 属性 | 取值 | 数学含义 |
|---|---|---|
| 基数 | `ℓₖ` | 这一位的**基数**（radix）= 本层循环长度 |
| 权重 | `∏_{j<k} ℓ_j` | 这一位的**权重**（所有更内层 length 之积） |
| 坐标 | `FloorDiv(idx, divisor)` 或 `ModularIndexing(idx, divisor, ℓₖ)` | 取这一位的坐标 |

**entry = 数位 = 一条迭代方向 = 子空间的一条轴。rank = 数位个数 = entry 个数。** Inductor 里 `lookup(divisor, length)`（simd.py:258）就是执行这个同构、按需创建 entry 的唯一入口。

> **合并与分裂互为逆映射**——这把 §2.4(c) 和本节缝在一起：
> - 合并（simplify 里）：`(i_a, i_b) ↦ i_a·s_b + i_b`，2 轴 → 1 轴，rank −1；
> - 分裂（lookup 里）：`idx ↦ (idx // s_b, idx % s_b)`，1 轴 → 2 坐标。
>
> lookup 用的 `divisor = s_b` 即"更内层 length 之积"——它正是合并时内维的 size（即便实际没发生合并，"divisor = 更内层 length 之积"也恒成立）。所以"一棵 root 有几个 entry"的最终答案：**simplify 的连续合并走了多远，rank 就降到多低；降不下去、剩下几条独立方向，lookup 就把它们分裂回几个 entry。**

**关键区分——权重 ≠ 数据 stride：** entry 的权重（divisor）是**混合进制权重**（内层 length 之积），**不是**数据 stride——二者只是恰好都出现在地址公式里。case-4 是铁证：慢维 entry 的 `divisor=8`（= 快维 length 之积），但它在输入地址里的系数是 `16`（`[::2]` 的真实 stride），`8 ≠ 16`。

**地址收束**（连回 §2.3）：r 个 entry 是 r 个坐标分量，最终地址把它们按各自**数据 stride** 加权求和，收敛到**一个 1D 偏移**：

```
offset = Σₖ entryₖ · (该维 data stride)
```

### 2.6 坍缩流程图

```
                 ┌──────────────────────────────────────────────────┐
  naive 迭代域    │  I = [0,d₀) × [0,d₁) × … × [0,d_{n-1})           │
  rank = n       │  numel = ∏ dₖ                                     │
                 └──────────────────────────────────────────────────┘
                                   │
                                   ▼  ① reduction 划分（直积分解）
                                       每维判 iter / reduce → 分到 X-root / R-root
                                       【不降秩，只把秩切分到不同 root】
                                   │
                                   ▼  ② 丢 size-1（商掉单点空间 [0,1) ≅ ∅）
                                       dₖ==1 的维直接删除
                                       【rank −= #singletons】
                                   │
                                   ▼  ③ 重排（把连续相邻维挪到一起）
                                   │
                                   ▼  ④ 连续合并（平坦同构塌缩）  ← 唯一的"两维合一"机会
                                       对每对相邻 (a外, b内)，逐 buffer 测：
                                          δ_a == δ_b · s_b  且  sound？
                                            是 → [0,s_a)×[0,s_b) ≅ [0,s_a·s_b)，rank−1
                                            否 → 保留两维（各成一个 entry）
                                       【rank −= #成功合并】
                                   │
                                   ▼
                 ┌──────────────────────────────────────────────────┐
  坍缩后         │  每 root rank r  →  r 个 entry（混合进制数位）     │
                 │  地址 = Σ entryₖ · strideₖ  →  1D 物理偏移           │
                 └──────────────────────────────────────────────────┘
```

①~④ 是一串"门"：②④ 降秩（② 丢 size-1、④ 连续合并；③ 只是重排、为 ④ 铺路，本身不降秩），① 把秩分到不同 root。走到底剩几条独立方向，就有几个 entry。

**走一遍 case-4（非连续，坍不动 → 2 entry）：**

```
big = randn(32768,8)[::2,:]    shape (16384,8)    strides (16,1)    pointwise

naive:  I = [0,16384) × [0,8)                              rank 2
  ① 无 reduction → 两维都进 X-root
  ② 无 size-1
  ③ (16384外, 8内) 已相邻
  ④ δ_外=16,  δ_内·s_内 = 1·8 = 8  ≠ 16  → 不连续 → 不合并
▼
坍缩后: X-root rank 2  →  2 entries
   entry 快维: ℓ=8,    divisor=1,  c = idx mod 8        ← 低位
   entry 慢维: ℓ=16384, divisor=8,  c = idx // 8        ← 高位
   地址 = 快维·1 + 慢维·16  →  1D 偏移   （16 = [::2] 的真实 stride）
```

**对照（连续，能塌成 1 entry）：** `T = randn(16384,8)` 连续 strides `(8,1)`，naive rank 2 → ④ `δ_外=8 = 1·8` 连续 → 合并 `[0,16384)×[0,8) ≅ [0,131072)` → rank 1，1 entry（GPU 上；NPU 因 `should_merge_loops=False` 仍保持 2 entry）。

---

有了这个数学模型——迭代空间、张量空间、内存空间三个坐标系的链式映射，融合多面体的秩，坍缩的几何条件（§2.4），混合进制的规范分解（§2.5）——一个 tiling 系统要把它翻译成可运行 kernel，**必须决定哪几件事**？Inductor 又是**用什么数据结构来承载这些决策**、并让它们互相咬合的？

---

## 3. 从数学到软件——Inductor 的 tiling 系统设计

### 3.1 从数学导出的四个决策要素

一个 tiling 要变成可运行 kernel，必须回答四个决策。它们**来源不同，但互相咬合**：**(a)(d) 由 §2 的数学强制**——积空间每条轴的大小、地址的仿射映射；**(b) 由 §1 的硬件约束强制**——UB/寄存器有限，一个 program 只能看见积空间的一小块；**(c) 由 program 如何覆盖坍缩后的积空间强制**——块间分派 + 块内 arange。

| # | tiling 要素 | 为什么必须决定（来源） | 何时定 |
|---|---|---|---|
| **(a)** | **每维迭代空间 numel** | 积空间每条轴的大小 `dₖ`（§2.1）；坍缩后每棵 root 的 `numel`（§2.4） | codegen 期 |
| **(b)** | **tile 形状 / block size** | 硬件约束：一个 program 受 UB/寄存器所限只能看见积空间的一小块（§1 + §2.1） | **launch 期**（autotune） |
| **(c)** | **遍历计划** | program 如何覆盖坍缩后的积空间：块间（跨 program 分派）+ 块内（一条轴上的 arange）（§1 + §2.1） | codegen 期 |
| **(d)** | **块内寻址** | tile 内的坐标怎么收敛到 1D offset：`offset = Σ entryₖ · strideₖ`（§2.3 + §2.5） | codegen 期 |

**关键区分（极易误解）**：(a)(c)(d) 在 **codegen 期（编译时）** 决定并固化进数据结构；(b) block size 在 **launch 期** 由 triton autotune/heuristics 决定。`range_tree` 里**没有** block size 的具体数值，只有一个占位符号 `XBLOCK`——具体值是 launch 时塞进 kernel 的 constexpr。这条区分贯穿后面所有讨论。

**要素 (c) 要拆成三个正交的子决策**：

- **(c1) 块间并行**：哪些轴跨 program 并行分派（grid 的 `program_id` 维）——"一长串只领一段回来干"；
- **(c2) 块间串行**：哪些轴在一个 program 内用 `for` 循环串行走完——"全领回来，一段一段干"；
- **(c3) 块内并行**：每棵 root 的 arange 铺在多维 tile 的**哪条轴**上——块内并行通道。

(c1)(c2) 是**块间**遍历的两种互斥模式（一条轴要么跨核并行领、要么单核串行循环领）；(c3) 是**块内**的事，与 (c1)(c2) 正交，一棵 root 可以同时有块间角色和块内轴。

**三种典型 tiling 形态**——同一个 tiling 系统对不同算子产出不同形状的 tiling dict（`simd.py:3872 create_tiling`）：

| 算子 | tiling dict | 形态 |
|---|---|---|
| pointwise | `{'x': numel}` | 一个存活维（一棵活树） |
| reduction | `{'x': numel, 'r0_': rnumel}` | 一个存活维 + 一个归约维 |
| matmul（native） | `{'y': M, 'x': N, 'r0_': K}` | 两个存活维 + 一个归约维（`simd.py:4366-4371`） |

本文主线用 **reduction**（最常见、综合度够）；pointwise / matmul 差异在 §5.2 简述。

### 3.2 核心数据结构：数学概念的软件载体

四要素的决策定了之后，必须有数据结构把它们"携带"到 codegen。Inductor 的设计非常直接：**一个数学量 → 一个字段**。下表列出了承载 §3.1 四要素的核心字段（`SIMDKernel` 实例字段，`simd.py:522`），以及 codegen 期需要的簿记字段（以"簿记"标出，不承载数学量）：

| 字段 | 类型 | 承载的要素 / 数学量 | init |
|---|---|---|---|
| `features` | `SIMDKernelFeatures` | 整个 tiling **问题**（积空间规模 `numel`/`reduction_numel`、`node_schedule`） | `simd.py:535` |
| `numels` | `dict[str, sympy]` | **simplify 后的 tile 几何**（prefix→numel）= 要素 (a) | `simd.py:539` |
| `tiling` | `dict[str, sympy]` | **原始 tiling 决策**（prefix→numel，未 simplify；给 memory_stats/meta） | `simd.py:552` |
| `range_trees` | `list[IterationRangesRoot]` | 每个 tile 维一棵根树，承载要素 (c)（`grid_dim`/`is_loop`/`tensor_dim`）+ (d) 的入口 | `simd.py:542` |
| `range_tree_nodes` | `dict[Symbol, Entry]` | **kernel 级全局反向索引**（符号→entry），承载要素 (d) 的查询 | `simd.py:543` |
| `tiling_scores` | `dict \| None` | （簿记）每 prefix 的访存合并打分；简单 case 为 `None` | `simd.py:551` |
| `iter_vars_count` | `count()` | （簿记）生成 `x0/x1/r0_1` 命名的全局计数器 | `simd.py:544` |
| `inside_reduction` | `bool` | （状态）当前 codegen 是否在 reduction 维打开 | `simd.py:545` |
| `persistent_reduction` | `bool` | （状态）reduction 走 persistent 还是 looped（`for`） | `simd.py:553` |

下面展开三个最核心的结构：root（承载 c）、entry（承载 d）、range_tree_nodes（d 的查询通道）。

#### root：一个坍缩后的迭代轴，承载要素 (c)

一棵 `IterationRangesRoot` = tiling dict 里的一个分块维度 = 坍缩后的一条迭代轴。它不是一个张量轴——§2.4 已说明一棵 root 可以**跨**多个张量轴（case-4 的 x-root 就覆盖了 `(16384,8)` 两轴）。root 的三个字段分别承载要素 (c) 的三个子决策：

| 字段 | 承载 | 直觉 | case-3 取值 |
|---|---|---|---|
| `grid_dim` (c1) | 块间并行 | "一长串切成 block，每个 program **只领一段**回来干"（跨核 grid 分派） | x-root: `0`；r0_-root: `None` |
| `is_loop` (c2) | 块间串行 | "**全领回来**，用 `for` **一段一段**串行干" | x-root: `False`；r0_-root: `True` |
| `tensor_dim` (c3) | 块内 arange 轴 | 这棵 root 的迭代，铺在多维 block tile 的**第几条轴**上 | x-root: `0`（行）；r0_-root: `1`（列） |

三者关系：`grid_dim` 与 `is_loop` 是**块间遍历的两种互斥模式**（断言 `simd.py:206`：loop 维必须 `grid_dim=None`）；`tensor_dim` 是**块内**的事，与二者**正交**——一棵 root 可以同时有 `grid_dim`（块间并行领）和 `tensor_dim`（块内沿某轴铺），x-root 就是 `grid_dim=0 + tensor_dim=0` 并存。`tensor_dim` 不是"root 是 tiling 里第几个轴"（那是 root 的身份 `prefix`），而是"root 的 arange 在块内多维 tile 里占第几轴"——它直接索引块形状数组（`shape[tensor_dim] = block_size`，`simd.py:239`）；`triton_tensor_ndim()`（`triton.py:3155`）计的就是非 `None` 的 `tensor_dim` 个数——即块内 tile 的维数。

#### entry：混合进制的一个数位，承载要素 (d)

§2.5 已经从数学上把 entry 定义清楚：**一棵 root 的 entry 数 = 它的 rank（§2.4 坍缩后剩下的独立方向数）；每个 entry 是混合进制分解里的一个数位。** 这里把"一棵 root 挂几个 entry、entry 不携带步幅"两件容易混的事钉死：

**一棵 root 挂几个 entry = 它的 rank**，由 §2.4 的坍缩决定，**不是**由"张量维数"或"步幅条数"决定：

- case-3 reduction：X 组 rank-1 `[256]`、R 组 rank-1 `[4096]` → 每棵 root **1 个** entry（`x0`、`r0_1`，`divisor=1`）。
- case-4 pointwise：X 组 rank-2 `[16384,8]`（合并被非连续 stride 挡掉）→ x-root **2 个** entry（快维 + 慢维）。

**entry 不携带步幅。** entry 的 `divisor` 是**混合进制权重**（内层 length 之积），**不是**数据 stride。数据 stride 活在另一条线——load/store 的**数据地址表达式**——里；entry 只提供坐标分量，地址由 `Σ entryₖ·strideₖ` 组合（§2.5）。case-4 的 `divisor=8` vs 真实 stride `16` 就是证据。

**entry 的字段**（每个都是 §2.5 数学量的载体）：

| 字段 | 含义 | 数学量 |
|---|---|---|
| `name` | 块内索引变量名 `f"{prefix}{N}"`（`N` 来自全局 `iter_vars_count`） | 生成代码里的变量（`x0`/`r0_1`/`x1`） |
| `length` | 本层循环长度 | 这一位的基数 `ℓₖ` |
| `divisor` | 本层在扁平索引里的步长基数 = 所有更内层 entry 的 `length` 之积（`simd.py:292-296`） | 这一位的权重 `∏_{j<k}ℓ_j` |
| `expr` | 把扁平逻辑索引 `xindex` 解码成本层坐标的公式 | 取这一位坐标的同构 |

**`expr` 的两种形式**（`lookup` `simd.py:262-265` 按 `divisor*length==numel?` 判定，正是 §2.5 的 FloorDiv vs ModularIndexing 分支）：

```
若 divisor*length == numel :  expr = FloorDiv(xindex, divisor)                 # 最外位：整维无余数
否则                         :  expr = ModularIndexing(xindex, divisor, length) # 内层位：余数子块
```

`divisor=1` 时 `FloorDiv(xindex,1)` 被 sympy 简化成裸 `xindex`——这是为什么 case-3 的 entry `expr` 看起来是恒等，而非 `FloorDiv` 字面量。

**entry 的两张表**：`root.nodes: dict[expr, entry]`（正向，按 expr 去重）和 `range_tree_nodes: dict[symbol, entry]`（反向，按符号）。同一个 entry 对象在两张表里是同一份。

#### range_tree_nodes：为什么需要一个 kernel 级反向索引

它是 `dict[sympy.Symbol, IterationRangesEntry]`（`simd.py:543`），**全 kernel 全局、跨所有树共用一张表**，**唯一写点是 `lookup` 的 `simd.py:275`**。

**为什么需要它**：codegen 时 load/store 的索引表达式是 sympy 形式，里面会出现 entry 的符号（如 `x0`、`r0_1`）。要把表达式翻译成代码、判断 broadcast、算 stride、取 block 形状，codegen 必须从符号**反查**回 entry——拿它的 `.expr`/`.root`/`.length`/`.divisor`。`range_tree_nodes` 就是这张反查表。读点遍布 `codegen_indexing`（`simd.py:1184`）、`is_broadcasted`（`:1067`）、`get_strides_of_load`（`:1250`）、`get_block_shape`（`triton.py:304`）。

它和每棵 root 自己的正向表 `root.nodes`（`simd.py:198`，键是 FloorDiv/ModularIndexing 表达式，用于"给定切分 (divisor,length) 找/建 entry"）互补：正向按 expr 建，反向按符号查，两者在 `lookup` 里同时写入（正向 `:278`、反向 `:275`）。

### 3.3 数据结构全景：填上真实值

在讲具体 case 的完整流程之前，先把所有数据结构在主线 case `out = randn(256,4096).sum(-1)` 上**填上真实值**，用一棵树串起来。**要素 (b) block size 故意不在图里**——它不存这些结构中，是 launch 期 autotune 注入的 constexpr。

```
SIMDKernel 实例   ◄── case: out = randn(256,4096,npu).sum(-1)   （一个 kernel 对象）
│
├─ features : SIMDKernelFeatures          ◄── 把 tiling 问题带进 kernel
│    ├─ numel            = 256            ── 存活维迭代空间（积空间存活轴大小）
│    ├─ reduction_numel  = 4096           ── 归约维迭代空间
│    └─ is_reduction()   = True
│
├─ tiling   (dict) = {'x': 256, 'r0_': 4096}   ◄── select_tiling 原始决策
├─ tiling_scores    = None                 ── 简单 reduction 走 default_tiling，不产 scores
├─ numels   (dict)  = {'x': 256, 'r0_': 4096}   ◄── simplify 后的 tile 几何          [要素 a]
├─ inside_reduction = True
├─ persistent_reduction = False            ── NPU 强制 looped（走 for，不走 persistent）
│
├─ range_trees : list[root_x, root_r0_]    ◄── 每个 tile 维一棵 root
│    │
│    ├─ root_x  (IterationRangesRoot)      ── 存活维（rank 1 → 1 entry）
│    │    ├─ prefix     = 'x'
│    │    ├─ numel      = 256              ◄── [要素 a]
│    │    ├─ grid_dim   = 0   ──► tl.program_id(0) 跨核并行          ◄── [要素 c1]
│    │    ├─ tensor_dim = 0   ──► tl.arange 放张量轴 0                ◄── [要素 c3]
│    │    ├─ is_loop    = False            ◄── 不走 for（这维走并行，非串行）
│    │    └─ nodes : { xindex ─► entry_x0 }   ◄── 正向表（按 expr）
│    │         └─ entry_x0 (IterationRangesEntry)
│    │              ├─ name    = 'x0'      ◄── 全局 iter_vars_count = 0
│    │              ├─ divisor = 1         ◄── 混合进制权重（≠ data stride！）
│    │              ├─ length  = 256       ◄── 基数
│    │              └─ expr    = xindex    ◄── 连续恒等：扁平索引即坐标   [要素 d]
│    │
│    └─ root_r0_ (IterationRangesRoot)     ── 归约维（rank 1 → 1 entry）
│         ├─ prefix     = 'r0_'
│         ├─ numel      = 4096             ◄── [要素 a]
│         ├─ grid_dim   = None             ◄── 不占 grid（这维走 loop）   [要素 c1]
│         ├─ tensor_dim = 1   ──► tl.arange 放张量轴 1                ◄── [要素 c3]
│         ├─ is_loop    = True ──► for r0_offset in tl.range(0,...)  ◄── [要素 c2]
│         └─ nodes : { r0_index ─► entry_r0_1 }
│              └─ entry_r0_1
│                   ├─ name    = 'r0_1'    ◄── iter_vars_count = 1（x0 占了 0）
│                   ├─ divisor = 1
│                   ├─ length  = 4096
│                   └─ expr    = r0_index  ◄── 连续恒等                [要素 d]
│
└─ range_tree_nodes : { x0 ─► entry_x0 ,  r0_1 ─► entry_r0_1 }   ◄── 反向索引（按符号）
```

> **读图**：`features` 把"256×4096 的 reduction 问题"带进来 → `tiling`/`numels` 承载要素 (a) 每维 numel → `range_trees` 每棵 root 用 `grid_dim`/`is_loop`/`tensor_dim` 承载要素 (c) → root 下挂的 entry 用 `expr` 承载要素 (d) 块内寻址 → `range_tree_nodes` 是 codegen 反查 entry 的表。**要素 (b) block size 缺席**——它不在这些结构里，是 launch 期 autotune 注入的 constexpr（`XBLOCK`/`R0_BLOCK`）。`entry_x0`/`entry_r0_1` 在 `root.nodes`（正向）和 `range_tree_nodes`（反向）里是**同一份对象**。

### 3.4 两段映射在 triton 代码中的体现

至此，entry 和 stride 还是抽象概念。它们在实际生成的 triton 代码中到底长什么样？拿 case-4 `out = big[::2,:]` 来演示——这个 case 足够简单（pointwise，无 reduction），又足够有信息量（非连续 → 2 entries，两个张量不同 strides）。

这个 kernel 融合了两个张量：

| 张量 | 形状 | strides |
|---|---|---|
| 输入 `in`（`big[::2,:]` 的 view） | `(16384, 8)` | `(16, 1)` |
| 输出 `out` | `(16384, 8)` | `(8, 1)` |

由于 strides 不同，两维都无法合并（§2.4）→ x-root rank = 2，迭代空间是 2D 的：`I_kern = [0, 16384) × [0, 8)`，挂两个 entry——快维（列，`ℓ=8, divisor=1`）和慢维（行，`ℓ=16384, divisor=8`）。

取 `I_kern` 的一个具体格点——**第 10368 行、第 5 列**。在 kernel 的扁平编号里，`idx = 10368·8 + 5 = 82949`。下面是 kernel 真正生成的 Triton 代码，以及它对应哪段映射：

```python
# ═══════════════════════════════════════════════════════════════════
# 空间 1 — 迭代空间 I_kern：kernel 的格子，1D 扁平编号 xindex
# ═══════════════════════════════════════════════════════════════════
# xindex 来自 tile header：kernel 给 I_kern 里每个工位一个 1D 扁平编号
xindex  = x0offset + tl.arange(0, real_block_x0)[:, None]
# 例：工位 (行=10368, 列=5) → idx = 10368·8 + 5 = 82949

# ═══════════════════════════════════════════════════════════════════
# 第一段映射：entry — 把 I_kern 的 1D 扁平 idx 拆成 r 个 1D 标量分量
# 每个 entry 输出一个标量；r 个标量合在一起才是 I_T 里的 r 维坐标
# ═══════════════════════════════════════════════════════════════════
#   xindex = 82949
#   entry 快维(ℓ=8, divisor=1):      x1  = idx mod 8   = 5      ← 列坐标分量（标量）
#   entry 慢维(ℓ=16384, divisor=8):  x0  = idx // 8    = 10368  ← 行坐标分量（标量）
#   ── (x0=10368, x1=5) 合在一起 = I_T 里的 2D 坐标 ──
x0 = xindex // 8          # 慢维 entry: ℓ=16384, divisor=8,  expr = FloorDiv(xindex,8)
x1 = xindex % 8           # 快维 entry: ℓ=8,      divisor=1, expr = ModularIndexing(xindex,1,8)
#                     ▲ entry 映射到此结束：拿到了每个张量坐标系里的坐标分量

# ═══════════════════════════════════════════════════════════════════
# 空间 2 → 空间 3 — 第二段映射：strides 仿射
# 把张量坐标 (x0,x1) 映射到 1D 内存地址，每个张量用自己的 strides
# ═══════════════════════════════════════════════════════════════════

# 张量「输入 in」: strides = (16,1) → addr_in = x1·1 + x0·16
# （in 是 big[::2,:] 的 view，行 stride = 16 = 原 tensor 行 stride 8 × [::2] 步长 2）
tmp0 = tl.load(in_ptr0  + (x1 + 16*x0), ...)
#                          ──┬──  ──┬──
#                   x1·1 = col stride   x0·16 = row stride

# 张量「输出 out」: strides = (8,1) → addr_out = x1·1 + x0·8
tl.store(out_ptr0 + (x1 + 8*x0), tmp0, ...)
#                     ──┬──  ─┬─
#              x1·1 = col stride  x0·8 = row stride（连续，8 = 内轴长度）
```

**关键——容易踩的坑：entry 不是"把 1D 变 2D"，而是"把 1D 拆成 2 个 1D 标量"**

`x0` 和 `x1` 各自是 1D 标量（一个整数），看代码里 `x0 = xindex // 8` 这行——赋值结果就是单个整数。但两个整数**放在一起**——`(x0=10368, x1=5)`——才是张量空间 `I_T = [0,16384) × [0,8)` 里的**一个 2D 坐标点**。这就像你的房间号编号是 1D 的 `82949`，但你实际位置是"第 10368 排、第 5 列"——排号和列号各自是 1D 标量，合在一起才是 2D 坐标。

entry 的数量 = rank = I_T（及 I_kern）的维数。每个 entry 输出**一个轴**的 1D 坐标分量；所有 r 个 entry 的输出合在一起才是 r 维坐标。这就是 §2.5 混合进制：r 个 entry = r 个数位，每个数位是一个 1D 标量，合起来构成 r 维坐标。entry 的 `divisor` 是这一位在 flat index 里的权重，不是张量的 stride。

**两段映射的分工**：第一段（`xindex → x0,x1`）是 **kernel 共享的**——in 和 out 用同一组 entry、同一组 `x0=xindex//8, x1=xindex%8` 赋值。第二段（`x0,x1 → 地址`）是 **每个张量独立的**——in 的代码里 `x0` 乘 `16`、out 的代码里 `x0` 乘 `8`，因为 strides 不同。同一个点 `(x0=10368, x1=5)` 在 in 的内存里偏移 165893 字节、在 out 的内存里偏移 82949 字节——差了 82944，正好是 `x0·(16−8) = 10368·8`。

### 3.5 生命周期：DECISION → BUILD → POPULATE → CONSUME

上面讲的是数据结构"是什么"。下面把它们放进时间线——从 tiling 决策到代码生成，数据结构经历了四个阶段：

```
=========== DECISION（决策 tiling dict）===========
 select_tiling(simd.py:4324) ──► tiling dict {'x':N,'r0_':K}

=========== BUILD（建空架）===========
 SIMDKernel.__init__(simd.py:522) ──► self.numels  = simplify(tiling)        ◄── 要素(a)
                                      self.tiling  = tiling                  ◄── 原版(给 meta)
                                      self.tiling_scores                     ◄── 访存打分(簿记)
                                      self.range_trees = []                  ◄── 待填
                                      self.range_tree_nodes = {}             ◄── 反向索引,待填
       initialize_range_tree(simd.py:703) → construct_range_trees(simd.py:648)
         每个 prefix 建一棵 IterationRangesRoot（标好 grid_dim/is_loop/tensor_dim） ◄── 要素(c)
         root.nodes = {}   ◄── 本树正向表(待填)

=========== POPULATE（填 entry）===========
 set_ranges(simd.py:811) → construct_entries → lookup(simd.py:258)        ★唯一建 entry 点
   按 §2.5 混合进制：expr = FloorDiv/ModularIndexing(index_sym, divisor, length)  ◄── 要素(d)
   entry = IterationRangesEntry(name, divisor, length, expr)
   root.nodes[expr] = entry              ◄── 正向登记(按 expr)
   V.kernel.range_tree_nodes[entry.symbol()] = entry   ★simd.py:275 反向登记(按符号)

=========== CONSUME（翻译成 triton 源码）===========
 codegen_indexing(simd.py:1181): 拿到含 x0 符号的索引表达式
   → range_tree_nodes[x0] 反查回 entry → entry.root → root.block_size()  ◄── 反向索引的用途
   → entry.codegen() 发 "x0 = xindex"
 codegen_node_schedule(NPU triton.py:4428) → POPULATE → _apply_linearize(:5381)
   → codegen_range_tree(NPU triton.py:2902): x 树→npu_header:207 group-dispatch;
                                            r0_ 树→自定义 elif(:2941)
   → codegen_kernel(NPU triton.py:3228) → codegen_body(:3624→上游:5790)
```

> NPU codegen 与上游 GPU 路径的关键差异：x 树不走上游 `codegen_range_tree`（triton.py:3287）和 `iteration_ranges_codegen_header`（:6960），而是走 NPU 自己的 group-dispatch（`npu_header.py:207`）；entry 赋值也不走上游 `codegen_iteration_ranges_entry`（:6738），而是走 NPU `triton.py:2965`。r0_ 树的 loop header 仍沿上游路径。

这个生命周期贯穿了 tiling 的全部阶段。下一章拿一个真实的 reduction case，跟着这个生命周期，**一步一步**看到每个阶段的数据结构值和最终的代码产物。

---

## 4. 贯穿实例——reduction `(256,4096).sum(-1)` 全程

前三章建立了完整的理论框架：tiling 是什么（Ch1）、迭代域的数学结构（Ch2）、Inductor 如何用数据结构承载 tiling 决策（Ch3）。这一章拿最典型的 reduction case，从决策到代码生成，把整条链走通。

### 4.1 问题设定

- 输入：`out = randn(256, 4096).sum(-1)`
- 迭代域：`[0,256) × [0,4096)`，归约轴 dim=1
- 期望 tiling：存活轴 256 → `'x'` prefix，归约轴 4096 → `'r0_'` prefix
- 下面六步覆盖完整管线：DECISION → BUILD → POPULATE → block size → CONSUME

每步格式：停 `<file:line>`｜发生什么｜数据结构值｜原料→产物。

### 4.2 六步单步走

**Step 1｜问题**

输入张量迭代域 `[0,256)×[0,4096)`（§2.1），归约轴 dim=1。tiling 要决定：256 和 4096 各怎么切、怎么遍历、块内怎么寻址。

**Step 2｜DECISION —— `select_tiling` 决策出 tiling dict**

- 停 `simd.py:4324` `SIMDKernel.select_tiling` → `get_tiling_and_scores`（`:4336`）→ `create_tiling`（`:3872`）。
- 真实值：`tiling = {'x': 256, 'r0_': 4096}`，`features.numel=256`，`features.reduction_numel=4096`。
- 干什么：调度器按 §2.4(a) 把积空间直积分解——存活轴 `[256]` 右对齐到 `'x'`、归约轴 `[4096]` 给 `'r0_'`（`create_tiling` 用 `pw_prefixes=("z","y","x")`、`reduction_prefixes=("r0_","r1_")`，`simd.py:3878-3888`）。
- 原料→产物：`features.numel/reduction_numel` → tiling dict `{'x':256,'r0_':4096}`（要素 a 决策完成）。

**Step 3｜BUILD —— tiling dict 固化成数据结构空架**

- 停 `simd.py:522` `SIMDKernel.__init__`（经 `initialize_range_tree` → `construct_range_trees` `:648`）。
- 真实值：`self.numels={'x':256,'r0_':4096}`，`self.range_trees=2 棵`（空壳），`self.range_tree_nodes={}`。
  - root `x`：`numel=256, tensor_dim=0, grid_dim=0, is_loop=False`（存活维 → 跨核并行）
  - root `r0_`：`numel=4096, tensor_dim=1, grid_dim=None, is_loop=True`（归约维 → 单核 `for` 串行）
- 干什么：把 tiling dict 固化成 `numels`（要素 a），为每个 prefix 建一棵空壳 root 并标好遍历角色（要素 c）。
- 原料→产物：tiling dict → `numels` + 2 棵带元数据、`nodes={}` 的空壳 root。

**Step 4｜POPULATE —— `set_ranges` 建 entry，写反向索引**

- 停 `simd.py:811` `set_ranges`（`lengths=((256,),(4096,))`）→ `simd.py:258` `lookup`。
- 真实值：`lookup(1,256)` → entry `x0`（`expr=xindex`）；`lookup(1,4096)` → entry `r0_1`（`expr=r0_index`）。
- 干什么：每个 node codegen 时，把"每棵树覆盖的长度"告诉 root，触发 `lookup` 按 §2.5 混合进制建 entry（两维都 rank-1，整维覆盖走 `FloorDiv`，被 sympy 简化成裸符号），**同时**写正向表 `root.nodes` 和反向索引 `range_tree_nodes`（`simd.py:275/278`）。
- 原料→产物：`(divisor=1,length=256/4096)` → entry `x0`/`r0_1` + `range_tree_nodes` 填充（要素 d 决策完成）。

**Step 5｜block size 估值 —— launch 期 autotune 定 XBLOCK/R0_BLOCK**

- 停 `triton_heuristics.py` 的 `reduction()`（`:4571`）→ `Grid1D.generate`（`:5270`）。
- 干什么：codegen 期 `XBLOCK`/`R0_BLOCK` 只是符号（`root.block_size()` `simd.py:217`）；launch 期 heuristics 生成若干 `{XBLOCK, R0_BLOCK}` 候选 configs，autotune 选一个（如 `R0_BLOCK=1024`），`Grid1D.generate` 用 `ceildiv("xnumel", XBLOCK)`（`:5271`/`:5156`）算出 grid 大小。
- 原料→产物：`numels` + 候选 configs → 具体的 `XBLOCK`/`R0_BLOCK` + grid（要素 b 决策完成）。
- 关键：这一步**不改动** `range_trees`——block size 是 launch 期 constexpr，range_tree 始终只存符号。

**Step 6｜CONSUME —— 遍历 range_trees 翻译成 `@triton.jit` 源码**

codegen 的真实触发点是调度器的 `codegen_node_schedule`（NPU `triton.py:4428`），它在 POPULATE 之后调 `_apply_linearize`（`:5381`）重发真实 header，再调 `codegen_kernel`（`:3228`）。调用顺序：

```
codegen_node_schedule (NPU triton.py:4428)
  → [Step4 POPULATE 已完成]
  → _apply_linearize (:5381) → 重调 codegen_range_tree (:2902) 发真实 header
       x 树 (is_loop=F) → _codegen_header_npu_for_tree (npu_header.py:207)  ★group-dispatch
       r0_ 树 (is_loop=T) → 自定义 elif (:2941-2953) 发 r0_base arange
       （上游 iteration_ranges_codegen_header 在此不被调用——x 树被 NPU 绕过）
  → codegen_kernel (:3228) → codegen_body (:3624 → super() 上游 :5790)
       上游 codegen_body 发 `for r0_offset in tl.range(...)` (上游:5894)，再调
       iteration_ranges_codegen_header (上游:6960) 仅给 r0_ 树发 r0_index
```

> 注：上游 `codegen_range_tree`（triton.py:3287）和 `codegen_iteration_ranges_entry`（:6738）被 NPU 对应函数取代，调用次数 = 0。上游 `iteration_ranges_codegen_header`（:6960）和 `codegen_body`（:5790）仍被调用（经 NPU super()），但只服务 r0_ 树。

### 4.3 生成的 triton 代码逐行注解

下面是 case-3 真实生成的 kernel（节选），每行标注它由哪个数据结构的哪个字段发出：

```python
def KERNEL(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    #                      ─┬──  ──┬──  ──┬──          ──┬──
    #            numels['x']  numels['r0_']  root.block_size()(符号)
    ...
    for i in range(group_size):                            # NPU linearize group-dispatch
        x0offset = (group_base + i) % x0_blocks * real_block_x0          # ← root.grid_dim=0 (c1) ★NPU group-dispatch
        x0index  = x0offset + tl.arange(0, real_block_x0)[:, None]       # ← root.tensor_dim=0 (c3) + XBLOCK
        x0 = x0index; x0mask = x0index < x0numel                         # ← root.numel=256 (a) → mask
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):                # ← root.is_loop=True (c2) + R0_BLOCK
            r0_index = r0_offset + r0_base                               # ← r0_ 树头 (上游:6960)
            r0_1 = r0_index                                              # ← entry_r0_1.codegen() (d, NPU:2965)
            tmp0 = tl.load(in_ptr0 + (r0_1 + 4096*x0), ...)             # ← Σ entry·stride: 4096=x0 行 stride, 1=r0_ stride (d)
        tmp2 = tl.sum(_tmp2, 1)[:, None]
        tl.store(out_ptr0 + (x0), tmp2, x0mask)
```

- 原料→产物：`range_trees`（含 entry）+ launch 期 block size → 完整 kernel 源码。

每行都能回指到某个数据结构的某个字段——这正是"四要素被数据结构承载、最终翻译成代码"的闭环。

### 4.4 Case-4 对照

同样的机制放到 case-4 `big[::2,:]` 上——这是 pointwise，无 reduction，所以只有一棵 x-root（无 `r0_` 树），没有 `for r0_offset` 循环。但因为 strides `(16,1)` 非连续（§2.4），x-root 坍缩到 rank 2，挂了 2 个 entry：

- 快维 entry：`x1 = xindex % 8`（ℓ=8, divisor=1）→ ModularIndexing
- 慢维 entry：`x0 = xindex // 8`（ℓ=16384, divisor=8）→ FloorDiv

两个张量共享同一组 entry，但各自用不同的 strides 做地址仿射（in 乘 16，out 乘 8）。详细代码见 §3.4 的完整演示。

对比 case-3 和 case-4：
- case-3：2 棵 root（X + R），每棵 rank 1（1 entry），不同遍历模式（grid 并行 + for 串行）
- case-4：1 棵 root（X），rank 2（2 entries），纯 grid 并行，但 entry 拆分更复杂

**同样的数据结构、同样的生命周期，坍缩结果不同，生成的代码形态就不同。** 这正是 tiling 系统的设计意图：把差异化的数学结构（连续 vs 非连续、有无 reduction）装进统一的数据结构框架，由 codegen 机械地翻译。

---

## 5. 总结与总表

### 5.1 总表：要素 → 决策点 → 数据结构 → 代码

下面这张表是全文的索引。它把四要素的完整链路——从决策点到数据结构到发出的代码——一次列出。NPU 后端会**绕过**上游部分函数（凡标 ★ 处，上游对应行号被 NPU override 取代，不再调用）。

| tiling 要素 | 决策点 | 承载数据结构 | 发出的代码（NPU 实际位置） |
|---|---|---|---|
| **(a) 每维 numel** | `get_tiling_and_scores` `simd.py:4336` | `kernel.numels` `simd.py:539`；`root.numel` `simd.py:142`（写）/ `:191`（转发） | 形参 `xnumel/r0_numel`（`SizeArg` 上游 `triton.py:6415`）；`{p}mask = {p}index < {p}numel`（上游 `triton.py:7009`） |
| **(b) block size** | **launch 期**：heuristics 产 configs `triton_heuristics.py:4571`；autotune 选 | `Config.kwargs["XBLOCK"]`；`root.block_size()` 仅符号 `simd.py:217` | `XBLOCK: tl.constexpr=…`（上游 `triton.py:6438`）；`tl.arange(0,XBLOCK)`；`tl.range(…,R0_BLOCK)`；launcher grid `ceildiv(xnumel,XBLOCK)` `triton_heuristics.py:5271` |
| **(c1) 并行 grid 维** | `construct_range_trees` 分配 `grid_dim` `simd.py:666/685` | `root.grid_dim` `simd.py:211` | ★`x0offset=(group_base+i)%x0_blocks*real_block_x0` **NPU `npu_header.py:207`**（替代上游 `program_id*XBLOCK` `triton.py:7001`） |
| **(c2) 串行 loop 维** | `is_loop = is_reduction and not persistent_reduction` `simd.py:695` | `root.is_loop` `simd.py:207` | `for r0_offset in tl.range(0,r0_numel,R0_BLOCK):` 上游 `triton.py:5894-5895`（NPU `codegen_body` `:3624` super() 进入） |
| **(c3) 块内 arange 维** | 每棵 `tensor_dim!=None` 的树 | `root.tensor_dim` `simd.py:209` | x 树：★`x0index=x0offset+tl.arange(...)[:,None]` **NPU `npu_header.py:207`**；r0_ 树：NPU `triton.py:2941-2953` + 上游 header |
| **(d) 块内寻址** | `lookup` 建 entry `simd.py:258`（按 §2.5 混合进制） | `IterationRangesEntry(divisor,length,expr)` `simd.py:345`；`root.nodes` `:198`；`range_tree_nodes` `:543` | ★`{name}={expr}`（如 `r0_1=r0_index`）**NPU `codegen_iteration_ranges_entry` `triton.py:2965`**（替代上游 `:6738`）；地址项由 `indexing()` 组合（`Σ entry·stride`） |

### 5.2 其他 tiling 形态

本文主线是 reduction，但它只是 tiling 系统的三种基本形态之一。同一个系统对不同的算子产出不同形状的 tiling dict，但四要素不变、数据结构不变、生命周期不变：

- **pointwise**：`tiling={'x':N}`，少一棵活树（无 `r0_`），没有 `for r0_offset` 归约循环。Ch2 的坍缩理论、Ch3 的数据结构、Ch4 的生命周期完全适用。
- **matmul（native）**：`tiling={'y':M,'x':N,'r0_':K}`（`simd.py:4366-4371`），多一棵 `y` 存活树（`grid_dim=1`），三维 tile 是三个 block 的笛卡尔积。机制相同，只是多一棵 root。注意 matmul 可能走模板路径（`fixed_config`），block size 由模板定而非通用 autotune。
- **嵌套 / grouped reduction / scan / TMA**：不在本文主线内，仅列入口——它们都对应 Ch2/Ch3 某个要素的扩展：嵌套 / grouped reduction 在**归约轴上叠了第二层坍缩**（`DerivedIterationRangesRoot` `simd.py:407` = 第二棵 root；`_GroupedReductionLayout` `:1540`），此时 `SIMDKernel` 激活额外字段：`mix_order_reduction`、`rsplit_size`、`min_xblock`、`saved_partial_accumulate`；scan 用 `TritonSplitScanKernel`。本文的系统主线（DECISION→BUILD→POPULATE→CONSUME）对它们同样适用，只是 BUILD/POPULATE 阶段多了派生树/分块。

### 5.3 全文回顾

这篇文章从一条主线把 Inductor tiling 系统完整串了起来：

1. **Ch1 立起问题**：硬件有限 → 必须切 block → tiling 是"把迭代域翻译成并行执行计划"。
2. **Ch2 拆开数学**：迭代域不是一条扁平的 `[0,numel)`，它有三个坐标系（I_kern、I_T、内存 1D），两两之间靠 entry 和 strides 连接。融合核的 I_kern 是一个覆盖多面体，它的 rank 经过坍缩（reduction 划分 → 丢 size-1 → 连续合并）被尽可能降低。降不掉的独立方向就是混合进制的数位——每个数位 = 一个 entry。
3. **Ch3 架起软件**：四个决策要素（numel / block size / 遍历 / 寻址）直接从数学导出。Inductor 用 `numels` + root（`grid_dim`/`is_loop`/`tensor_dim`）+ entry（`divisor`/`length`/`expr`）+ `range_tree_nodes` 这一组数据结构把它们逐一编码。两段映射（entry 分解 + strides 仿射）在 triton 代码里清清楚楚：第一段 kernel 共享，第二段每个张量独立。
4. **Ch4 走完实例**：拿 reduction `(256,4096).sum(-1)` 从 DECISION 到 CONSUME 单步走了一遍——每个阶段的 file:line、数据结构值、原料→产物，最后生成的 kernel 源码每行都能回指到某个字段。

tiling 系统的设计意图至此清晰：**把差异化的数学结构装进统一的数据结构框架，由 codegen 机械地翻译成并行代码。** 理解了这条链，无论遇到 pointwise、reduction、matmul 还是更复杂的嵌套归约，都能用同一套思维工具去拆解。
