# 第 5 章 线性代数核心（matmul 旗舰）

> 矩阵乘是整个深度学习里**最重**的单个计算原语。一个 Transformer 训练 step 的 FLOPS 里，矩阵乘（及其近亲卷积）通常占 70%~95%。这一类算子不"搬运数据"、不"重排结构"，纯粹是**密集浮点乘加**——但正因为密，它对**访存模式**、**累加精度**、**Tiling** 的敏感度，远超前面所有逐元素与规约算子。本章是 Part I 的压轴，也是全书"数学类"里最值得深讲的一章。

矩阵乘家族在模型里的角色一句话可概括：**几乎所有"学到的线性变换"都是它**。全连接层（`y = xWᵀ + b`）、注意力里的 QKV 投影与 `softmax(QKᵀ)V`、FFN 的两次升维/降维、LoRA 的低秩分支、MoE 的路由打分……背后的核心计算都是 `mm`/`bmm`/`addmm`。理解它们的**形状契约**（谁乘谁、批维怎么对齐、bias 怎么融进来）和**数值/精度契约**（累加用哪种 dtype），是读懂任何 inductor 计算图的前提。

> 本章公共概念（广播、类型提升、SSA 契约）详见第 0 章 §0.4 / §0.5 / §0.7。本章所有算子都是**位置保持**（输出第 `i` 行只依赖输入第 `i` 行相关数据），数学公式是全部内容。

---

## 本章速查（Tier C）

> 本章无 Tier C 算子——`mm`/`bmm`/`addmm` 三个全是旗舰/难点，全部进 Tier D 详解。

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| — | （本章 Tier C 为空） | — | — |

> 相邻复合算子提示（不展开，详见附录 B）：`matmul` → `mm`/`bmm` + 批维广播；`mv`/`dot` → 把向量当 `[n,1]`/`[1,n]` 的 `mm`；`outer` → 外积；`einsum` → 分解成若干 `mm`/`bmm` + 广播/规约；`linear` → `addmm`/`mm` + bias。这些上层算子在 tracing/分解阶段就会被改写成 `mm`/`bmm`/`addmm`，不会原样出现在 inductor 的核心计算图里。

---

## 深入算子（Tier D）

### `aten.mm(input, other)` — 两个 2D 矩阵的矩阵乘

**作用与语义**　标准的矩阵乘法。输入是两个**严格 2 维**的张量 `A: [M, K]` 与 `B: [K, N]`，输出 `C: [M, N]`，定义：

```
C[i, j] = Σ_{k=0}^{K-1}  A[i, k] * B[k, j]
```

契约要点：
- **形状硬约束**：`A.shape[1] == B.shape[0]`（即内维 `K` 必须相等），否则报错。**不接受广播**——既不广播 batch 维、也不广播 1 维（向量要用 `mv`/`dot`，那是复合算子）。
- **维度硬约束**：输入必须是恰好 2 维。1 维或 3 维都会报错（1 维→用 `mv`/`dot`；3 维及以上→用 `bmm`/`matmul`）。
- 输入→输出形状：`[M,K] × [K,N] → [M,N]`。
- 返回**新张量**（不别名、不修改输入）。dtype 按类型提升规则取（实操上几乎总是 `float32`/`float16`/`bfloat16`，整数矩阵乘罕见且性能差）。

**示例**　`[2,3] @ [3,2] → [2,2]`，内维 `K=3` 被消去，外维 `M=2 / N=2` 留下来——矩阵乘的形状契约一目了然：

```python
>>> a = torch.tensor([[1, 2, 3],
...                    [4, 5, 6]])        # [2,3]
>>> b = torch.tensor([[7, 8],
...                    [9, 10],
...                    [11, 12]])         # [3,2]
>>> torch.mm(a, b)                        # [2,3] @ [3,2] -> [2,2]
tensor([[ 58,  64],
        [139, 154]])
```

**为什么需要这个算子 / 数值与精度**　矩阵乘是"密集浮点乘加"，数值上的关键不是"会不会爆炸"，而是**累加精度**：
- 形状 `[M,N] × [N,P]` 的 `mm`，每个输出元素是 `K` 个乘积的累加。当 `K` 很大（如 4096）时，累加器的有效位数决定了相对误差。
- **精度陷阱**：输入 `bfloat16` 时，若**累加也在 bfloat16** 上做，`K>1024` 后误差会显著放大（bfloat16 只有 7 位尾数）。因此**几乎所有硬件/库都在内部把累加器升到 `float32`**——这正是 inductor 选 Triton 模板而非手写 pointwise 核的根本原因（Triton 允许指定 `allow_tf32`/`acc_dtype`，pointwise lower 路径无此能力）。
- TF32：在 Ampere+/Ascend 上，`float32` 输入可启用 TF32（把每元素截成 19/10 位再乘），`mm` 在 `tf32` 模式下快数倍但精度下降，这是用户可感知的最大"相同算子、不同结果"来源之一。

**实现逻辑与复杂度**

```python
def mm(A, B):                                   # A: [M,K], B: [K,N]
    M, K = A.shape; K2, N = B.shape
    assert K == K2                              # 内维必须相等
    C = empty((M, N), dtype=promote(A.dtype, B.dtype))
    for i in range(M):
        for j in range(N):
            acc = 0.0                           # 实际里这里用 float32 累加器
            for k in range(K):
                acc += A[i, k] * B[k, j]
            C[i, j] = acc
    return C
```

- 时间复杂度：**O(M·N·K)** 乘加（FLOPS 计口径是 `2·M·N·K`，因为一次乘+一次加算 2 次浮点操作）。
- 空间复杂度：O(M·N) 输出 + O(1)~O(tile²) 临时（分块大小）。**不零拷贝**——必须分配输出。
- **为什么这是性能关键**：`mm` 的算术强度 (arithmetic intensity = FLOPS / bytes) 远高于逐元素算子。逐元素算子每个元素读一次写一次（`2 bytes/元素`），算术强度 < 1，是**访存受限**；而 `mm` 读 `A` 一行（K 个元素）可复用于 N 列输出、读 `B` 一列可复用于 M 行输出，理想算术强度 ≈ `2MNK / (MK+KN+MN)` 个元素字节，随 M/N/K 增大趋向 ∞，是**计算受限**。这意味着 `mm` 能把硬件算力吃满，是测 NPU 峰值 FLOPS 的标尺算子。
- **Tiling** 是把上述复用落到真实内存层次的核心：把 `[M,K]×[K,N]` 切成 `[BM,BK]×[BK,BN]` 的小块，让每个小块的 `A_tile`/`B_tile` 留在片上 SRAM/Cache，反复参与 `BM·BN` 次输出累加——这正是 Triton 模板干的事，也是 autotune 要搜索 `BM/BN/BK` 的原因。

**边界与陷阱**
- **非连续输入**：`mm` 的两个输入会在 lowering 里被标记为 `needs_realized_inputs`（见第 0 章连续性），非连续的张量会先被物化成连续副本。这是少数几个**强制连续化**的算子——因为 Triton GEMM 模板假设行/列主序寻址，非连续会破坏 tiling 假设。代价是：对一个转置过的输入（如 `mm(A.t(), B)`）可能产生一次额外拷贝；inductor 有专门 pass（`b2b_gemm`/`decompose_mem_bound_mm`）尝试消解这种拷贝。
- **空维度**：`K=0`（空内积）合法，输出全 0；`M=0` 或 `N=0` 合法，返回空矩阵 `[0,N]`/`[M,0]`。
- **整数矩阵乘**：`aten.mm` 支持整数输入，但累加用同 dtype，`K` 大时极易溢出（int32 累加到 ~2³¹ 就翻转）。模型里几乎不出现，调试时易踩。
- **NaN 传播**：任何 `A[i,k]` 或 `B[k,j]` 为 NaN，对应 `C[i,j]` 为 NaN；`inf × 0 = NaN`（与 IEEE 754 一致）。
- **类型提升**：`float16 @ float16` 默认仍是 `float16` 输出（但内部累加 `float32`）；`bfloat16 @ bfloat16` 同理。`float16 @ float32` 会先把 `float16` 提升到 `float32`。

**Inductor 视角**　**template**（`select_algorithm` 选 Triton GEMM 模板 + autotune 搜 tile/blocking）——`mm`/`bmm`/`addmm`/`conv` 是 inductor 里少数走 template 路径的算子，因为它们需要硬件特定的 tiling 与累加精度控制，pointwise/reduction 路径都给不了。

---

### `aten.bmm(input, other)` — 批矩阵乘（每个 batch 独立做一次 2D 矩阵乘）

**作用与语义**　对**一批**矩阵两两做矩阵乘。输入 `A: [B, M, K]` 与 `B: [B, K, N]`，输出 `C: [B, M, N]`：

```
C[b, i, j] = Σ_{k=0}^{K-1}  A[b, i, k] * B[b, k, j]      # 对每个 batch b 独立
```

契约要点：
- **首维是 batch 维**：两边的 `B`（batch 数）必须相等，**batch 维不做广播**（要广播请用 `matmul`，它是复合算子，会先 broadcast 再调 `bmm`）。
- **后两维是严格矩阵乘契约**：`A.shape[2] == B.shape[2]`（即 `K` 相等），与 `mm` 相同。
- 输入→输出形状：`[B,M,K] × [B,K,N] → [B,M,N]`。
- 输入必须是恰好 3 维（2 维请先 `unsqueeze` 成 `[1,M,K]` 或直接用 `mm`）。
- 返回新张量，不别名。

**示例**　`(2,2,3) @ (2,3,2) → (2,2,2)`，`B=2`：每个 batch **独立**做一次 `[2,3]@[3,2]`，结果互不混合——batch 1 换了不同的元素，得到的 `[[1,2],[3,4]]` 与 batch 0 的 `[[58,64],[139,154]]` 各算各的：

```python
>>> A = torch.tensor([[[1, 2, 3],
...                    [4, 5, 6]],         # batch 0: [2,3]
...                   [[1, 0, 0],
...                    [0, 1, 0]]])        # batch 1: [2,3]
>>> B = torch.tensor([[[7, 8],
...                    [9, 10],
...                    [11, 12]],          # batch 0: [3,2]
...                   [[1, 2],
...                    [3, 4],
...                    [5, 6]]])           # batch 1: [3,2]
>>> torch.bmm(A, B)                        # (2,2,3) @ (2,3,2) -> (2,2,2)
tensor([[[ 58,  64],
         [139, 154]],

        [[  1,   2],
         [  3,   4]]])
```

**为什么需要这个算子 / 数值与精度**　`bmm` 的数值与 `mm` 完全一致（每个 batch 内部就是一次 `mm`），精度陷阱同上（bfloat16 累加、TF32、整数溢出）。值得单独强调的是它在**注意力机制**中的地位：`softmax(Q @ Kᵀ / √d) @ V` 里的两次矩阵乘，在 batch 多头设定下就是 `bmm`（`[B*H, S, S] × [B*H, S, d]`），这是 LLM 训练里除 FFN 的 `mm` 外另一个 FLOPS 大头。

**实现逻辑与复杂度**

```python
def bmm(A, B):                                  # A: [B,M,K], B: [B,K,N]
    Bsz, M, K = A.shape; Bsz2, K2, N = B.shape
    assert Bsz == Bsz2 and K == K2
    C = empty((Bsz, M, N), dtype=promote(A.dtype, B.dtype))
    for b in range(Bsz):                        # 各 batch 独立
        for i in range(M):
            for j in range(N):
                acc = 0.0
                for k in range(K):
                    acc += A[b, i, k] * B[b, k, j]
                C[b, i, j] = acc
    return C
```

- 时间复杂度：**O(B·M·N·K)**（即 B 倍于单次 `mm`）。
- 空间复杂度：O(B·M·N) 输出。
- **与 `mm` 的本质区别**：仅多一个 batch 循环。实现上，优秀的 GEMM 模板会把 batch 维并进 grid 的一个轴（每个 block 处理 `(b, i_tile, j_tile)`），让不同 batch 之间互不干扰地并行——所以 `bmm` 的峰值吞吐 ≈ `mm` 的 B 倍（直到打满硬件并发度）。

**边界与陷阱**
- **batch 不广播**：`[4,M,K]` 与 `[1,K,N]` 会**报错**，不会自动把 batch=1 复制成 4（这是 `bmm` 与 `matmul` 的关键区别）。若需要，先用 `expand`（零拷贝，stride=0）把 batch 维扩开再 `bmm`。
- **非连续**：同 `mm`，两个输入在 lowering 阶段被 `needs_realized_inputs` 强制物化。对注意力里常见的转置输入（`Kᵀ`）同样可能触发额外拷贝。
- **空 batch**：`B=0` 合法，返回 `[0,M,N]` 空张量。
- **dtype/NaN/整数**：规则与 `mm` 完全相同。

**Inductor 视角**　**template**——与 `mm` 同走 `select_algorithm` 的 Triton batched GEMM 模板（多一个 batch grid 维 + autotune）。对 NPU，`bmm` 是 `_inductor_new` 后端必须重点调优的算子之一。

---

### `aten.addmm(input, mat1, mat2, beta=1, alpha=1)` — 融合 GEMM：α·(mat1 @ mat2) + β·input

**作用与语义**　把"矩阵乘 + 加偏置/残差"融合成单个算子，定义：

```
C = alpha * (mat1 @ mat2) + beta * input
```

签名注意：第一个参数 `input` 是"被加项"（bias/残差），`mat1`/`mat2` 才是相乘的两个矩阵——顺序与数学直觉相反，是历史遗留，是这一族里最易写反的坑。

契约要点：
- `mat1: [M, K]`，`mat2: [K, N]`，`input: [M, N]`（必须能广播成 `[M,N]`，常见是 `[M,N]` 或 `[N]` 的 bias）。
- 输出 `C: [M, N]`。
- `alpha`/`beta` 是标量，默认都是 1。
- 返回新张量。`beta=input=0` 时退化为纯 `mm`（`alpha·(mat1@mat2)`），实现常走快路径。

**示例**　让 `mat2` 取单位阵，则 `mat1@mat2 == mat1`，叠加关系最清楚：`beta=2, alpha=10` 时结果是 `2·input + 10·mat1`，每个输出元素都能在 `input` 与 `mat1` 里对号入座——`[0,0]` 位 `2*10+10*1=30`，`[1,1]` 位 `2*10+10*4=60`：

```python
>>> input_ = torch.tensor([[10, 0],
...                        [0, 10]])       # 明显非零的"被加项"
>>> mat1 = torch.tensor([[1, 2],
...                      [3, 4]])          # [2,2]
>>> mat2 = torch.tensor([[1, 0],
...                      [0, 1]])          # 单位阵 -> mat1@mat2 == mat1
>>> torch.addmm(input_, mat1, mat2, beta=2, alpha=10)   # 2*input_ + 10*(mat1@mat2)
tensor([[30, 20],
        [30, 60]])
```

**为什么需要这个算子 / 数值与精度**　`addmm` 的核心价值不在数学（它显然等价于 `mm` + `add`），而在**融合 (fusion)**：
- 全连接层 `y = xWᵀ + b` 的标准写法就是 `addmm(bias, x, Wᵀ)`。若拆成 `mm` + `add`，会多一次 `[M,N]` 张量的写出/读回——对大 `M·N`（如 4096×4096）这是一次不小的访存开销，且阻塞了 GEMM 后的流水。
- `addmm` 让"乘累加 + 加 bias"在一次核里完成，**省掉一次中间张量的全程访存**。这是 inductor 反复强调"融合 GEMM"的原因。
- 数值上与 `mm` 一致（累加精度、TF32、整数溢出同上）；额外注意 `beta=0` 时 `input` 的 NaN/inf 会被乘 0——`0 * NaN = NaN`，所以 `beta=0` **不会**屏蔽 `input` 的 NaN，这是一个反直觉的坑（想屏蔽 NaN 要用 `where`）。

**实现逻辑与复杂度**

```python
def addmm(input, mat1, mat2, beta=1, alpha=1):   # input: [M,N], mat1:[M,K], mat2:[K,N]
    M, K = mat1.shape; K2, N = mat2.shape
    assert K == K2
    C = empty((M, N), dtype=promote(input.dtype, mat1.dtype, mat2.dtype))
    bias = broadcast_to(input, (M, N))           # bias 常是 [N]，广播到 [M,N]
    for i in range(M):
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += mat1[i, k] * mat2[k, j]
            C[i, j] = alpha * acc + beta * bias[i, j]
    return C
```

- 时间复杂度：**O(M·N·K)**（与 `mm` 同阶，加 `input` 的 O(M·N) 被乘加项淹没）。
- 空间复杂度：O(M·N) 输出，**只分配一次**（对比 `mm`+`add` 要分配两次 + 一次中间读写）。
- `beta=0` 时通常走"无 input 依赖"快路径，等价于 `alpha * mm(mat1, mat2)`。

**边界与陷阱**
- **参数顺序坑**：`addmm(input, mat1, mat2)` —— `input` 在前。写成 `addmm(mat1, mat2, input)` 会形状对不上而报错，或更糟，形状恰好都对上但语义全错（静默错误）。
- **bias 广播**：`input` 可以是 `[N]`（最常见，行广播）或 `[1,N]`/`[M,1]`/`[M,N]`，但不能是 `[M]`（列向量 bias，与 GEMM 输出形状不兼容，需先 `unsqueeze`）。
- **非连续**：`mat1`/`mat2` 强制连续化（同 `mm`）；`input`（bias）通常允许非连续/广播（stride=0 零拷贝）。
- **`alpha`/`beta` 类型**：可以是 Python 标量或 0 维张量；整数 `alpha`/`beta` 配浮点矩阵会按类型提升走。
- **NaN 传播**：`beta·input` 项里，`beta=0` 不能消除 `input` 的 NaN（见上）；`alpha=0` 同理对乘积项。
- **与 `addbmm` 的关系**：`addbmm` 是 `addmm` 的批版本（`β·input + α·Σ_b (A_b@B_b)`），它不在本书 Tier D（core 但冷门），inductor 对其 `make_fallback`。

**Inductor 视角**　**template**——`addmm` 走与 `mm` 相同的 `select_algorithm` 路径，但 Triton GEMM 模板多一个"加 bias/残差"的 epilogue（fusion 的落脚点）。`beta=1, alpha=1` 是最常见的全连接层形态，autotune 会针对它选模板；`addmm` 还参与 `binary_folding` 等 pass（把相邻的 add 折进 GEMM）。

---

## 本章小结

`mm`/`bmm`/`addmm` 是全书"数学类"里**计算最重、精度最敏感、inductor 调优投入最大**的一族：三者形状契约清晰（`mm`=2D 严格、`bmm`=多一个不广播的 batch 维、`addmm`=带 bias/残差融合），核心复杂度都是 O(M·N·K)，但真正的难点在**累加精度**（bfloat16/TF32）与**访存-计算比**（tiling/autotune）。在 inductor 里它们是少数走 template 路径的算子，`addmm` 的价值在于把 bias 加法融进 GEMM 的 epilogue、省掉一次中间张量全程访存。理解这一章，是读懂 LLM/注意力/FFN 计算图 FLOPS 分布与精度问题的前提。

下一章 [第 6 章 激活函数](06-activations.md) 进入 `relu`/`gelu`/`sigmoid`/`_softmax`/`native_dropout`，仍是位置保持的数学类，但重心从"密集乘加"转到"非线性与数值稳定性"。

---

[上一章 第 4 章](04-reductions.md) 　|　 [下一章 第 6 章](06-activations.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
