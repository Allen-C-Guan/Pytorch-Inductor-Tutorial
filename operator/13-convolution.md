# 第 13 章 卷积

卷积 (convolution) 是整个深度学习视觉/序列模型里**计算量最集中**的原语。一个 ResNet-50 里 90% 以上的 FLOPs 花在卷积上，Transformer 把它换成 matmul 后，它仍然是 CNN 类模型、检测/分割头、上采样解码器（转置卷积）的核心。它的数学定义其实只有一行（滑窗内积），但 PyTorch 把 **1D/2D/3D、普通/转置、空洞 (dilated)、分组 (grouped)、带 `output_padding`** 全部折叠进**同一个**底层算子 `aten.convolution`，用六组参数区分这些变体。本章的任务就是把这"一个算子、六组参数"彻底拆开，让你看到一张 `aten.convolution` 调用图时，能立刻读出它是哪种卷积、输入输出形状如何推导、inductor 会怎么降级它。

理解卷积降级 (lowering) 可懂性的钥匙是 **im2col（image-to-column）** 这个重排技巧：它把"滑窗内积"重写成一次矩阵乘（→ 第 5 章 `mm`），从而让卷积复用高度优化的 GEMM 实现。`col2im` 是它的逆。卷积反向 `convolution_backward` 是另一个独立 core 算子——它不能简单交给 autograd 分解（梯度本身又是一个卷积），所以作为基础原语存在。

本章属 **Part III（模型层计算原语）**：inductor 对它走 **template** 路径（与 `mm` 同类，靠 `select_algorithm` 选核），而不是 pointwise 或 reduction。

---

## 本章速查（Tier C）

本章所有卷积相关 core 算子均为 Tier D 详解或附录一行带过，**无 Tier C 条目**。相邻的 `conv1d/conv2d/conv3d/conv_transpose*` 全部是复合算子（分解到本节的 `aten.convolution`），不在 core 主章节展开。

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.convolution` | 通用卷积（普通/转置/空洞/分组统一底层） | 输出形状由六组参数推导；分组需 `C_in`、`C_out` 都能被 `groups` 整除 | template（`select_algorithm`） |
| `aten.convolution_backward` | 卷积反向（同时算 grad_input/grad_weight/grad_bias） | 与正向形状对偶；输入多、易踩 `output_padding`/`groups` 不一致 | fallback（`make_fallback`） |
| `aten.col2im` / `aten.im2col` | im2col 的列→图（逆）/ 图→列 | 仅 2D；附录 A 标 ○ | 可分解（`_decomp` 内有实现） |

---

## 深入算子（Tier D）

### `aten.convolution(input, weight, bias, stride, padding, dilation, transposed, output_padding, groups)` — 六组参数折叠所有卷积变体的统一底层

**作用与语义**　卷积是 CNN 的核心特征提取操作：**一个权重核（kernel）在输入特征图上滑动，每个位置算一次"核 ⊙ 局部窗口"的内积，拼出输出特征图。** 这一个算子用六组参数（`stride/padding/dilation/transposed/output_padding/groups`）统一表达 1D/2D/3D、普通/转置、空洞、分组所有变体。

最直白的情形（2D、`groups=1`）——每个输出元素就是一个 kH×kW 窗口与核的内积加偏置：

```
out[n, co, i, j] = bias[co] + Σ_{ci,p,q} input[n, ci, i·stride+p, j·stride+q] · weight[co, ci, p, q]
```

形状契约：
- 空间维数 `= len(stride)`（1/2/3 对应 Conv1d/2d/3d）；输入 `(N, C_in, *spatial_in)` → 输出 `(N, C_out, *spatial_out)`。
- 权重形状：普通卷积 `(C_out, C_in/groups, *kernel_size)`；**转置卷积**（`transposed=True`，做上采样）通道维对调成 `(C_in, C_out/groups, *kernel_size)`。
- **输出尺寸**（每个空间维 `L`）：
  - 普通卷积：`L_out = ⌊(L_in + 2·pad − dil·(k−1) − 1) / stride⌋ + 1`
  - 转置卷积：`L_out = (L_in−1)·stride − 2·pad + dil·(k−1) + output_padding + 1`
- `groups>1`：通道分组、组内独立卷积（通道不全交叉，极端即 depthwise）；`dilation>1`：核内插空隙"撑开"感受野；`output_padding` 仅转置卷积用，消除"多个输入尺寸映射到同一输出尺寸"的歧义。

**示例**　最小的 2D 卷积：`weight=[[1,0],[0,1]]` 相当于"取每个 2×2 滑窗的主对角线之和"，输出尺寸 `L_out=⌊(3+0−1)/1⌋+1=2`：

```python
>>> x = torch.tensor([[[[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]]])   # (1,1,3,3)
>>> w = torch.tensor([[[[1.,0.],[0.,1.]]]])                     # (1,1,2,2)
>>> out = torch.nn.functional.conv2d(x, w, stride=1, padding=0)  # 落 aten.convolution
>>> out[0, 0]                                                    # 去 batch/通道维看结果
tensor([[ 6.,  8.],
        [12., 14.]])
```

```
x (1,1,3,3)        w (1,1,2,2)        stride=1, pad=0        out (1,1,2,2)
[[1 2 3]           [[1 0]             L_out = ⌊(3−1)/1⌋+1=2  [[ 6  8]
 [4 5 6]      ⊙    [0 1]]   (只选窗口                          [12 14]]
 [7 8 9]]                    主对角元素)
# out[i,j] = x[i,j] + x[i+1,j+1]
# out[0,0]=1+5=6   out[0,1]=2+6=8   out[1,0]=4+8=12   out[1,1]=5+9=14
```

**为什么需要这个算子 / 数值与精度**　它是 CNN 的特征提取原语：局部连接 + 权重共享。从功能上看，普通卷积是"下采样/保持分辨率的特征变换"，转置卷积是"上采样"（分割 U-Net 的解码器、GAN 生成器）。`groups > 1`（分组卷积，极端即 `depthwise`）用更少参数/计算量换取通道间解耦，MobileNet/ResNeXt 的核心。

**数值与精度陷阱**（Part III 重点）：
1. **累加精度**：卷积内层是 `C_in/groups * ∏kernel` 个乘加，在 `float16`/`bfloat16` 下直接累加会丢精度甚至溢出。硬件/库（cuDNN、oneDNN、NPU 的 CANN）普遍在内部把累加提升到 `float32`，但**输出 dtype** 由调用方决定；inductor 在 NPU 上的 Triton 核要显式选 `tf32`/`fp32` 累加器，否则 bf16 训练会发散。
2. **空滑窗**：当 `(L_in + 2*pad - dilation*(k-1) - 1) < 0` 时 `L_out ≤ 0`，运行期报错（"output size too small"）。这不是 NaN，是硬错误。
3. **非对称 padding**：`padding` 是对称两侧；若模型需要非对称（如 `SAME` 下采样奇数尺寸），PyTorch 先用 `constant_pad_nd`（→ 第 16 章）补一边，再调本算子。本算子本身**不支持**非对称 pad。

**实现逻辑与复杂度**　朴素六重循环是 `O(N * C_out * Cout_sp * C_in * ∏kernel)`（`Cout_sp = ∏L_out`），实测慢到不可用。**实际库都用 im2col + GEMM**：

```python
def convolution_via_im2col(x, w, stride, pad, dilation, groups):
    # 1. im2col：把每个滑窗重排成矩阵的一列
    #    x: (N, Cin, *L_in) -> cols: (N, Cin*∏k, ∏L_out)
    cols = im2col(x, kernel=w.shape[2:], stride, pad, dilation)   # 见下
    # 2. 权重 reshape 成 (Cout, Cin*∏k)
    w2d = w.reshape(C_out, -1)
    # 3. 一次批矩阵乘：每行 = 一个输出通道对一个滑窗的内积
    out = bmm(w2d[None], cols)            # (N, C_out, ∏L_out)，详见第 5 章 bmm
    # 4. reshape 回 (N, C_out, *L_out) + bias
    return out.reshape(N, C_out, *L_out) + bias[:, None]
```

`im2col` 本身（图→列重排）的伪代码（2D、单样本单通道）：

```python
def im2col(x, k, stride, pad, dilation):              # x: (Cin, H, W)
    Hout, Wout = floor((H + 2*pad - d*(k-1) -1)/s)+1, ...   # 同上
    cols = empty(Cin*k*k, Hout*Wout)
    for ci in range(Cin):
      for p in range(k):
        for q in range(k):
          row = ci*k*k + p*k + q
          # 这一行的数据 = 在 x 上以 dilation 步长、stride 滑窗取的一条对角线
          for i in range(Hout):
            for j in range(Wout):
              ii = i*stride - pad + p*dilation
              jj = j*stride - pad + q*dilation
              cols[row, i*Wout+j] = x[ci, ii, jj] if 0<=ii<H and 0<=jj<W else 0
    return cols
```

**关键洞察**：`im2col` 把卷积**降维成第 5 章的 `mm`/`bmm`**。这就是为什么 im2col 是"理解卷积降级可懂性的关键"——卷积不是独立原语，是"GEMM + 一次重排"。代价是 im2col 占用 `O(Cin*∏k * ∏Lout)` 额外内存（核越大、特征图越大越夸张），所以现代库在核小时走 winograd/直接卷积核，核大或分组时才退回 im2col。

时间复杂度（im2col 路径）：im2col 重排 `O(N * Cin*∏k * ∏Lout)`（内存搬运）+ GEMM `O(N * C_out * Cin*∏k * ∏Lout)`（主导项）。空间：im2col 缓冲 `O(N * Cin*∏k * ∏Lout)`（**非零拷贝，必然分配**）。

`groups` 的实现：对每组的输入/权重切片独立做上述流程，或把权重 block-diagonal 化后一次 GEMM。`transposed=True` 的实现：本质是"前向卷积的梯度"——即对 `grad_output` 做一次带 `output_padding`、stride 反向的卷积，再翻核。所以**转置卷积 = 卷积反向的 forward**，二者互为对偶。

**边界与陷阱**
1. **通道整除约束**：`C_in % groups == 0` 且 `C_out % groups == 0` 必须成立，否则运行期报错。depthwise（`groups == C_in == C_out`）是最常见特例。
2. **`output_padding` 只在转置卷积合法**：普通卷积传非零 `output_padding` 会报错。它必须 `< stride`（否则相邻滑窗在输出上重叠错位）。**它的唯一作用是消除歧义**：转置卷积的 `L_out` 公式里，`(L_in-1)*stride` 可能让多个不同的 `L_in'` 映射到同一个 `L_out`，`output_padding` 显式补足这"多出来的尺寸"。
3. **`transposed` 与 weight 布局对调**：普通卷积 `weight=(C_out, C_in/g, ...)`；转置卷积 `weight=(C_in, C_out/g, ...)`。直接把普通卷积的 weight 喂给转置卷积形状对不上，是最常见的新手错误。
4. **`bias` 可选**：`bias=None` 时不加偏置；`bias` 形状必须 `(C_out,)`。
5. **padding 越界产生负 `L_out`**：如 `L_in=3, k=5, pad=0, stride=1` → `L_out = floor((3-4)/1)+1 = 0` → 报错（"输入太小"）。不是 NaN，是硬错误。
6. **类型提升**：`input`/`weight`/`bias` 必须同 dtype 且同 device；不支持跨设备（先 `to`，→ 第 12 章）。`int` 输入不被支持（卷积只对浮点/复数定义）。
7. **非连续输入**：`input` 可以非连续（库内部按 stride 寻址），但连续输入更快（im2col 重排是缓存友好的线性扫描）。

**Inductor 视角**　`aten.convolution` 走 **template** 路径（与 `mm` 同类）：它被 `select_algorithm` 收口，候选核来自 `max_autotune_conv_backends`（GPU 上是 cuDNN/CUTLASS template，CPU 上是 oneDNN 的 `mkldnn._convolution_pointwise`，NPU `_inductor_new` 后端委托给 Triton/CANN 路径），autotune 选最优算法并缓存；输入需 `needs_realized_inputs`（必须物化，不能是 fused view），无法和相邻 pointwise 融合进同一个核。`convolution_backward` 因形状对偶复杂，inductor 当前对它 `make_fallback`（直接调 aten eager，不降级成 Triton）。

---

## 本章小结

`aten.convolution` 是把"1D/2D/3D、普通/转置、空洞、分组"六种变体折叠进**六组参数**的统一原语，掌握它的输出形状公式（普通 vs 转置两条）和 `groups`/`transposed` 的语义就掌握了全部卷积调用的读图能力。理解它在 inductor 里如何降级，关键是 **im2col** 把卷积重排成矩阵乘（→ 第 5 章 `mm`），从而归入 template + `select_algorithm` 路线——这也是为什么卷积和 matmul 是 Part III 里"最难也最像"的两个原语。

下一章我们看卷积的"近亲"：池化（`max_pool2d`/`avg_pool2d`/`_adaptive_avg_pool2d`）——它复用了卷积的滑窗参数（`kernel/stride/padding`），但没有可学权重，inductor 同样走 template/fallback。

---

[上一章 第 12 章](12-memory-layout-dtype.md) 　|　 [下一章 第 14 章](14-pooling.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
