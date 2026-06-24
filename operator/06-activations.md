# 第 6 章 激活函数

激活函数（activation function）是深度神经网络引入非线性的核心机制。一个多层感知机若没有激活层，无论堆多少层线性矩阵乘法，整体仍是仿射变换（affine），无法逼近任意函数；激活算子把每层线性输出逐元素"扭折"，使网络具备万能逼近（universal approximation）能力。这一类算子在领域上的角色高度统一：接在 Linear/Conv/Gemm 之后、归一化层之前，作用在张量最后一维（隐藏维）或通道维上。

从算子语义看，激活函数几乎都是**逐元素（pointwise）的非线性映射**——`relu`/`tanh`/`gelu` 把输入张量按位变换成同形状输出，无跨元素依赖、无归约、无内存格式重排。唯一的两个例外是本章的难点：`_softmax`/`_log_softmax` 沿某一维做归约（先求最大值再归一），`native_dropout` 在训练期按概率随机置零（引入跨样本的随机路由）。正因为这三类语义差异，它们在 Inductor 的归类也各不相同——pointwise / reduction / 走随机数生成的 fallback。本章是 Part I（数学语义类），因此重点放在数学公式、数值稳定性与精度陷阱上。

## 本章速查（Tier C）

| 算子 | 功能一句话 | 形状 / 风险 | Inductor 归类 |
|---|---|---|---|
| `aten.relu` | `max(0, x)`，截断负值 | 输出同形状；不抗 `+inf`/`NaN`（直接透传） | pointwise |
| `aten.leaky_relu` | `x≥0 ? x : slope·x`，带负斜率的 ReLU | 输出同形状；`slope` 默认 0.01 | pointwise |
| `aten.hardtanh` | `clip(x, min_val, max_val)`，默认 `[-1,1]` | 输出同形状；`min_val/max_val` 对称时即 `tanh` 的廉价替身 | pointwise |
| `aten.tanh` | `tanh(x) = (e^x - e^-x)/(e^x + e^-x)` | 输出同形状；大 `|x|` 饱和→梯度趋零 | pointwise |

> `relu`/`leaky_relu`/`hardtanh`/`tanh` 的 forward 都是纯 pointwise 映射，数学平凡，故只进 Tier C 速查；其 `_backward` 版本同样是 pointwise（详见各算子反向的"导数"），不单列 Tier D。本章 Tier D 重点放在数学非平凡或语义特殊的四个算子。

## 深入算子（Tier D）

### `aten.gelu(input, approximate="none")` — 高斯误差线性单元，Transformer 激活的事实标准

**作用与语义**　GELU（Gaussian Error Linear Unit）将输入按标准正态分布的累积分布函数（CDF）平滑门控：

```
近似="none"  (erf 精确版):   GELU(x) = x · Φ(x) = x · ½·(1 + erf(x / √2))
近似="tanh" (tanh 近似版):  GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
```

输入 `x: Tensor`（浮点），标量 `approximate: str ∈ {"none","tanh"}`。输出与输入同形状、同 dtype。返回新张量（非 in-place；`aten.gelu_` 是原地版，core 161 不含）。

**示例**　同一输入 `[-1, 0, 1]`，`approximate="none"`（erf 精确版）与 `"tanh"`（近似版）结果**略有不同**——差异在第 4 位小数，正是两套近似不可混用的实证：

```python
>>> x = torch.tensor([-1.0, 0.0, 1.0])
>>> torch.nn.functional.gelu(x, approximate="none")   # erf 精确版
tensor([-0.1587,  0.0000,  0.8413])
>>> torch.nn.functional.gelu(x, approximate="tanh")   # tanh 近似版
tensor([-0.1588,  0.0000,  0.8412])
```

**为什么需要这个算子 / 数值与精度**　GELU 由 Hendrycks & Gimpel (2016) 提出，是 BERT/GPT/LLaMA 等主流 Transformer 的默认激活。相比 ReLU 的硬截断，GELU 在 `x≈0` 处是**光滑**的（导数连续），对小幅扰动更鲁棒；相比 `tanh`/`sigmoid` 又不会过早饱和，负半轴保留少量信息（`x→-∞` 时 `GELU→0`，但 `x<0` 一段仍有非零响应）。

两个版本的精度差异是核心要点：
- **erf 版**（`"none"`）数学精确，依赖 `erf`（误差函数，回链第 2 章 §2.x）。在 `float16` 下 `erf` 通常需升到 `float32` 计算，否则边缘精度损失明显。
- **tanh 版**（`"tanh"`）是工程近似，避免昂贵的 `erf`（某些后端没有原生 `erf` 指令）。两者在 `|x|>3` 处偏差可忽略，但在 `x≈0` 附近绝对误差约 `1e-3` 量级——对量化或低精度推理敏感的模型需统一版本，**混用会导致前后向数值对不上**。

选择策略：后端有高效 `erf`（如 GPU TensorCore、Ascend 部分核）用 `"none"`；否则用 `"tanh"`。权重 checkpoint 必须记录 `approximate` 值。

**实现逻辑与复杂度**　两种近似的 pointwise 实现（NumPy 风格伪代码）：

```python
# erf 版
def gelu_erf(x):
    cdf = 0.5 * (1 + erf(x * M_SQRT1_2))        # Φ(x), M_SQRT1_2 = 1/√2
    return x * cdf

# tanh 版
def gelu_tanh(x):
    kBeta  = sqrt(2/pi)                          # ≈ 0.7978845608
    kKappa = 0.044715
    inner  = kBeta * (x + kKappa * x**3)
    return 0.5 * x * (1 + tanh(inner))
```

时间复杂度 `O(n)`（逐元素），空间 `O(n)`（分配一个输出；中间 `erf`/`tanh`/`x³` 在 fused kernel 内复用寄存器，不落显存）。无零拷贝路径——`relu`/`hardtanh` 理论上可原地（core 161 含 `relu` 但 `relu_` 不在本章），但 `gelu` 必须新分配。

**边界与陷阱**
- **dtype 提升陷阱**：`gelu` 在 `float16/bfloat16` 输入时，反向公式中的 `exp(-x²/2)`、`tanh` 极易在 `|x|` 较大时下溢。安全做法是 forward 在计算 dtype（通常升到 `float32`）算，再 `to(result_dtype)` 回落——这与 `_softmax` 的 `computation_dtype` 策略一致（详见第 0 章 §0.5 类型提升）。
- **`approximate` 字符串大小写/拼写**：传 `"None"`、`"NONE"`、`"exact"` 都会静默落到非预期分支或报错，务必用字面量 `"none"`/`"tanh"`。
- **NaN 透传**：`erf(NaN)=NaN`、`tanh(NaN)=NaN`，输入含 `NaN` 不会因激活而被"洗掉"。
- **`+inf` 行为**：erf 版 `gelu(+inf)=+inf`、`gelu(-inf)=0`；tanh 版同（`tanh(+inf)=1`），数值安全。
- **反向（backward）不是基础算子**：`aten.gelu_backward` 是复合算子（→ 分解成 `erf`/`exp`/`mul`/`tanh` 等 pointwise 基础算子，分解式见附录 B）。其 erf 版导数为 `grad·(Φ(x) + x·φ(x))`，`φ(x)` 是标准正态 PDF；tanh 版导数涉及 `1 - tanh²(inner)` 的链式求导。

**Inductor 视角**　pointwise——`gelu` 是纯逐元素映射，落到 `pointwise` template，由前端的 `erf`/`exp`/`tanh`/`mul` 原语 fuse 进同一个 Triton kernel。

---

### `aten._softmax(x, dim, half_to_float)` — 数值稳定的归一化指数（softmax 前向）

**作用与语义**　沿 `dim` 维做归一化指数变换，使该维各元素变为非负且和为 1：

```
softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
```

签名 `_softmax(x: Tensor, dim: int, half_to_float: bool) -> Tensor`。输入 `x`（任意浮点），`dim` 指定归约轴，`half_to_float` 仅当 `x` 为 `float16` 时可设 `True`——此时输出升到 `float32`（softmax 中间值范围窄，half 易溢出）。输出形状同输入；`dim` 可为负（按第 0 章 §0.x 负索引规则换算）。返回**连续（contiguous）**张量（eager 语义如此，decomp 显式调用 `.contiguous()` 保证）。

**示例**　沿 `dim=0` 归一化后，输出各元素非负且**和为 1**——大值（`3`）拿到最大权重：

```python
>>> x = torch.tensor([1.0, 2.0, 3.0])
>>> torch._softmax(x, dim=0, half_to_float=False)
tensor([0.0900, 0.2447, 0.6652])
```

**为什么需要这个算子 / 数值与精度**　softmax 是注意力机制（attention）与多分类交叉熵的基石。原始定义 `exp(x)/Σexp(x)` 有致命的数值缺陷：`exp(1000)` 直接 `inf`，而 `exp(x_i - max)` 恒在 `(0,1]`。因此**所有正确实现都必须先减去该维最大值**（max-shift）：

```
softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
```

减去常数不影响结果（分子分母同乘 `exp(-max)` 抵消），但把指数幅值压到 `(-∞, 0]`，`exp` 结果落在 `(0,1]`，彻底避免上溢。这是"log-sum-exp 技巧"的幂形式（详见 `_log_softmax`）。`half_to_float=True` 是另一道保险：softmax 的求和在 `float16` 下精度极差（和可能远大于单个元素），升到 `float32` 求和再归一化是工程标配。

**实现逻辑与复杂度**　官方 decomp（`torch/_decomp/decompositions.py`）实现：

```python
def _softmax(x, dim, half_to_float):
    x = x.contiguous()
    if half_to_float: assert x.dtype == float16
    computation_dtype, result_dtype = elementwise_dtypes(x)   # 默认提升
    x = x.to(computation_dtype)
    if x.numel() == 0:                       # 空张量直接 exp（无 max 可减）
        unnormalized = exp(x)
    else:
        x_max = amax(x, dim, keepdim=True)   # 关键：减最大值
        unnormalized = exp(x - x_max)
    result = unnormalized / sum(unnormalized, dim, keepdim=True)
    if not half_to_float:
        result = result.to(result_dtype)
    return result
```

时间 `O(n)`（每个元素常数次 exp + 一次归约），空间 `O(n)`（分配 `x_max`、`unnormalized`、`result`；`x_max` 因 `keepdim=True` 是 `O(n/dim_size)`）。归约由 `amax`/`sum` 两个 reduction op 实现。

**边界与陷阱**
- **`numel()==0` 分支必须存在**：空输入若直接 `amax` 会得到 `-inf`（空集 max），`x - (-inf) = +inf`，`exp(+inf)=inf`，归一化得 `NaN`。decomp 用空集特判跳过 max-shift（此时直接 `exp(x)` 即 `exp(空)=空`）。
- **`dim` 负值**：`dim=-1` 合法，按 `dim + ndim` 换算。
- **`half_to_float` 与 dtype 不匹配**：`x` 非 `float16` 却传 `True` 触发 `assert` 失败。
- **类型提升**：输入 `bfloat16` 不会触发 `half_to_float`，但中间计算仍按 `computation_dtype` 提升到 `float32`（详见第 0 章 §0.5）。
- **非连续输入**：decomp 强制 `.contiguous()`，会触发一次拷贝；这是 eager 语义的一部分，Inductor 若不希望拷贝需自行优化。
- **含 `inf`/`NaN`**：`x=[inf, inf]` 经 max-shift 变 `[0,0]`，归一得 `[0.5,0.5]`（合理）；`x=[inf, -inf]` 变 `[0, -inf]`，`exp(-inf)=0`，得 `[1, 0]`（合理）。但 `x=[NaN, ...]` → `max=NaN` → 全 `NaN`。

**Inductor 视角**　reduction——max-shift 需要沿 `dim` 的两趟归约（`amax` + `sum`），落到 `reduction` template。Inductor 通常把 `amax`、`exp`、`sum`、`div` fuse 成单个两-pass kernel（splitk 式），避免中间张量物化。

---

### `aten._log_softmax(x, dim, half_to_float)` — 数值稳定的 log-softmax

**作用与语义**　对 softmax 取对数，输出沿 `dim` 维各元素的 log-概率：

```
log_softmax(x)_i = log( softmax(x)_i ) = x_i - log( Σ_j exp(x_j) )
```

签名与 `_softmax` 完全一致：`_log_softmax(x: Tensor, dim: int, half_to_float: bool) -> Tensor`。返回同形状、连续张量；`half_to_float` 语义同上（仅 `float16` 输入可升 `float32`）。

**示例**　log-softmax 值**每个 ≤ 0**（log 域）；同一输入 `[1,2,3]` 下，三个值之差恰为原 `x` 之差（`±1`），因 log-softmax = `x - 常数`：

```python
>>> x = torch.tensor([1.0, 2.0, 3.0])
>>> torch._log_softmax(x, dim=0, half_to_float=False)
tensor([-2.4076, -1.4076, -0.4076])
```

**为什么需要这个算子 / 数值与精度**　`log_softmax` 是交叉熵损失（cross-entropy loss）的标准内部表示。**为什么不直接 `log(softmax(x))`？** 因为 `softmax` 输出值可能极小（接近 0），`log(0)=-inf`；而 `log_softmax` 用 log-sum-exp 直接计算，把减最大值与取对数合并：

```
log_softmax(x)_i = (x_i - max(x)) - log( Σ_j exp(x_j - max(x)) )
```

记 `shifted = x - max(x)`，则结果 = `shifted_i - logsumexp(shifted)`。整个计算中 `exp` 的输入恒 `≤0`（值域 `(0,1]`），`log` 的输入恒 `≥ exp(-∞)` 的正数——**全程无溢出、无 `log(0)`**。这是数值分析里教科书级的稳定化技巧（log-sum-exp trick），任何手写 `log(softmax(x))` 都是 bug 的温床。

精度上，`log_softmax` 比 `log(_softmax(x))` 多一个量级的有效位（前者避免了一次中间 `1/N` 归一带来的舍入放大），在低精度（`float16`）训练里这个差异直接决定梯度是否爆炸。

**实现逻辑与复杂度**　官方 decomp：

```python
def _log_softmax(x, dim, half_to_float):
    x = x.contiguous()
    computation_dtype, result_dtype = elementwise_dtypes(x)
    x = x.to(computation_dtype)
    if x.numel() == 0:
        shifted = x                          # 空集：无需 shift
    else:
        x_max = amax(x, dim, keepdim=True)
        shifted = x - x_max
    shifted_logsumexp = log( sum( exp(shifted), dim, keepdim=True ) )
    result = shifted - shifted_logsumexp
    if not half_to_float:
        result = result.to(result_dtype)
    return result
```

时间 `O(n)`（与 `_softmax` 同阶），空间 `O(n)`。注意 `@out_wrapper(exact_dtype=True)`：当 `half_to_float=False` 时输出 dtype 严格等于输入（不像 `_softmax` 用默认提升，`_log_softmax` 强制 `exact_dtype`——因为 log 域值域宽，half 也能装下负数）。

**边界与陷阱**
- **空输入**：与 `_softmax` 一样必须特判，否则 `amax` 空 = `-inf` → `shifted = x - (-inf) = +inf` → `exp(+inf)=inf` → `log(inf)=inf` → `+inf - inf = NaN`。
- **`-inf` 输入（masked 位置）**：常见于 attention 的 padding mask。`x=[5, -inf]` → `max=5` → `shifted=[0, -inf]` → `sum(exp)=[1]` → `logsumexp=[0]` → `result=[0, -inf]`。即被 mask 位置输出 `-inf`（log-prob 为负无穷），**这是期望行为**，下游交叉熵会忽略。但若整维全是 `-inf`，`sum(exp)=0`，`log(0)=-inf`，`-inf - (-inf)=NaN`——属于用户 mask 全错，不是算子 bug。
- **`exact_dtype` 语义**：`half_to_float=False` 时强制输出与输入同 dtype，即便中间用 `float32` 算。这意味着 `float16` 输入会产生一次精度有损的回落 `to(float16)`。
- **反向 `_log_softmax_backward_data`**：`(grad - softmax·Σgrad)`，依赖前向 softmax 值，是复合算子（附录 B）。

**Inductor 视角**　reduction——同样需要两趟归约（`amax` + `sum(exp)`），落到 `reduction` template，与 `_softmax` 共享 kernel 结构，仅末端 `log` + `sub` 不同。

---

### `aten.native_dropout(input, p, train)` — 训练期随机置零 + 缩放（随机路由）

**作用与语义**　Dropout 按伯努利概率 `p` 把输入元素**随机置零**，幸存元素乘以 `1/(1-p)` 缩放以**保持期望**：

```
训练 (train=True, 0 < p < 1):
    mask_i ~ Bernoulli(1 - p)          # 以概率 (1-p) 取 1，概率 p 取 0
    out_i = mask_i · x_i / (1 - p)
推理 (train=False 或 p == 0):
    out_i = x_i                         # dropout 失效，原样返回
```

签名 `native_dropout(input: Tensor, p: float, train: Optional[bool]) -> (out: Tensor, mask: Tensor)`——**返回元组**：第一项是缩放后的输出，第二项是 `bool` 掩码（反向传播需要）。`p∈[0,1]` 为丢弃概率；`train=None` 时按模块全局 `self.training` 决定。

**示例**　对 `ones(10)` 以 `p=0.5` dropout（**已设种子，实际为随机**）：部分元素被置 0，存活元素被放大 `1/(1-p)=2`——返回 `(out, mask)` 元组，mask 标出存活位置：

```python
>>> torch.manual_seed(0)                          # 固定 RNG，结果可复现
>>> out, mask = torch.native_dropout(torch.ones(10), 0.5, True)
>>> out                                           # 存活元素 ×2，其余 0
tensor([0., 0., 2., 0., 0., 0., 2., 2., 0., 2.])
>>> mask                                          # 4 个 True = 存活
tensor([False, False,  True, False, False, False,  True,  True, False,  True])
```

**为什么需要这个算子 / 数值与精度**　Dropout 是 Hinton et al. (2012) 的正则化（regularization）利器：训练时随机"删神经元"防止共适应（co-adaptation），强迫网络学到冗余表示。缩放因子 `1/(1-p)` 是关键——它使 `E[out] = (1-p)·x/(1-p) + p·0 = x`，**输出期望等于输入**，因此推理时直接恒等映射无需调整。数学上 dropout 极简单（就是"乘掩码 + 缩放"），难点全在**随机性语义与训练/推理模式切换**：

- **随机性**：每次前向掩码不同，导致同一输入在训练期输出抖动。这对图编译（graph compile）是麻烦——Inductor 必须把随机数生成（RNG）从 dropout 内部"提取"出来（functionalization of RNG），变成显式的 `rand_like` + 比较，否则重放（replay）结果不可复现。`decompositions_for_rng.py` 正是为此存在。
- **generator/seed**：dropout 含隐式全局 RNG；ATen 另有 `native_dropout` 的 generator 变体（core 161 的 `native_dropout` 不直接暴露 `generator` 参数，但通过 RNG 钩子注入）。
- **训练/推理二态**：`train=False` 时整个算子退化为 `identity`（返回 `(input, ones_mask)`），编译图需据此消除分支。

**实现逻辑与复杂度**　官方 decomp（`torch/_decomp/decompositions.py`）：

```python
def native_dropout(input, p, train):
    if train and p != 0:
        if p == 1:
            return (zeros_like(input), zeros_like(input, dtype=bool))
        if not input.dtype.is_floating_point:
            raise RuntimeError("dropout 仅支持浮点输入")
        bool_mask = rand_like(input) > p        # 伯努利采样，>p 即丢弃
        res = bool_mask * input * (1.0 / (1.0 - p))
        return (res, bool_mask)
    else:
        return (input, ones_like(input, dtype=bool))
```

时间 `O(n)`（一次随机采样 + 一次逐元素乘缩放），空间 `O(n)`（分配 `bool_mask` 与 `res`）。`p==1` 特判返回全零掩码，避免 `1/(1-1)` 除零。

**边界与陷阱**
- **必须浮点输入**：decomp 显式 `raise`——dropout 的缩放涉及小数，整型输入无意义。手写整数 dropout 会被拒。
- **`p==1` 除零**：若不特判，`1/(1-1)=inf`，输出全 `inf`/`NaN`。特判返回全零是数学一致的（丢弃概率 100% → 全部置零）。
- **`p` 越界**：`p<0` 或 `p>1` 行为未严格定义（伯努利参数非法），依赖 RNG 实现，可能产生全 1 或全 0 掩码。
- **训练/推理不一致**：用户若 `model.eval()` 忘了但传 `train=True`，或反之，编译缓存会击穿（shape 一样但行为不同）。Inductor 在 graph capture 时把 `train` 作为常量特化（specialize）。
- **`mask` 返回值的复用**：反向 `native_dropout_backward(grad, mask, scale) = grad * mask * scale` 直接复用前向 `mask`，不重新采样——这是 dropout 数值正确的核心契约。
- **随机数提取（RNG functionalization）**：`decompositions_for_rng.py` 把 `native_dropout` 列入 `extra_random_decomps`，确保 `rand_like` 被"函数化"成带种子的显式 RNG 节点，否则 `torch.compile` 的图重放会破坏随机性语义。
- **非连续输入**：`rand_like(input)` 与输入同 stride，逐元素乘天然兼容非连续，无需 `.contiguous()`。

**Inductor 视角**　fallback（随机数路径）——forward 表面是 pointwise（`mul` + `scale`），但内部 `rand_like` 引入 RNG 节点，Inductor 把它分解为"显式随机数生成 + pointwise 乘缩放"两段。随机数生成落 `rng` codegen，pointwise 部分与其他算子 fuse；训练态整段算子无法纯 pointwise 优化，故归类为含 RNG 的 fallback 路径。

## 本章小结

激活函数以**逐元素非线性**为绝对主体（`relu`/`leaky_relu`/`hardtanh`/`tanh`/`gelu` 都是 pointwise），它们让网络摆脱纯线性陷阱；其难点不在数学而在**精度**——`gelu` 两套近似版本不可混用、`float16` 下需提升计算 dtype。两个例外是 `_softmax`/`_log_softmax`（reduction，靠 max-shift / log-sum-exp 保证数值稳定，空输入与 `-inf` mask 是高频坑）与 `native_dropout`（含 RNG，靠 `1/(1-p)` 缩放保持期望，训练/推理二态与 RNG functionalization 是编译期主战场）。掌握这八个算子的数值契约与 Inductor 归类，就掌握了深度网络前向最密集的那层计算。下一章我们离开"数学语义"，转向**形状与视图**类算子（reshape/view/permute/contiguous），它们不改数值、只改张量的"看方式"。

---

[上一章 第 5 章](05-linear-algebra-core.md) 　|　 [下一章 第 7 章](07-shape-and-view.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 公共概念：广播 (broadcast)、类型提升 (type promotion)、SSA 契约 详见第 0 章 §0.4 / §0.5 / §0.7。`erf`/`sigmoid` 的语义回链第 2 章 §2.x；`silu`/`mish`/`elu`/`hardsigmoid`/`hardswish`/`softplus` 等复合激活的分解式见附录 B。
