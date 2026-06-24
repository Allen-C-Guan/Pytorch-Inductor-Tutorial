# 附录 B 复合算子 → 基础算子分解 cookbook

本附录的全部论点用一句话就能说完：**core ATen 那 161 个基础算子已经足以表达所有上层算子**。你在 PyTorch 前端看到的 `silu`、`gelu`、`softmax`、`binary_cross_entropy`……这些"复合算子"并不是新的硬件指令，它们只是基础算子（`mul`/`add`/`exp`/`sub`/`where`/`amax`/`cat`/`mm`…）按固定公式拼出来的组合。

下面每一节就是一张"配方"：一句话功能、一条可直接读的分解公式、列出用到的基础算子（标注对应章节）。这些配方与 `torch/_decomp/decompositions.py` 里 inductor 在编译期真正执行的代码一一对应，我把它们用自己的话写清楚，目的是让你建立"复合 = 基础的拼凑"这种直觉。读完之后你会相信：**只要把这 161 个基础算子降级到 Triton/硬件，所有上层模型都能跑**——这正是 inductor 的核心编译策略。

---

## silu

一句话：Swish 激活函数，输入乘以它自己的 sigmoid。

```python
def silu(x):
    return x * sigmoid(x)
```

用到：`mul`（第 1 章）、`sigmoid`（第 2 章）。

---

## hardsigmoid

一句话：ReLU6 风格的廉价 sigmoid 近似，值域 [0, 1]。

```python
def hardsigmoid(x):
    return clamp(clamp(x + 3, min=0), max=6) / 6
```

用到：`add`（第 1 章）、`clamp`（第 1 章）、`div`（第 1 章）。这里两层 clamp 先把 `(x+3)` 截到 `[0, 6]`，再除 6 归一化。

---

## hardswish

一句话：MobileNetV3 用的激活，就是把 `hardsigmoid` 当门控乘回输入。

```python
def hardswish(x):
    return x * hardsigmoid(x)
    # = x * clamp(clamp(x + 3, 0), 6) / 6
```

用到：`mul`（第 1 章）+ 上一条的 `add`/`clamp`/`div`。

---

## softplus_backward

一句话：softplus 的反向，对超过 threshold 的硬上限段透传梯度、否则走 `exp/(1+exp)` 的软段。

```python
def softplus_backward(grad, x, beta, threshold):
    z = exp(x * beta)
    return where((x * beta) > threshold, grad, grad * z / (z + 1.0))
```

用到：`mul`（第 1 章）、`exp`（第 2 章）、`where`（第 3 章）、`gt`（第 3 章）、`div`/`add`（第 1 章）。

---

## leaky_relu_backward

一句话：Leaky ReLU 反向，正段全通、负段缩放一个斜率。

```python
def leaky_relu_backward(grad, x, negative_slope, self_is_result):
    return where(x > 0, grad, grad * negative_slope)
```

用到：`where`（第 3 章）、`gt`（第 3 章）、`mul`（第 1 章）。

---

## mish_backward

一句话：Mish = `x * tanh(softplus(x))` 的反向，用 `tanh` 的导数 `1 - tanh²` 展开。

```python
def mish_backward(grad, x):
    t = tanh(softplus(x))      # softplus = log(1 + exp(x))
    s = sigmoid(x)
    inner = x * s * (1 - t * t)   # tanh 的导数链
    return grad * (t + inner)
```

用到：`tanh`（第 2 章）、`sigmoid`（第 2 章）、`mul`/`sub`/`add`（第 1 章）。

---

## gelu_backward

一句话：GELU 反向，两种近似各对应一条解析导数链。

tanh 近似：

```python
def gelu_backward_tanh(grad, x):
    kBeta  = sqrt(2 / pi) * 0.5      # ≈ 0.7978
    kKappa = 0.044715
    inner = kBeta * (x + kKappa * x**3)
    t = tanh(inner)
    left_d  = 0.5 * (1 + t)
    right_d = 0.5 * x * (1 - t*t) * kBeta * (1 + 3*kKappa*x*x)
    return grad * (left_d + right_d)
```

erf 近似（默认）：

```python
def gelu_backward_erf(grad, x):
    cdf = 0.5 * (1 + erf(x / sqrt(2)))
    pdf = (1/sqrt(2*pi)) * exp(-x*x/2)
    return grad * (cdf + x * pdf)
```

用到：`erf`/`tanh`（第 2 章）、`exp`（第 2 章）、`mul`/`add`/`sub`（第 1 章）。

---

## gelu（两种前向近似）

一句话：高斯误差线性单元，"平滑的 ReLU"。

erf 版（精确）：

```python
def gelu_erf(x):
    return 0.5 * x * (1 + erf(x / sqrt(2)))
```

tanh 版（近似）：

```python
def gelu_tanh(x):
    return 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
```

用到：`erf`/`tanh`（第 2 章）、`mul`/`add`（第 1 章）。

---

## softmax

一句话：把任意实数向量变成概率分布（和为 1）。关键技巧是**减最大值**保数值稳定。

```python
def softmax(x, dim=-1):
    m = amax(x, dim=dim, keepdim=True)   # 每行最大值
    e = exp(x - m)                        # 减最大值，exp 不会溢出
    return e / sum(e, dim=dim, keepdim=True)
```

用到：`amax`（reduction 章）、`sub`（第 1 章）、`exp`（第 2 章）、`sum`（reduction 章）、`div`（第 1 章）。不减最大值时 `exp(large)` 会 inf；减了之后最大项的 exp 恰为 1，分布形状不变。

---

## log_softmax

一句话：softmax 的对数，直接按 logsumexp 写更稳。

```python
def log_softmax(x, dim=-1):
    m = amax(x, dim=dim, keepdim=True)
    shifted = x - m
    return shifted - log(sum(exp(shifted), dim=dim, keepdim=True))
    # 等价于 logsumexp: log(Σexp) - m 的展开
```

用到：`amax`、`sub`、`exp`、`sum`、`log`（第 2 章）。这就是 logsumexp 思想——先减 max，避免大数 exp。

---

## mse_loss

一句话：逐元素差的平方，再做 reduction（mean/sum/none）。

```python
def mse_loss(x, target, reduction="mean"):
    loss = (x - target) ** 2
    return reduction(loss)   # mean → mean(loss), sum → sum(loss), none → loss
```

用到：`sub`（第 1 章）、`pow`/`mul`（第 1 章）、`mean`/`sum`（reduction 章）。反向同理：`2/N * (x - target) * grad`。

---

## binary_cross_entropy

一句话：二分类交叉熵 `- [t·log(x) + (1-t)·log(1-x)]`，用 `maximum(..., -100)` 裁掉 `-inf`。

```python
def binary_cross_entropy(x, target, weight=None, reduction="mean"):
    loss = (target - 1) * maximum(log1p(-x), -100) \
         - target * maximum(log(x), -100)
    if weight is not None:
        loss = loss * weight
    return reduction(loss)
```

用到：`sub`（第 1 章）、`mul`（第 1 章）、`log`/`log1p`（第 2 章）、`maximum`（第 1 章）、`mean`（reduction 章）。`maximum(..., -100)` 是手动下限裁剪，防止 `log(0) = -inf`。

---

## std / var

一句话：方差是"平方偏差的均值"（无偏版用 N-1 校正），标准差是其开方。

```python
def var(x, unbiased=True):
    mu = mean(x)
    n  = x.numel()
    denom = (n - 1) if unbiased else n
    return sum((x - mu) ** 2) / denom

def std(x, unbiased=True):
    return sqrt(var(x, unbiased))
```

用到：`mean`/`sum`（reduction 章）、`sub`/`pow`/`div`（第 1 章）、`sqrt`（第 2 章）。

---

## stack

一句话：沿**新维度**把一组同形 tensor 拼起来。

```python
def stack(tensors, dim=0):
    return cat([unsqueeze(t, dim) for t in tensors], dim=dim)
```

用到：`unsqueeze`（shape 章）、`cat`（shape 章）。`unsqueeze` 先给每个 tensor 插一个长度 1 的新轴，`cat` 再沿该轴拼。

---

## unbind

一句话：沿某维把 tensor 切成一组 slice，并把那一维 squeeze 掉。

```python
def unbind(x, dim=0):
    n = x.shape[dim]
    return [squeeze(slice(x, dim=dim, start=i, end=i+1), dim) for i in range(n)]
```

用到：`slice`（shape 章）、`squeeze`（shape 章）。本质是"逐片 slice + 去 size-1 维"。

---

## roll

一句话：沿某维循环移位——把后段切下来拼到前段前面。

```python
def roll(x, shifts, dims):
    # 沿 dims 的每个轴，把 x 切成 [后 shifts 个, 前 N-shifts 个]，再 cat 回去
    back  = slice(x, dim=d, start=-shift, end=size)
    front = slice(x, dim=d, start=0,     end=size-shift)
    return cat([back, front], dim=d)
```

用到：`slice`、`cat`（shape 章）。

---

## matmul

一句话：通用矩阵乘，按维度分派到 `mm` / `bmm`，更高维靠 `expand` 广播。

```python
def matmul(a, b):
    if a.dim() == 2 and b.dim() == 2:
        return mm(a, b)                       # 纯 2D
    # 批量情形：把批量维 reshape/expand 对齐后走 bmm
    a_b, b_b = align_batch(a, b)              # expand 广播批量维
    return bmm(a_b, b_b).reshape(out_shape)
```

用到：`mm`/`bmm`（matmul 章）、`expand`（shape 章）、`reshape`（见下）。

---

## transpose

一句话：交换两个指定维度，是 `permute` 的特例。

```python
def transpose(x, dim0, dim1):
    perm = list(range(x.dim()))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return permute(x, perm)
```

用到：`permute`（shape 章）。

---

## reshape

一句话：改形状不改数据；连续时等价于 `view`，否则先 `contiguous` 再 `view`。

```python
def reshape(x, shape):
    if x.is_contiguous():
        return view(x, shape)
    else:
        return view(contiguous(x), shape)
```

用到：`view`（shape 章）、`contiguous`/`clone`（见下）。

---

## contiguous

一句话：把非连续张量重排成连续布局，底层就是一次 `clone`。

```python
def contiguous(x, memory_format=torch.contiguous_format):
    if x.is_contiguous():
        return x
    return clone(x, memory_format=memory_format)
```

用到：`clone`（memory 章）。

---

## to(dtype / device)

一句话：类型/设备转换统一走 `_to_copy`（拷贝 + 转换）。

```python
def to(x, dtype=None, device=None):
    return _to_copy(x, dtype=dtype, device=device)
```

用到：`_to_copy`（memory 章）。

---

## zeros / ones / eye / linspace / randint

一句话：常量张量构造，全部归结到 `full` 或 `arange`。

```python
zeros(shape)  = full(shape, 0)
ones(shape)   = full(shape, 1)
eye(n)        = full((n, n), 0) 然后对角填 1   # 用 scatter / index_put
linspace(s, e, n) = arange(n) * ((e-s)/(n-1)) + s
randint(low, high, shape) = (rand(shape) * (high-low) + low).to(int)
```

用到：`full`（creation 章）、`arange`（creation 章）、`mul`/`add`（第 1 章）、`rand`（creation 章）。

---

## argsort

一句话：返回排序后的索引序列，就是 `sort` 的 `.indices` 分量。

```python
def argsort(x, dim=-1):
    values, indices = sort(x, dim=dim)
    return indices
```

用到：`sort`（reduction 章）。

---

## 末尾说明

这些配方不是教学示意——它们就是 inductor 在 `torch/_decomp/decompositions.py` 里**真正执行**的东西。编译期，复合算子先被换成上面这些基础算子的组合，然后基础算子再被 pointwise / reduction / template 三大降级通道翻译成 Triton kernel（进而落到硬件）。换句话说：**你写的是 2000 个算子，编译器只关心 161 个**。

继续阅读：
- [附录 C：Inductor cheatsheet](appendix-c-inductor-cheatsheet.md) —— pointwise/reduction/template 三大降级通道速查。
- [README](README.md) —— 回到本书目录。
