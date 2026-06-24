# 第 2 章 三角与超越函数（Transcendental Functions）

> 本章是 **Part I（数学语义类）**的第二章。所有算子都是**位置保持**的逐元素数学函数：`out[i] = f(in[i])`，形状平凡可推，没有别名风险，inductor 一律落在 **pointwise** 类。但和第 1 章的算术不同，这里的函数在模型里扮演的不是"算数值"，而是**几何相位**（`atan2`）、**概率门控**（`sigmoid`）、**误差积分**（`erf`）三类"基础数学部件"的角色——它们是更上层复合算子（GELU、softmax、角度回归、位置编码…）的**数学基石**。

本章先一张速查表把三角/双曲族打包带过（公式一目了然），再用 Tier D 深讲三个**数值/语义非平凡**的算子：`atan2`（两参象限正确，是 `atan` 补不上的洞）、`sigmoid`（激活与逻辑斯蒂，第 6 章会回链）、`erf`（GELU 与 `_log_softmax` 的数学依赖）。

---

## 本章速查（Tier C）

> 三角/双曲族统一签名：`aten.<f>(Tensor self) -> Tensor`（另有 `.out` 原地重载与若干标量重载，本书不展开）。公式一目了然，**全部 pointwise、全部逐元素、全部输出与输入同形、分配新张量**（无别名）。下表同类打包。

| 算子 | 功能一句话 | 形状 / 风险 | inductor 归类 |
|---|---|---|---|
| `aten.sin` `aten.cos` `aten.tan` | 正弦 / 余弦 / 正切（弧度制） | 同形；`tan` 在 `π/2 + kπ` 附近**数值爆炸**（非 bug，数学本身无界） | pointwise |
| `aten.asin` `aten.acos` | 反正弦 / 反余弦 | 同形；**输入必须在 `[-1,1]`**，否则 NaN（定义域外）；`acos` 在 `±1` 处梯度无穷 | pointwise |
| `aten.atan` | 反正切（单参，**不区分象限**，输出 `(-π/2, π/2)`） | 同形；无定义域限制；但**丢失象限信息**（见下方 `atan2`） | pointwise |
| `aten.sinh` `aten.cosh` | 双曲正弦 / 双曲余弦 | 同形；`cosh` 恒 ≥ 1；`sinh`/`cosh` 对大输入**指数溢出**（`|x|>20` 时 float32 下 `exp` 已接近上溢） | pointwise |
| `aten.asinh` `aten.acosh` `aten.atanh` | 反双曲正弦 / 余弦 / 正切 | 同形；`acosh` 要求 `x ≥ 1`；`atanh` 要求 `|x| < 1`，否则 NaN | pointwise |

**类型提升与广播**：同第 0 章 §0.4 / §0.5。整数输入会先被提升到浮点（三角函数值域是实数，整数域无意义）。本章所有算子对 NaN 输入返回 NaN、对 ±Inf 视数学定义而定（如 `atan(±Inf) = ±π/2`，`tanh(±Inf) = ±1`）。

> **inductor 视角（全族一句话）**：全部降级为 **pointwise**，生成单逐元素 Triton 核，常与加减乘除、`exp`/`log` 融合成一个核。后端（NPU 上 Triton-Ascend）对 `sin`/`cos` 多有硬件近似指令；`tan`/双曲/反三角通常编译为 `sin`/`cos`/`exp`/`log` 的组合或调数学库。本章重点不在"怎么降级"，而在**为什么需要这些函数**和**数值陷阱**。

---

## 深入算子（Tier D）

### `aten.atan2(Tensor self, Tensor other) -> Tensor` — 两参反正切（象限正确）

**作用与语义**　给定点 `(other, self) = (x, y)`（注意 PyTorch 参数顺序：**第一个参数是 y，第二个是 x**，与 `y/x` 的书写顺序相反，这是经典踩坑点），返回该点在极坐标下的辐角 `θ`，值域 `(-π, π]`（完整一圈，覆盖四个象限）。数学定义：

```
θ = atan2(y, x)
  # (x>0)            -> atan(y/x)                         第一/四象限
  # (x<0,  y≥0)      -> atan(y/x) + π                     第二象限
  # (x<0,  y<0)      -> atan(y/x) - π                     第三象限
  # (x=0,  y>0)      ->  π/2
  # (x=0,  y<0)      -> -π/2
  # (x=0,  y=0)      ->  0      （IEEE/PyTorch 约定；部分库返回 NaN）
```

逐元素作用于两个同形张量（支持广播，详见第 0 章 §0.4）；输出形状为广播后的形状，dtype 为两输入提升后的浮点类型。返回新张量，无别名。

**示例**　`atan2(self=y, other=x)` 返回点 `(x, y)` 的辐角，能区分象限——三个点 `(0,1)`/`(1,0)`/`(0,-1)` 分别落到 `π/2`/`0`/`-π/2`，而单参 `atan(y/x)` 把 `(1,1)` 和 `(-1,-1)` 算成同一个比值 `1`、一律给 `π/4`，丢失象限：

```python
>>> y = torch.tensor([1., 0., -1.])     # self = 分子 y
>>> x = torch.tensor([0., 1., 0.])      # other = 分母 x
>>> torch.atan2(y, x)                   # (x,y)=(0,1)/(1,0)/(0,-1) -> π/2, 0, -π/2
tensor([ 1.5708,  0.0000, -1.5708])
>>> torch.atan2(torch.tensor([1., -1.]), torch.tensor([1., -1.]))   # (1,1)->π/4, (-1,-1)->-3π/4
tensor([ 0.7854, -2.3562])
>>> torch.atan(torch.tensor([1., -1.]) / torch.tensor([1., -1.]))   # atan(y/x) 都给 π/4 -> 丢象限
tensor([0.7854, 0.7854])
```

**为什么需要这个算子 / 数值与精度**　这是本章的旗舰难点。`atan(y/x)` 之所以不够用，有两层原因，都必须讲清：

1. **象限丢失（语义层）**：`y/x` 把点 `(1,1)` 和 `(-1,-1)` 算成同一个比值 `1`，`atan` 一律返回 `π/4`。但前者辐角是 `π/4`、后者是 `-3π/4`。`atan2` 靠**分别**看 `y` 和 `x` 的符号恢复象限信息。模型里凡是涉及**角度回归**（目标检测的旋转框角度、机器人朝向、复数相位 `angle()`）、**位置编码**（RoPE / 旋转注意力里的 `freqs`）都直接需要它。
2. **`x=0` 除零（数值层）**：`atan(y/0)` 当 `y≠0` 时是 `±Inf`，`atan(±Inf)=±π/2` 能凑出正确结果但中间值是 Inf，容易在后续融合核里污染计算；`atan(0/0)=atan(NaN)=NaN`。`atan2` 把这些**特例内化**为分支，全程不产生 Inf/NaN。

**实现逻辑与复杂度**　逐元素，无数据依赖：

```python
def atan2(y, x):
    out = empty(broadcast_shape(y.shape, x.shape), dtype=promote(y,x))
    for i in all_indices(out.shape):
        yi, xi = y[idx_align(i, y.shape)], x[idx_align(i, x.shape)]
        # 数学库内部用 sign(x), sign(y), abs(y), abs(x) 的组合 + 象限修正
        out[i] = libm_atan2(yi, xi)        # 单元素 atan2，含全部分支
    return out
```

时间 O(N)、空间 O(N)（N=元素数），分配一份输出张量。底层调的是 C/CUDA/NPU 数学库的 `atan2`，**不是** `atan` + 手写修正——因为象限分支在硬件数学库里已是单条指令/单段近似多项式。

**边界与陷阱**
- **参数顺序**：`atan2(self, other)` 即 `atan2(y, x)`，`self` 是分子 y、`other` 是分母 x。写成 `atan2(x, y)` 会得到 `π/2 - 正确值` 的系统性错误（除轴上点外全错），且不会报错——这是最隐蔽的 bug。
- **`(0,0)`**：PyTorch 返回 `0`（不报错、不 NaN）；与部分数学库的 NaN 约定不同，跨框架移植时要留意。
- **Inf 参与**：`atan2(±Inf, x)` 按 IEEE 约定返回 `±π/2`（x 有限正）；`atan2(±Inf, ∓Inf)` 返回 `±3π/4`。这些是定义良好的，不会出错。
- **`±0`**：`atan2(±0, -0)` 这类有符号零的组合会落到不同象限边界，IEEE 有完整规定；实务中罕有人依赖。
- **类型**：两参广播（第 0 章 §0.4）；整数输入先提升到浮点（第 0 章 §0.5）。

**Inductor 视角**　**pointwise**（两输入逐元素，带广播），与 `sin`/`cos`/`mul`/`add` 同类，常融合进位置编码或角度计算的大 pointwise 核。

---

### `aten.sigmoid(Tensor self) -> Tensor` — 逻辑斯蒂函数（标准激活）

**作用与语义**　逐元素计算 S 型曲线：

```
σ(x) = 1 / (1 + e^{-x})        值域 (0, 1)，严格单调递增
```

输出同形；整数输入提升到浮点；返回新张量，无别名。

**示例**　`sigmoid` 把任意实数压进 `(0,1)`：`0→0.5`，严格单调递增（`-1→0.2689`、`+1→0.7311`）：

```python
>>> torch.sigmoid(torch.tensor([-1., 0., 1.]))   # 0->0.5，单调
tensor([0.2689, 0.5000, 0.7311])
```

**为什么需要这个算子 / 数值与精度**　`sigmoid` 是**门控**与**二分类概率**的标准部件：二分类 logits→概率、LSTM/GRU 的输入/遗忘/输出门、attention 里的门控变体都靠它把任意实数压到 `(0,1)` 区间。它还有两个数值特性使它"比手写 `1/(1+exp(-x))` 更值得做成独立算子"：

1. **数值稳定性（前向）**：朴素实现 `1/(1+exp(-x))` 在 `x` 很负时 `exp(-x)` 上溢（`exp(40)≈2.4e17`，`exp(90)` float32 上 Inf），导致 `1/Inf=0`——结果碰巧对，但中间值 Inf 在融合核里可能污染相邻计算。数学库的 `sigmoid` 用**分段等价式**规避：
   ```
   σ(x) = 1/(1+e^{-x})          当 x ≥ 0
        = e^{x}/(1+e^{x})        当 x < 0      ← 这一支 exp 的参数恒 ≤ 0，永不溢出
   ```
2. **梯度形式优雅（反向）**：`σ'(x) = σ(x)·(1-σ(x))`——**前向值本身就是反向梯度**。这使 autograd/inductor 在写 `sigmoid_backward` 时无需重算 `exp`，直接复用前向输出即可。`sigmoid` 的反向因此常被实现成独立的 pointwise 核（`mul(σ, 1-σ)` 形式）。

> 关于 `sigmoid` 作为**激活函数**的角色（与 relu/gelu/tanh 的对比、为什么深层网络里少用、梯度消失的来由），详见第 6 章。本章只锁定它的**数学定义与导数形式**——因为 `_log_softmax`（第 6 章）和二分类 `bce`（附录 B）都直接依赖它。

**实现逻辑与复杂度**　逐元素：

```python
def sigmoid(x):
    out = empty(x.shape, dtype=promote(x))
    for i in all_indices(x.shape):
        xi = x[i]
        out[i] = 1.0/(1.0+exp(-xi)) if xi >= 0 else exp(xi)/(1.0+exp(xi))  # 分支避免上溢
    return out
```

时间 O(N)、空间 O(N)，分配一份输出。零拷贝与否：**否**（恒新分配）。

**边界与陷阱**
- **bfloat16 精度**：`sigmoid` 在 `|x|` 较大（如 `|x|>20`）时输出饱和到 0 或 1，bfloat16 的 7 位尾数使"接近 0/1 的中间值"丢失严重；在 attention 门控里偶见训练不稳定，是 NPU 上常被 profile 的点。
- **`sigmoid(±Inf)`**：返回 `0`/`1`（定义良好）；`sigmoid(NaN)=NaN`。
- **不要**把它和 `softplus`/`logsigmoid` 混淆——后两者是复合（`log(1+exp)` / `log(sigmoid)`），数值上各有更稳的实现，附录 B 给分解式。
- **复合而非基础**：`silu(x)=x·sigmoid(x)` 是复合（→ `mul` + `sigmoid`，见附录 B），**不是** core 基础算子，本章不单列。

**Inductor 视角**　**pointwise**，常与后续 `mul`/`add`/`tanh` 融合（LSTM 门控是典型融合大户）；反向有专用 `sigmoid_backward` 融合路径，复用前向输出避免重算 `exp`。

---

### `aten.erf(Tensor self) -> Tensor` — 误差函数（GELU / softmax 的数学基石）

**作用与语义**　逐元素计算高斯误差函数（error function）：

```
erf(x) = (2/√π) · ∫₀ˣ e^{-t²} dt        奇函数：erf(-x) = -erf(x)
                                          值域 (-1, 1)，单调递增
                                          erf(0)=0, erf(±∞)=±1
```

输出同形；整数输入提升到浮点；返回新张量，无别名。`erf` **没有初等闭式**——它是一个积分定义的特殊函数，只能数值近似（Abramowitz-Stegun 有理近似 / 有理 Chebyshev 多项式）。

**示例**　`erf` 是奇函数（`erf(-x) = -erf(x)`），`erf(0)=0`，是 `gelu` 精确版里标准正态 CDF `Φ(x)=½(1+erf(x/√2))` 的积分部件：

```python
>>> torch.erf(torch.tensor([0., 0.5, -0.5]))   # erf(0)=0，奇函数：erf(-0.5)=-erf(0.5)
tensor([ 0.0000,  0.5205, -0.5205])
```

**为什么需要这个算子 / 数值与精度**　`erf` 几乎**只在两个地方**被模型用到，但这两处都是核心：

1. **GELU（精确版）**：`gelu(x) = x · Φ(x)`，其中 `Φ(x) = ½(1 + erf(x/√2))` 是标准正态 CDF。这是 BERT/GPT 等大模型激活的默认选择，第 6 章详解。**注意**：`gelu` 有两个变体——`tanh` 近似版（`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`，**不**用 `erf`）和精确 `erf` 版；inductor 要按 `approximate` 参数区分（见第 6 章）。这就是"它在这里被用到"的第一处。
2. **`_log_softmax` / 数值稳定的 softmax 反向**：softmax 的 log 域实现（第 6 章 `_softmax`/`_log_softmax`）在反向里出现形如 `erf` 的积分项吗？严格说 softmax 本身只用 `exp`/`sum`，**不直接用 `erf`**——但**logistic-Normal 采样**、某些概率损失（如 high-dimensional normal CDF 近似）、以及和 `gelu` 共享的 `erf` 内核都依赖它。更准确地说：`erf` 在 core 161 里存在，**主要服务对象就是 `gelu(erf 版)`**。

`erf` 值得做成独立基础算子（而非每次展开成数值积分）的理由：①它是标准数学库（libm/CUDA/NPU）的单条硬件/库函数；②`gelu` 反向 `gelu_backward` 需要 `erf` 和 `erf` 的导数 `erf'(x) = (2/√π)e^{-x²}`（注意后者正好是高斯钟形），独立算子让反向也能走快路径。

**实现逻辑与复杂度**　逐元素，调数学库：

```python
def erf(x):
    out = empty(x.shape, dtype=promote(x))
    for i in all_indices(x.shape):
        out[i] = libm_erf(x[i])     # 内部为有理多项式近似（无闭式）
    return out
```

时间 O(N)、空间 O(N)，分配一份输出。零拷贝与否：否。

**边界与陷阱**
- **无闭式 / 硬件依赖**：`erf` 的精度**完全取决于后端数学库**。CPU libm 通常 ~1 ULP；GPU/NPU 的硬件 `erf` 在尾部（`|x|>4`）精度可能差几 ULP，但此处 `erf` 已饱和到 ±1，相对误差可接受。在 `gelu` 里这点误差通常无所谓，但在概率计算里要留意。
- **`erf` vs `erfc`**：`erfc(x) = 1 - erf(x)` 是互补误差函数，在 `x` 大正时 `erf→1` 而 `erfc→0`，后者保留小数值精度。core 161 **没有** `erfc`（需要时用 `1 - erf`，但大 x 下会丢精度——这是真实存在的坑，`_log_softmax` 等场景偶尔要绕开）。
- **`erf(±Inf)`**：返回 `±1`（定义良好）；`erf(NaN)=NaN`。
- **整数输入**：先提升到浮点（`erf` 值域是实数）。

**Inductor 视角**　**pointwise**，主要出现在 `gelu`（第 6 章）展开图里，常与 `mul`/`add`/常数缩放融合成单核；NPU 上若后端无原生 `erf` 指令，会被 Triton-Ascend 编译为有理多项式近似的 `exp`/`div`/`mul` 组合，此时核体较大、融合收益明显。

---

## 本章小结

本章覆盖了 core aten 的三角、双曲与超越函数族。要点三条：①三角/双曲/反三角族公式一目了然、全是 pointwise，**真正需要警惕的是定义域外的 NaN 和大输入的数值溢出**；②`atan2` 是 `atan` 补不上的洞——靠分别看 `y`/`x` 符号恢复象限，是角度回归与位置编码的标准部件，**参数顺序 `atan2(y,x)` 是经典踩坑点**；③`sigmoid` 与 `erf` 是激活/概率域的数学基石，前者靠分段等价式保数值稳定、后者无闭式而服务于 `gelu(erf 版)`，二者都将在第 6 章激活函数里再次出现。

下一章[第 3 章 比较与布尔/位运算](03-comparison-boolean.md)离开"实数数学"进入"逻辑/判定"——`where` 是那里唯一的 Tier D，它把布尔掩码翻译成数据选择，是掩码语言模型和条件计算的根。

---

[上一章 第 1 章](01-elementwise-arithmetic.md) 　|　 [下一章 第 3 章](03-comparison-boolean.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 本章涉及广播、类型提升、SSA 契约等公共概念，详见第 0 章 §0.4 / §0.5 / §0.7。
