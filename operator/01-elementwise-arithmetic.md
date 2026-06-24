# 第 1 章 逐元素算术（Elementwise Arithmetic）

> 本章是 **Part I（数学语义类）** 的第一章，也是全书算子数量最多、出现频率最高的一类。它们全部是**位置保持 (position-preserving)** 算子：`out[i]` 只依赖 `in[i]`（含广播对齐），形状平凡可推，数学公式就是算子的全部内容。

逐元素算术是神经网络计算图的"水泥"——权重与激活的逐点相乘（`mul`）、残差连接的逐点相加（`add`）、学习率作用到梯度上的标量除法（`div`）、激活前的尺度裁剪（`clamp`）、指数缩放（`exp`）……几乎所有 Part II/III 的复杂算子展开到最底层，都由这一章的算子与第 2 章的超越函数拼接而成。换句话说：**这一章的算子单个看都简单，但它们的数量级是全图节点数的大头**——inductor 的 pointwise 融合 (pointwise fusion) 主要就是在打它们的主意，把一长串 `add/mul/div/exp` 焊进一个 Triton 核里。所以对 inductor 开发者，本章除了"算子在干什么"，更要盯住两件事：①**数值与精度**（bfloat16 下 `div`/`exp`/`rsqrt` 的累加误差、整数除的类型提升陷阱）；②**广播与类型提升这两条隐形规则如何作用于每个算子**（详见第 0 章 §0.4 / §0.5）。

本章把公式一目了然的算子（`add`/`mul`/`exp`/`log`/`maximum`/`floor`…）压缩进一张速查表（Tier C），把**数学非平凡或带经典坑**的五个算子（`div`、`pow`、`fmod`/`remainder`、`clamp`）拎出来逐个详解（Tier D）。

---

## 本章速查（Tier C）

> 以下算子公式一目了然，逐元素作用于广播对齐后的形状，输出 dtype 由类型提升规则决定（详见第 0 章 §0.5）。inductor 视角一律归 **pointwise**（标量映射，可与其他 pointwise 融合），不再逐个重复。

### 加减乘与符号族

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.add(a, b, *, alpha=1)` | `a + alpha*b`（带标量缩放） | 广播；`alpha` 是 Python 标量 | pointwise |
| `aten.sub(a, b, *, alpha=1)` | `a - alpha*b` | 广播 | pointwise |
| `aten.mul(a, b)` | 逐元素相乘 | 广播 | pointwise |
| `aten.neg(a)` | 取反 `-a` | 同形；整数取反不变 dtype | pointwise |
| `aten.abs(a)` | 绝对值 `\|a\|` | 同形 | pointwise |
| `aten.sign(a)` | 符号函数（-1/0/+1） | 同形；复数返回 `a/\|a\|` | pointwise |
| `aten.reciprocal(a)` | 倒数 `1/a` | 同形；`0` → `inf`，**无零保护** | pointwise |

> 注：`add`/`sub` 的 `alpha` 参数是"省一次 `mul`"的语法糖，tracing 后通常被拆成 `mul`+`add`。`reciprocal` 是 `div(1, a)` 的等价特化，但**不**做零除保护——若 `a` 含 0，结果直接是 `inf`/`nan`。

### 指数与对数族

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.sqrt(a)` | 平方根 `√a` | 同形；`a<0` → `nan` | pointwise |
| `aten.rsqrt(a)` | `1/√a` | 同形；**反向传播主力**（比 `sqrt`+`div` 快一拍） | pointwise |
| `aten.exp(a)` | `e^a` | 同形；大正数 → `inf`（溢出） | pointwise |
| `aten.expm1(a)` | `e^a - 1` | 同形；**小 `a` 数值稳定**（避免大数相消） | pointwise |
| `aten.log(a)` | 自然对数 `ln(a)` | 同形；`a<=0` → `-inf`/`nan` | pointwise |
| `aten.log2(a)` / `aten.log10(a)` | 以 2 / 以 10 为底 | 同形；同上 | pointwise |
| `aten.log1p(a)` | `ln(1+a)` | 同形；**小 `a` 数值稳定**（避免 `1+a` 丢精度） | pointwise |

> 重点：`expm1`/`log1p` 是为**小绝对值**输入设计的数值稳定版本——当 `a≈0` 时，`exp(a)-1` 直接算会因大数相消 (catastrophic cancellation) 丢失几乎全部有效位，`expm1` 用专门的泰勒展开实现避免这个问题。`log1p` 同理。在 `gelu`/`softplus`/损失函数里它们是常客。

### 逐元素极值与取整族

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.maximum(a, b)` | 逐元素取大（**返回非 NaN 那个**） | 广播；`maximum(nan, 1)=1` | pointwise |
| `aten.minimum(a, b)` | 逐元素取小（同上 NaN 规则） | 广播 | pointwise |
| `aten.floor(a)` | 向 −∞ 取整 | 同形；浮点才合法 | pointwise |
| `aten.ceil(a)` | 向 +∞ 取整 | 同形 | pointwise |
| `aten.round(a)` | 四舍六入五成双（banker's rounding） | 同形；**非"四舍五入"** | pointwise |
| `aten.trunc(a)` | 向 0 取整（截断小数） | 同形 | pointwise |

> 重点 ① **`maximum`/`minimum` 的 NaN 语义**与 `where(a>b, a, b)` **不一致**：前者"传染 NaN"的反向是"非 NaN 优先"（只要有一个不是 NaN 就返回那个），后者严格传播 NaN。模型里写 `relu` 等价物时这个差异偶尔咬人。② **`round` 用的是银行家舍入**（round-half-to-even），`round(0.5)=0`、`round(1.5)=2`、`round(2.5)=2`——这是 IEEE 754 默认，与中小学"四舍五入"不同，移植代码时注意。③ `floor`/`ceil`/`trunc` 对 `±0.5` 各不同：`trunc(-0.5)=0`、`floor(-0.5)=-1`、`ceil(-0.5)=-0`。

---

## 深入算子（Tier D）

### `aten.div(a, b, *, rounding_mode=None)` — 逐元素除法，可选取整模式

**作用与语义**　`a / b` 的逐元素版本，形状按广播对齐（详见第 0 章 §0.4）。`rounding_mode` 决定如何处理商：

- `None`（默认，"真除 true division"）：`out = a / b`，**浮点结果**。
- `"trunc"`：`out = trunc(a / b)`，向零取整。
- `"floor"`：`out = floor(a / b)`，向 −∞ 取整。

输入两个张量；输出张量与广播后的形状一致，dtype 由 `rounding_mode` 和类型提升共同决定（见下）。无别名、产出新张量。

**示例**　同一对整数 `[7, -7] / [2, 2]`，三种 `rounding_mode` 结果截然不同——注意默认真除把整数**提升成了浮点**：

```python
>>> a, b = torch.tensor([7, -7]), torch.tensor([2, 2])
>>> torch.div(a, b)                            # 默认真除 -> 提升到 float
tensor([ 3.5000, -3.5000])
>>> torch.div(a, b, rounding_mode="trunc")     # 向 0 截断 -> 留在整数
tensor([ 3, -3])
>>> torch.div(a, b, rounding_mode="floor")     # 向 -∞ -> 负数处与 trunc 不同
tensor([ 3, -4])
```

**为什么需要这个算子 / 数值与精度**　`div` 是少数**一个算子三种语义**的 core 算子——`rounding_mode` 把"真除 / 截断除 / 地板除"合并到一个签名下。这一点在数值上的连锁后果如下，是经典坑集中地：

| 输入 dtype | `rounding_mode=None`（真除） | `rounding_mode="trunc"` | `rounding_mode="floor"` |
|---|---|---|---|
| 整数 / 整数（如 `int64 / int64`） | **先提升到默认浮点**（通常 `float32`）再除 → 浮点结果 | 留在整数域，向零截断 → **整数结果** | 留在整数域，向 −∞ 取整 → **整数结果** |
| 浮点 / 浮点 | 浮点结果 | 浮点（小数部分被截断） | 浮点（向 −∞ 取整） |
| 整数 / 浮点 | 类型提升到浮点 → 浮点结果 | 浮点截断 | 浮点地板 |

**最关键的坑**：两个 `int64` 张量做 `div`，默认 `rounding_mode=None` 时，结果**不是**整数而是 `float32`（"真除"语义强制提升到浮点）。这与 NumPy 的 `np.true_divide` 一致，但与 Python 2 / C 的整数除法直觉相反。如果想要整数整除，**必须**显式传 `rounding_mode="trunc"` 或 `"floor"`——否则你会无声地得到一个浮点张量，下游期望整数的算子（如 `index`）就会报错。

- `trunc` vs `floor` 对**负数**不同：`div(-7, 2, trunc) = -3`（向 0），`div(-7, 2, floor) = -4`（向 −∞）。这是 `fmod` vs `remainder` 差异的同一根源（见下）。
- 浮点 `div` 的精度：bfloat16 下 `a/b` 当 `b` 很小时误差显著放大；若 `b` 可能为 0，结果 `inf`/`nan` 会向后传染。inductor 在 NPU 上对带 `div` 的 pointwise 段通常保留 `float32` 累加。
- `b=0` 的行为：浮点 `1.0/0.0 = inf`、`0.0/0.0 = nan`、`-1.0/0.0 = -inf`（IEEE 754）；整数 `1/0` 在 `rounding_mode="trunc"/"floor"` 下抛运行时错误。

> 复合算子提示：`torch.floor_divide(a, b)` 等价 `aten.div(a, b, rounding_mode="floor")`；`torch.true_divide` 等价 `rounding_mode=None`。本书不展开这两个 Python 层别名，见附录 B。

**实现逻辑与复杂度**　NumPy 风格伪代码：

```python
def div(a, b, *, rounding_mode=None):
    out_shape = broadcast(a.shape, b.shape)
    # 关键：dtype 求解
    if rounding_mode is None:                      # 真除
        out_dtype = promote_to_float(a.dtype, b.dtype)   # 整数会被提升到浮点
    else:                                          # trunc / floor
        out_dtype = promote(a.dtype, b.dtype)      # 可能仍是整数
    out = empty(out_shape, dtype=out_dtype)
    for idx in all_indices(out_shape):
        ai = a[broadcast_idx(idx, a.shape)]
        bi = b[broadcast_idx(idx, b.shape)]
        q   = ai / bi                              # 真值
        if rounding_mode == "trunc":
            q = trunc(q)
        elif rounding_mode == "floor":
            q = floor(q)
        out[idx] = cast(q, out_dtype)
    return out
```

时间复杂度 O(N)（N 为输出元素数），空间 O(N)（必分配新张量）。无零拷贝路径。整数取整模式在硬件上通常走"除法 + 修正"两条指令（因为整除向 0 与向 −∞ 对负数不同）。

**边界与陷阱**

1. **整数 dtype 提升**：`int64 / int64`（`rounding_mode=None`）→ `float32`（不是 `float64`！PyTorch 默认浮点是 `float32`）。若想要 `float64`，需先把输入 `.double()`。
2. **`rounding_mode` 默认值的历史**：早期 `torch.div` 对整数输入默认是 `floor`（向 −∞），2.0 起改为 `None`（真除）。移植老代码务必检查。
3. **`0/0` 与 `x/0`**：浮点下分别是 `nan` 和 `±inf`（不报错）；整数取整模式下 `x/0` 抛 `RuntimeError`。
4. **非连续输入**：`div` 对非连续张量无特殊要求，按 stride 寻址即可；inductor 会把它和邻居一起融合，stride 在编译期静态求解。
5. **NaN 传播**：`nan / x = nan`、`x / nan = nan`，与 IEEE 754 一致。

**Inductor 视角**　pointwise——标量映射 `a/b`（加可选 `trunc`/`floor` 后处理），与其他 pointwise 算子一视同仁地融合进同一个 Triton 核；整数取整变体在 lowering 时映射到对应的 Triton 整除内建 (`llvm.sdiv` / 自定义 floor-div 助手)。

---

### `aten.pow(a, exponent)` — 逐元素幂 `a ** exponent`

**作用与语义**　逐元素计算 `a` 的 `exponent` 次方。两个输入都可以是张量（逐元素）或其中一个是标量；形状按广播对齐。`out = a ** exponent`。返回新张量，无别名。

**示例**　整数底的三种指数命运——非负整幂留在整数域，负整幂**报错**，浮点/分数指数**提升到浮点**：

```python
>>> a = torch.tensor([2, 3])
>>> torch.pow(a, 2)          # 非负整指数 -> 留在 int64
tensor([4, 9])
>>> torch.pow(a, -1)         # 负整数指数 -> 报错（不是悄悄提升！）
RuntimeError: Integers to negative integer powers are not allowed.
>>> torch.pow(a, -1.0)       # 浮点指数 -> 提升到 float32
tensor([0.5000, 0.3333])
```

**为什么需要这个算子 / 数值与精度**　`pow` 的"数学"很简单，但**整数底的类型提升**是它入选 Tier D 的唯一原因——这是另一个经典坑：

| 底 `a` 的 dtype | 指数 `exponent` | 结果 dtype | 说明 |
|---|---|---|---|
| 整数（如 `int64`） | **非负整数** | 同底（`int64`） | 整数幂，留在整数域 |
| 整数（如 `int64`） | **负整数** | **报错**（`RuntimeError`） | 整数域无 `1/x`，PyTorch 直接拒绝 |
| 整数（如 `int64`） | **浮点 / 分数**（`-1.0`、`0.5`…） | **提升到浮点**（`float32`） | 强制提升到浮点域计算 |
| 浮点 | 任意 | 同底浮点 | 正常 |
| 负浮点底 + 分数指数 | — | `nan` | 实数域无定义 |

**最关键的坑**：整数底的指数有**两种**截然不同的命运——**负整数指数直接报错**（`torch.pow(int_tensor, -1)` 抛 `RuntimeError: Integers to negative integer powers are not allowed.`），而**浮点 / 分数指数静默提升到 `float32`**（`torch.pow(int_tensor, -1.0)` → `tensor([0.5000, 0.3333])`）。后者才是"悄悄改 dtype"的 silent bug：下游期望整数的算子会因 dtype 不匹配断链。想要负整数幂，先把底 `.float()`。这与 `div` 的整数提升是同一类机制（详见第 0 章 §0.5）。

- **负底 + 分数指数 → `nan`**：`pow(-2.0, 0.5) = nan`（实数域下根号负数无定义）。模型里若可能出现负底（如未加保护的 `pow(x, 0.5)` 当 `sqrt` 用），务必先 `clamp_min` 或用 `abs`。
- **`pow(x, 2)` vs `x*x`**：数值结果相同，但 `x*x` 通常更快（一条 `mul`），`pow(x,2)` 在某些后端会走通用幂路径（`exp(2*log(x))`），既慢又在 `x<=0` 时给出 `nan`。inductor 一般会把整数指数的 `pow` 特化成连乘，但**不要依赖**——写 `x*x` 更稳。
- **`pow(0, 0) = 1`、`pow(0, -1) = inf`**：遵循 C/IEEE 约定。
- **精度**：bfloat16 下 `pow` 的误差比 `mul`/`div` 大得多（内部走 `exp(b*log(a))`），关键路径上能用整数指数的连乘就用连乘。

**实现逻辑与复杂度**　NumPy 风格伪代码：

```python
def pow(a, exponent):
    a_t, e_t, out_shape = broadcast(a, exponent)
    # dtype 决策
    if is_integer(a.dtype):
        if (is_integer(e_t.dtype) and all(e >= 0 for e in e_t)) \
           or (is_scalar(exponent) and exponent == int(exponent) and exponent >= 0):
            out_dtype = a.dtype                       # 留在整数域
        else:
            out_dtype = default_float_dtype()         # 提升到 float32
    else:
        out_dtype = promote(a.dtype, e_t.dtype)
    out = empty(out_shape, dtype=out_dtype)
    for idx in all_indices(out_shape):
        out[idx] = a_t[idx] ** e_t[idx]               # 硬件 pow 或 exp/log 复合
    return out
```

时间 O(N)，空间 O(N)。底层实现分两条路：① 整数指数走"平方-乘"快速幂（log(指数) 次乘法）；② 通用浮点幂走 `exp(e * log(a))`。后者在 `a<=0` 且 `e` 非整数时返回 `nan`。

**边界与陷阱**

1. **整数底 × 负整数指数 → 报错**（`RuntimeError`，非 silent）；**整数底 × 浮点/分数指数 → 静默提升到 `float32`**——这才是最常见的 silent dtype bug，下游整数算子会因此报 dtype 不匹配。
2. **负底分数指数 → `nan`**：`pow(-1.0, 0.5)` 不报错，返回 `nan`。
3. **`0 ** 负数 = inf`**、`0 ** 0 = 1`：与 NumPy 一致，但与某些数学教材不同。
4. **复数底**：core aten 的 `pow` 不直接处理复数（复数有专门算子，超出本章），传复数张量会落到非 core 路径。
5. **大指数溢出**：`pow(10.0, 100)` → `inf`（float32 上限约 `3.4e38`）。

**Inductor 视角**　pointwise——标量映射，与其他 pointwise 融合；整数指数常被特化为乘法链，通用情况映射到 Triton 的 `libdevice.pow`（底层 `exp+log`）。dtype 提升在 tracing 时已求解，核内 dtype 确定。

---

### `aten.fmod(a, b) 与 aten.remainder(a, b)` — 两种"取余"，符号约定相反

**作用与语义**　两个算子都做"取余"，但对**负数**的符号约定截然不同，这是它们并列存在的唯一理由：

- `aten.fmod(a, b)`：结果**符号同被除数 `a`**（C/`%` 语义）。定义为 `a - trunc(a/b) * b`。
- `aten.remainder(a, b)`：结果**符号同除数 `b`**（Python `%` 语义）。定义为 `a - floor(a/b) * b`。

两者形状按广播对齐，输出 dtype = `promote(a.dtype, b.dtype)`（**不会**强制提升到浮点，与 `div` 默认不同）。返回新张量。

对照表（`a, b` 为整数时同样适用，下例用浮点便于看符号）：

| `a` | `b` | `fmod(a,b)`（同 `a`） | `remainder(a,b)`（同 `b`） |
|---|---|---|---|
| `7` | `3` | `1` | `1` |
| `-7` | `3` | **`-1`** | `2` |
| `7` | `-3` | `1` | **`-2`** |
| `-7` | `-3` | `-1` | `-1` |

记忆法：**`fmod` 把商向 0 截断（`trunc`），`remainder` 把商向 −∞ 取整（`floor`）**——这与 `div` 的 `rounding_mode="trunc"` vs `"floor"` 完全对应。所以恒等式 `a == fmod(a,b) + trunc(a/b)*b` 与 `a == remainder(a,b) + floor(a/b)*b` 恒成立。

**示例**　同一对 `(-7, 3)`，两者符号相反——`fmod` 同被除数 `a`，`remainder` 同除数 `b`：

```python
>>> a, b = torch.tensor([-7]), torch.tensor([3])
>>> torch.fmod(a, b)        # 商向 0 截断 -> 符号同 a
tensor([-1])
>>> torch.remainder(a, b)   # 商向 -∞ -> 符号同 b
tensor([2])
```

**为什么需要这个算子 / 数值与精度**　模型里取余主要用于：①周期性索引（位置编码的周期展开、环形缓冲）；②梯度裁剪里的符号保持（`fmod` 保留梯度方向）；③与 Python `%` 语义对齐的代码移植（用 `remainder`）。**选哪个完全取决于你想要结果的符号跟谁**——这是移植 C 代码（`fmod`）vs Python 代码（`remainder`）时最容易踩的语义坑。

数值上两者都简单（一次除法 + 一次乘法 + 一次减法），无特殊稳定性问题；但浮点下 `fmod(1.0, 0.1)` 这类"数学上整除但浮点上有残余"的情况，结果可能不是 0 而是 `0.0999...`——这是浮点表示固有的，不是算子 bug。

**实现逻辑与复杂度**　NumPy 风格伪代码（两者并列）：

```python
def fmod(a, b):                           # 符号同 a
    out_shape = broadcast(a.shape, b.shape)
    out = empty(out_shape, dtype=promote(a.dtype, b.dtype))
    for idx in all_indices(out_shape):
        out[idx] = a[idx] - trunc(a[idx] / b[idx]) * b[idx]
    return out

def remainder(a, b):                      # 符号同 b
    # 同上，把 trunc 换成 floor
    for idx in ...:
        out[idx] = a[idx] - floor(a[idx] / b[idx]) * b[idx]
    return out
```

时间 O(N)，空间 O(N)。整数版可直接用硬件 `srem`/`urem` 指令（截断语义对应 `fmod`，地板语义需要一次条件修正）。浮点版走 `div`+`trunc`/`floor`+`fma`。

**边界与陷阱**

1. **负数符号差异**（见上表）：这是 90% 的 bug 来源。移植 Python `%` 必须用 `remainder`，移植 C `%` 必须用 `fmod`。
2. **`b == 0`**：浮点下 `fmod(a, 0) = nan`、`remainder(a, 0) = nan`；整数下两者都抛 `RuntimeError`。
3. **结果为零时的符号**：`fmod(5, 5) = 0`（不是 `-0`）；但当 `a` 为负且整除时，`fmod(-5, 5) = -0`（IEEE 带符号零），`remainder(-5, 5) = 0`。极少数代码会区分 `+0`/`-0`，注意。
4. **`fmod`/`remainder` 不做类型提升到浮点**：`int64 % int64` 结果仍是 `int64`（与 `div` 默认行为不同，与 `floor_divide` 一致）。
5. **NaN 传播**：任一输入 `nan` → 结果 `nan`。

> 复合算子提示：Python 的 `%` 运算符在张量上映射到 `aten.remainder`；C 风格的取余用 `torch.fmod`。两者都是 core，无需分解。

**Inductor 视角**　pointwise——标量映射，与其他 pointwise 融合；整数版 lowering 到硬件取余指令或 `sdiv`+修正序列，浮点版走 `div`+`floor`/`trunc`+`fma`。

---

### `aten.clamp(input, min=None, max=None)` — 把元素限制在 `[min, max]` 区间内

**作用与语义**　逐元素把 `input` 限制在 `[min, max]` 区间内：小于 `min` 的置为 `min`，大于 `max` 的置为 `max`，其余不变。`min`/`max` 可以都是 `None`（但**不能同时**为 `None`，否则无意义），也可以只给一个：

| 调用形式 | 语义 | 等价 |
|---|---|---|
| `clamp(x, min, max)` | 双边裁剪 | `min(maximum(x, min), max)` |
| `clamp(x, min=m)` | 下界裁剪（仅限下界） | `maximum(x, m)` |
| `clamp(x, max=M)` | 上界裁剪（仅限上界） | `minimum(x, M)` |

`min`/`max` 可以是 Python 标量，也可以是与 `input` 可广播的张量（core 重载里 `clamp` 有张量版本 `aten.clamp` + 标量版本，tracing 后统一）。输出形状 = 广播后的形状，dtype = `input` 的 dtype（标量边界会被 cast 到 `input.dtype`，**不**做提升）。

**示例**　`clamp(x, min=0)` 即 `relu`；双边 `clamp(x, 0, 1)` 把值夹进 `[0, 1]`：

```python
>>> x = torch.tensor([-1.0, 0.5, 3.0])
>>> torch.clamp(x, min=0.0)      # 下界裁剪 == relu
tensor([0.0000, 0.5000, 3.0000])
>>> torch.clamp(x, 0.0, 1.0)     # 双边裁剪到 [0, 1]
tensor([0.0000, 0.5000, 1.0000])
```

**为什么需要这个算子 / 数值与精度**　`clamp` 在模型里无处不在：

- **梯度裁剪 / 激活裁剪**：`relu` 本质是 `clamp(x, min=0)`；`hardtanh` 是 `clamp(x, -1, 1)`；很多自定义激活的"饱和段"都是 `clamp`。
- **数值稳定**：`log`/`sqrt` 前先 `clamp_min` 避免 0/负数；`softmax`/`attention` 里对 score 做 `clamp_max` 防止 `exp` 溢出；`pow(x, 0.5)` 前先 `clamp_min(0)` 防止负底 `nan`。
- **物理约束**：概率值 `clamp(p, 0, 1)`、角度归一化等。

数值上 `clamp` 是纯比较 + 选择，**无精度损失**（不改变落在区间内的值）。唯一注意：`min > max` 时行为是"**后者优先**"——`clamp(x, min=5, max=3)` 对所有元素返回 `3`（先取 `maximum(x,5)` 再 `minimum(·,3)`，结果恒为 `3`），不报错。这是"先 min 后 max"实现顺序的直接后果，偶尔会让用户困惑。

**实现逻辑与复杂度**　NumPy 风格伪代码：

```python
def clamp(x, min=None, max=None):
    assert min is not None or max is not None
    out_shape = broadcast(x.shape, min.shape if is_tensor(min) else x.shape,
                                 max.shape if is_tensor(max) else x.shape)
    out = empty(out_shape, dtype=x.dtype)
    for idx in all_indices(out_shape):
        v = x[idx]
        if min is not None:
            v = maximum(v, min_idx(idx))    # 标量或广播
        if max is not None:
            v = minimum(v, max_idx(idx))
        out[idx] = v
    return out
```

时间 O(N)，空间 O(N)。无零拷贝路径（即便 `min`/`max` 都是 `None` 的退化情况也要求至少给一个边界）。

**边界与陷阱**

1. **`min == max`（同值）**：结果对所有元素返回该值（恒等于先 `maximum` 再 `minimum`）。
2. **`min > max`**：不报错，结果恒为 `max`（见上）。
3. **`min`/`max` 为 `None`**：必须至少给一个；两个都 `None` 在 Python API 抛 `ValueError`。
4. **整数 `input` + 浮点边界**：边界被 cast 到整数 dtype（向 0 截断），**不**提升 `input`——这点与 `add`/`mul` 的类型提升规则不同，是 `clamp` 的特殊设计（裁剪不应改变 dtype）。若想保留浮点边界，先把 `input` 转 `float`。
5. **NaN 处理**：`clamp(nan, min, max)` → `nan`（比较运算对 NaN 返回 False，所以 NaN 既不被 min 替换也不被 max 替换，原样保留）。这与 `maximum`/`minimum` 的"非 NaN 优先"语义**不同**——`clamp` 内部虽然用 `maximum`/`minimum` 实现，但 ATen 的 `clamp` 对 NaN 有专门处理使其传播 NaN。
6. **张量边界（Tensor overload）**：`min`/`max` 是张量时按广播对齐，常用于逐通道裁剪（如按通道的 `clip_grad_norm`）。

**Inductor 视角**　pointwise——本质是两次比较 + 选择，与其他 pointwise 融合；张量边界版本在 lowering 时把广播求解成 stride，核内仍是标量 `max(min(x, hi), lo)`。

---

## 本章小结

逐元素算术是计算图的"水泥"：单个看都是一道数学公式，但它们贡献了全图绝大多数节点数，是 inductor pointwise 融合的主要对象。本章把公式一目了然的 20 个算子（`add`/`mul`/`exp`/`log`/`maximum`/`floor`…）压进速查表，重点深挖了五个带经典坑的：`div`（三种 `rounding_mode` + 整数提升陷阱）、`pow`（整数底的隐式浮点提升）、`fmod` vs `remainder`（符号同被除数 vs 同除数）、`clamp`（min/max 三种变体 + NaN 传播）。贯穿全章的两条隐形规则——**广播**与**类型提升**——详见第 0 章 §0.4 / §0.5；所有算子在 inductor 侧都归 pointwise，可被自由融合。下一章我们进入三角与超越函数（`sin`/`cos`/`atan2`/`sigmoid`/`erf`），它们数学上更"重"，但形态与本章完全一致——依然是位置保持的标量映射。

---

[上一章 第 0 章](00-tensor-substrate.md) 　|　 [下一章 第 2 章](02-transcendental.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
