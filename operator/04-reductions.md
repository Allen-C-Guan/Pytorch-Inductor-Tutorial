# 第 4 章 规约（Reductions）

规约算子把一个张量的若干元素**折叠**成更少的元素——多对一的聚合。它是模型计算里"信息压缩"的原语：损失函数对一个 batch 求均值（`mean`）、softmax 分母归一化（`sum`）、BatchNorm/LayerNorm 算方差（`var`）、`argmax` 取预测类别、梯度累加成全剧和（`cumsum`）…… 没有规约，网络就只能逐元素运算，永远无法把"多个样本的信息"汇聚成"一个统计量"。本章是 Part I（数学语义类）的成员，重心放在**数学定义与数值精度**：尤其 `var`（无偏/有偏估计）与 `cumsum`（scan/前缀和），它们是规约家族里数值非平凡的代表。

> 一句话归类：规约是**位置保持**算子——`out[i]` 读的是 `in` 里一个**正则块** `in[i,k]`，地址闭式仿射、与数据值无关、不重解释存储。注意"位置保持"指**地址映射的规律性**，**不是**"输出个数不变"：规约输出更小（多对一，N:1）属于正交的**基数轴**——逐元素（1:1）/ 规约（N:1）/ 广播（1:N）三者都位置保持。详见第 0 章 §0.6。

---

## 本章速查（Tier C）

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.sum(input, dim, keepdim, dtype)` | 沿 `dim` 求和 | 沿规约维被压缩；空集 → `0`；`dtype` 提升陷阱（int→int64，bool→int64） | reduction |
| `aten.prod(input, dim, keepdim, dtype)` | 沿 `dim` 求积 | 沿规约维被压缩；空集 → `1`；浮点连乘易上溢/下溢 | reduction |
| `aten.mean(input, dim, keepdim, dtype)` | 沿 `dim` 求均值（`sum/count`） | 沿规约维被压缩；空集 → `nan`；整型输入需先提升到浮点 | reduction |
| `aten.amax(input, dim, keepdim)` | 沿 `dim` 取最大**值** | 不返回下标；空集报错；NaN 传播 | reduction |
| `aten.amin(input, dim, keepdim)` | 沿 `dim` 取最小**值** | 不返回下标；空集报错；NaN 传播 | reduction |
| `aten.max(input)` / `aten.max(input, dim)` | 全规约取最大值（带 `dim` 时**同时返回下标**） | 两种重载：标量版只返回值；`dim` 版返回 `(values, indices)` 命名元组 | reduction |
| `aten.min(input)` / `aten.min(input, dim)` | 同上，取最小 | 同上；与 `amax`/`amin` 的差别是**返回下标**与否 | reduction |
| `aten.any(input, dim, keepdim)` | 沿 `dim` 做**逻辑或**（任一为真） | 输入 bool；空集 → `False` | reduction |

> **`max`/`min` vs `amax`/`amin`**：这是最容易混的一对。`amax`/`amin` **只返回值**，且支持任意 `dim` 列表（可一次规约多维）；`max`/`amin` 的 `dim` 版**同时返回下标**（反向传播用），且 `dim` 只接受单个整数或 None。语义上 `max(x)` ≡ `amax(x)`，`max(x,dim).values` ≡ `amax(x,dim)`——多出来的 `.indices` 是关键差异。模型代码取极值下标（如预测类别）必须走 `max`/`argmax` 系。

---

## 深入算子（Tier D）

### `aten.var(input, dim=None, *, correction=1, keepdim=False)` — 沿 dim 计算方差

**作用与语义**　计算沿规约维的方差。对长度为 `n` 的子序列 $x_1, \dots, x_n$，均值为 $\bar{x} = \frac{1}{n}\sum_i x_i$，方差定义为：

$$\mathrm{Var}_{\text{correction}=c}(x) = \frac{1}{n-c}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

- `correction=1`（**默认**，历史称 `unbiased=True`）：分母 $n-1$，即**样本方差 / 无偏估计 (Bessel 校正, Bessel's correction)**。
- `correction=0`（历史称 `unbiased=False`）：分母 $n$，即**总体方差 / 有偏估计 (biased)**。
- 输入→输出形状：沿 `dim`（可列表）规约后压扁（或 `keepdim=True` 时保留为 1）。`dim=None` 时对全体元素求一个标量。返回**新张量**，不别名。

**示例**　同一输入 `correction=1`（除以 n-1）与 `correction=0`（除以 n）结果明显不同：

```python
>>> a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
>>> torch.var(a, dim=1, correction=1)   # 无偏 -> 除以 n-1=2
tensor([1., 1.])
>>> torch.var(a, dim=1, correction=0)   # 有偏 -> 除以 n=3
tensor([0.6667, 0.6667])
```

**为什么需要这个算子 / 数值与精度**　`var` 之所以是规约里的"数值难点旗舰"，是因为它**嵌套了两层规约**（先求均值，再求平方偏差和），而朴素实现会因**大数相减抵消 (catastrophic cancellation)** 严重损失精度。两个实现版本对比：

| 实现 | 公式 | 精度风险 |
|---|---|---|
| **两遍法 (two-pass)** | 先扫一遍算 $\bar{x}$，再扫一遍算 $\sum(x_i-\bar{x})^2$ | 精度好；但要读两遍数据，不友好 to fusion/inductor |
| **一遍法 (naive, Welford 之前)** | $\sum x_i^2 - \frac{(\sum x_i)^2}{n}$（用和与平方和直接凑） | **危险**：$\sum x_i^2$ 与 $(\sum x_i)^2/n$ 都很大且接近，相减后有效位数所剩无几；bfloat16 下甚至可能得到**负数** |

ATen 的 CPU/CUDA 后端默认走两遍法保证精度。对 inductor 开发者的关键提醒：**当融合图里出现 `var`（或其分解 `mean(x)` + `(x-mean)^2` + `mean`）时，bfloat16 累加极易出负方差或 NaN**——这是 NPU 上 LayerNorm/BatchNorm 反复踩的坑（详见第 15 章）。累加应在更高精度（float32）的累加器里做，这正是 `_softmax`、`native_layer_norm` 内部要单独处理的事。

**实现逻辑与复杂度**　两遍法 NumPy 风格伪代码：

```python
def var(x, dim, correction=1, keepdim=False):
    n = x.shape[dim]                         # 规约维长度
    mean = x.mean(dim, keepdim=True)         # 第一遍：求均值
    sq = (x - mean) ** 2                     # 平方偏差（依赖广播，见第 0 章 §0.4）
    var = sq.sum(dim, keepdim=keepdim)       # 第二遍：求平方偏差和
    return var / (n - correction)            # Bessel 校正（默认除 n-1）
```

- 时间 O(N)，空间 O(1)（若融合）/ O(规约外形状)（若不融合，需存中间 `mean`）。
- **分配**：必然产出新张量；融合进 reduction 核时中间 `mean` 不落盘。

**边界与陷阱**
1. **`correction` 必须 < n**：单元素张量（n=1）用默认 `correction=1` → 分母为 0 → 结果 `nan`（会打 warning "degrees of freedom <= 0"）。要避免就传 `correction=0`（结果 0.0）。
2. **空集**：规约维长度 0 → `nan`（带 warning）。
3. **`dim` 可为列表**：如 `var(x, dim=[0,2])`，一次规约多维（`max`/`min` 的 `dim` 版不支持，这是个常被忽略的不对称）。
4. **dtype 提升陷阱**：整数/布尔输入会被提升到默认浮点（`float32` 或 `get_default_dtype`）。`var` **没有 `dtype` 出参**——想控精度要先把输入 `to(float64)`。
5. **NaN 传播**：规约维含任一 NaN → 结果 NaN。

**Inductor 视角**　reduction（走 `ir.Reduction` 核）。`var` 在 export 后通常被分解成 `mean` + `sub` + `mul` + `sum` + `div`，其中两个 `mean`/`sum` 是 reduction、中间运算是 pointwise；融合器会把它们拼成"reduction + pointwise"组合核。

---

### `aten.argmax(input, dim=None, keepdim=False)` — 沿 dim 返回最大值所在**下标**

**作用与语义**　返回**使 `input` 取最大值**的下标（而非值本身）。`dim=None` 时返回整张展平后的标量下标；给定 `dim` 时返回沿该维的下标张量。输出 `dtype=int64`。返回**新张量**。

**示例**　返回的是**下标**（`1`），不是最大值本身（`5`）：

```python
>>> b = torch.tensor([[1., 5., 3.]])
>>> torch.argmax(b, dim=1)   # 最大值 5 在位置 1 -> 返回索引
tensor([1])
```

**为什么需要这个算子 / 数值与精度**　`argmax` 是**分类模型推理的终点**：logits 形如 `[batch, num_classes]`，预测类别就是 `argmax(dim=-1)`——模型把"一组分数"压缩成"一个类别 id"。它也出现在 attention 的掩码选取、`max_pool` 反向（`max_pool2d_with_indices`）、`topk`/`sort` 的实现里。与 `amax` 的本质区别：**它返回位置，不返回值**——位置信息无法从值恢复，所以这是一个独立算子，不是 `amax` 的薄包装。

**实现逻辑与复杂度**　一遍扫描，维护"当前最大值 + 当前下标"：

```python
def argmax(x, dim=None, keepdim=False):
    if dim is None:
        flat = x.reshape(-1)                 # 展平（复合：view+copy，见第 7 章）
        best_idx, best_val = 0, flat[0]
        for i in range(1, len(flat)):
            if flat[i] > best_val:           # 严格 > ：平手取首个
                best_val, best_idx = flat[i], i
        return tensor(best_idx, dtype=int64)
    # 给定 dim：沿 dim 并行扫，每个非 dim 下标独立取 argmax
    out_shape = [s if d != dim else (1 if keepdim else None)
                 for d, s in enumerate(x.shape)]
    out = empty(out_shape, dtype=int64)
    for idx in all_indices(out_shape, skip_dim=dim):
        best, best_val = -1, -inf
        for k in range(x.shape[dim]):
            v = x[idx_with(idx, dim=k)]
            if v > best_val:                 # 严格 > ：平手取首个（约定）
                best_val, best = v, k
        out[idx] = best
    return out
```

- 时间 O(N)，空间 O(1)。
- **平手 (tie-breaking) 约定**：取**第一个**最大值的位置（严格 `>` 比较更新）——这是 PyTorch 的一致约定，对确定性输出很重要。

**边界与陷阱**
1. **空输入报错**：`argmax(empty)` 直接抛 `RuntimeError`（不像 `sum` 返回 0）。
2. **NaN 行为**：含 NaN 的规约维 → NaN 在比较中"不可比较"，结果下标**未定义**（依赖后端，PyTorch 不保证；实践中常落在 NaN 位置或保持初始值，**不要依赖**）。需要确定行为应先 `nan_to_num`。
3. **平手取首个**：见上。若算法依赖"取最后一个"，需自行 `flip` + `argmax` + 反算下标。
4. **不返回值**：要值请 `x.gather(dim, argmax)` 或直接用 `max(x, dim)`（一次拿值+下标）。

**Inductor 视角**　reduction（`ir.Reduction`，但产出 int64 下标而非数值和）。融合性差：下标结果无法与后续 pointwise 自然合并（dtype 不同、语义是 argmin/argmax 原语），通常独立成核；`argmax` 也会出现在 `max_pool2d_with_indices` 的 template 里（第 14 章）。

---

### `aten.argmin(input, dim=None, keepdim=False)` — 沿 dim 返回最小值所在**下标**

**作用与语义**　与 `argmax` 完全对称，返回**最小值**所在下标。`dtype=int64`，返回新张量。

**示例**　同样返回**下标**（`0`），不是最小值本身（`1`）：

```python
>>> b = torch.tensor([[1., 5., 3.]])
>>> torch.argmin(b, dim=1)   # 最小值 1 在位置 0 -> 返回索引
tensor([0])
```

**为什么需要这个算子 / 数值与精度**　用于"找最优/最差位置"——强化学习里 `argmin` 选最小损失动作、距离矩阵里找最近邻、损失函数里找最小值位置。语义与数值讨论同 `argmax`，不再赘述。

**实现逻辑与复杂度**　把 `argmax` 伪代码里的 `-inf` 初值改成 `+inf`、`>` 改成 `<` 即可。O(N) 时间、O(1) 空间。平手取**第一个**最小值。

**边界与陷阱**　与 `argmax` 完全一致：空输入报错；NaN 行为未定义；平手取首个；只返回下标不返回值。

**Inductor 视角**　reduction（`ir.Reduction`，产出 int64 下标）。实现上常复用 argmax 核，对输入取负后做 argmax——这是后端常见的取巧，注意因此带来的浮点符号/精度细微差异。

---

### `aten.cumsum(input, dim, *, dtype=None)` — 沿 dim 的**前缀和**（scan / 累积求和）

**作用与语义**　计算**前缀和 (prefix sum / cumulative sum / scan)**：输出每个位置等于输入从该维起点到当前位置的**累积和**。这是规约家族里的**特殊类**——普通规约把整维压成一个数，`cumsum` 把整维变成**等长的另一维**，每个输出元素是输入的一个**前缀**的聚合。对长度 n 的序列 $x_0, \dots, x_{n-1}$：

$$\mathrm{out}[k] = \sum_{i=0}^{k} x_i, \quad k=0,\dots,n-1$$

`dim` 必填（不可为 None，这是与 `sum`/`var` 的差别）。输出形状与输入**完全相同**。`dtype` 可指定输出累加精度（常用于把 int 累加提升到 int64 防溢出、或把 bfloat16 提升到 float32 累加）。返回**新张量**。

**示例**　每个位置 = 从起点到当前位置的累加和：

```python
>>> c = torch.tensor([1., 2., 3., 4.])
>>> torch.cumsum(c, 0)   # 1, 1+2=3, 3+3=6, 6+4=10
tensor([ 1.,  3.,  6., 10.])
```

**为什么需要这个算子 / 数值与精度**　`cumsum` 是**scan（扫描）算法族**的代表。普通规约是"树形两两合并"，而 scan 要保留**每一个前缀的结果**——这要求算法是**顺序依赖**的：`out[k]` 依赖 `out[k-1]`，不能像 `sum` 那样任意重排结合。这种"前缀依赖"特性使它在 GPU/NPU 上需要专门的 **parallel prefix sum（Blelloch / work-efficient scan）** 算法，而不是简单的 reduction 树。

模型域用途：
- **因果 attention** 的累加掩码、长度不均的 `segment_sum`（配合 `_embedding_bag`/`scatter`）。
- **时间序列 / RNN** 状态累积、积分近似。
- **反向传播**：某些规约的反向梯度是 `cumsum`（如 `flip(cumsum(flip(grad)))`）。

数值精度要点：累加顺序固定（不是任意重排），所以**误差是确定性的**，但仍会随长度累积——长序列上 int8/int16 必溢出（务必传 `dtype=int64`），bfloat16 累加漂移严重（传 `dtype=float32`）。

**实现逻辑与复杂度**　顺序版（教学）与并行 scan 版对比：

```python
# 朴素顺序版（CPU 参考实现，O(n) 但无法并行）
def cumsum_seq(x, dim):
    out = empty_like(x)
    for idx in all_indices(x.shape, skip_dim=dim):
        acc = 0
        for k in range(x.shape[dim]):
            acc = acc + x[idx_with(idx, dim=k)]   # 累加：out[k] = out[k-1] + x[k]
            out[idx_with(idx, dim=k)] = acc
    return out

# 并行 scan（GPU/NPU 实际用，Blelloch 风格）
# up-sweep 建部分和树，down-sweep 把前缀摊到每个位置
# work = O(n), steps = O(log n)，可大规模并行
```

- 时间：顺序 O(N)；并行 scan work O(N)、深度 O(log N)。
- 空间 O(N)（必须存每个前缀，无法压缩——这是与普通 `sum` 的根本差别）。
- **必然分配**新张量，不能零拷贝。

**边界与陷阱**
1. **`dim` 必填**：`cumsum(x)`（不带 dim）会报错——这与 `sum`/`var` 允许 `dim=None` 不同。
2. **溢出**：整数 `cumsum` 不自动提升（除非传 `dtype`）。`int8` 的 `[1,1,1,...]` 累加到第 128 个就溢出。**务必 `cumsum(x, dim, dtype=int64)`**。
3. **`dtype` 的语义**：是**输出/累加器**的精度，不是输入的（输入原样读）。这恰好用来"输入低精度、累加高精度"。
4. **非连续输入**：支持任意 stride，但若 `dim` 维 stride 不为 1，性能可能下降（需分散访存）。
5. **NaN 传播且不可恢复**：一旦遇到 NaN，从该位置往后全是 NaN（前缀和一旦被污染，后续全脏）。
6. **空维**：沿长度 0 的维 cumsum → 输出空张量（无元素可累加），不报错。

**Inductor 视角**　reduction（`ir.Reduction`，但是 **scan 变体**，不是普通 tree-reduction）。在 Triton 里用 `tl.cumsum` 实现；NPU 上 `_inductor_new` 后端走专门的 scan 核。`cumsum` **难以与前后 pointwise 融合**（顺序依赖打断 fusion 边界），通常独立成核。

---

### （点名）scan 家族的其余成员为何不在 core 161

core ATen 把 scan 家族**只保留了 `cumsum`** 作为代表，其余成员——`cumprod`（累积积）、`cummax`/`cummin`（累积极值，同时返回值与下标）、`logcumsumexp`（数值稳定的累积 `logsumexp`，softmax 的累积变体）——**都不在 core 161 基础集**里。它们属于更高层的"复合 scan 算子"：

| 算子 | 语义 | 复合分解（示意） |
|---|---|---|
| `cumprod(x)` | 累积积 $\prod_{i\le k}x_i$ | `exp(cumsum(log(x)))`（仅正数；一般需符号处理） |
| `cummax`/`cummin` | 累积极值 + 下标 | scan with `(max, argmax)` 合并幺半群，无法纯 `cumsum` 表达 |
| `logcumsumexp(x)` | $\log\sum_{i\le k}e^{x_i}$（数值稳定） | scan with `logaddexp` 合并，等价稳定版 `log(cumsum(exp(x)))` |

它们在 `native_functions.yaml` 里没有 `core` 标签，故按本书范围规则只在附录 A 一行带过、不写 Tier D。理解了 `cumsum` 的 scan 本质（前缀依赖、顺序结合、并行 Blelloch 算法），这些变体只是把"合并算子"从 `+` 换成 `*` / `max` / `logaddexp` 而已。

---

## 本章小结

规约算子把"多个元素压成一个/少数几个"，是模型里信息汇聚的核心机制。本章把家族分成三层：**普通聚合**（`sum`/`prod`/`mean`/`amax`/`amin`/`max`/`min`/`any`，走标准 reduction 树，注意 `max`/`min` 比 `amax`/`amin` 多返回下标）；**带统计语义的方差**（`var`，难点在 Bessel 校正与两遍法防抵消）；**scan 类前缀和**（`cumsum`，前缀依赖、独立 scan 核，core 161 唯一的 scan 代表）。共同的形状规则是 `keepdim`——保留为 1 的维度能让规约结果**直接广播**回原形状（如 `(x - x.mean(dim, keepdim=True))`），这是规约与广播配合的标准模式（广播规则见第 0 章 §0.4）。

下一章 [第 5 章 线性代数核心](05-linear-algebra-core.md) 进入 `mm`/`bmm`/`addmm`——这是规约的"二维推广"（内积是乘加规约），也是 inductor 用 template 降级的旗舰算子。

---

[上一章 第 3 章](03-comparison-boolean.md) 　|　 [下一章 第 5 章](05-linear-algebra-core.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 公共概念：广播（`keepdim` 与规约结果的回扩）详见第 0 章 §0.4；类型提升（规约时整型→浮点、bool→int64）详见 §0.5；位置保持 vs 位置变化的分类轴（规约属位置保持、多对一）详见 §0.6；SSA 契约（规约算子均为纯函数、产出新张量）详见 §0.7。
