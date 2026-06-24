# 第 9 章 索引家族（全书最难章）

## 导言

索引家族是 Part II（操作类）里最反直觉、最容易踩坑的一类算子。前面几章（concat/split、typecast、数学运算）功能描述即语义——"把两个张量沿某轴拼起来"你一眼就能在大脑里跑出来；但索引家族不是。"沿 dim=1 用 index 张量从 src gather 到 out"这句话，直到你在纸上画完两个箭头（index 张量的"语义双重性"：它的**下标 `[i][j]` 标定的是 out 的位置**（out 与 index 同形），而它的**值 `index[i][j]` 是去 input(src) 里取数的坐标**——下标和值分属两个不同张量的坐标空间）之前，功能描述几乎无法直觉理解。这是全书重心：把"index 张量的语义双重性"刻进骨头里。

从模型域看，索引家族是现代深度学习的隐形地基：**embedding 查表**（`weight[idx]`）、**梯度累加**（`scatter_add` 把多份梯度写回同一参数）、**attention 的 token gather**（按 causal/index 屏蔽抽取 KV）、**MoE 的路由分发**（scatter 把 token 派到专家）——这些都只有索引类算子能高效表达。它不像 pointwise 那样可并行无依赖地铺开，而是"读/写到由另一个张量决定的位置"，天然带数据依赖、带冲突（同位置多次写）、带形状推断困难（nonzero 的输出形状依赖数据）。理解了这一章，你才真正理解 inductor 里 `template` / `fallback` 这两条路径为什么必须存在。

本章的难点排序：gather/scatter 的轴语义（第一性指标）→ scatter 家族的写冲突语义（覆盖 vs 累加 vs 归约）→ index_put 的 accumulate 与类型提升经典 bug → embedding 作为 gather 的特化 → nonzero 的 dynamic output shape。

## 本章速查（Tier C）

本章无 Tier C 算子，全部 Tier D 深入讲解。下列 10 个算子按"轴语义 / 写冲突 / 数据依赖形状"三个维度归类：

| 算子 | 功能一句话 | 形状 / 风险 | inductor 归类 |
|---|---|---|---|
| `aten.gather` | 按 index 张量从 src 沿 dim 抽取到 out（**只读**） | out.shape == index.shape；index 值需 ∈ src.dim_size | template（常走 gather kernel） |
| `aten.scatter` | gather 的逆：按 index 把 src 写到 out（**后写覆盖**） | out.shape == index.shape 的目标；同位置写冲突→未定义胜者 | template / fallback |
| `aten.scatter_add` | scatter，但同位置**累加**而非覆盖 | 同上；语义确定（commutative 加） | reduction-类 template |
| `aten.scatter_reduce` | scatter + 指定归约（sum/prod/amin/amax） | 同上；`include_self` 决定初值 | reduction-类 template |
| `aten.index` | 用 tuple of 1D index 张量（每轴独立）索引，等价 `out[:, idx0, idx1]` | 高级索引：广播、可能降维 | fallback（混合索引形状复杂） |
| `aten.index_select` | 沿单一 dim 用 1D index 抽取（gather 的窄化版） | 输出沿 dim 长度 == len(index) | template |
| `aten.index_put` | 按 tuple of index 张量（可广播）写入 values；accumulate=True → 累加 | 写冲突；accumulate 的类型提升是经典 bug | reduction-类 template / fallback |
| `aten.masked_scatter` | mask 为 True 的位置填入 1D values 的下一个元素 | values 长度 ≥ mask.sum() | fallback（数据依赖） |
| `aten.nonzero` | 返回非零元素的坐标矩阵 | **输出形状依赖输入数据** → dynamic_output_shape | fallback（必须） |
| `aten.embedding` | `weight[idx]`：embedding 查表 = gather(dim=0) 特化 | idx 任意形状，输出 = idx.shape + weight.shape[1:] | template（gather 特化） |

> 相邻复合算子（**非 core，本章不展开**）：`masked_fill` → `where`（见第 4 章）；`masked_select` → `nonzero` + `index`；`index_copy` → `scatter` 的整块写入；`embedding_dense_backward` / `_embedding_bag` 见附录 A（标注 ○）。

## 深入算子（Tier D）

### `aten.gather(input, dim, index, *, sparse_grad=False, out=None) -> Tensor` — 按 index 张量沿 dim 从 input 抽取元素到输出

**作用与语义**　`out` 与 `index` **同形状**。一句话：**沿 `dim` 轴，用 `index` 里的值去 `input` 取数，其余轴就按下标本身取。** 核心公式（二维、`dim=0`）：

```
out[i][j] = input[ index[i][j] ][j]        # index 的值替换第 0 轴下标；第 1 轴沿用 j
```

换成 `dim=1`，就是把 `index` 的值挪到第二个下标槽（即下面示例的情形）：

```
out[i][j] = input[i][ index[i][j] ]
```

推广到 n 维：输出下标 `J` 中只有第 `dim` 轴被替换成 `index[J]` 的值，其余轴照抄 `J` 本身——`out[J] = input[J, 但第 dim 轴换成 index[J]]`。

形状约束：`out.shape == index.shape`；除 `dim` 轴外要求 `index.shape[k] ≤ input.shape[k]`，且每个 `0 ≤ index[J] < input.shape[dim]`。返回新张量（不别名 input）；`sparse_grad=True` 仅影响反向（输出 sparse coalesce 梯度）。

**示例**　`dim=1` 时，`index` 的**值**决定沿列方向读 input 的哪一列：

```python
>>> inp = torch.tensor([[10, 11, 12], [20, 21, 22]])
>>> idx = torch.tensor([[2, 0], [1, 1]])
>>> torch.gather(inp, 1, idx)
tensor([[12, 10],
        [21, 21]])
```

```
inp (2,3)         idx (2,2)         out (2,2)  与 idx 同形
[[10 11 12]       [[2 0]            [[12 10]   out[0,0]=inp[0, idx[0,0]=2]=12
 [20 21 22]]       [1 1]]            [21 21]]   out[0,1]=inp[0, idx[0,1]=0]=10
                                              out[1,0]=inp[1,1]=21 ; out[1,1]=inp[1,1]=21
# index 的【值】索引的是 dim 轴；其余轴按下标本身取（这是 gather 的语义双重性）
```

**为什么需要这个算子**　gather 解决的核心结构问题是"**沿某一轴，用一个张量的值来动态决定读哪个位置**"——这是不可被 pointwise / 切片替换的，因为读地址是数据依赖的。典型模型用途：

- **Attention / KV-cache 抽取**：给定 token 位置 `pos`，从缓存 `K` 里抽出历史 key → `K.gather(0, pos.expand(...))`。
- **Top-k 选词 / beam search**：logits topk 后再 gather 词表 embedding。
- **序列重排（permutation by index）**：把乱序的 index 当作地址表，把输入重排成目标顺序。
- **NLP 的 negative sampling**：按负样本索引 gather 词向量。

与相邻算子的边界：`index_select` 是 gather 的**窄化版**（只允许单一 dim、index 必须 1D），gather 允许 index 与 input 同维度、多维形状，表达力更强；`index`（高级索引）允许"每轴独立给 1D index 然后广播"，语义不同（见下）。

**实现逻辑与复杂度**　伪代码（NumPy 风格）：

```python
def gather(input, dim, index):
    out = empty_like(index)            # out.shape == index.shape
    for J in ndindex(index.shape):
        J_src = list(J)
        J_src[dim] = index[J]
        out[J] = input[tuple(J_src)]
    return out
```

时间复杂度 O(∏ index.shape)，空间 O(∏ index.shape)（必须分配 out，**非零拷贝**）。

**边界与陷阱**

1. **index 值范围**：必须 `0 ≤ index[J] < input.shape[dim]`（负索引也合法，按 Python 语义解析），越界 → RuntimeError。
2. **index 与 input 形状约束**：除 `dim` 轴外，`index.shape[k] ≤ input.shape[k]`（不是相等！允许 index 比 input 短，但不能长）。
3. **dtype**：index 必须是整数类型（int64 / int32），float 索引报错。
4. **非连续输入**：gather 不要求 input 连续，但有些后端（如某些 Triton kernel）会先 contiguous()，引入额外拷贝——inductor 优化时需注意。
5. **反向**：`gather_backward` 是 `scatter_add`（多对一映射 → 累加梯度），这也是为什么 sparse_grad 选项存在（稀疏场景下避免物化大稀疏梯度）。

**Inductor 视角**　template（gather kernel）——形状静态、读地址数据依赖但**无写冲突**，可生成专用 gather Triton kernel；非连续或形状怪异时退化为 fallback。

---

### `aten.scatter(dim, index, src, *, reduce=None) -> Tensor`（注：PyTorch 新签名 self-first） — gather 的逆：按 index 把 src 写入 self（后写覆盖）

> PyTorch 实际签名为 `aten.scatter(input, dim, index, src, *, reduce=None) -> Tensor`，input 作为 self 被原地写入（functional 视图返回 input 本身或副本）。

**作用与语义**　gather 的逆操作——**把 `src` 的元素按 `index` 指定的位置写进 `self`**（`self` 是被改写的目标张量）。写到哪一列/行由 `index` 的值决定（沿 `dim` 轴），其余轴按位置本身。核心公式（二维、`dim=1`）：

```
self[i][ index[i][j] ] = src[i][j]        # 把 src[i][j] 写进 self 的第 index[i][j] 列
```

`self` 形状不变、`src.shape == index.shape`；同一位置被多次写时**后写覆盖**（结果未定义，要累加请用 `scatter_add`）。返回 `self` 的 functional 副本。

**示例**　`dim=1`，把 `src` 的元素写到 `self` 里 `index` 指定的列——未被命中的位置保持原值：

```python
>>> self = torch.zeros(2, 3, dtype=torch.long)
>>> torch.scatter(self, 1, torch.tensor([[0, 1], [1, 2]]), torch.tensor([[1, 2], [3, 4]]))
tensor([[1, 2, 0],
        [0, 3, 4]])
```

```
self (2,3)=0      index (2,2)       src (2,2)        out (2,3)
[[0 0 0]          [[0 1]            [[1 2]           [[1 2 0]   self[0][0]=1, self[0][1]=2
 [0 0 0]]          [1 2]]            [3 4]]           [0 3 4]]   self[1][1]=3, self[1][2]=4
# self[i][ index[i][j] ] = src[i][j]；同位置多次写 -> 后写覆盖（结果未定义，见 scatter_add）
```

**为什么需要这个算子**　scatter 解决"**把一个张量的元素散布（scatter）到另一个张量的指定位置**"，是 gather 的对偶。模型域关键用途：

- **梯度写回 / 参数更新（朴素 SGD/AdaGrad 的稀疏更新）**：只更新被 gather 过的几行参数 → `param.scatter_add(dim=0, index=idx, src=grad)`。
- **MoE 路由分发**：把 token 按门控路由 scatter 到各专家的输入缓冲。
- **One-hot 编码 / 填表**：按类别 index 把值填到 one-hot 矩阵。
- **Histogram / bincount 的批量版**：scatter_add 实现分箱求和。

与 `index_put` 的边界：scatter 只能沿**单一 dim** 做索引（其他维用 identity），且 index 与 src 同形状；index_put 允许每轴独立给 index 张量、能广播、能 accumulate——表达力更强但更慢。

**实现逻辑与复杂度**

```python
def scatter(self, dim, index, src):
    out = self.clone()                 # 注意：functional 语义下要拷贝
    for J in ndindex(index.shape):
        J_dst = list(J)
        J_dst[dim] = index[J]
        out[tuple(J_dst)] = src[J]     # 同位置多次写 → 后写胜（覆盖）
    return out
```

时间 O(∏ index.shape)，空间 O(self.shape)（需拷贝 self，否则会别名修改原 input）。**注意"后写覆盖"语义**：若两个 J 映射到同一个 `J_dst`，最终值是遍历顺序里最后一个写——这是**未定义**的（依赖实现），不可依赖。

**边界与陷阱**

1. **同位置多次写 → 后写覆盖、结果未定义**。这是 scatter 最大的坑：模型里若用裸 scatter 写梯度且同一参数被多份 src 命中，**梯度会丢**。**必须用 scatter_add**。
2. **index 范围**：`0 ≤ index[J] < self.shape[dim]`。
3. **dtype 提升**：src 的 dtype 必须能安全转成 self.dtype，否则报错（不会隐式窄化提升）。
4. **reduce 参数**（v1.8+）：传 `"add"` / `"multiply"` 把覆盖语义改成累加/累乘，等价 scatter_add/scatter_reduce 的语法糖——但**不传时仍是覆盖**。
5. **非连续 self**：scatter 会在 logical 形状上工作，但输出可能非连续；若下游依赖连续需显式 contiguous()。

**Inductor 视角**　template（scatter kernel），但因**写冲突**（同位置多写）无法直接并行——若 reduce 模式为累加则归到 reduction-类 template（需要原子加 / 两遍归约）；裸覆盖 scatter 在 Triton 里通常退化为 fallback 以保证正确性。

---

### `aten.scatter_add(dim, index, src) -> Tensor` — scatter，但同位置**累加**（语义确定）

**作用与语义**　和 `scatter` 一模一样地散布，区别只在：**写同一位置时是"累加"而非"覆盖"**——多个 `src` 元素落到同一 `self` 位置时把它们加起来。加法可交换可结合，所以结果与写入顺序无关、完全确定（这正是它相对裸 `scatter` 的价值：不会丢数据）。公式：

```
self[i][ index[i][j] ] += src[i][j]       # 同位置多次写 -> 相加
```

返回 `self` 的 functional 副本。

**示例**　同样写入，但同位置**累加**——`self[0][0]` 被 `1`、`2` 命中两次，相加得 `3`（裸 scatter 在这里只会留后写的 `2`，丢梯度）：

```python
>>> self = torch.zeros(2, 2, dtype=torch.long)
>>> torch.scatter_add(self, 1, torch.tensor([[0, 0, 1], [0, 1, 1]]), torch.tensor([[1, 2, 3], [4, 5, 6]]))
tensor([[ 3,  3],
        [ 4, 11]])
```

```
index (2,3)       src (2,3)         out (2,2)
[[0 0 1]          [[1 2 3]          [[ 3  3]   self[0][0] = 1+2 = 3  (两次命中, 累加)
 [0 1 1]]          [4 5 6]]          [ 4 11]]   self[0][1] = 3 ; self[1][0] = 4
                                              self[1][1] = 5+6 = 11 (累加)
# 裸 scatter 在 self[0][0] 只会保留后写的 2 -> 梯度丢失；累加务必用 scatter_add
```

**为什么需要这个算子**　这是 scatter 的**正确版**——只要你担心"同位置会被多次写"，就该用 scatter_add。模型域几乎一切的"散布式累加"都用它：

- **稀疏梯度写回**：embedding lookup 的反向——多个 query 查同一行 embedding，梯度要在该行累加 → scatter_add。
- **Segment sum / group-by 聚合**：按 group index 把特征向量累加到 group 表征。
- **Histogram、scatter-plot 求和**。
- **PointNet++ / GNN 的邻居聚合**：把邻居特征 scatter_add 到中心节点。

边界：与 `index_put(accumulate=True)` 在功能上高度重叠——scatter_add 限制为单一 dim，index_put 允许每轴独立 index 且能广播，但底层语义一致（都是 atomic add）。

**实现逻辑与复杂度**

```python
def scatter_add(self, dim, index, src):
    out = self.clone()
    for J in ndindex(index.shape):
        J_dst = list(J); J_dst[dim] = index[J]
        out[tuple(J_dst)] += src[J]    # 顺序无关
    return out
```

时间 O(∏ index.shape)，空间 O(self.shape)。底层常用 atomic add（GPU/Triton）或两遍（先 sort 后 segment sum）。

**边界与陷阱**

1. **NaN 传播**：src 中的 NaN 会污染目标位置；self 的 NaN 也会被保留（`x + NaN = NaN`）。
2. **整型溢出**：int 累加不饱和、不报错，溢出环绕（与 C 一致）——大 batch 整型累加要小心，最好用 int64。
3. **dtype 提升坑**：self 与 src 必须同 dtype（不提升），若 src 是 float32 而 self 是 float16，会**报错**而非自动提升——这跟 index_put 的经典 bug 同源（见下）。
4. **原子加精度**：GPU atomicAdd 对 float16 在老硬件上不支持，常退化为 float32 中间累加再转回。

**Inductor 视角**　reduction-类 template（atomic add kernel）——因累加语义确定，可放心并行；Triton 提供 `atomic_add`，是 NPU/Inductor 里 scatter_add 的标准实现路径。

---

### `aten.scatter_reduce(dim, index, src, reduce, *, include_self=True) -> Tensor` — scatter + 通用归约（sum/prod/amin/amax）

**作用与语义**　`scatter_add` 的推广——**散布到同一位置时，按指定归约合并**，而不是固定成"加"。`reduce` 取 `"sum"`/`"prod"`/`"amin"`/`"amax"`：多个 `src` 落到同一 `self` 位置时，取它们的和/积/最小/最大。`include_self`（默认 `True`）决定 `self` 原值是否参选（`False` = 先清成归约单位元再合并）。公式：

```
self[i][ index[i][j] ] = reduce( self[i][index[i][j]], src[i][j] )
```

**示例**　`reduce="amax"`：同位置取最大（而非相加）——`self[0][0]` 命中 `5` 和 `1`，取大得 `5`：

```python
>>> self = torch.zeros(2, 2)
>>> torch.scatter_reduce(self, 1, torch.tensor([[0, 0], [1, 1]]),
...                      torch.tensor([[5., 1.], [3., 8.]]), "amax")
tensor([[5., 0.],
        [0., 8.]])
```

```
index (2,2)       src (2,2)         out (amax, include_self=True)
[[0 0]            [[5 1]            [[5 0]   self[0][0] = amax(初值0, 5, 1) = 5
 [1 1]]            [3 8]]            [0 8]]   self[1][1] = amax(初值0, 3, 8) = 8
# 换 reduce="sum" 即等价 scatter_add；未被命中的 self[0][1]/self[1][0] 保留初值 0
```

**为什么需要这个算子**　当你需要的不是"加"而是"取最大/最小/乘积"的散布聚合时——scatter_add 表达不了。模型域用途：

- **GNN / 集合操作的 max-pool 聚合**：邻居特征 scatter 后取 max（PointNet 的 max-pool）。
- **带掩码的 confidence 累乘**。
- **去重 / 取每行最优**：按 index 把候选写到目标位置，取 amax 留下"每桶最优"。
- **bag-of-words 的 max-pool 表征**。

边界：与 `index_put(accumulate=True, reduce=...)` 高度重叠，scatter_reduce 限制单一 dim。`include_self=False` 是相对晚加入的语义，等价于先 zero 再 reduce——这是与 scatter_add 最大的行为差异（scatter_add 隐式 include_self=True）。

**实现逻辑与复杂度**

```python
def scatter_reduce(self, dim, index, src, reduce, include_self=True):
    out = self.clone()
    if not include_self:
        # 先把所有 index 命中的位置置为归约单位元
        out = scatter_zero(out, dim, index, reduce)   # sum→0, prod→1, amin→+inf, amax→-inf
    for J in ndindex(index.shape):
        J_dst = list(J); J_dst[dim] = index[J]
        out[tuple(J_dst)] = REDUCE[reduce](out[tuple(J_dst)], src[J])
    return out
```

时间 O(∏ index.shape)，空间 O(self.shape)。`amax/amin` 不支持原子操作，常需两遍或 hash-map 中间结构，比 scatter_add 慢。

**边界与陷阱**

1. **include_self 语义**：默认 True 会把 self 原值"参选"——若你不希望，必须显式传 `include_self=False`，否则 amax 可能永远保留一个大的旧值。
2. **amin/amax 的 NaN**：与 max/min 一致，NaN 传染。
3. **prod 的数值稳定性**：大 batch 累乘易溢出，工程上罕见，慎用。
4. **未命中位置**：include_self=False 时，没被任何 index 命中的位置保留单位元（如 sum→0）——这是设计如此，不是 bug。

**Inductor 视角**　reduction-类 template——sum/prod 可走 atomic 路径，amin/amax 因无原生原子常退化为两遍或 fallback。

---

### `aten.index(input, indices) -> Tensor`（即 `input[indices_tuple]`） — 高级索引：每轴独立给 1D index，广播后抽取

**作用与语义**　这就是 Python 里 `x[i, j]` 这种**高级索引**的算子形式——**给每个轴一个索引数组，抽出这些坐标交叉位置上的元素**。各轴索引数组先广播对齐，每个输出位置就取"所有索引数组在该位置共同给出的那个坐标"上的输入元素。

二维情形：输入 `Input`，两个轴各给一个索引数组 `rows`、`cols`，先把它们广播到同形 `rows_broadcasted`、`cols_broadcasted`，则输出形状 = 该广播形状，且每个元素：

```
Output[i][j] = Input[ rows_broadcasted[i][j], cols_broadcasted[i][j] ]
```

一句话：**输出第 `[i][j]` 个元素 = 输入在 `(rows_broadcasted[i][j], cols_broadcasted[i][j])` 这个坐标上的值。** 多于两个被索引轴时同理（每个轴一个索引数组、全部广播对齐）；没给索引（`None`/省略）的轴整段保留在输出末尾。

例：`a` 形状 `(B, N, D)`，`a[idx0, idx1]`（`idx0`、`idx1` 各长 `K`）→ 输出 `(K, D)`：`idx0/idx1` 广播成 `(K,)` 作前导维，`D` 维保留。`indices` 是 tuple of 1D LongTensor，`None` 表示该轴全取。

**示例**　每轴给一个 1D index，**广播后成为前导维**，结果是 1D：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a[torch.tensor([0, 2]), torch.tensor([1, 3])]   # 即 a[[0,2],[1,3]]
tensor([ 1, 11])          # = [a[0,1], a[2,3]]，两个轴的 index 广播成 (2,) 前导维
```

**为什么需要这个算子**　这是 Python `[]` 语义的算子化，覆盖"**每轴独立用整数索引**"的全部场景，是写 PyTorch 模型时最自然的索引方式：

- **Batch 内按 ID 抽取样本**：`feat[batch_idx, sample_idx]`。
- **按类别 mask 抽取**：`data[rows, cols]`。
- **重排 / dedup 后回查**。
- **Cython / NumPy 习惯迁移**：用户写 `x[i, j]` 几乎都走 index。

边界：与 gather 的区别——gather 沿单一 dim，index 多轴独立且会**改变输出形状**（前导维度）；与 index_select 的区别——index_select 只一轴、不改变秩。

**实现逻辑与复杂度**

```python
def index(input, indices):
    # 1. 把非 None 的 index 广播到同一形状 B
    # 2. 输出形状 = B.shape + 保留轴的形状
    B = broadcast(*[i for i in indices if i is not None])
    out_shape = B.shape + tuple(input.shape[k] for k where indices[k] is None or Ellipsis)
    out = empty(out_shape)
    for idx in ndindex(B.shape):
        multi_idx = tuple(B[idx] if indices[k] is not None else slice_all
                          for k in range(input.ndim))
        out[idx, ...] = input[multi_idx]
    return out
```

时间 O(out.numel())，空间 O(out.numel())。

**边界与陷阱**

1. **布尔索引走另一条路径**：`x[mask]`（mask 为 bool）实际是 `masked_select`，不是 index。
2. **混合 advanced + basic 索引**：形状规则非常绕（advanced 索引在前、basic 在后），是用户最常写错的地方。
3. **负索引 / 越界**：负索引按 Python 解析，越界报错。
4. **None / 省略号**：`indices` 里可含 None（新增轴）和 Ellipsis（占位）——inductor 通常在前端把 index 调用规范化掉。

**Inductor 视角**　fallback——形状规则、混合索引、广播使得生成专用 kernel 收益低，常被前端 decompose 成 gather/index_select/reshape 的组合，或直接 fallback 到 ATen eager。

---

### `aten.index_select(input, dim, index) -> Tensor` — 沿单一 dim 用 1D LongTensor 抽取（gather 的窄化版）

**作用与语义**　**沿单一轴、用一维索引数组挑行（或列）**——`gather` 的窄化易用版。`index` 是 1D 整数数组，输出即沿 `dim` 轴按 `index` 的值抽取并重排，其余轴原样（允许重复抽同一行）。严格定义（`k` 是输出沿 `dim` 的下标）：

```
out[..., k, ...] = input[..., index[k], ...]                # 仅 dim 轴被 index[k] 替换
```

例：`a.index_select(0, [2,0,0])` 抽出 `a` 的第 2、0、0 行；输出沿 `dim` 的长度 = `len(index)`。

**示例**　沿 `dim=1` 用 1D index 抽列，秩不变（`embedding(idx)` ≡ `weight.index_select(0, idx)`）：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a.index_select(1, torch.tensor([0, 2]))     # 取列 0 与 2
tensor([[ 0,  2],
        [ 4,  6],
        [ 8, 10]])
```

**为什么需要这个算子**　gather 的**易用窄化版**——当索引只需要沿一个轴、且 index 是 1D（最常见场景）时，index_select 比 gather 直观得多，且语义无歧义。模型域：

- **按行/列子采样**：`X.index_select(0, keep_idx)`。
- **数据加载的 batch 组装**底层（其实 DataLoader 用 stack，但单 tensor 子采样是 index_select）。
- **Beam search 的状态收集**：按 beam index 抽 hidden state。
- **稀疏场景取参数行**：`weight.index_select(0, idx)`（语义等价 embedding，但 embedding 有 padding_idx 等附加）。

边界：`embedding(idx)` ≡ `weight.index_select(0, idx)`（外加 padding_idx / sparse 处理）。gather 表达力 ⊃ index_select。

**实现逻辑与复杂度**

```python
def index_select(input, dim, index):
    out_shape = list(input.shape)
    out_shape[dim] = len(index)
    out = empty(out_shape)
    for k in range(len(index)):
        out[..., k, ...] = input[..., index[k], ...]
    return out
```

时间 O(out.numel())，空间 O(out.numel())。

**边界与陷阱**

1. **index 必须 1D**：传多维报错（这是与 gather 的硬边界）。
2. **重复 index 合法**：index 可以含重复值，会重复抽取（这是 embedding 的核心需求）。
3. **负索引**：合法。
4. **非连续 input**：支持，性能可能受影响。

**Inductor 视角**　template（gather 的特化 kernel，沿单轴、index 1D）——常被前端 decompose 成 gather，或生成专用 index_select Triton kernel。

---

### `aten.index_put(input, indices, values, accumulate=False, *, unsafe=False) -> Tensor` — 按 tuple of index 张量写入 values（accumulate=True → 累加）

**作用与语义**　**按一组坐标把 `values` 写回 `self` 的指定位置**——`scatter` 的多轴广播版。`indices` 是每轴一个索引数组（与 `self` 同维数、可广播）。严格定义（二维 `(rows, cols)` 情形，与高级索引同构）：

```
self[ rows[k], cols[k] ] = values[k]        # accumulate=True 时改成 += 同位置累加
```

推广到 n 维：`self[ indices[0][k], …, indices[n-1][k] ] = values[k]`。`accumulate=True` 时同位置累加（如把多份梯度加到同一参数），`False` 时覆盖。

**示例**　同样写入下标 `[0,0,1]`：`accumulate=True` 在 `[0]` 处累加得 `3`，`False` 则**后写覆盖**得 `2`：

```python
>>> idx  = (torch.tensor([0, 0, 1]),)            # 行下标
>>> vals = torch.tensor([1., 2., 3.])
>>> torch.zeros(3).index_put(idx, vals, accumulate=True)
tensor([3., 3., 0.])      # [0] = 1+2 = 3 (累加)
>>> torch.zeros(3).index_put(idx, vals, accumulate=False)
tensor([2., 3., 0.])      # [0] = 2 (后写覆盖, 丢掉了 1)
```

```
indices=([0,0,1],)  values=[1,2,3]      out (长度 3)
accumulate=True :  self[0]+=1, self[0]+=2, self[1]+=3  -> [3, 3, 0]
accumulate=False:  self[0]=1, self[0]=2(覆盖), self[1]=3 -> [2, 3, 0]
# 梯度累加 / Adam 类优化器走 accumulate=True；经典 bug: fp16 self += fp32 values 报错(见边界)
```

**为什么需要这个算子**　scatter 的**多轴广播版**——当索引需要"每轴独立给地址、还能广播"时，scatter 的单 dim 限制就不够用。模型域：

- **稀疏 embedding 梯度更新**：`w.index_put((rows, cols), grads, accumulate=True)`。
- **稀疏矩阵赋值**：`(i, j, v)` 三元组写回稠密矩阵。
- **强化学习的 Q-table / 经验回放写入**。
- **scatter-add 的多轴泛化**——任何"按坐标批量写入"的场景。

边界：与 scatter 的关系——scatter 是 index_put 的**单轴 + 不广播**特化；与 index_put_ 的区别只在原地 vs functional。

**实现逻辑与复杂度**

```python
def index_put(self, indices, values, accumulate=False):
    out = self.clone()
    B = broadcast(*indices, values)        # 广播所有 index 和 values
    for J in ndindex(B.shape[:-values.ndim]):
        dst = tuple(indices[k][J] for k in range(self.ndim))
        if accumulate:
            out[dst] += values[J]
        else:
            out[dst] = values[J]
    return out
```

时间 O(∏ broadcast_shape)，空间 O(self.shape)。

**边界与陷阱**

1. **【经典 bug】accumulate=True 的类型提升**：`out[dst] += values[J]` 在 fp16 self + fp32 values 时会**报错而非提升**（因为 in-place `+=` 不允许窄化赋值）。常见错误代码：

   ```python
   # w 是 fp16，grad 是 fp32
   w.index_put((idx,), grad, accumulate=True)   # RuntimeError: result type Float can't be cast to Half
   ```
   修复：先把 w 转 fp32 累加，或把 grad 转 fp16。这是 embedding 反向最常见的崩溃点。

2. **同位置多次写、accumulate=False → 未定义**（与 scatter 同），累加时确定。
3. **indices 与 input 维度数必须一致**（或用 None 占位）。
4. **unsafe=True**：跳过 index 越界检查换性能——慎用。

**Inductor 视角**　accumulate=True 时归 reduction-类 template（atomic add，需处理类型提升陷阱）；accumulate=False 时易走 fallback（多轴广播 + 写冲突）。

---

### `aten.masked_scatter(input, mask, source) -> Tensor` — mask 为 True 处按行序填入 source 的下一个元素

**作用与语义**　`mask` 是 bool 张量，与 input 同形状；`source` 是 1D（或可展平）张量。输出 = input 的拷贝，但**按行主序（C order）遍历 mask，每遇到一个 True 就把 source 的下一个元素填进去**。要求 `source.numel() ≥ mask.sum()`。

**示例**　按行主序遍历 `mask`，每遇 `True` 就填入 `source` 的下一个元素：

```python
>>> a = torch.zeros(2, 3)
>>> mask = torch.tensor([[True, False, True], [False, True, False]])
>>> torch.masked_scatter(a, mask, torch.tensor([10., 20., 30., 40., 50.]))
tensor([[10.,  0., 20.],
        [ 0., 30.,  0.]])    # True 位 [0,0],[0,2],[1,1] 依次填 10,20,30
```

**为什么需要这个算子**　当你需要"**按一个布尔模板，把一批值依次灌进指定位置**"——比如：

- **数据填充 / 缺失值补齐**：mask 标出缺失位，source 给出填充值序列。
- **Teacher forcing / schedule sampling**：mask 决定哪些时间步用 ground-truth token 而非模型输出。
- **Speculative decoding 的接受位写入**。
- **Attention 的 padded 位置写入 pad token**。

边界：与 `masked_fill` 的区别——masked_fill 是"mask True 处填**同一个标量值**"，masked_scatter 填的是 **source 的不同值**（按序）。masked_fill 复合为 `where(mask, value, input)`，见第 4 章。

**实现逻辑与复杂度**

```python
def masked_scatter(input, mask, source):
    out = input.clone()
    src_flat = source.flatten()
    p = 0
    for idx in ndindex(input.shape):       # 行主序
        if mask[idx]:
            out[idx] = src_flat[p]; p += 1
    return out
```

时间 O(input.numel())，空间 O(input.numel())。**遍历顺序敏感**是核心特征。

**边界与陷阱**

1. **source 长度不足**：RuntimeError。
2. **遍历顺序**：严格 C order（行主序），不要假设成其他顺序。
3. **mask 必须是 bool**：非 bool（如 uint8）行为依版本而异。
4. **source 多于 mask.sum()**：多余的元素被忽略。

**Inductor 视角**　fallback——遍历顺序 + mask.sum() 的数据依赖使其难于生成静态 kernel，通常 fallback 到 ATen eager，或 decompose 成 `nonzero(mask)` + `index_put`。

---

### `aten.nonzero(input, *, as_tuple=False) -> Tensor | Tuple[Tensor, ...]` — 返回非零元素的坐标

**作用与语义**　**找出张量里所有非零元素的坐标**。默认返回 `(非零个数, 维度数)` 的矩阵，每行是一个非零元素的多维坐标（按行主序）；`as_tuple=True` 则按轴拆成一维数组，便于直接 `input[nonzero(..., as_tuple=True)]`。**行数取决于输入有多少非零元——运行时才知道**（`dynamic_output_shape`）。

**示例**　返回非零元素的**坐标矩阵**——行数取决于数据（dynamic shape）：

```python
>>> b = torch.tensor([[1, 0, 2], [0, 3, 0]])
>>> b.nonzero()
tensor([[0, 0],     # 值 1 在 (0,0)
        [0, 2],     # 值 2 在 (0,2)
        [1, 1]])    # 值 3 在 (1,1)
```

**为什么需要这个算子**　这是把"布尔/数值条件"转成"可索引坐标"的桥梁，所有"按条件抽取元素位置"的逻辑都依赖它：

- **Mask R-CNN / 检测头的 proposal 抽取**：confidence > thr 的 anchor 坐标。
- **稀疏场景：构造 (row, col) 坐标对**。
- **Loss masking**：找出被 pad 的位置并屏蔽。
- **实现 masked_select / index 的内部依赖**。

边界：nonzero 是 `masked_select` / 布尔 `index` 的实现基础（先 nonzero 拿坐标，再 gather）。

**实现逻辑与复杂度**

```python
def nonzero(input, as_tuple=False):
    coords = [idx for idx in ndindex(input.shape) if input[idx] != 0]
    if as_tuple:
        return tuple(LongTensor([c[k] for c in coords]) for k in range(input.ndim))
    return LongTensor(coords)   # (num_nonzero, ndim)
```

时间 O(input.numel())，空间 O(num_nonzero × ndim)。底层通常是两遍：先数个数、再填坐标（或用 prefix sum 单遍）。

**边界与陷阱**

1. **dynamic output shape**：图编译（torch.compile / inductor）里输出形状未知，必须走 dynamic shape 路径或 fallback。
2. **`input != 0` 的精确语义**：float 0.0、int 0、复数 (0+0j) 都算零；NaN 非零。
3. **空输入 / 全零**：返回 `(0, ndim)` 形状——下游必须处理空张量。
4. **as_tuple 的空维度**：单维度全零时对应轴返回空 LongTensor。

**Inductor 视角**　**必须 fallback**——dynamic_output_shape 与 inductor 的静态形状假设冲突，几乎所有 nonzero 调用都退回 ATen eager；这是 inductor graph break 的常见来源。

---

### `aten.embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False) -> Tensor` — embedding 查表：weight[indices]，即 gather(dim=0) 的特化

**作用与语义**　**查表**：给一组整数 ID（`indices`），从权重表 `weight`（每行一个向量）里取出对应的行。严格定义（本质 = `gather(dim=0)`）：

```
out[j, :] = weight[ indices[j], : ]                        # 沿第 0 轴按 ID 取行；输出 = indices.shape + (embedding_dim,)
```

单独成算子是因为调用量极大且带训练专用选项——`padding_idx` 让某行梯度恒为 0（pad token），`sparse=True` 反向输出稀疏梯度。例：`embedding(weight, [0,2,2,1])` 取出第 0、2、2、1 行。

**示例**　`weight[idx]` 查表 = `gather(dim=0)` 特化，输出 = `idx.shape + (embedding_dim,)`：

```python
>>> weight = torch.tensor([[10., 11.], [20., 21.], [30., 31.]])   # 3 个 embedding, dim=2
>>> torch.nn.functional.embedding(torch.tensor([0, 2, 2, 1]), weight)
tensor([[10., 11.],     # weight[0]
        [30., 31.],     # weight[2]
        [30., 31.],     # weight[2] (重复 id 合法, 这是 embedding 的核心需求)
        [20., 21.]])    # weight[1]
```

**为什么需要这个算子**　这是 NLP / 推荐系统最频繁的算子之一——**词表 / item 表查表**。所有"离散 ID → 稠密向量"的映射都靠它：

- **Transformer / RNN / Word2Vec 的 token embedding**：`embed = embedding(weight, token_ids)`。
- **推荐系统的 user/item embedding**。
- **GNN 的节点特征查表**。
- **MoE 的专家 embedding**。

它本质是 gather 的特化（dim=0、index 任意形状、外加 padding_idx/sparse 语义），单独存在是因为：(1) 调用量极大，值得专门优化；(2) padding_idx / scale_grad_by_freq / sparse 是 embedding 特有的训练语义，gather 没有这些选项。

边界：`embedding(idx) ≡ weight.index_select(0, idx) ≡ gather(weight, 0, idx.unsqueeze(-1).expand(-1, D))`。`embedding_dense_backward` / `_embedding_bag`（变长 padding 的 bagged embedding）见附录 A（标注 ○，非 core 基算子）。

**实现逻辑与复杂度**

```python
def embedding(weight, indices, padding_idx=-1, **kw):
    out_shape = indices.shape + (weight.shape[1],)
    out = empty(out_shape, dtype=weight.dtype)
    for J in ndindex(indices.shape):
        out[J, :] = weight[indices[J], :]
    return out
```

时间 O(indices.numel() × embedding_dim)，空间同上。底层就是 gather kernel，常与权重矩阵的 tiling / 分块 cache 优化结合。

**边界与陷阱**

1. **indices 范围**：`0 ≤ idx < num_embeddings`，越界 RuntimeError。
2. **padding_idx 反向**：该行梯度恒为零（前向值不变）。
3. **sparse=True 反向**：输出 sparse coalesce 梯度，节省内存但部分后端不支持。
4. **dtype**：weight 通常 fp32/fp16，indices 必须 int64/int32。
5. **性能**：embedding 常是模型 IO/访存瓶颈，inductor 一般不融合、保留专用 kernel。

**Inductor 视角**　template（gather 特化）——形状静态、读无冲突，是 inductor 里少数稳定走 template 的索引类算子；但因访问模式（按 ID 行访问）通常不与其它算子融合，保持独立 kernel。

## 本章小结

索引家族的难点全部集中在两个第一性问题上：(1) **index 张量的语义双重性**——它编址的是 out 的轴，它的值索引的是 input 的轴：**gather** 里index下标对应 out的位置、值是 input(src) 的坐标（`out[i][j]=input[i][index[i][j]]`）；**scatter** 正好对偶，下标对应 src、值是 self(target) 的坐标（`self[i][index[i][j]]=src[i][j]`）；index_select/embedding 是 gather 的特化；(2) **同位置多次写的语义**——裸 scatter 是后写覆盖（未定义），scatter_add / scatter_reduce / index_put(accumulate=True) 是累加/归约（确定），这决定了 inductor 是走 atomic template 还是 fallback。类型提升（index_put fp16 += fp32）与 dynamic shape（nonzero）是两个反复出现的实战坑。下一章我们离开"按地址读写"的世界，进入【第 10 章 排序与 TopK】——另一类形状动态、依赖归约/分治的算子。

---

[上一章 第 8 章](08-concat-split.md) 　|　 [下一章 第 10 章](10-sort-topk.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 公共概念（广播规则 / 类型提升 / SSA 契约 / dynamic_output_shape）详见第 0 章 §0.4 / §0.5 / §0.7。
