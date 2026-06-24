# 第 8 章 拼接与切分（Concat & Split）

拼接与切分是 Part II（位置变化类）里最"朴素"的一族：它们不做数学运算，只负责把多个张量**粘成一块**（`cat`），或把一块张量**切成多块**（`split_with_sizes`），或把一块张量**像瓷砖一样平铺复制**（`repeat`）。这些操作之所以"从功能描述很难直觉理解"，是因为它们**不在算数学**——它们的难点不在公式（公式平凡到就是"拷贝/重排"），而在**为什么模型需要这种特定的粘合方式、它和零拷贝视图（第 7 章）的边界在哪**。

本章三个 Tier D 算子共有一根暗线：它们**都是分配新张量的位置变化算子**（`cat`/`repeat` 必拷贝，`split_with_sizes` 返回视图列表）。其中 `repeat` 与第 7 章 `expand` 的对比是全书的关键对照——同样"把张量变大"，`expand` 靠 `stride=0` 零分配，`repeat` 真复制，这正是一道常考的面试题与 inductor 优化点。

> 公共概念：广播、类型提升、SSA 契约分别详见第 0 章 §0.4 / §0.5 / §0.7。本章所有算子均为**位置变化**类（第 0 章 §0.6 的右列）。

---

## 本章速查（Tier C）

本章**无 Tier C 算子**——三个算子全部 Tier D 深入讲解。原因：它们都是位置变化类，"为什么存在、与零拷贝视图的边界"远比功能一句话重要，速查表装不下。

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `cat` | 沿指定维把多个张量拼接成一个 | 除拼接维外其余维须相同；**必分配新张量** | fallback（含 cat 的 split_cat 优化） |
| `split_with_sizes` | 沿指定维按尺寸列表切成多块 | 返回**视图列表**（共享 storage，只读别名）；尺寸和须等于该维长度 | fallback（视图，编译期静态求解） |
| `repeat` | 沿每个维按重复次数平铺复制 | repeats 长度可 ≥ 张量维数（左侧补维）；**必分配内存** | pointwise（`ModularIndexing` 取模寻址） |

> 相邻复合算子（不在本章展开）：`stack` = `unsqueeze` + `cat`（→ 第 7 章 + 本章 `cat`）；`unbind` = `slice` + `squeeze` 循环（→ 第 7 章）；`chunk` = `split`（→ 本章 `split_with_sizes` 的均匀切分特化）。分解式详见附录 B。

---

## 深入算子（Tier D）

### `aten.cat.default(tensors, dim=0)` — 沿指定维把多个张量拼成一个

**作用与语义**　给定张量列表 `tensors = [T₀, T₁, ..., T_{k-1}]` 和拼接维 `dim`，输出一个新张量 `out`：沿 `dim` 维依次首尾相连，其余维与各输入相同。

数学上，设各输入沿 `dim` 的长度为 `l_i`，记前缀和 `L_j = Σ_{i<j} l_i`，则：

```
out[..., L_j ≤ idx_dim < L_{j+1}, ...] = T_j[..., idx_dim - L_j, ...]
```

输出形状：`out.shape[dim] = Σ l_i`，其余维 `out.shape[d] = T₀.shape[d]`（所有输入须相同）。返回**新张量**（不别名共享输入 storage）。

**示例**　两个 `1×2` 张量，`dim=0` 沿行拼成 `2×2`，`dim=1` 沿列拼成 `1×4`——形状只沿拼接维变长：

```python
>>> a = torch.tensor([[1, 2]])
>>> b = torch.tensor([[3, 4]])
>>> torch.cat([a, b], dim=0)   # 沿第 0 维拼 -> 1+1=2 行
tensor([[1, 2],
        [3, 4]])
>>> torch.cat([a, b], dim=1)   # 沿第 1 维拼 -> 2+2=4 列
tensor([[1, 2, 3, 4]])
```

**为什么需要这个算子**　模型里"把分散表示合并"的需求无处不在：

- **多分支融合**：残差网络里 shortcut 与主干的特征图相加是逐元素（第 1 章 `add`），但若两路特征**维度不同**（如通道数 256 + 256 → 512），就用 `cat` 沿通道维拼接——这是 ResNet/Inception/DenseNet 的标准操作。`cat` 因此是"特征拼接"的代名词。
- **序列拼接**：把多个 token 序列沿 seq 维拼成一条（batch 内 or batch 间）。
- **多卡/多块输出汇总**：数据并行或张量并行下，各卡算完一片，最后 `cat` 回完整张量。

**边界**：`cat` 与逐元素 `add` 的区别是——`add` 要求**形状完全相同**且结果同形（逐位置相加），`cat` 要求**除拼接维外相同**且结果更大。与 `stack`（复合）的区别是：`stack` 会先 `unsqueeze` 出一个**新维**再 `cat`，所以 `stack([a,b])` 的输出比输入多一维；`cat([a,b])` 不增维。记住：**同形→add；同形但要加一维存"来自哪个"→stack；同形且想变大不增维→cat**。

**实现逻辑与复杂度**

```python
def cat(tensors, dim=0):
    k = len(tensors)
    assert all(t.shape[d] == tensors[0].shape[d] for t in tensors for d in range(t.ndim) if d != dim)
    out_shape = list(tensors[0].shape)
    out_shape[dim] = sum(t.shape[dim] for t in tensors)
    out = empty(out_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    offset = 0
    for t in tensors:
        n = t.shape[dim]
        out_slice = out.narrow(dim, offset, n)      # 切出目标位置的一块
        out_slice.copy_(t)                           # 拷贝（内部处理非连续）
        offset += n
    return out
```

时间复杂度 **O(N)**（N = 输出总元素数，必读必写一次）；空间 **O(N)**（**必分配新张量**）。是否零拷贝：**否**，必然分配并拷贝所有数据。若某输入在 `dim` 维上恰好连续且对齐，`copy_` 可走 memcpy 快路径；否则逐元素拷贝。

**边界与陷阱**

1. **空输入过滤**：`torch.cat([randn(2,3), randn(0), randn(3,3)])` 在旧版是合法的（中间那个 1 维 size-0 张量被默默丢弃）。inductor 的 `cat` 分解会**过滤掉这类 size-0 输入**再 redispatch，下游 `split_cat` 等 pass 依赖此规范化。
2. **除拼接维外须严格相同**：`cat([zeros(2,3), ones(2,4)], dim=0)` 报错（dim=0 拼接要求 dim=1 都是 3，但一个是 3 一个是 4）。常见误用：忘了第 0 维是 batch、第 1 维才是通道，沿错维 cat。
3. **负 `dim`**：支持，`dim=-1` 即最后一维。
4. **单一输入的退化**：`cat([x])` 在 inductor 分解里被优化成 `x.clone()`（避免无谓拼接）。
5. **非连续输入**：`copy_` 内部会处理任意 stride，但若输入是转置/切片视图，可能退化成逐元素拷贝，性能下降。连续输入走 memcpy。

**Inductor 视角**　**fallback**。inductor 对 `cat` 没有专门的 pointwise/reduction/template 核——它落到设备后端的原生 `cat` 实现。但 inductor 的**分解 pass** 会先规范化 cat（去空输入、单输入退化），并在"含 cat 的图"上做 `split_cat` 优化（如把 `cat` 后立即 `split`/沿拼接维 `unbind` 的模式识别出来消除冗余拷贝）。channels-last 输入有专门的 `require_channels_last` 处理路径。

---

### `aten.split_with_sizes.default(self, split_sizes, dim=0)` — 沿指定维按尺寸列表切成多块

**作用与语义**　把 `self` 沿 `dim` 维切成若干块，每块长度由 `split_sizes` 列表指定，返回**张量列表**（视图，共享 storage）。要求 `Σ split_sizes == self.shape[dim]`。

```
split_with_sizes(T, [s_0, s_1, ..., s_{k-1}], dim)
  → [ T.narrow(dim, 0,           s_0),
      T.narrow(dim, s_0,          s_1),
      T.narrow(dim, s_0+s_1,      s_2),
      ... ]
```

每个返回张量沿 `dim` 的长度就是对应的 `s_i`，其余维同 `self`。返回值是**视图列表**——每个视图共享 `self` 的 storage，仅 `offset/shape` 不同，**零拷贝**。

**示例**　长度 10 的张量按 `[3, 2, 5]` **不等分**切成三段（和 = 10），返回的是 `list`，各块长度恰好是 3/2/5：

```python
>>> t = torch.arange(10)
>>> parts = torch.split(t, [3, 2, 5])   # 不等分 -> 返回 list[Tensor]
>>> for p in parts: print(p.shape, p)
... 
torch.Size([3]) tensor([0, 1, 2])
torch.Size([2]) tensor([3, 4])
torch.Size([5]) tensor([5, 6, 7, 8, 9])
```

**为什么需要这个算子**　"把一个张量按已知边界拆开"的需求：

- **多任务/多头拆分**：一层线性层的输出沿通道维 `split_with_sizes` 成多段，分别喂给不同任务头或不同 attention head。比如 `[hidden, hidden, hidden, pooled]` 四段一次性切出。
- **逆操作 of `cat`**：上游 `cat` 起来的张量，下游按相同的尺寸表拆回去——`split_with_sizes` 是 `cat` 的天然逆运算。
- **变长切分**：与 `split`（等长切分）/`chunk`（按块数切分，复合）相对，`split_with_sizes` 处理**各块不等长**的情况，这是它的独特价值——等长用 `chunk`，不等长才用它。

**边界**：`split_with_sizes` 与 `slice`（第 7 章）的关系——前者是"按尺寸列表**批量**切片"，等价于一串 `slice`/`narrow`。与 `unbind`（复合）的区别：`unbind` 沿某维**逐元素**拆（每块 size=1）并 `squeeze` 掉那一维；`split_with_sizes` 每块可以是任意尺寸且**保留** `dim` 维。与 `chunk` 的区别：`chunk` 给块数 `n` 自动算每块大小（可能最后一块更小），`split_with_sizes` 给完整尺寸列表。

**实现逻辑与复杂度**

```python
def split_with_sizes(self, split_sizes, dim=0):
    assert sum(split_sizes) == self.shape[dim]
    out, offset = [], 0
    for s in split_sizes:
        out.append(self.narrow(dim, offset, s))   # narrow = view，零拷贝
        offset += s
    return out
```

时间复杂度 **O(k)**（k = 块数，仅算 offset/shape 元数据）；空间 **O(k)** 个视图对象，**零数据拷贝**。是否零拷贝：**是**（纯视图，与第 7 章 `slice`/`select` 同源）。

**边界与陷阱**

1. **尺寸和必须精确等于该维长度**：`split_with_sizes(randn(10), [2,3,6])` 报错（和=11≠10）。少了多了都不行。
2. **视图的别名风险**：返回的块共享 `self` 的 storage，**修改任一块会影响 `self` 和其它块**（只要落在同一 storage 地址）。在函数式 SSA 图里这不是问题（inductor 会 functionalize），但在 eager 或带原地操作时要小心。
3. **负 `dim`**：支持。
4. **空块**：`split_sizes` 里允许 0（返回一个该维为 0 的空视图），但下游算子可能不接受空张量。
5. **非连续 `self`**：仍可切，视图继承 `self` 的 stride；只读访问无碍，但若要喂给要求连续的算子需先 `contiguous()`。

**Inductor 视角**　**fallback**（视图类）。`split_with_sizes` 在 lowering 时被静态求解成一串 `as_strided`/`slice` 视图元数据，编译期就确定每块的 offset/shape/stride，运行期零开销。它是 inductor `split_cat` 优化的关键拼图——"cat 后 split_with_sizes 还原"模式会被识别并消除中间的大张量拷贝。

---

### `aten.repeat.default(self, repeats)` — 沿每个维按重复次数平铺复制

**作用与语义**　把 `self` 像**铺瓷砖**一样，沿每个维重复 `repeats[i]` 次，得到一个放大后的新张量。`repeats` 是长度等于（或大于）`self.ndim` 的整数列表。

```
repeat(T, [r_0, r_1, ..., r_{n-1}])
  → out.shape = [T.shape[0]*r_0, T.shape[1]*r_1, ..., T.shape[n-1]*r_{n-1}]
    out[i_0, ..., i_{n-1}] = T[i_0 mod T.shape[0], ..., i_{n-1} mod T.shape[n-1]]
```

即：输出的每个下标对**对应输入维长度取模**，回到原张量取值。若 `len(repeats) > self.ndim`，`self` 左侧补 size-1 维再重复（等价先 `view` 补维）。返回**新张量**（不共享 storage）。

**示例**　`1×2` 张量 `.repeat(2, 3)` 像贴瓷砖一样平铺成 `2×6`——行复制 2 倍、列复制 3 倍，**真复制**（与第 7 章 `expand` 的零拷贝对比）：

```python
>>> r = torch.tensor([[1, 2]])
>>> r.repeat(2, 3)         # 1x2 -> (2, 6)：行×2、列×3
tensor([[1, 2, 1, 2, 1, 2],
        [1, 2, 1, 2, 1, 2]])
```

**为什么需要这个算子**　"把一个张量当成基本单元，整块整块地复制粘成更大的块"——这是 `expand` 做不到的：

- **广播做不到的真复制**：`expand` 只能把 size-1 的维"虚拟放大"（stride=0，永远读同一元素）。若想沿一个**非 1 维**重复（如把 `[3]` 变成 `[1,2,3,1,2,3]`），`expand` 无能为力——它不会真的复制内容。`repeat` 才能做这种"块级平铺"。例：把一组位置编码 `[seq, d]` 沿 batch 维 repeat 成 `[batch, seq, d]`，每个 batch 副本都是完整的一份。
- **构造周期性模式**：tile-based 卷积、位置编码广播、attention mask 的分块重复，都需要"整块复制"语义。
- **上采样/分块**：把小特征图 repeat 放大成大特征图（虽然上采样有专门算子，第 16 章，但 repeat 是最朴素的手段）。

**与 `expand` 的边界（关键对比）**：两者都"把张量变大"，但机制截然不同。下表是全书的一道核心对照题：

| 维度 | `aten.expand`（第 7 章） | `aten.repeat`（本章） |
|---|---|---|
| 机制 | 改 stride（被扩展维 `stride=0`） | 真复制数据（取模寻址） |
| 内存分配 | **零分配**（视图，共享 storage） | **必分配**（新张量） |
| 可扩展的维 | 只能扩展 **size=1** 的维 | 可重复**任意 size** 的维 |
| 结果与原张量别名 | 是（视图） | 否（独立张量） |
| 典型用途 | 广播对齐（加 bias 到每行） | 块级平铺（位置编码复制到每 batch） |
| inductor 降级 | view 元数据（编译期） | pointwise（`ModularIndexing` 取模） |

一句话：**`expand` 是"虚胖"（零拷贝，只对 1 维有效）；`repeat` 是"真胖"（复制，对任意维有效）**。模型里若你只需要广播对齐，永远优先 `expand`——它免费；只有"非 1 维也要重复"时才用 `repeat`，且要意识到它真花钱。

**实现逻辑与复杂度**

```python
def repeat(self, repeats):
    n = len(repeats)
    if self.ndim < n:
        self = self.view([1]*(n-self.ndim) + list(self.shape))   # 左侧补 1 维
    out_shape = [s * r for s, r in zip(self.shape, repeats)]
    out = empty(out_shape, dtype=self.dtype, device=self.device)
    for idx in all_indices(out_shape):
        src_idx = [i % s for i, s in zip(idx, self.shape)]        # 每维取模
        out[idx] = self[tuple(src_idx)]
    return out
```

时间复杂度 **O(N)**（N = 输出总元素数）；空间 **O(N)**（**必分配**）。是否零拷贝：**否**。一个重要的常数因子：输出元素数是输入的 `∏ repeats` 倍，`repeat([100,100,100])` 会把张量放大 10⁶ 倍——`mark_reuse` 会因此触发输入的 realize。

**边界与陷阱**

1. **`repeats` 长度语义**：`len(repeats) >= self.ndim`。若 `len(repeats) > self.ndim`，`self` 左侧补 size-1 维（不是右侧！）。例：`tensor([1,2,3]).repeat(2,3)` → shape `[2,9]`（先把 `[3]` 补成 `[1,3]`，再各维乘 `[2,3]`）。**极易踩坑**：以为是"先 repeat 第 0 维再第 1 维"，实际是左侧补维。
2. **`repeats` 含 0**：`repeat([0,...])` 该维变 0，输出是空张量（合法，inductor 走 `empty` 快路径）。
3. **`repeats` 全为 1 或原维为 1 的快路径**：若 `all(a==1 or b==1 for a,b in zip(repeats, old_size))`，inductor 会优化成 `clone(expand(...))`——即先用零拷贝 expand 虚拟放大，再 clone 落地。这是 `repeat` 的"退化成 expand+clone"优化。
4. **代价意识**：`repeat` 是"隐形的大内存消耗源"。一个 `[1, 1024]` 的 bias `repeat(batch, 1)` 会分配 `batch × 1024` 个元素——而等价的 `expand(batch, 1024)` 零分配。code review 见到 `repeat` 先问"能不能换成 expand"。
5. **非连续输入**：取模寻址在任意 stride 上都成立，但缓存不友好（跳跃访问），性能可能差于连续情形。

**Inductor 视角**　**pointwise**。`repeat` 被 lowering 成一个 pointwise 核：输出形状 = `new_size`，`inner_fn` 对每个下标做 `ModularIndexing(idx, 1, old_size[i])`（即 `idx % old_size[i]`）去原张量取值。这正是"取模寻址平铺"的直接翻译。退化情形（全 1 或原维 1）走 `clone(expand(x, new_size))` 零拷贝快路径。因为 repeat 会让数据被复用多次，`mark_reuse` 会按 `new_size/old_size` 比例提示输入是否需要 realize。

---

## 本章小结

拼接（`cat`）与切分（`split_with_sizes`）是互逆的结构变换：前者把多块粘成一块并**必分配**，后者把一块切成多块但**零拷贝视图**。`repeat` 则是"真复制式放大"，它与第 7 章 `expand` 的零拷贝虚拟放大形成全书最重要的内存语义对照——记住"**1 维才能 expand 免费扩，任意维 repeat 必花钱**"这一条，就抓住了这一族算子的性能灵魂。复合算子 `stack`/`unbind`/`chunk` 都分解到本章与第 7 章的基础算子（见附录 B），不另立门户。

下一章（第 9 章 索引家族）将进入全书最难的 scatter/gather/index 家族——它们的轴语义 (axis semantics) 与位置变换比本章的拼接切分复杂得多，是 inductor fallback 与 NPU 适配的重灾区。

---

[上一章 第 7 章](07-shape-and-view.md) 　|　 [下一章 第 9 章](09-indexing-family.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
