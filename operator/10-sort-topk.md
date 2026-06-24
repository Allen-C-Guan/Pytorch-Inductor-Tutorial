# 第 10 章 排序与选取

排序与选取这一类算子的特点是：**它们输出的值不再位于输入的原始位置上**。前面几章（算术、逐点、规约、索引）要么逐元素保留位置，要么沿某轴塌缩长度——输出的每个元素都能在输入里找到一一对应的"源"。排序则打破了这个对应：输出第 `i` 个元素来自输入第 `indices[i]` 个位置，下标本身成了一等公民、被算子显式返回。这正是为什么这类算子从功能描述（"返回排序后的张量"）很难直觉理解——它的语义必须连着 indices 一起看，元组拆包是第一道门槛。

更关键的是，排序与选取在数值上几乎总是 data-dependent 的：sort 的输出位置依赖元素大小关系，topk 的输出形状依赖运行时传入的 `k`。它们不是规约（不塌缩维度），不是逐点（位置被打乱），不是索引（不查表）——它们是一类独立的"顺序变换 (order-transform)"算子，在模型里承担"挑出最重要的若干个""按置信度排序候选""beam search 维护候选池"这类结构任务。本章覆盖两个 core 基础算子 `aten.sort` 与 `aten.topk`（`argsort` 是它们的复合衍生物，见章末）。对 inductor 开发者而言，这两个算子还有一个共同命运：**它们在大多数情况下会落到 fallback，而不是被 codegen 成 Triton 核**——开发者调试时经常会撞上，本章会点明原因。

## 本章速查（Tier C）

本章无 Tier C 算子——`sort` 与 `topk` 均为 Tier D（语义足够独立、需要专门展开）。

## 深入算子（Tier D）

### `aten.sort` — 沿指定轴对张量做全量排序，返回 (values, indices) 元组

**签名**：`aten.sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)`
（另有 `.stable` 变体：`aten.sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False)`；以及 out 变体 `.values`。）

**作用与语义**　沿 `dim` 轴对该轴上的每个一维切片做（升序或降序）排序。输出是**两个**形状与输入完全相同的张量：

- `values`：排好序后的值，沿 `dim` 单调（升序时 `values[..., i] <= values[..., i+1]`）。
- `indices`：`values[..., i]` 在原切片中的下标，即 `values[..., i] == self[..., indices[..., i]]`（沿 `dim` 取数）。

数学上，对输入切片 `x = [x_0, ..., x_{n-1}]`，`sort` 求一个置换 `π` 使得 `x_{π(0)} ≤ x_{π(1)} ≤ ... ≤ x_{π(n-1)}`，输出 `values = [x_{π(i)}]`、`indices = [π(i)]`。`descending=True` 时取反向序。**返回值是一个二元组，必须拆包使用**——这是新手最常踩的坑（见下文）。`indices` 的 dtype 恒为 `int64`。

**示例**　`v, i = torch.sort(...)` 返回的是元组——`v` 是值有序的副本，`i` 是这些值各自在原张量里的下标（两个 `1.` 分别来自原位置 1 与 3）：

```python
>>> v, i = torch.sort(torch.tensor([3., 1., 4., 1., 5.]))   # 默认 ascending
>>> v, i
(tensor([1., 1., 3., 4., 5.]), tensor([1, 3, 0, 2, 4]))
>>> v_desc, i_desc = torch.sort(torch.tensor([3., 1., 4., 1., 5.]), descending=True)
>>> v_desc, i_desc
(tensor([5., 4., 3., 1., 1.]), tensor([4, 2, 0, 1, 3]))
```

**为什么需要这个算子**　排序在模型里的角色不是"算一个数"，而是"建立顺序关系以做后续选取/重排"——典型用途：

- **按置信度排序候选**：分类/检测里把 logits 排序后取 top-N、绘制 PR 曲线；推荐系统里给召回结果排序。
- **为 gather/scatter 准备下标**：得到 `indices` 后配合 `gather` 可在另一张量上做同序重排（如把 attention 权重与 token 一起按某准则排序）。
- **beam search / k-NN 检索**：维护一个按分数有序的候选池。
- **统计分位数、去重前的预处理**。

它与相邻算子的边界很清晰：`sort` 是**全量**排序（整轴都参与、输出长度不变）；`topk`（见下一节）只取前 `k` 个、输出长度变 `k`，且复杂度更低。`argsort` 不是独立算子，它就是 `sort` 后丢弃 `values` 只取 `indices`（复合，分解到 `sort`）。

**实现逻辑与复杂度**

```python
def sort(self, dim=-1, descending=False):
    dim = canonicalize_dim(self.ndim, dim)
    out_values = empty_like(self)
    out_indices = empty(self.shape, dtype=int64)
    for idx in iter_indices(self.shape, skip=dim):       # 遍历除 dim 外的所有轴
        slc = self[idx]                                   # 长度 = self.shape[dim] 的一维切片
        perm = argsort_1d(slc, descending=descending)    # 求置换 π
        out_values[idx] = slc[perm]
        out_indices[idx] = perm
    return out_values, out_indices
```

时间复杂度 O(N · n log n)，其中 N 为除 `dim` 外元素总数、n 为 `dim` 轴长度；空间 O(N · n)（必须分配 `values` 与 `indices` 两个输出，**无法零拷贝**——因为位置被打乱，原地排会破坏输入别名契约）。`stable=True` 保证相等元素的相对顺序不变（稳定排序），实现上通常用合并排序或带原下标的比较；`stable=False`（默认）不保证，实现可选更快的非稳定算法。GPU/加速库实现（如 ACLNN）一般用 radix sort 或 bitonic sort。

**边界与陷阱**

1. **元组拆包**：`result = torch.sort(x)` 得到的是命名元组 `(values, indices)`；写 `v = torch.sort(x)` 再 `v + 1` 会直接报错（tuple 不可加）。务必 `values, indices = torch.sort(x)`。
2. **负索引 dim**：`dim=-1` 指最后一维，由 `canonicalize_dim` 归一；传越界值抛 `IndexError`。
3. **空输入**：对空切片（`shape[dim] == 0`）排序是合法的，返回空 `values` 与空 `indices`，不报错。
4. **0 维张量**：标量（`shape=()`）排序时 inductor 走特殊路径，返回 `(clone(self), full((), 0, int64))`。
5. **NaN 行为**：浮点切片含 NaN 时，排序结果中 NaN 的位置是**实现定义**的（PyTorch 通常把 NaN 排到末尾，但不保证跨后端一致）。跨后端比对排序结果时务必警惕。
6. **相等元素（tie）的顺序**：`stable=False` 时相等元素的相对顺序无定义——不同后端、不同批次可能给出不同的 `indices`，但 `values` 一致。需要可复现请显式 `stable=True`。
7. **非连续输入**：`dim` 若不是"最内维"会导致跨步访问，影响性能；ATen 层会正确处理但实现可能先做拷贝。
8. **`descending` 与 `stable` 的交互**：稳定 + 降序时，相等元素仍按"原顺序"输出（不是反序）。

**Inductor 视角**　Inductor 对 `sort` 的处理**分两条路径**，不是简单一句"fallback"能概括：

1. **Triton 模板核路径**（后端声明支持 `BackendFeature.SORT`、`dim_size` 静态已知 < `int16` 上限、且 `dim_size ≤ 512` 满足 persistent kernel 启发式时）：走 `ir.Sort`——这是一种**类规约的 template IR 节点**（`get_reduction_type()` 返回 `"sort"`），它构造 `iota` 生成 `[0,1,...,n-1]` 的 int16 下标，把 `(values_loader, indices_loader)` 作为键值对一起送进排序核，最后 `to_dtype` 把 indices 转回 int64。见 `torch/_inductor/ir.py::Sort` 与 `torch/_inductor/lowering.py::sort_stable`。
2. **fallback 路径**（不满足上述任一条件）：`make_fallback(aten.sort)` / `make_fallback(aten.sort.stable)`，降级到 eager/ACLNN 执行，**不参与 Triton 融合**。

对 Ascend NPU 后端：当前 `BackendFeature.SORT` 通常未声明，因此实际几乎总是走 fallback——开发者用 `TORCH_LOGS=inductor` 调试时会看到 `sort` 落在 unflattened fallback 区，这是预期行为而非 bug。要点是：`sort` 不是 pointwise 也不是普通 reduction，它是独立的 template 类（仅条件满足时启用），否则 fallback。

---

### `aten.topk` — 沿指定轴选取前 k 个最大/最小元素，返回 (values, indices) 元组

**签名**：`aten.topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)`
（另有 out 变体 `.values`。）

**作用与语义**　沿 `dim` 轴对每个一维切片，选出 `k` 个最大（`largest=True`，默认）或最小（`largest=False`）的元素及其原下标。输出形状：除 `dim` 轴长度由 `n` 变为 `k` 外，其余维度不变。

- `values`：选中的 k 个值；`sorted=True` 时沿 `dim` 单调降序（`largest=True`）或升序（`largest=False`），`sorted=False` 时顺序未定义。
- `indices`：每个选中值在原切片中的下标，dtype 恒为 `int64`。

注意 `k` 是 `SymInt`（符号整数），意味着输出形状依赖运行时 `k`——在动态形状 (dynamic shapes) 场景下输出维度可能是 unbacked symint，这会影响图捕获与融合。

**示例**　同样返回 `(values, indices)` 元组；`topk(..., k=2)` 取最大的两个值（`5.` 与 `4.`）及它们在原张量里的下标（`4` 与 `2`），并按从大到小排好：

```python
>>> v, i = torch.topk(torch.tensor([3., 1., 4., 1., 5.]), k=2)   # 默认 largest=True
>>> v, i
(tensor([5., 4.]), tensor([4, 2]))
```

**为什么需要这个算子**　topk 的核心价值是**"只关心前几名，不关心全局顺序"**——当你只需要 top-1/top-5/top-k 时，做一次全量 `sort` 是浪费。典型用途：

- **分类任务的 top-k 准确率**：top-1 / top-5 是图像分类的标准评测指标，直接 `topk(logits, k=5)`。
- **注意力稀疏化 / routing**：MoE 路由取 top-k 专家、Longformer/Big Bird 注意力取 top-k token，都是 topk 驱动。
- **beam search**：每步从词表分布里取 top-k 候选扩展 beam。
- **NMS（非极大值抑制）的前置**：按 score 取 top-k 框再去做 IoU 过滤，避免对全部框排序。

与 `sort` 的边界：`topk(x, k=n)`（k 等于轴长）等价于一次全排序，但复杂度更高、不该这么用；正常用 `k << n`。`topk` 与 `sort` 都返回 `(values, indices)` 元组，拆包坑相同。

**实现逻辑与复杂度**

```python
def topk(self, k, dim=-1, largest=True, sorted=True):
    assert 0 <= k <= self.shape[dim]
    dim = canonicalize_dim(self.ndim, dim)
    out_shape = list(self.shape); out_shape[dim] = k
    out_values = empty(out_shape, dtype=self.dtype)
    out_indices = empty(out_shape, dtype=int64)
    for idx in iter_indices(self.shape, skip=dim):
        slc = self[idx]                                    # 长度 n 的一维切片
        # 典型实现：partial selection + 小顶/大顶堆，O(n log k)
        perm = select_topk_1d(slc, k, largest=largest)     # 长度 k 的下标数组
        vals = slc[perm]
        if sorted:
            order = argsort_1d(vals, descending=largest)   # 对这 k 个再排序
            vals, perm = vals[order], perm[order]
        out_values[idx] = vals
        out_indices[idx] = perm
    return out_values, out_indices
```

时间复杂度 O(N · (n + k log k))（partial selection + 对 k 个结果排序），通常记为 O(N · n) 级别，比全量 `sort` 的 O(N · n log n) 在 `k << n` 时显著更快。空间 O(N · k)。`sorted=False` 省去最后的 k 排序，更快但输出顺序未定义——下游若依赖顺序须自己再排。底层加速库（ACLNN）一般用 partial radix sort 或堆 (heap)。

**边界与陷阱**

1. **`k` 越界**：`k > shape[dim]` 或 `k < 0` 抛 `RuntimeError: selected index k out of range`（注意不是 `IndexError`）。
2. **`sorted=False` 时顺序未定义**：`sorted=False` 的输出，其 `values` 沿 `dim` 不保证单调、相同输入不同后端可能给不同排列——只保证"这 k 个值被选中"，不保证"按这个顺序"。跨后端比对时若要求顺序一致，必须 `sorted=True`。
3. **并列元素（tie）行为是实现定义的**：当第 k 名与第 k+1 名值相等时，**选中哪一个完全由实现决定**，不保证稳定、不保证按原下标顺序。需要确定性结果请改用 `sort` 后取前 k（成本更高但可复现）。
4. **元组拆包**：同 `sort`，`v, i = torch.topk(x, k)`，不要把返回值当单张量用。
5. **`largest=False`（取最小）**：与 `largest=True` 对称，但有些后端实现走不同内核路径，性能可能与取最大不一致。
6. **动态 `k`**：`k` 为 `SymInt`，图编译期未知时会引入 unbacked symint，导致含 topk 的子图无法静态确定输出形状，影响后续算子的形状推导与融合边界。
7. **NaN**：含 NaN 的浮点切片，NaN 是否入选 top-k、排到哪，是**实现定义**的，跨后端不一致。
8. **`dim` 与非连续**：同 sort，跨步访问影响性能。

**Inductor 视角**　**fallback**。`make_fallback(aten.topk)`（见 `torch/_inductor/lowering.py`）——与 `sort` 不同，inductor **没有** topk 的 Triton 模板核路径，topk 在所有情况下都降级到 eager/ACLNN，不生成 Triton 核、不参与融合。原因：partial selection 的并行算法（堆/带剪枝的归约）在 Triton 里实现复杂且收益不确定，而 ACLNN 有专门优化的 topk 核。对 Ascend 后端，topk 直接落到 ACLNN 的 topk 算子。开发者调试时若发现图里 topk 两侧出现 `buffer_read/buffer_write` 断开融合，这是预期——它本质是一个不可融合的 data-dependent 选取算子。

---

### 相邻复合算子（非本章深入对象）

- `aten.argsort`（复合 → `aten.sort` + 取 `indices`）：即 `sort(x)[1]`。无需单独条目，详见本章 `aten.sort` 与附录 B。

## 本章小结

`sort` 与 `topk` 是 ATen 里"顺序变换"的两个核心基础算子：它们打破位置对应关系，把下标提升为一等输出，因此都返回 `(values, indices)` 元组——拆包是第一道坑，`stable`/`sorted`/tie 行为的可复现性是第二道。复杂度上，`topk`（O(n)，`k << n`）显著优于全量 `sort`（O(n log n)），但 tie 与 NaN 的跨后端一致性都不能保证，做精确比对时要警惕。对 inductor 开发者最重要的结论是：**这两个算子在大多数 Ascend 场景会落 fallback（topk 是无条件 fallback；sort 仅在满足 `BackendFeature.SORT` + 小轴 + persistent 启发式时才有 Triton 模板核）**，图里它们的两侧会断开融合，这是预期而非缺陷。

下一章我们将离开"操作变换"的范畴，进入 **Part III 创建与填充类算子**——从 `zeros`/`linspace`/`empty` 等"凭空造张量"的算子开始，讨论它们与上述所有变换算子的衔接契约。

---

[上一章 第 9 章](09-indexing-family.md) 　|　 [下一章 第 11 章](11-creation-filling.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 公共概念：广播 (broadcast)、类型提升 (type promotion)、SSA 契约详见第 0 章 §0.4 / §0.5 / §0.7。
