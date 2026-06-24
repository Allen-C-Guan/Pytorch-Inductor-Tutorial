# 第 3 章 比较与布尔/位运算

比较与布尔/位运算是 Part I（数学语义类）里最"轻量"的一类——它们不做连续的数学变换，只做逐元素的二元判定或位级逻辑。但正是这种"判定"语义，让它们成为模型中几乎所有控制流、掩码 (mask)、稀疏门控、损失过滤的基石：训练里的梯度掩码、注意力里的因果掩码、量化里的越界检测、生成模型里的早停条件，本质上都归约为一连串 `lt` / `eq` / `logical_and` 的组合。从 inductor 后端视角看，这类算子几乎清一色是 **pointwise**（逐元素、无邻域、可广播），是 Triton kernel 里最容易向量化的部分；唯一的旗舰 `where` 兼具 pointwise 与"位置保持选择"的双重身份，是 Part I 与 Part II（位置/索引类）的分界案例——本章结尾会把它与第 9 章的 `gather` 做一次刻意对比。

## 本章速查（Tier C）

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.eq` | 逐元素相等判定，返回 bool | 广播后形状；浮点比相等有精度风险 | pointwise |
| `aten.ne` | 逐元素不等判定，返回 bool | 同上 | pointwise |
| `aten.lt` | 逐元素小于，返回 bool | 同上 | pointwise |
| `aten.le` | 逐元素小于等于，返回 bool | 同上 | pointwise |
| `aten.gt` | 逐元素大于，返回 bool | 同上 | pointwise |
| `aten.ge` | 逐元素大于等于，返回 bool | 同上 | pointwise |
| `aten.logical_and` | 布尔逻辑与（操作 bool） | 操作数非 bool 先转 bool；广播 | pointwise |
| `aten.logical_or` | 布尔逻辑或（操作 bool） | 同上 | pointwise |
| `aten.logical_not` | 布尔逻辑非（操作 bool） | 一元；转 bool | pointwise |
| `aten.logical_xor` | 布尔逻辑异或（操作 bool） | 同上 | pointwise |
| `aten.bitwise_and` | 按位与（操作整数/bool） | 类型必须一致；不提升到 float | pointwise |
| `aten.bitwise_or` | 按位或（操作整数/bool） | 同上 | pointwise |
| `aten.bitwise_not` | 按位取反（整数/bool） | 一元；符号位翻转 | pointwise |
| `aten.bitwise_xor` | 按位异或（整数/bool） | 同上 | pointwise |
| `aten.isinf` | 逐元素判 ±Inf，返回 bool | 一元；对非浮点恒为 False | pointwise |
| `aten.isnan` | 逐元素判 NaN，返回 bool | 一元；对非浮点恒为 False | pointwise |

> 比较六元组 (`eq/ne/lt/le/gt/ge`) 语义高度同构，下文不再逐个展开 Tier D，只在 `where` 的语境里统一说明它们的共性陷阱。

## 深入算子（Tier D）

### `aten.where(condition, self, other)` — 按掩码逐元素选择，真取 self 假取 other

**作用与语义**

逐元素条件选择，是整章里唯一"产出非 bool 张量"的算子：

```
out[i] = condition[i] ? self[i] : other[i]
```

- 输入：`condition`（bool 张量）、`self`、`other`（任意数值类型）。
- 输出：类型 = `self` 与 `other` 的类型提升结果 (promoted dtype)；形状 = 三者广播后的公共形状。
- 三路操作数都参与广播（详见第 0 章 §0.4），不是只广播 `self`/`other`。
- 返回新张量（非视图、非别名）。

**示例**　2×2 布尔 `cond` 选 `self`/`other`：True 位取 `self`、False 位取 `other`；注意 `self`(int) 与 `other`(float) 混用导致**类型提升到 float**，输出带小数点：

```python
>>> cond = torch.tensor([[True, False],
...                      [False, True]])
>>> a    = torch.tensor([[1, 2],
...                      [3, 4]])
>>> b    = torch.zeros(2, 2)
>>> torch.where(cond, a, b)         # True 取 a，False 取 b；int+float -> 提升 float
tensor([[1., 0.],
        [0., 4.]])
```

下面是三路广播对齐：`cond`（1×2）、`self`（1×2）、`other`（2×1）三者广播成 2×2——`cond` 的每一行被复制成两行，`other` 的每一列被复制成两列：

```python
>>> torch.where(cond[0:1, :], torch.tensor([[10, 20]]), torch.tensor([[0], [100]]))
tensor([[ 10,   0],
        [ 10, 100]])
```

**为什么需要这个算子 / 数值与精度**

`where` 是把"布尔判定"重新接回数值流的唯一标准入口。没有它，掩码只能用来 reduce（统计真假个数）或索引（挑出真位），却无法原地构造"真位填 A、假位填 B"的同形张量。它覆盖三类高频场景：

1. **掩码填充**：注意力里的因果掩码、padding 掩码——`where(mask, -inf, 0)` 把不该看的位置压成 -inf，使 softmax 后归零。
2. **数值守卫**：除法前用 `where(denom==0, eps, denom)` 防止除零；log 前用 `where(x>0, log(x), -inf)` 防止 NaN。
3. **梯度路由**：自定义反向里按条件选择上游梯度。

> 相邻的 `masked_fill(mask, value)` 是**复合算子**（→ 分解为 `where(mask, value, self)`，详见附录 B），用户写 `masked_fill` 时 inductor 会落在同一个 `where` kernel 上。

精度上 `where` 本身不计算，只是搬运，故无累积误差；唯一要注意的是**类型提升**：当 `self`/`other` dtype 不同（如 int 与 float 混用），输出会按第 0 章 §0.5 的提升规则升档，可能悄悄把 int32 提升成 float32。

**实现逻辑与复杂度**

NumPy 风格伪代码：

```python
def where(condition, self, other):
    cond_bc, self_bc, other_bc = broadcast(condition, self, other)
    out = empty(cond_bc.shape, dtype=promote(self.dtype, other.dtype))
    for i in all_indices(cond_bc.shape):
        out[i] = self_bc[i] if cond_bc[i] else other_bc[i]
    return out
```

- 时间 O(n)，n = 元素总数。
- 空间 O(n)：必定分配一个与广播后形状等大的新张量，**无法零拷贝**（即使 `condition` 全真/全假，ATen 也不做这种短路优化，因为检查全真本身也是 O(n)）。
- 三路读取 + 一路写入，是典型 memory-bound pointwise。

**边界与陷阱**

- **`condition` 必须是 bool**：非 bool（如 uint8）会先被强制转成 bool，历史代码里 uint8 当掩码的写法在强类型后端上会报 dtype 错。
- **空张量**：`condition` 为空时输出为对应空形状，不报错；但若广播后形状不一致会抛 `broadcast error`。
- **NaN 语义不对称**：`condition` 来自 `lt/le/...` 时，涉及 NaN 的比较返回 False，于是 NaN 位会走 `other` 分支——这是 IEEE 754 的连带后果，`where` 本身不引入 NaN，但它会**放大**比较算子的 NaN 行为。
- **非连续输入**：`where` 接受任意 stride，kernel 内按逻辑索引访问；inductor 在融合时通常先做 stride 重排。
- **`self`/`other` 都会被求值**：`where` 不是惰性的，两个分支都会算（不同于 Python 的 `a if c else b`）。若想省掉昂贵分支，得在图级别手动门控。

**Inductor 视角**

pointwise（三输入、广播、逐元素选择，可与其他 pointwise 算子融合进同一个 Triton kernel）。它是 Part I 的"收尾案例"——位置保持（`out[i]` 只取决于同位 `a[i]/b[i]`），与第 9 章 `gather` 的位置变化（`out[i]` 取自 `a[index[i]]`）形成对照：前者仍是 pointwise，后者必须落到 indexing/template 类。

## 本章小结

比较六元组与 `logical_*`/`bitwise_*`/`isinf`/`isnan` 共同构成"判定层"——它们产出 bool、可广播、纯 pointwise，是 inductor 最易融合的一类；`where` 则是把判定结果重新接回数值流的唯一标准算子，三路广播 + 必分配，仍是 pointwise 但已是该类的复杂度上限。下一章进入 reduction 类（sum/mean/max/min），算子首次跨元素聚合，inductor 的归约轴与并行策略将登场。

---

[上一章 第 2 章](02-transcendental.md) 　|　 [下一章 第 4 章](04-reductions.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
