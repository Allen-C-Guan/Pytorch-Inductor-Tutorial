# 第 7 章 形状与视图（全书重心之一）

> 本章是 Part II（张量操作类）的开篇，也是全书被引用最多的一章。视图类算子的共同特征是：**它们不算任何数学，只改 `(offset, shape, stride)` 这三个元数据**——但正因为"不算数学"，从功能描述几乎看不出它在干什么、为什么模型需要它。一个刚入职的 inductor 开发者读完 `add` 的文档立刻就会用；但读完 `as_strided` 的文档，多半还是不知道它存在的理由。这一章的任务就是把"为什么"讲透。

所有视图算子都建立在第 0 章 §0.1 的四元组 `(storage, offset, shape, stride)` 之上，并依赖 §0.3 的连续性 (contiguity) 概念。读本章前请确保你记得这条总根：

```
storage_index = offset + i0*s0 + i1*s1 + ... + in-1*sn-1
```

视图算子做的事情，本质是**修改等号右边那三个元数据变量**（offset / shape / stride），而 storage 一字节都不动——这就是"零拷贝"的来源，也是别名 (aliasing) 风险的来源。代价是：改完 stride 的张量往往不再连续，某些算子（尤其是 `view` 本身和缓存敏感的 pointwise 核）会因此变慢或报错，需要先 `contiguous()`（第 12 章，必拷贝）。

本章先给一张速查表把 10 个算子全貌摆出来（Tier C），再逐个深讲（Tier D）。`as_strided` 是其中的旗舰——它是**所有视图算子的底层原语**，其他 9 个都可以用 `as_strided` 表达。

---

## 本章速查（Tier C）

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.as_strided` | 直接指定 (offset,shape,stride)，最底层的视图原语 | 别名共享 storage；stride 含 0 时为广播视图；越界/重叠会触发别名或运行期错误 | fallback（meta-only，不生成核；含别名需 functionalization 处理） |
| `aten.view` | 在**连续**张量上改 shape，自动按行主序重算 stride | 输入必须连续且元素数守恒，否则 `RuntimeError`；零拷贝 | fallback（meta-only） |
| `aten.expand` | 把长度为 1 的维扩成更大，靠 stride=0 实现零拷贝广播 | 只能扩 1 维、不能缩；产出非连续张量 | fallback（meta-only；广播在 pointwise 核里被编译期 stride 求解吸收） |
| `aten.permute` | 交换维度顺序，仅重排 stride 数组 | 元素数守恒；产出通常非连续 | fallback（meta-only） |
| `aten.squeeze` | 删去长度为 1 的维 | shape 改、stride 不变；零拷贝 | fallback（meta-only） |
| `aten.unsqueeze` | 在指定位置插入长度为 1 的维 | shape 改、插入位置 stride 取相邻维推断；零拷贝 | fallback（meta-only） |
| `aten.select` | 沿某维取一个切片（固定该维下标） | 降一维；产出靠 offset+stride 的视图，**通常非连续** | fallback（meta-only） |
| `aten.slice` | 沿某维取 [start:stop:step) 子区间 | 维度变短；靠 offset+stride 视图，step≠1 时非连续 | fallback（meta-only） |
| `aten.flip` | 沿指定维翻转元素顺序 | 靠负 stride 视图，**非连续**；但 many 路径会 materialize | fallback（meta-only；某些后端无负 stride 支持时退化为拷贝） |
| `aten.diagonal` | 取指定偏移的对角线 | 输出 2D，靠 offset+stride 视图，非连续 | fallback（meta-only） |

> 本章没有 Tier C 深度条目——10 个算子全部进 Tier D。原因是它们都属于 Part II 操作类，"为什么存在"是核心，速查表无法承担。

---

## 深入算子（Tier D）

### `aten.as_strided(input, size, stride, storage_offset=None)` — 直接指定 (offset,shape,stride) 的视图，是所有视图算子的底层原语

**作用与语义**　返回一个与 `input` **共享同一段 storage** 的新张量，其 `shape = size`、`stride = stride`、`offset = storage_offset`（缺省时沿用 input 的 offset）。数学上就是直接覆写第 0 章 §0.1 的三元组：

```
out.storage       = input.storage            # 同一段内存，零拷贝
out.offset        = storage_offset if given else input.offset
out.shape         = size
out.stride        = stride
out[i0,...,in-1]  = storage[ out.offset + Σ ik * stride[k] ]
```

不分配任何新内存；`out` 是 `input` 的**别名 (alias)**——两者任一被（原地）修改，另一个可见。元素数 `numel(out) = ∏size`，**不必**等于 `numel(input)`（这正是它能做 expand、slice 的原因）。

**示例**　同一块 storage `[0..11]`、同一段内存，用 `stride=(1,4)` 读出来就是**转置**——`as_strided` 只改 `(shape,stride)`，storage 一字节不动：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> torch.as_strided(a, (4, 3), (1, 4))     # shape=(4,3), stride=(1,4)
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
```

```
storage = [0,1,2,3,4,5,6,7,8,9,10,11]      a 的原始 (3,4) 视图
as_strided(shape=(4,3), stride=(1,4)):
  out[i,j] = storage[ i*1 + j*4 ]
  第 0 行读 storage[0,4,8] -> [0,4,8] ; 第 1 行读 [1,5,9] ; ...
  结果 == a.permute(1,0)  (见下，permute 就是它的受限糖)
```

**为什么需要这个算子**　这是**全书最高杠杆**的一个算子。它存在的理由是：**所有其他视图算子都是它的语法糖**。

- `view(s)` = `as_strided(t, s, contiguous_strides(s))`
- `expand(s)` = `as_strided(t, s, new_stride_with_zero_on_expanded_dims)`
- `permute(axes)` = `as_strided(t, permuted_shape, permuted_stride)`
- `select(dim, idx)` = `as_strided(t, shape_without_dim, same_stride, offset + idx*stride[dim])`
- `slice(dim, a,b,step)` = `as_strided(t, shorter_shape, stride_with_step, offset + a*stride[dim])`，且 `stride[dim] *= step`
- `squeeze`/`unsqueeze` = `as_strided` 改 shape（stride 数组对应增删）
- `flip(dim)` = `as_strided` 把 `stride[dim]` 取负、`offset += (size[dim]-1)*stride[dim]`
- `diagonal` = `as_strided` 用两条 stride 相加的组合构造对角步长

模型域的典型用途：**滑动窗口 (sliding window) / patch 提取**——把一个 `[H, W]` 的图像用 `as_strided` 一次性变成 `[num_h, num_w, kh, kw]` 的 patch 张量（零拷贝），是 conv im2col（第 13 章）、ViT 的 patch embedding、attention 的 sliding window 实现的底层技巧。任何"我希望用一段连续内存的不同视角去看"的需求，底层都是 `as_strided`。

与相邻算子的边界：`view`/`expand`/`permute` 等是**带合法性校验的受限 `as_strided`**（view 要求连续、expand 只能扩 1 维、permute 要求 axes 是排列）；`as_strided` 把所有校验都拆掉，让调用方完全负责——所以它最强大也最危险。**如果你能用受限算子表达，就不要用 `as_strided`**，因为它绕过了所有安全网。

**实现逻辑与复杂度**　纯元数据操作，无数据搬运：

```python
def as_strided(input, size, stride, storage_offset=None):
    out = empty_metadata_tensor()          # 不分配 storage
    out.storage       = input.storage      # 共享，零拷贝
    out.shape         = tuple(size)
    out.stride        = tuple(stride)
    out.offset        = storage_offset if storage_offset is not None else input.offset
    return out
```

时间复杂度 **O(1)**（只写常数个元数据字段），空间复杂度 **O(1)**（零分配）。

**边界与陷阱**　这是 core aten 里**别名风险最高**的算子，坑全在"共享 storage"这件事上：

1. **越界读**：`as_strided` 本身**不校验** `(offset, shape, stride)` 是否落在 storage 范围内。若算出的 `storage_index` 越界，行为是未定义的——读到相邻内存的垃圾值、或在某些后端上 segfault。PyTorch 在 `col2im`/反向等内部路径会先校验，但用户直接调用时不保证。**写 view 类代码时，越界检查是你的责任。**
2. **重叠 (overlapping) 与别名**：当不同 `out[i]` 映射到同一 `storage_index` 时（stride 不构成双射），原地写 `out` 会触发 PyTorch 的 `RuntimeError: unsupported operation: ...`——functionalization 会把这种写改写成 `copy_` 到新张量。
3. **stride 含 0 = 广播视图**：`as_strided(t, [4,3], [0,1])` 产出 4 行全相同的视图，**这就是 `expand` 的实现**。读永远返回同一元素，零分配。详见 §0.4 广播规则。
4. **负 stride = 翻转视图**：`flip` 就靠这个。但**很多后端（含部分 NPU path）不支持负 stride 寻址**，inductor 会先插一个 `contiguous()` 物化。
5. **storage_offset 与 dtype 的耦合**：offset 是**元素数**而非字节数，跨 dtype reinterpret（`view` 成别的 dtype）要自己算字节数除法，PyTorch 对此有严格校验。
6. **inductor / functionalization 关系**：因为 `as_strided` 产出别名，**它不能被原样保留进可融合的 pointwise 图**。functionalization pass 会把"读别名"保留、把"写别名"改写成显式 `copy_`，使后续图恢复 SSA/函数式（见第 0 章 §0.7）。

**Inductor 视角**　`fallback`（meta-only，不生成核）——`as_strided` 在 inductor 里只更新张量的元数据（shape/stride/offset），不出现在任何 Triton 核里；别名语义靠 functionalization pass 在编译期消解成对 storage 的直接读写或显式拷贝。

---

### `aten.view(input, size)` — 在连续张量上改 shape，零拷贝

**作用与语义**　把张量重新解释成 `size` 指定的形状，**自动按行主序重算 stride**（见 §0.3：行主序即从最后一维往前 `stride[k] = stride[k+1]*shape[k+1]`，末维 stride=1）。`-1` 表示该维由元素数守恒自动推断。

```
out.shape  = size          # 含一个 -1 则自动推断
out.stride = row_major_strides(size)
out.numel  == input.numel  # 守恒
out.storage, out.offset    # 与 input 相同（零拷贝别名）
out[i0,...] = input[flatten(i0,...) 按行主序]
```

元素数必须守恒；shape 是输入的**纯重排**（行主序下）。

**示例**　`view(2,6)` 把 12 个元素按行主序重新分组，**不搬数据**（注意：这与转置完全不同！`view(4,3)` 会得到 `[[0,1,2],[3,4,5],...]` 而非转置）：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a.view(2, 6)
tensor([[ 0,  1,  2,  3,  4,  5],
        [ 6,  7,  8,  9, 10, 11]])
```

**为什么需要这个算子**　`view` 解决的是**"同一段连续数据，换个维度去看"**的需求——这是几乎所有模型代码里最高频的结构变换。

典型场景：
- **线性层前的 flatten**：卷积/attention 输出 `[B, C, H, W]` → `view(B, C*H*W)` 喂给 `nn.Linear`。元素数守恒、行主序连续，`view` 零拷贝完成。
- **多头注意力的头拆分**：`[B, seq, n_head*d]` → `view(B, seq, n_head, d)` → `permute` 成 `[B, n_head, seq, d]`。这里 view 负责"在连续内存上把最后一维拆成两维"。
- **batch 制造**：`[B*T, D]` ↔ `[B, T, D]` 的来回切换。

与相邻算子的边界：
- **`view` vs `reshape`**（复合，→ `view` 或 `contiguous()+view`）：`reshape` 是 `view` 的"宽容版"——输入不连续时它自动 `contiguous()` 再 view（可能拷贝），而 `view` 在不连续时直接报错。**模型代码应优先用 `reshape`**（更不容易踩坑），`view` 留给"我明确知道输入连续、要强制零拷贝"的场景。
- **`view` vs `permute`/`transpose`**（复合，→ `permute`）：`view` 只能做**行主序意义上的重排**——它不会改变"哪个元素挨着哪个元素"的物理顺序，只改"怎么分组读"。要让"维度顺序"本身变（转置），必须用 `permute`，不能用 view。经典坑：`t.view(3,2)` 对一个 `[2,3]` 矩阵**不是**转置，而是把行主序数据重新按 `[3,2]` 切——结果与 `t.t()` 完全不同。
- **`flatten`**（复合，→ `view`）：`flatten(start,end)` = `view(*shape[:start], prod(shape[start:end+1]), *shape[end+1:])`。

**实现逻辑与复杂度**　纯元数据：

```python
def view(input, size):
    if not input.is_contiguous():
        raise RuntimeError("view size is not compatible with input tensor's size and layout")
    size = infer_minus_one(size, input.numel())
    if prod(size) != input.numel:
        raise RuntimeError("shape ... is invalid for input of size ...")
    out = share_storage(input)            # 零拷贝
    out.shape  = size
    out.stride = row_major_strides(size)  # 从后往前累乘
    return out
```

时间 O(1)，空间 O(1)（零拷贝）。

**边界与陷阱**：
1. **连续性硬约束**：对任何非连续张量（`.t()`、`permute` 后、`select`/`slice` 后、`expand` 后）调 `view`，**必报错**。这是新手最常见的 `RuntimeError`。解法：要么先 `.contiguous()`（花钱），要么改用 `reshape`。
2. **元素数必须守恒**：`view(5)` 对一个 6 元素张量报错；只能有一个 `-1`。
3. **0 维与 0 元素**：`view(())` 把张量压成标量（要 1 元素）；`view(0,3)` 合法（空张量）。
4. **跨 dtype view**：`tensor.view(dtype)` 要求两 dtype 每元素字节数整除关系（如 float32 ↔ int32 同 4 字节可直接 view；float32 → float16 需要 `8 % 2 == 0` 才行，且 storage_offset 语义变化）。PyTorch 在此有严格校验。
5. **别名**：`view` 的输出是输入别名；functionalization 会处理后续原地写。

**Inductor 视角**　`fallback`（meta-only）——只改 shape/stride 元数据，不生成核；与 `as_strided` 同属"被 functionalization 识别为别名源"的视图算子。

---

### `aten.expand(input, size)` — 把长度为 1 的维扩成更大，靠 stride=0 零拷贝广播

**作用与语义**　把 `input` 的某些长度为 1 的维度扩展成 `size` 指定的更大长度，**不复制数据**。机制是把这些维的 stride 置为 0（详见 §0.4）。

```
对每一维 k：
  if input.shape[k] == 1 and size[k] > 1:
      out.stride[k] = 0            # 永远读到同一元素
  elif input.shape[k] == size[k]:
      out.stride[k] = input.stride[k]
  else:
      raise RuntimeError("expand: the ... dimension ...")  # 不能缩、不能扩非1维
out.shape    = size
out.storage  = input.storage       # 零拷贝
```

允许 `size` 比 `input.ndim` 长（左边补的维度全设 stride=0，即"新维度全是广播"）。

**示例**　把第 0 行 `(1,4)` 扩成 `(3,4)`：三行**完全相同**（靠 stride=0 永远读同一行，零拷贝）：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a[0:1].expand(3, 4)          # (1,4) -> (3,4)，不复制数据
tensor([[0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3]])
```

**为什么需要这个算子**　`expand` 是**显式、可控地把广播规则物化成一个张量**的算子。它解决的是"我希望把这个张量当作更大形状来用，但不想真复制数据"。

典型场景：
- **手动广播后再做不可融合的操作**：大多数逐元素运算里广播是隐式的（pointwise 核自动处理）。但当你要把广播结果喂给 `cat`、`scatter`、卷积等"形状必须严格匹配"的算子时，就需要 `expand` 显式拉齐。
- **偏置加到每个位置**：`bias` 形状 `[C]`，`feature` 形状 `[N,C,H,W]`。`bias.view(1,C,1,1).expand(N,C,H,W)` 后即可与 feature 相加（虽然 pointwise 核会自动广播，但 `expand` 让形状匹配显式化，便于调试与某些后端的模板匹配）。
- **attention 的 mask 广播**：`mask` 形状 `[B,1,1,S]`，`expand` 成 `[B,H,S,S]` 与 attention score 相加。

与相邻算子的边界（这是本章强制对比之一）：

| 算子 | 分配内存？ | 别名？ | 维度变化 | 典型误用 |
|---|---|---|---|---|
| `expand` | **否**（stride=0） | 是（共享 storage） | 只能扩长度为 1 的维、不能缩 | 当成 repeat 用、之后原地写 |
| `repeat`（第 8 章） | **是**（整块复制） | 否 | 任意维按倍数重复，能整段拷贝多份 | 想要零拷贝却用了 repeat |
| `broadcast_to`（复合，→ `expand`） | 否 | 是 | 等价 expand | 与 expand 完全等价，只是名字不同 |
| 隐式广播（pointwise 规则） | 否 | — | 编译期求解成 stride | 不算"算子"，是规则 |

**一句话记住**：**`expand` 零拷贝、产出只读视图；`repeat` 真复制、产出独立张量。** 想要"重复多份且每份能独立改"，必须用 `repeat`（或 `contiguous()` 把 expand 物化）。

**实现逻辑与复杂度**：

```python
def expand(input, size):
    # 左对齐：input 维数少则在左边补长度1
    padded_shape  = (1,)*(len(size)-input.ndim) + input.shape
    padded_stride = (0,)*(len(size)-input.ndim) + input.stride
    out_stride = []
    for k, s in enumerate(size):
        if padded_shape[k] == s:
            out_stride.append(padded_stride[k])
        elif padded_shape[k] == 1:
            out_stride.append(0)             # 广播维
        else:
            raise RuntimeError(...)
    return as_strided(input, size, out_stride)   # 本质就是 as_strided
```

时间 O(1)，空间 O(1)。

**边界与陷阱**：
1. **只能扩、不能缩**：`expand([2])` 对一个 `[4]` 张量直接报错。要缩请用 `slice` 或 `index`。
2. **只能扩长度为 1 的维**：`[3]` 不能 `expand([4])`（3≠1 且 3≠4）。
3. **原地写 expand 的输出是错的**：因为 stride=0，写一个位置会污染所有"广播出来"的位置。PyTorch 检测到这种写会报错；functionalization 会把它改写成拷贝。
4. **expand 后跟 view 会失败**：expand 产出非连续张量（stride 有 0），对其 `view` 会触发 view 的连续性约束报错。要先 `contiguous()` 或 `reshape`。

**Inductor 视角**　`fallback`（meta-only）——`expand` 在 inductor 里通常不单独生成核：广播维度在 pointwise 核里被编译期求解成 stride=0，运行期零开销。`expand` 算子本身只更新元数据，供后续算子的形状推导用。

---

### `aten.permute(input, dims)` — 交换维度顺序，仅重排 stride 数组

**作用与语义**　按 `dims` 指定的顺序重排张量的维度。机制是**把 shape 和 stride 数组按同一置换重排**，storage 完全不动。

```
out.shape[k]  = input.shape[dims[k]]
out.stride[k] = input.stride[dims[k]]
out.storage, out.offset = input.storage, input.offset   # 零拷贝
```

`dims` 必须是 `[0,1,...,ndim-1]` 的一个排列（每个维出现且仅出现一次）。元素数守恒。

**示例**　`permute(1,0)` 交换两维 = 转置，结果与上面 `as_strided(...,(1,4))` **逐元素相同**——印证"permute 就是 as_strided 的受限糖"：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a.permute(1, 0)              # (3,4) -> (4,3)
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11]])
```

**为什么需要这个算子**　`permute` 解决的是**"我想换维度顺序，但不搬数据"**——它只是改"哪个轴叫 axis 0"。

典型场景：
- **矩阵乘前的转置**：`bmm` 要求 `[B, M, K] × [B, K, N]`。如果第二个张量是 `[B, N, K]`，`permute(0,2,1)` 零拷贝得到 `[B, K, N]`。绝大多数 `matmul`/`linear` 里的转置都是 `permute`（或其复合）。
- **注意力里的 head 维换位**：`[B, seq, n_head, d]` → `permute(0,2,1,3)` → `[B, n_head, seq, d]`，让 head 成为 batch 内的独立维。
- **NCHW ↔ NHWC**：卷积框架的两种内存布局切换。

与相邻算子的边界：
- **`permute` vs `transpose`**（复合，→ `permute`）：`transpose(dim0, dim1)` = `permute(...)` 的特化——只交换两个维。`transpose` 在 core 里被分解成 `permute`，二者语义完全一致，`permute` 是基础算子。
- **`permute` vs `view`**：`permute` 改"维度顺序"（哪个轴是轴 0），物理内存顺序不变；`view` 改"分组方式"，假设物理内存是行主序。对一个 `[2,3]` 张量，`.t()`（即 `permute(1,0)`）得到 stride `[1,3]`（非连续），而 `.view(3,2)` 得到 stride `[2,1]`（连续但语义完全不同）。**这是新手最容易混淆的一对。**
- **`permute` 后通常不连续**：除恒等置换外，`permute` 的输出几乎都不连续；后续要做 `view` 必须先 `contiguous()`。

**实现逻辑与复杂度**：

```python
def permute(input, dims):
    assert sorted(dims) == list(range(input.ndim))    # 必须是排列
    out = share_storage(input)
    out.shape  = tuple(input.shape[d]  for d in dims)
    out.stride = tuple(input.stride[d] for d in dims)
    return out
```

时间 O(ndim)，空间 O(1)。

**边界与陷阱**：
1. **dims 必须是排列**：缺维、重复维、越界都报错。
2. **输出非连续**：除 `dims == [0,1,...,n-1]` 外，`permute` 的输出都不满足行主序。后续 `view` 会失败，需先 `contiguous()`（必拷贝）或改用 `reshape`。
3. **原地写**：与所有视图算子一样，写 permute 的输出是别名修改，functionalization 会改写。
4. **某些后端的 stride 顺序约束**：NPU/CUDA 的 matmul 模板可能要求某一维连续，inductor 会自动插入 `contiguous()` 物化。

**Inductor 视角**　`fallback`（meta-only）——只重排 stride/shape 数组，不出现在核里；permute 的"代价"全部转嫁到后续算子（pointwise 核按 permute 后的 stride 寻址，可能缓存不友好；matmul 模板可能要求物化）。

---

### `aten.squeeze(input, dim=None)` — 删去长度为 1 的维

**作用与语义**　移除张量中长度为 1 的维度。`dim=None` 时删去**所有**长度为 1 的维；指定 `dim` 时仅当该维长度为 1 才删（否则原样返回，不报错）。

```
若 dim is None：
    out.shape = [s for s in input.shape if s != 1]
    out.stride = 对应保留维的 stride
否则：
    if input.shape[dim] == 1:
        out.shape = input.shape 删去第 dim 维
        out.stride = input.stride 删去第 dim 维
    else:
        out = input              # 不报错，原样返回
out.storage = input.storage      # 零拷贝
```

**示例**　`(1,3,1)` 去掉两个长度为 1 的维 → `(3)`：

```python
>>> t = torch.arange(3).reshape(1, 3, 1)   # shape (1,3,1)
>>> t.squeeze().shape
(3,)
```

**为什么需要这个算子**　`squeeze` 解决的是**"去掉冗余的尺寸为 1 的维度，让形状更紧凑"**——这是模型代码里清理中间形状的标准动作。

典型场景：
- **规约后去维**：`sum(dim=1, keepdim=True)` 产出 `[B,1,D]`，`squeeze(1)` 变回 `[B,D]`。虽然现代代码常用 `keepdim=False`（不产生那维），但在广播对齐的中间步骤里经常 keepdim 后再 squeeze。
- **标量提取前**：把 `[1]` 压成 `[]`（0 维标量张量）。
- **batch 维清理**：单样本推理时 `[1, C, H, W]` → `squeeze(0)` → `[C, H, W]`。

与相邻算子的边界：
- **`squeeze` vs `view`**：`squeeze` 可看作"删 1 维的受限 view"——它不要求连续（因为它删的是 stride 不影响其它维寻址的 1 维），但本质等价于一次 shape 重写。`view` 也能删 1 维，但 view 要求连续。
- **`unsqueeze`** 是逆操作（见下条）。

**实现逻辑与复杂度**：

```python
def squeeze(input, dim=None):
    if dim is None:
        keep = [k for k in range(input.ndim) if input.shape[k] != 1]
    else:
        keep = list(range(input.ndim))
        if input.shape[dim] == 1:
            keep.remove(dim)
    out = share_storage(input)
    out.shape  = tuple(input.shape[k]  for k in keep)
    out.stride = tuple(input.stride[k] for k in keep)
    return out
```

时间 O(ndim)，空间 O(1)。

**边界与陷阱**：
1. **指定维不等于 1 时不报错**：`squeeze(t, dim=2)` 当 shape[2]=5 时原样返回 t（不抛异常），这是设计而非 bug，但容易让调试困惑。
2. **零拷贝但语义上是别名**：写 squeeze 的输出仍会改原张量，functionalization 处理。
3. **空张量与 0 维**：`squeeze` 对 0 维张量无操作；对全 1 形状 `[1,1,1]`，`squeeze()` 得 `[]`。
4. **梯度里 keepdim 的耦合**：反向传播时 `squeeze` 与 `unsqueeze` 配对出现以恢复形状，这是 autograd shape 推导的常见模式。

**Inductor 视角**　`fallback`（meta-only）——只删 shape/stride 数组的一项，零开销。

---

### `aten.unsqueeze(input, dim)` — 在指定位置插入长度为 1 的维

**作用与语义**　在 `dim` 位置插入一个长度为 1 的新维度。允许 `dim` 为负（`-(ndim+1) <= dim <= ndim`）。

```
out.ndim    = input.ndim + 1
out.shape   = input.shape[:dim] + (1,) + input.shape[dim:]
out.stride  = input.stride[:dim] + (推断值,) + input.stride[dim:]
              # 插入维的 stride 不影响寻址（该维长度为1，永不被越过），
              # PyTorch 取一个合理占位值（通常 = 相邻维 stride * shape）
out.storage = input.storage       # 零拷贝
```

**示例**　给 `(3,4)` 在最前插一个长度为 1 的维 → `(1,3,4)`：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a.unsqueeze(0).shape
(1, 3, 4)
```

**为什么需要这个算子**　`unsqueeze` 解决的是**"在某个位置加一个尺寸为 1 的维，为后续广播/拼接对齐维度数"**。它是 `squeeze` 的逆，也是 `stack`（复合，→ `unsqueeze`+`cat`）的构成元素。

典型场景：
- **为广播对齐维度数**：偏置 `[C]` → `unsqueeze(0)` → `[1,C]`，或 `unsqueeze(1)` → `[C,1]`，便于与 2D 张量广播相加。
- **`stack` 的内部构成**：`stack([a,b,c], dim)` = `cat([unsqueeze(a,dim), unsqueeze(b,dim), unsqueeze(c,dim)], dim)`。
- **把标量抬成张量**：`[]` → `unsqueeze(0)` → `[1]`。

与相邻算子的边界：
- **`unsqueeze` vs `view`**：`unsqueeze(input, dim)` 等价于 `input.view(*shape[:dim], 1, *shape[dim:])`——当输入连续时二者可互换。`unsqueeze` 的优势是不要求连续（它只是给 shape 数组插项，不动其它维的 stride 寻址）。
- **`expand` 的前驱**：`unsqueeze` 出来的 1 维往往是 `expand` 的输入（先 unsqueeze 再 expand 是"加一个广播维"的标准两步）。

**实现逻辑与复杂度**：

```python
def unsqueeze(input, dim):
    dim = wrap_dim(dim, input.ndim + 1)
    out = share_storage(input)
    out.shape  = input.shape[:dim] + (1,) + input.shape[dim:]
    out.stride = input.stride[:dim] + (placeholder,) + input.stride[dim:]
    return out
```

时间 O(ndim)，空间 O(1)。

**边界与陷阱**：
1. **dim 范围比 view 宽**：允许 `dim == ndim`（插在最末）和 `dim == -ndim-1`（插在最前）。
2. **零拷贝别名**：同 squeeze。
3. **不影响连续性判断**：unsqueeze 一个本来连续的张量，结果仍连续（插入的 1 维不影响行主序递推）。

**Inductor 视角**　`fallback`（meta-only）——只插 shape/stride 数组一项。

---

### `aten.select(input, dim, index)` — 沿某维取一个切片（固定该维下标），降一维

**作用与语义**　等价于 `input.index_select(dim, index)` 但 `index` 是单个整数——返回沿 `dim` 第 `index` 个切片，**该维被消去**（输出比输入少一维）。

```
out.shape  = input.shape 去掉第 dim 维
out.stride = input.stride 去掉第 dim 维
out.offset = input.offset + index * input.stride[dim]    # 关键：挪 offset
out.storage = input.storage                              # 零拷贝
```

`index` 支持负索引（`-1` 表最后一个）。

**示例**　`select(1, 1)` 取第 1 列、自动降一维 → 1D（靠挪 offset，零拷贝）：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a.select(1, 1)              # 取第 1 列 -> 1D
tensor([1, 5, 9])
```

**为什么需要这个算子**　`select` 解决的是**"取某维上的一个切片，作为低维张量"**——它是 `slice`（取一段）的极端情形（取长度 1 的一段并顺手降维）。

典型场景：
- **取 batch 中的一个样本**：`select(0, i)` 得到第 i 个样本。
- **取序列的某一步**：`select(1, t)` 在 `[B, T, D]` 上得到 `[B, D]`。
- **取对角元素**（与 `diagonal` 区分）：2D 矩阵 `select(0, i)` 得第 i 行（1D）。

与相邻算子的边界：
- **`select` vs `slice`**：`select(dim, i)` ≈ `slice(dim, i, i+1)` 再 `squeeze(dim)`。select 是"取一个 + 自动降维"的快捷写法。
- **`select` vs `index`**（第 9 章）：`index` 是高级索引（支持任意下标张量、可降多维、可能拷贝）；`select` 是单下标的零拷贝视图特化。
- **`select` vs `narrow`**（复合，→ `slice`）：`narrow(dim, start, length)` = `slice(dim, start, start+length)`，不降维。

**实现逻辑与复杂度**：

```python
def select(input, dim, index):
    dim = wrap_dim(dim, input.ndim)
    if index < 0: index += input.shape[dim]
    assert 0 <= index < input.shape[dim]
    new_offset = input.offset + index * input.stride[dim]
    keep = [k for k in range(input.ndim) if k != dim]
    return as_strided(input,
                      [input.shape[k]  for k in keep],
                      [input.stride[k] for k in keep],
                      storage_offset=new_offset)
```

时间 O(ndim)，空间 O(1)（零拷贝）。

**边界与陷阱**：
1. **输出通常非连续**：例如 `[3,4]` 矩阵 `select(1, 1)`（取第 1 列）得到 stride `[4]` 的 1D 张量——本该连续（末维 stride=1）的 1D 张量这里 stride=4，所以**非连续**。后续 `view` 会失败。
2. **负索引**：`select(dim, -1)` 取最后一个，符合 Python 惯例。
3. **越界报错**：`index` 超出 `[0, shape[dim])` 抛 `IndexError`。
4. **零元素维**：若 `shape[dim]==0`，select 必然越界报错。

**Inductor 视角**　`fallback`（meta-only）——通过 offset + stride 删项构造视图，不出核。

---

### `aten.slice(input, dim=0, start=None, end=None, step=1)` — 沿某维取 [start:end:step) 子区间

**作用与语义**　Python 切片语义在单维上的物化。`start/end` 为 `None` 时取该维端点；支持负索引、负 step（注意 PyTorch 的 `slice` 算子 `step` 通常为正，负 step 走 `flip`）。

```
new_shape_dim = ceil((end - start) / step)        # 该维新长度
out.shape     = input.shape，但 shape[dim] = new_shape_dim
out.stride    = input.stride，但 stride[dim] *= step
out.offset    = input.offset + start * input.stride[dim]
out.storage   = input.storage                     # 零拷贝
```

其余维度不变。

**示例**　`a[:, ::2]`（落 `aten.slice(dim=1, start=0, end=4, step=2)`）每隔一列取一列——`step=2` 把该维 stride 乘以 2：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a[:, ::2]                   # dim=1, step=2 -> 取列 0,2
tensor([[ 0,  2],
        [ 4,  6],
        [ 8, 10]])
```

```
a (3,4)            列:   0   1   2   3        a[:, ::2]  (取列 0 与 2)
[[ 0  1  2  3]          ↑       ↑             [[ 0  2]
 [ 4  5  6  7]   step=2 跳过 1   跳过 3   -->   [ 4  6]
 [ 8  9 10 11]]         ↑       ↑              [ 8 10]]
# stride[dim=1] 由 1 变成 2；offset 不变；零拷贝视图
```

**为什么需要这个算子**　`slice` 解决的是**"取一段连续/等距子区间"**——它是序列截断、KV-cache 填充、训练数据切分等无数场景的底层。

典型场景：
- **Transformer 的 KV-cache**：`past_key` 形状 `[B, S_past, H, D]`，每步把新算的 `[B, 1, H, D]` 拼上去；读取时 `slice(1, 0, S_past+1)` 取已缓存部分。
- **序列 padding/截断**：`slice(1, 0, max_len)` 把变长序列统一截到 `max_len`。
- **滑动窗口数据加载**：在时间序列上 `slice(0, t, t+w)` 取一个窗口。
- **drop_last 的 batch 切分**：`split_with_sizes`（第 8 章）内部就是多次 `slice`。

与相邻算子的边界：
- **`slice` vs `narrow`**（复合，→ `slice`）：`narrow(dim, start, length)` = `slice(dim, start, start+length, step=1)`，是 step=1 的特化。
- **`slice` vs `select`**：select 是 length=1 + 降维；slice 保留维。
- **`slice` vs `index`**：slice 只沿单维取连续/等距区间（零拷贝）；index 支持任意下标集合（通常拷贝）。

**实现逻辑与复杂度**：

```python
def slice(input, dim=0, start=None, end=None, step=1):
    dim = wrap_dim(dim, input.ndim)
    n = input.shape[dim]
    start = 0    if start is None else clamp(wrap(start, n), 0, n)
    end   = n    if end   is None else clamp(wrap(end,   n), 0, n)
    if end < start: end = start
    new_len = (end - start + step - 1) // step
    new_offset = input.offset + start * input.stride[dim]
    new_stride = list(input.stride); new_stride[dim] *= step
    new_shape  = list(input.shape);  new_shape[dim]  = new_len
    return as_strided(input, new_shape, new_stride, storage_offset=new_offset)
```

时间 O(ndim)，空间 O(1)（零拷贝）。

**边界与陷阱**：
1. **越界自动裁剪**：`slice(1, 0, 99999)` 在长度 10 的维上不报错，等价于 `slice(1, 0, 10)`（Python 切片语义）。
2. **step≠1 产生非连续**：stride[dim] 被乘以 step，破坏行主序递推，输出非连续。
3. **start > end**：得到空张量（长度 0），不报错。
4. **负 step 的处理**：core `aten.slice` 的 step 主要为正；要反向取区间应先 `slice` 再 `flip`，或直接用 `as_strided` 配负 stride。
5. **零元素切片**：`slice(dim, 0, 0)` 合法，得到该维为 0 的张量，后续很多算子对空张量有专门路径。

**Inductor 视角**　`fallback`（meta-only）——通过 offset + stride 修改构造视图，零开销；切片信息被后续 pointwise/reduction 核的循环边界吸收。

---

### `aten.flip(input, dims)` — 沿指定维翻转元素顺序

**作用与语义**　沿 `dims` 列出的每个维度翻转元素顺序（首尾互换）。多维翻转 = 各维独立翻转的组合。

```
对每个 d in dims：
    out.stride[d] = -input.stride[d]
    out.offset   += (input.shape[d] - 1) * input.stride[d]
out.shape   = input.shape
out.storage = input.storage       # 理论上零拷贝（靠负 stride）
```

**示例**　`flip(0)` 沿行维翻转（首尾行互换），靠**负 stride** 实现：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a.flip(0)
tensor([[ 8,  9, 10, 11],
        [ 4,  5,  6,  7],
        [ 0,  1,  2,  3]])
```

```
a (3,4)            flip(0) 沿行翻转           结果
[[ 0  1  2  3]     stride[0]: 4 -> -4         [[ 8  9 10 11]
 [ 4  5  6  7]     offset += (3-1)*4 = 8       [ 4  5  6  7]
 [ 8  9 10 11]]    从最后一行倒着读            [ 0  1  2  3]]
# 理论零拷贝（负 stride）；某些后端不支持会退化为拷贝
```

**为什么需要这个算子**　`flip` 解决的是**"反转某一维的元素顺序"**——它在信号处理、可逆网络、双向 RNN/attention 里频繁出现。

典型场景：
- **双向模型的反向序列**：反向 GRU/LSTM 把输入序列 flip 后跑一遍，再 flip 回来与前向结果拼接。
- **可逆残差网络 (RevNet)**：正反向都需要翻转某些维。
- **卷积的对称 padding 推导**：某些 padding 分解会用 flip。
- **数学上的卷积 vs 互相关**：严格数学卷积定义含一个翻转，深度学习里的"卷积"其实是互相关，需要 flip 才回到数学定义。

与相邻算子的边界：
- **`flip` vs `slice(step=-1)`**：语义上等价，但 PyTorch 的 `flip` 走专门的负 stride 路径，而 `slice` 的 step 主要为正。
- **`flip` vs `roll`/`rot90`**（非 core，复合）：`roll` 是循环移位（要拷贝），`flip` 是镜像（理论上零拷贝）。

**实现逻辑与复杂度**　理论上靠负 stride 实现，O(ndim) 元数据、零拷贝：

```python
def flip(input, dims):
    dims = {wrap_dim(d, input.ndim) for d in dims}
    new_stride = list(input.stride)
    new_offset = input.offset
    for d in dims:
        new_offset += (input.shape[d] - 1) * input.stride[d]
        new_stride[d] = -new_stride[d]
    return as_strided(input, input.shape, new_stride, storage_offset=new_offset)
```

**但**：很多后端（含部分 NPU path、某些 pointwise 核）**不支持负 stride 寻址**，这时 inductor 会自动插入 `contiguous()` 物化，退化为 O(numel) 时间 + O(numel) 空间的真拷贝。验证基准下 `flip` 的输出实测 `is_contiguous() == True`，正是因为已被物化。

**边界与陷阱**：
1. **负 stride 不被普遍支持**：上述"零拷贝"是理论上的；实际是否物化取决于后端。在 NPU 上不要假设 flip 一定零拷贝。
2. **dims 不能重复**：重复维翻转两次等于不翻转，PyTorch 会报错或去重（版本相关）。
3. **空维或长度 1 的维**：flip 无效（空集或单元素翻转等于自身）。
4. **原地写 flip 输出**：别名问题，functionalization 处理。

**Inductor 视角**　`fallback`（meta-only，可能物化）——若后端支持负 stride 则零开销元数据视图；否则 inductor 自动插 `contiguous()`，此时产生一次 O(numel) 拷贝核。

---

### `aten.diagonal(input, offset=0, dim1=0, dim2=1)` — 取指定偏移的对角线

**作用与语义**　返回 `input` 在 `dim1`、`dim2` 两维构成的"矩阵平面"上的对角线。`offset>0` 取上对角、`offset<0` 取下对角。其余维度作为 batch 维保留。

```
设 a = shape[dim1], b = shape[dim2]
对角线长度 L = max(0, min(a, b) - |offset|)
out.shape = input.shape 去掉 dim1、dim2 两维，再在最前追加 L
            （实际实现里 L 通常作为最后一维或第一维，依实现）
out 沿对角方向 stride = stride[dim1] + stride[dim2]
out.storage = input.storage          # 零拷贝
```

对 `offset=0`、`dim1=0`、`dim2=1` 的 2D 输入，结果就是主对角线 `[a_00, a_11, ..., a_(L-1)(L-1)]`。

**示例**　主对角线 `[0,5,10]`；上对角线（`offset=1`）`[1,6,11]`：

```python
>>> a = torch.arange(12).reshape(3, 4)
>>> a.diagonal()                # 主对角
tensor([ 0,  5, 10])
>>> a.diagonal(offset=1)        # 上对角
tensor([ 1,  6, 11])
```

```
a (3,4)                     diagonal(): 沿"行stride + 列stride"方向取
[[ 0  1  2  3]              diag_stride = 4 + 1 = 5
 [ 4  5  6  7]   主对角  -> storage[0],[5],[10]  =>  [0, 5, 10]
 [ 8  9 10 11]]   offset=1 -> storage[1],[6],[11] =>  [1, 6, 11]
# 这是唯一用"两条 stride 之和"寻址的视图算子，故无法被 view/slice/permute 表达
```

**为什么需要这个算子**　`diagonal` 解决的是**"沿一条斜线取元素"**——这是唯一一个用**两条 stride 之和**构造新 stride 的视图算子，因此它无法用 view/permute/slice 等单独表达，必须有专门算子（或退化为 `as_strided`）。

典型场景：
- **取矩阵的迹 (trace)**（复合，→ `diagonal` + `sum`）：`trace(A) = diagonal(A).sum()`。
- **批量取对角**：`[B, N, N]` 的 batch 矩阵取每片的主对角 → `[B, N]`。
- **构造对角矩阵的反向**：某些 loss 把对角元素单独取出。
- **协方差/注意力里提取对角项**：attention 的 self-score 对角线 masking 前的提取。

与相邻算子的边界：
- **`diagonal` vs `select`/`slice`**：select/slice 沿坐标轴方向取；diagonal 沿**斜方向**取（stride 是两条轴 stride 的和），这是它独有的能力。
- **`diag`**（复合）：一维 → 二维对角阵、二维 → 一维对角线，是 `diagonal` 与 `scatter`/填充的组合。

**实现逻辑与复杂度**：

```python
def diagonal(input, offset=0, dim1=0, dim2=1):
    dim1, dim2 = wrap_dim(dim1), wrap_dim(dim2)
    a, b = input.shape[dim1], input.shape[dim2]
    if offset >= 0:
        start1, start2 = offset, 0
        L = max(0, min(a - offset, b))
    else:
        start1, start2 = 0, -offset
        L = max(0, min(a, b + offset))
    diag_stride = input.stride[dim1] + input.stride[dim2]
    new_offset  = input.offset + start1*input.stride[dim1] + start2*input.stride[dim2]
    # 其余维作为 batch，构造 as_strided
    return as_strided(input, batch_shape + (L,), batch_strides + (diag_stride,),
                      storage_offset=new_offset)
```

时间 O(ndim)，空间 O(1)（零拷贝）。

**边界与陷阱**：
1. **L 可能为 0**：`|offset| >= min(a,b)` 时对角线长度 0，返回空张量（合法）。
2. **输出非连续**：对角 stride 是两条轴 stride 之和，几乎不可能满足行主序，输出必然非连续。
3. **dim1 == dim2 报错**：两个轴必须是不同的轴。
4. **offset 符号**：`offset>0` 上三角（行起点右移），`offset<0` 下三角（列起点下移），与 NumPy 一致。
5. **batch 维的位置**：实现不同版本会把 L 放在最后一维或第一维，inductor 里以元数据为准，不要假定。

**Inductor 视角**　`fallback`（meta-only）——通过"两条 stride 相加"构造斜向寻址的视图；后续算子按这个 diag stride 寻址，不出专门核。

---

## 本章小结

1. **视图算子的本质是改 `(offset, shape, stride)` 三元组，storage 一字节不动**——所有"零拷贝"都来自这一点；所有"别名风险"也来自这一点（见第 0 章 §0.1）。
2. **`as_strided` 是底层原语**，其余 9 个都是它的语法糖；能用受限算子（view/expand/permute/slice...）表达就不要直接 `as_strided`。
3. **三条关键对比要记死**：`view`（要求连续、零拷贝）vs `reshape`（不要求、可能拷贝，复合）vs `contiguous()`（必拷贝）；`expand`（零拷贝、stride=0、只读视图）vs `repeat`（真复制、独立张量，第 8 章）vs 隐式广播（编译期 stride，规则）；`permute`（换维序、不搬数据、通常非连续）vs `view`（换分组、假设行主序、要求连续）。
4. **视图的代价转嫁给后续算子**：非连续视图会让 `view` 报错、让缓存敏感的核变慢、让某些不支持负/零 stride 的后端被迫物化。inductor 视角下这些算子都是 meta-only 的 fallback，但它们影响的 stride 会被后续 pointwise/reduction/template 核继承。

下一章 [第 8 章 拼接与切分](08-concat-split.md) 讲 `cat`/`split_with_sizes`/`repeat`——`repeat` 是 `expand` 的"真复制"对照，`cat` 是把多个张量拼成一段新 storage（必分配），与本章"零拷贝视图"形成 Part II 的另一半。

---

[上一章 第 6 章 激活函数](06-activations.md) 　|　 [下一章 第 8 章 拼接与切分](08-concat-split.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 本章频繁回链第 0 章：§0.1（`(storage, offset, shape, stride)` 四元组）、§0.3（连续性 contiguity）、§0.4（广播 broadcasting 规则）、§0.7（SSA 契约与 alias/functionalization）。这些是全书被引用最多的概念，详见第 0 章对应小节。
