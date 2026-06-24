# 第 12 章 内存布局与类型转换（Memory Layout & Dtype Conversion）

> **Part II 末章。** 这一类的算子不"算数学"——它们做的是**拷贝**与**存储重排**：把数据从一个张量搬到另一个、把 dtype 从 `float32` 改成 `bfloat16`、把张量从 CPU 搬到 NPU、把不连续的视图压平成连续。功能描述听上去全是"它就是复制"，但"复制"这件事在 SSA/函数式编译框架里恰恰是最微妙的——**什么时候必须复制、复制在编译图里产生什么副作用、与 `view`/`reshape` 的边界在哪里**——这些才是本章的第一性问题。
>
> 本章承接第 0 章 §0.7 的 SSA 契约裂缝：core aten 整体是纯函数式的（不修改输入），但 `resize_` / `alias` 两个"逃生舱"是例外。读完本章你会理解：**为什么一个"理想"的 inductor 图里几乎不该出现真正的 `copy`——绝大多数拷贝会被融合、消除或改写成 pointwise 核。**

---

## 本章速查（Tier C）

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.resize_(self, size, *, memory_format=None)` | **原地**改 `self` 的 `shape/stride`（可能重分配 storage） | 违反 SSA；形状缩小是安全的视图截断，扩大可能读到未初始化内存；functionalization 通常改写掉它 | **fallback**（`make_fallback(aten.resize_)`） |
| `aten.alias(self)` | 恒等返回 `self`（零操作，仅占位） | 完全不搬数据、不改元数据；SSA 框架内的"别名锚点" | **nop**（`register_lowering` 直接 `return x`） |

> **关于 `resize_` / `alias`**：它们是第 0 章 §0.7 定义的"SSA 逃生舱"。普通模型代码几乎不直接写它们；它们出现在底层管道、autograd 内部、或被 functionalization（函数化 pass）**消除**。本章不展开为 Tier D，下面只在"Inductor 视角"里一句话点明：`resize_` 因为**原地修改 storage 语义无法被 Triton 核表达**，inductor 把它整个丢回 aten eager 当 fallback 执行；`alias` 在 lowering 阶段就退化为 `return x`，图里彻底消失。所以**对一个干净的 inductor 计算图，你基本看不到这两个算子**——这正是 functionalization 想要的结果。

---

## 深入算子（Tier D）

### `aten.clone(self, *, memory_format=None)` — 返回一个独立 storage 的深拷贝

**作用与语义**　`clone` 是 core aten 里**唯一**的"显式深拷贝"原语。它分配一段**全新的、独立的 storage**，把 `self` 的所有元素按指定 `memory_format` 重排后写入，返回一个**与 `self` 形状、dtype、device 完全相同，但 storage 完全独立**的新张量。

- 数学上是个恒等：`out[i] = self[i]`，对所有下标 `i`。
- 形状：`out.shape == self.shape`，`out.dtype == self.dtype`，`out.device == self.device`。
- 返回值：**新张量**，不修改 `self`，不与 `self` 共享 storage（这是它和 `view`/`alias` 的根本区别）。
- `memory_format`：可选 `Preserve`（默认，沿用 `self` 的 stride）/ `Contiguous`（压成行主序）/ `ChannelsLast`（NHWC 排布）。**这是 `clone` 区别于普通 `copy` 的关键参数**——它能在拷贝的同时强制改变内存布局。

**示例**　`clone` 产出**独立 storage** 的副本：数据相同但 `data_ptr()` 不同，改 `b` 不影响 `a`：

```python
>>> a = torch.tensor([[1, 2, 3], [4, 5, 6]])
>>> b = a.clone()                  # 深拷贝
>>> b.data_ptr() == a.data_ptr()
False                              # 独立内存地址
>>> b[0, 0] = 99                   # 改 b ...
>>> a                              # ... 不影响 a
tensor([[1, 2, 3],
        [4, 5, 6]])
>>> b
tensor([[99,  2,  3],
        [ 4,  5,  6]])
```

**为什么需要这个算子**　这是本章第一性指标，必须讲透。

1. **打破别名（aliasing）**。视图类算子（`view`/`slice`/`expand`/`as_strided`，第 7 章）都共享 storage，多个张量指向同一段内存。当你需要**写一份独立副本**——比如保存检查点、对一份中间激活做后续 in-place 操作而不污染原始数据——就必须 `clone`。这是模型里最常见的用途：训练时的梯度检查点、attention 的 KV cache 初始化、EMA（指数滑动平均）权重的副本。
2. **强制连续性（contiguity）的底层原语**。`contiguous()` 是复合算子（→ 不连续时内部调 `clone(memory_format=Contiguous)`）。任何"把一个非连续视图压平成连续 buffer"的需求，最终都落到 `clone` 上。这就是为什么第 7 章反复说"`reshape` 在不能 `view` 时会触发拷贝"——那个拷贝就是 `clone`。
3. **Inductor 的"物化"开关**。在 inductor IR 里，绝大多数张量是惰性的（lazy pointwise 表达式）。当编译器**必须**把一个惰性张量物化成一块真实可寻址的 buffer（例如要做 scatter、要做跨核边界、要送给一个不支持融合的 fallback 算子）时，lowering 代码里会显式 `clone(x)`。所以 `clone` 在 inductor 内部既是用户语义，也是**编译器的物化原语**——这是它的双重身份。
4. **与相邻算子的边界**：`view`/`reshape` 改元数据（可能零拷贝），`clone` 必拷贝；`copy_`（即 `aten.copy`）写**既有**张量，`clone` **新分配**目标张量。一句话：**`clone` = `empty_like` + `copy_` 的语义糖，但带 `memory_format` 参数。**

**实现逻辑与复杂度**

```python
def clone(self, *, memory_format=None):
    # 1. 求目标 stride（按 memory_format 决定）
    if memory_format is Contiguous:
        out_stride = row_major_stride(self.shape)
    elif memory_format is ChannelsLast:
        out_stride = channels_last_stride(self.shape)
    else:  # Preserve（默认）
        out_stride = self.stride   # 沿用原 stride（可能非连续）
    # 2. 分配新 storage
    out = empty_strided(self.shape, out_stride,
                        dtype=self.dtype, device=self.device)
    # 3. 按下标逐元素拷贝（out[i] = self[i]）
    for idx in all_indices(self.shape):
        out[idx] = self[idx]
    return out
```

- 时间复杂度：**O(numel)**——必须读每个元素、写每个元素，无法省略。
- 空间复杂度：**O(numel)**——新分配一份等大的 storage。
- **永不零拷贝**：这是 `clone` 的定义性特征。

**边界与陷阱**

- **`memory_format=Preserve`（默认）会保留非连续 stride**：`x.t().clone()` 的结果**仍然是非连续的**（stride 沿用 `[1,3]`）。很多人误以为"clone 就连续了"——错，要连续必须显式 `clone(memory_format=torch.contiguous_format)` 或用复合算子 `contiguous()`。这是个高频踩坑点。
- **空张量**：`numel == 0` 时 `clone` 返回一个零元素的新张量，不报错，但会走特殊路径（避免零长度分配的 UB）。
- **不连续输入 + 默认 `Preserve`**：虽然 stride 保留，但元素的**值**仍是按逻辑下标拷贝的（`out[i] = self[i]`），所以语义正确；只是输出的物理布局和输入一样"散"。
- **dtype/device 不变**：`clone` **不做**类型转换或设备搬运。要转 dtype 用 `_to_copy`（见下），要跨设备用 `_to_copy(device=...)`。
- **稀疏张量 / NestedTensor**：有专门的 dispatch（`clone_sparse` / `clone_sparse_compressed`），本书不展开。

**Inductor 视角**　`clone` 被降级成一个**pointwise 核**（`Pointwise.create(...)`，`inner_fn = x.make_loader()`）——因为"逐元素恒等拷贝"恰好是 pointwise 的最简形式。这意味着 `clone` **可以被水平/垂直融合**：如果前后都是 pointwise 算子，编译器会把 `clone` 的恒等函数 inline 掉，**实际不生成单独的拷贝核**。所以图里看到的 `clone` 不等于运行期的一次真实内存拷贝——这是 inductor 性能的关键，也是"clone 在 lowering 里既是物化原语又常被消除"的张力所在。

---

### `aten.copy(self, src, *, non_blocking=False) -> Tensor` — 把 `src` 拷贝到既有张量 `self`（可跨设备、可广播）

**作用与语义**　`copy`（用户侧写作 `self.copy_(src)`）把 `src` 的元素写入**已经存在**的张量 `self`，返回 `self` 本身。它是**原地 (in-place)** 操作——这是它和 `clone` 的根本区别（`clone` 新分配，`copy` 写既有）。

- 数学上：`self[i] = src[broadcast(i)]`，对所有 `self` 的下标 `i`。
- 形状：**`self` 的形状不变**；`src` 的形状允许与 `self` **可广播**即可（不必完全相同）。
- dtype：`self` 的 dtype **不变**；`src` 的元素在写入时被**隐式 cast** 成 `self` 的 dtype（不提升 `self`，这是 inductor 显式关掉类型提升的原因）。
- device：**允许跨设备**——`self` 在 NPU、`src` 在 CPU 时，`copy_` 会做一次隐式的 H2D 搬运。`non_blocking=True` 时尝试异步搬运（仅对支持 pinned memory / overlap 的后端有意义）。
- 返回值：`self` 本身（已就地修改）。注意签名 `copy(self, src)` 里 `self` 是**输出**，不是输入——这是它读起来反直觉的原因。

> **命名陷阱**：`aten.copy` 在 PyTorch Python 层对应的方法是 **`Tensor.copy_`**（带下划线，表示 in-place）。inductor 图里你会看到 `aten.copy.default`，它就是 `copy_`。不要和 `clone` / `_to_copy` 混淆——三者职责完全不同（见本章末对比表）。

**示例**　`copy_` 是**原地**操作——把 `src` 写进已存在的 `self`，`dst` 被改写：

```python
>>> dst = torch.zeros(2, 3, dtype=torch.int64)
>>> src = torch.tensor([[10, 20, 30], [40, 50, 60]])
>>> dst.copy_(src)                 # 原地把 src 拷进 dst
>>> dst
tensor([[10, 20, 30],
        [40, 50, 60]])
```

**为什么需要这个算子**　这是本章第一性指标。

1. **`out=` 接口的底层实现**。很多算子（`add(..., out=buf)`、`mm(..., out=buf)`）允许用户**预分配输出 buffer**，把结果直接写进去而不是新分配。这在反复迭代（训练循环、推理 batch 循环）里能避免成千上万次的内存分配。这些 `out=` 语义在分解后，最终都落到一个 `copy` 或等价的 in-place 写入。
2. **跨设备搬运的最简形式**。当你只需要把一个 CPU 张量"灌"进一个已存在的 NPU buffer（而不是新分配一个 NPU 张量），`npu_buf.copy_(cpu_tensor)` 是最直接的。比 `to()` 更省一次分配。`non_blocking=True` 是流水线重叠（computation/communication overlap）的关键钩子。
3. **填充预分配 buffer / 权重加载**。模型加载时，框架先 `empty()` 出权重 buffer，再从磁盘/CPU 把数据 `copy_` 进去。Optimizer 状态、梯度累加 buffer 也是这个模式。
4. **广播写入**。`buf` 形状 `[B, N, D]`，`src` 形状 `[D]`，`buf.copy_(src)` 会把 `[D]` 广播填满整个 buffer——这是"用标量/低维张量刷掉一个高维 buffer"的标准手法。
5. **与相邻算子的边界**：`copy` 写既有张量（in-place），`clone` 新分配张量，`_to_copy` 在新张量上做 dtype/device 转换。三者构成"拷贝三兄弟"——`copy` 是唯一一个**原地 + 可跨设备 + 可广播**的。

**实现逻辑与复杂度**

```python
def copy(self, src, *, non_blocking=False):
    # 1. 若跨设备，先把 src 搬到 self 的 device（隐式 _to_copy）
    if src.device != self.device:
        src = src_to_device(src, self.device, non_blocking)
    # 2. 若 dtype 不同，隐式 cast（注意：不提升 self.dtype，是把 src 降/升到 self.dtype）
    # 3. 广播 src 到 self.shape
    out_shape = self.shape
    # 4. 逐元素写入（含广播对齐）
    for idx in all_indices(out_shape):
        self[idx] = src[broadcast_idx(idx, src.shape)]
    return self
```

- 时间复杂度：**O(self.numel)**（按 `self` 的元素数，不是 `src`）。
- 空间复杂度：**O(1)**（不分配新张量，原地写 `self`）——跨设备时会有**临时**搬运 buffer，但语义上不持有新张量。
- **零分配**（对 `self` 而言）：这是它比 `clone` 省的关键。

**边界与陷阱**

- **`non_blocking=True` 不保证异步**：只在 pinned memory / 支持重叠的后端上有效；普通 CPU 张量搬到 NPU 时 `non_blocking` 常被静默忽略。误以为它一定异步会导致同步错误难以排查。
- **dtype 降精度不报错**：`float32_buf.copy_(float64_tensor)` 会**静默截断**精度，不抛异常。反向（低精度 → 高精度）也静默。这是 `copy` 区别于 `_to_copy` 的一个微妙点：`copy` 的目标 dtype 由 `self` 决定，源 dtype 被默默 cast。
- **广播但不自动转置**：`self` 形状 `[3,2]`、`src` 形状 `[2,3]` 会**报错**（不可广播，最后一维 2≠3）。要转置必须先在 `src` 上 `.t()`。
- **重叠别名（overlapping alias）**：如果 `self` 自身是一个有重叠 memory 的视图（如对角线视图），`copy` 在重叠区域的写入顺序未定义——PyTorch 会检测并报错或给出未定义结果。
- **`self` 和 `src` 是同一张量的视图（部分重叠）**：写入语义未定义，是经典 bug 源。需要先 `src = src.clone()` 解除别名。
- **空张量**：合法，no-op。

**Inductor 视角**　`copy` 的 lowering 显式关掉类型提升（`type_promotion_kind=None`），因为"`fp16.copy_(fp32)` 不应提升 `self` 的 dtype"。它的实现是一条小流水线：跨设备则先 `to_device`，dtype 不同则 `to_dtype`，形状不同则 `expand`，最后 `clone`（即生成一个 pointwise 写入核）。所以 **`copy` 在 inductor 里被拆解成"搬运 + cast + 广播 + pointwise 拷贝"的组合**，而不是单个 fallback 核——这让它能部分享受 pointwise 融合。

---

### `aten._to_copy(self, *, dtype=None, layout=None, device=None, pin_memory=None, non_blocking=False, memory_format=None) -> Tensor` — dtype/device/layout 转换的底层拷贝

**作用与语义**　`_to_copy` 是 `to()` / `.float()` / `.half()` / `.npu()` / `.contiguous()` 等**所有"转换类"复合算子的底层实现**。它根据指定的 `dtype` / `device` / `layout` / `memory_format`，分配一个**符合目标规格的新张量**，把 `self` 的元素转换后拷贝过去。

- 数学上：`out[i] = cast_to(self[i], target_dtype)`，对所有下标 `i`（dtype 转换是逐元素的）。
- 形状：`out.shape == self.shape`（不改形状，只改 dtype/device/layout/stride）。
- 返回值：**新张量**（独立 storage），dtype/device/layout 按参数指定，未指定的项沿用 `self`。
- `memory_format`：和 `clone` 一样，可在转换的同时强制布局（默认 `Preserve`）。
- `device_check: NoCheck` / `device_guard: False`：这个算子**本身**就是用来跨设备的，所以跳过设备一致性检查——这是签名里一个值得注意的细节。

**示例**　`_to_copy` 在用户层走 `.to(dtype=...)`——产出**新张量**、转 dtype，整数变浮点（带小数点），**不改原张量**：

```python
>>> i = torch.tensor([1, 2, 3])               # 整数张量
>>> f = i.to(dtype=torch.float32)             # 落 aten._to_copy，拷贝并转 dtype
>>> f
tensor([1., 2., 3.])
>>> i
tensor([1, 2, 3])                             # 原张量未变
```

**为什么需要这个算子**　这是本章第一性指标，且要讲清"它为什么是 `_` 前缀的内部算子"。

1. **`to()` 的真实本体**。用户写 `x.to(torch.bfloat16)`、`x.to("npu:0")`、`x.to(dtype=torch.float32, device="cpu")`，这些调用在分解后**全部**变成 `aten._to_copy(...)`。`to()` 本身是复合（→ 见附录 B），`_to_copy` 是它的唯一基础算子落地。下划线前缀表示"这是内部调度入口，普通用户应调 `to()`"，但在 inductor 图里你看到的**就是** `_to_copy`。
2. **`.float()` / `.half()` / `.bfloat16()` / `.double()` / `.int()` 的底层**。这些便捷方法在分解后都是 `_to_copy(dtype=...)`。所以一个 bf16 训练图里，所有 `.bfloat16()` 调用都汇聚到同一个 `_to_copy` 节点。
3. **`.npu()` / `.cuda()` / `.cpu()` 的底层**。设备搬运同理，是 `_to_copy(device=...)`。在 torch-npu 的 `_inductor_new` 后端，CPU→NPU 的数据灌入就是这条路径。
4. **`contiguous()` 的底层之一**。`contiguous()` 在输入非连续时，分解成 `clone(memory_format=Contiguous)`；但更广义的"转 dtype + 转连续"组合（`to(..., memory_format=Contiguous)`）会直接走 `_to_copy`。所以 `_to_copy` 是 `clone` + dtype cast 的**超集**。
5. **与相邻算子的边界**：
   - `clone`：只拷贝不改 dtype/device（恒等值拷贝，可选改布局）。
   - `copy`：原地写既有张量，可跨设备/cast，但**不分配**新张量。
   - `_to_copy`：**分配新张量** + 可改 dtype/device/layout，是"转换 + 物化"的复合底层。
   - 三者关系：`_to_copy(self, dtype=d, device=dev)` ≈ `empty_like(self, dtype=d, device=dev)` + `copy_(self)`；当 dtype/device 都不变时，`_to_copy` 退化为 `clone`。

**实现逻辑与复杂度**

```python
def _to_copy(self, *, dtype=None, layout=None, device=None,
             pin_memory=None, non_blocking=False, memory_format=None):
    # 1. 解析目标规格（未指定的项沿用 self）
    dst_dtype  = dtype  or self.dtype
    dst_device = device or self.device
    dst_layout = layout or self.layout
    dst_stride = compute_stride(self.shape, memory_format or Preserve)
    # 2. 分配目标张量
    out = empty_strided(self.shape, dst_stride,
                        dtype=dst_dtype, device=dst_device, layout=dst_layout,
                        pin_memory=pin_memory)
    # 3. 跨设备搬运 + dtype cast + 逐元素拷贝
    if self.device != dst_device:
        out.copy_(self, non_blocking=non_blocking)   # copy_ 内部含搬运 + cast
    else:
        out.copy_(self)                                # 同设备，仅 cast + 布局重排
    return out
```

- 时间复杂度：**O(numel)**（搬运 + cast 都是线性的）。
- 空间复杂度：**O(numel)**（新分配目标张量；跨设备时两端各占一份）。
- **永不零拷贝**：只要 `dtype` 或 `device` 真的变了，就必须分配 + 搬运。当所有参数都等于 `self` 时，`_to_copy` 仍会分配一份副本（不短路成恒等）——这点和 `to()` 的"copy=False 时同规格返回原张量"不同，差异由 `to()` 这层复合逻辑处理。

**边界与陷阱**

- **dtype 截断静默**：`_to_copy(dtype=torch.float16)` 从 `float64` 转过来会**静默丢精度**，不报错。bf16 尤其容易在累加场景溢出（详见第 0 章 §0.5 类型提升）。
- **跨设备 + `non_blocking`**：和 `copy` 一样的陷阱——`non_blocking=True` 只在支持的后端有效，CPU↔CPU 无意义。
- **`layout` 转换**：稠密 ↔ 稀疏的转换是另一条路径，本书不展开；绝大多数 inductor 图里 `layout` 保持稠密不变。
- **`pin_memory=True`**：仅在 CPU 张量上有意义，用于加速后续 H2D 搬运；对 NPU/CUDA 张量无效。
- **complex ↔ real**：complex 转实数会丢虚部（实数转 complex 虚部置零），有专门路径。
- **bitcast（同位宽 reinterpret）**：`view(dtype=...)` 走的是 `to_dtype_bitcast`（不改比特，只改解读），**不是** `_to_copy`——这是另一个常见混淆点。

**Inductor 视角**　`_to_copy` 不走单一 lowering。inductor 把它的语义拆开：**dtype 转换**走 `to_dtype`（生成 pointwise cast 核，可融合）、**设备搬运**走 `to_device`（生成 `ir.DeviceCopy`，是显式的设备边界节点，不融合）、**布局重排**走 `clone`（pointwise）。所以图里 `_to_copy` 通常**不会作为一个整体出现**，而是被分解成上述三件套。其中 **`to_dtype` 因为是 pointwise，常常被融合进相邻的 pointwise 算子**——这就是为什么 `x.to(bf16) + y` 在 inductor 里常常只生成一个融合核。而 `to_device` 是不可融合的硬边界（它对应一次真实的跨设备 DMA）。

---

## 拷贝三兄弟对比（一张表收束）

| 维度 | `clone` | `copy`（即 `copy_`） | `_to_copy` |
|---|---|---|---|
| 是否分配新张量 | ✅ 必分配 | ❌ 写既有 `self` | ✅ 必分配 |
| 是否 in-place | ❌ | ✅ | ❌ |
| 改 dtype | ❌ | ✅（隐式 cast 到 `self.dtype`） | ✅（按参数） |
| 改 device | ❌ | ✅（可跨设备） | ✅（按参数） |
| 改 memory_format | ✅（参数） | ❌ | ✅（参数） |
| 改形状 | ❌ | ✅（可广播 `src` 到 `self`） | ❌ |
| 是否零拷贝 | ❌ 必拷贝 | 对 `self` 零分配 | ❌ 必拷贝 |
| inductor 降级 | pointwise（可融合） | pointwise + to_device + expand 组合 | to_dtype + to_device + clone 分解 |
| 用户层入口 | `.clone()` | `.copy_()` | `.to()` / `.float()` / `.npu()` / `contiguous()` |

**记忆口诀**：`clone` 是"复印一份一模一样的"（可挑布局）；`copy` 是"把那份倒进我这个现成杯子"（杯子形状/材质不变，可跨设备倒）；`_to_copy` 是"复印并顺便换个材质/换个杯子"（`to()` 的真身）。

---

## 复合算子（相邻，非基础，仅指向）

> 本书范围规则：主章节只讲基础算子。下列复合算子的分解式统一在附录 B，这里仅一句话带过并指向对应基础算子。

- **`to(...)`**（复合 → `_to_copy`）：dtype/device/layout/memory_format 转换的用户入口；`copy=False` 且规格相同时短路返回原张量，否则落 `_to_copy`。详见附录 B。
- **`contiguous()`**（复合 → 不连续时 `clone(memory_format=Contiguous)`，连续时返回原张量）：强制行主序。第 0 章 §0.3、第 7 章反复引用。详见附录 B。
- **`.float()` / `.half()` / `.bfloat16()` / `.double()` / `.int()` / `.long()` / `.bool()`**（复合 → `_to_copy(dtype=...)`）：dtype 便捷方法，全是 `_to_copy` 的薄包装。详见附录 B。
- **`.npu()` / `.cuda()` / `.cpu()`**（复合 → `_to_copy(device=...)`）：设备便捷方法。详见附录 B。
- **`Tensor.to(dtype=..., copy=True)`**（复合 → `_to_copy`）：强制拷贝版 `to`。详见附录 B。
- **`torch.tensor(...)` / `Tensor.clone` 的上层 API**：最终也落 `clone` / `_to_copy`，不展开。

---

## 本章小结

1. **拷贝三兄弟职责分明**：`clone`（新分配、恒等值、可选改布局）、`copy`（原地、可跨设备/cast/广播）、`_to_copy`（新分配、可改 dtype/device/layout，是 `to()` 的本体）。记忆口诀见上表。
2. **`resize_` / `alias` 是 SSA 逃生舱**（第 0 章 §0.7）：`resize_` 因原地改 storage 语义被 inductor 当 fallback，`alias` 在 lowering 退化为 `return x`——干净的 inductor 图里基本看不到它们，functionalization 会消除。
3. **拷贝在 inductor 里常被"消除"**：`clone` 和 `to_dtype` 都是 pointwise，会被融合进相邻核；只有 `to_device`（跨设备 DMA）是不可融合的硬边界。理解这一点是看懂 inductor 性能特征的关键。
4. **复合算子的世界以此章为底**：`to` / `contiguous` / `.float()` / `.npu()` 全部建立在 `_to_copy` + `clone` + `copy` 之上。

下一章进入 **Part III 模型层计算原语**：[第 13 章 卷积](13-convolution.md)，从 `convolution` 的 im2col 降级讲起，`clone`/`copy` 将作为 im2col buffer 物化的底层原语再次登场。

---

[上一章 第 11 章 创建与填充](11-creation-filling.md) 　|　 [下一章 第 13 章 卷积](13-convolution.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 涉及广播、类型提升、SSA 契约等公共概念时，详见第 0 章 §0.4（广播）/ §0.5（类型提升）/ §0.7（SSA 契约与 `resize_`/`alias` 逃生舱）。
