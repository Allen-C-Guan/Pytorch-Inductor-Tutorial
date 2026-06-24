# 第 14 章 池化（Pooling）

池化是卷积网络（CNN）里与卷积并列的两大"局部窗口"算子之一。卷积（第 13 章）在窗口内做**加权求和**（权重可学），池化则在窗口内做**固定规则的聚合**——取最大值（max pooling）或取平均值（average pooling）。两者的共同骨架是同一条：把输入特征图按一个滑动的 `kernel_size` 窗口、`stride` 步长、`padding` 边缘展开，每个窗口塌缩成一个标量，从而**下采样**空间分辨率（H、W 维），同时（大体）保留空间结构。

在模型里池化承担两个角色：①**降分辨率**——减小后续层的算力、扩大感受野；②**平移不变性**——小幅平移输入不改变 max/avg 的输出，提升鲁棒性。现代架构（ResNet/ViT）很多用 stride>1 的卷积替代池化来降分辨率，但池化在分类头（global average pooling）、检测/分割的 neck、以及算力受限场景仍大量使用。

本章三个算子都是 Part III 的**模型层计算原语**：inductor 不把它们分解成 pointwise/reduction，而是走专门的 **template**（或降级到 `mkldnn`/`aten` native **fallback**）——因为滑窗逻辑 + padding 边界 + 反向所需的 argmax 都需要专用核，无法用通用规约模板覆盖。它们与第 13 章卷积共享同一种"局部窗口"思想（im2col 也是把窗口展平成列），但池化的聚合规则是固定的（无权重），所以实现更简单、数值更平凡。

> 本章属 Part III。算子的数学本身简单（取 max / 取均值），重点因此放在：**滑窗几何（输出尺寸公式、ceil_mode 的坑）**、**反向所需的 indices**、**自适应 (adaptive) 池化如何反推 stride**、以及 **count_include_pad 对数值的影响**。广播与类型提升详见第 0 章 §0.4 / §0.5；SSA 契约与张量基底详见 §0.7 与第 0 章全章。

## 本章速查（Tier C）

本章无 Tier C 算子——三个池化算子全部 Tier D 深入讲解（见下）。3D 变体（`max_pool3d` / `avg_pool3d` / `_adaptive_avg_pool3d`）与 2D 同构，仅附录 A 一行标 ○，正文不展开。

## 深入算子（Tier D）

### `aten.max_pool2d_with_indices(self, kernel_size, stride=[], padding=0, dilation=1, ceil_mode=False) -> (Tensor, Tensor)` — 2D 最大值池化，并返回每个输出的来源下标

**作用与语义**　对 4D 输入 `(N, C, H, W)`（或 3D `(C, H, W)`）在每个 `(C, H, W)` 切片上，沿 H/W 两维滑动一个 `kernel_size=(kH, kW)` 的窗口，步长 `stride=(sH, sW)`，边缘补 `padding=(pH, pW)` 个 `-inf`（语义上让 padding 区不参与 max），窗口内可选 `dilation=(dH, dW)` 空洞展开。每个窗口输出其元素最大值。输出空间尺寸：

```
oH = floor((H + 2*pH - dH*(kH-1) - 1) / sH) + 1     # ceil_mode=False
oH = ceil ( (H + 2*pH - dH*(kH-1) - 1) / sH) + 1     # ceil_mode=True
```
oW 同理。输出张量 `out` 形状 `(N, C, oH, oW)`，与输入同 dtype。**关键**：本算子返回**二元组** `(out, indices)`，其中 `indices` 形状与 `out` 相同、dtype 为 `int64`，记录每个输出元素对应的**输入展平下标**（按 `(C*H*W)` 维展平的一维下标，不是 (h,w) 二元组）。

> `stride=[]` 的空默认值表示 `stride = kernel_size`（无重叠池化），这是 PyTorch 的约定。

**示例**　1×1×4×4 输入上 2×2 无重叠窗口取 max，返回**元组** `(output, indices)`——`indices` 是每个输出来源的展平下标（如左上窗口 `{0,1,4,5}` 最大值 5 位于展平下标 5）：

```python
>>> x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
>>> out, idx = torch.nn.functional.max_pool2d_with_indices(x, kernel_size=2, stride=2)
>>> out            # 每个窗口的最大值
tensor([[[[ 5.,  7.],
          [13., 15.]]]])
>>> idx            # 每个输出的来源展平下标（沿 C*H*W 维）
tensor([[[[ 5,  7],
          [13, 15]]]])
```

**为什么需要这个算子 / 数值与精度**　max pooling 的前向本身平凡（窗口取 max）。**真正需要解释的是为什么前向要返回 `indices`**：max pooling 的反向是"把梯度路由回前向取 max 的那个位置、其余位置梯度为 0"——即 `grad_input[indices] += grad_output`。如果反向不知道每个输出来自输入哪个位置，就无法正确回传梯度（average pooling 不需要，因为它对所有位置均分梯度）。因此前向必须把 argmax 一并算出并随图传递，反向算子 `max_pool2d_with_indices_backward` 直接消费这个 `indices`。这就是算子名带 `_with_indices` 后缀的全部原因。从数值精度看，max pooling 是精确操作（无累加、无除法），不存在 bfloat16 的累加误差问题；唯一的"数值陷阱"是 padding 处填什么——实现填 `-inf`，所以即使整个窗口都在 padding 区，输出也是 `-inf`（而非 0），下游若未预期会出 NaN。

**实现逻辑与复杂度**　NumPy 风格伪代码（前向）：

```python
def max_pool2d_with_indices(x, kH, kW, sH, sW, pH, pW, dH, dW, ceil_mode):
    N, C, H, W = x.shape
    oH = floor((H + 2*pH - dH*(kH-1) - 1) / sH) + 1
    oW = floor((W + 2*pW - dW*(kW-1) - 1) / sW) + 1
    # 若 ceil_mode=True 且最后窗口起点落在 padding 内，补一格（见"边界与陷阱"）
    out      = empty(N, C, oH, oW, dtype=x.dtype)
    indices  = empty(N, C, oH, oW, dtype=int64)
    xp = pad(x, ((0,0),(0,0),(pH,pH),(pW,pW)), value=-inf)   # 仅概念；实现常按需计算索引
    for n, c, oh, ow:
        best_val, best_idx = -inf, 0
        for kh in range(kH):
            for kw in range(kW):
                ih = oh*sH - pH + kh*dH          # 映射回未 padding 的输入坐标
                iw = ow*sW - pW + kw*dW
                if 0 <= ih < H and 0 <= iw < W:   # 越界 = padding 区，跳过（视为 -inf）
                    v = x[n, c, ih, iw]
                    flat = (c*H*W + ih*W + iw)    # 输入沿 (C,H,W) 展平的一维下标
                    if v > best_val:
                        best_val, best_idx = v, flat
        out[n,c,oh,ow]     = best_val
        indices[n,c,oh,ow] = best_idx
    return out, indices
```
时间复杂度 `O(N·C·oH·oW·kH·kW)`，空间 `O(N·C·oH·oW)`（输出 + indices 各一份，**会分配新张量**，非零拷贝）。高效实现（im2col 或直接滑窗核）会避免上面双层 Python 循环，但语义等价。`indices` 存的是**展平下标**而非 (h,w)，是为了反向用一个 `scatter` 即可回传。

**边界与陷阱**
- **ceil_mode=True 的"最后一窗"**：当向上取整让 `oH` 多出一行时，该行窗口可能整行落在 padding 区，输出为 `-inf`。PyTorch 额外要求"最后一个窗口的起点必须落在输入范围内"，否则抛错（`Too large kernel/padding/stride`）。这是 ceil_mode 最常见的坑。
- **stride 空默认**：`stride=[]` ⇒ `stride = kernel_size`。传 `stride=None` 而非空列表会报错。
- **dilation 空洞**：`dilation>1` 时窗口内元素间隔 `dilation`，等效扩大感受野不增参数；输出尺寸公式里 `dH*(kH-1)` 项随之放大。
- **非连续输入**：算子接受任意 stride 的输入（内部按 stride 寻址），但极度非连续（如转置后的图）可能在某些后端触发隐式 contiguous 拷贝。
- **类型提升**：max 不涉及类型提升（输入输出同 dtype）；`indices` 恒为 `int64`。
- **NaN 传播**：窗口内若有 NaN，`max` 在多数实现里返回 NaN（NaN 与任何值比较均为 false，但 PyTorch 的 max pooling 核对 NaN 的行为是"NaN 胜出"），与 `fmax`（忽略 NaN）相反。

**Inductor 视角**　**template / fallback**——max pooling 有专用滑窗模板（inductor 的 `mkldnn`/cpu 路径）或直接降级到 `aten` native 实现；它**不是** reduction（窗口内有空间几何 + dilation + padding，规约轴不规整）、**不是** pointwise（输出尺寸与输入不同），因此无法走通用 pointwise/reduction 模板，必须走专门 template 或 fallback。带 `indices` 的返回也使其无法被当作纯 reduction 融合。

---

### `aten.avg_pool2d(self, kernel_size, stride=[], padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor` — 2D 平均值池化

**作用与语义**　输入 `(N, C, H, W)`（或 3D），在每个 `(C,H,W)` 切片上沿 H/W 滑动 `kernel_size=(kH,kW)` 窗口、步长 `stride`、边缘补 `padding` 个 0（注意：**avg pool 补 0**，不是 `-inf`），窗口内元素取**算术平均**。输出空间尺寸公式与 max pooling 完全相同（含 ceil_mode 规则）。输出形状 `(N, C, oH, oW)`，与输入同 dtype。**只返回一个张量**（无 indices）——因为反向对所有位置均分梯度，不需要 argmax。

平均的分母由 `count_include_pad` 和 `divisor_override` 控制（见下）。

**示例**　1×1×4×4 输入上 2×2 无重叠窗口取平均——左上窗口 `(0+1+4+5)/4 = 2.5`：

```python
>>> x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
>>> torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
tensor([[[[ 2.5000,  4.5000],
          [10.5000, 12.5000]]]])
```

**为什么需要这个算子 / 数值与精度**　avg pooling 的"平均"看似平凡，**真正需要解释的是分母怎么算**——这正是 avg 与 max 在数值上的核心差异，也是 `count_include_pad` 参数存在的理由：

- `count_include_pad=True`（默认）：分母 = `kH*kW`（固定窗口大小）。padding 补的 0 参与求和（贡献 0）但**计入分母**，因此靠近边缘的窗口平均值被 padding 的 0 拉低。
- `count_include_pad=False`：分母 = 窗口与**真实输入**重叠的元素数（即排除 padding 区）。这样边缘窗口的平均不被 padding 稀释，数值更"诚实"。
- `divisor_override=d`（若非 None）：直接用固定整数 `d` 做分母，**忽略** `count_include_pad`。用于让用户完全掌控除数。

精度陷阱：avg pooling 涉及**累加 + 除法**，在 `bfloat16`/`float16` 下累加大窗口（如 7×7=49 个元素）会有精度损失。PyTorch 的 CUDA 实现通常在 `float32` 内部累加再转回；但某些低精度路径或自定义核可能直接在低精度累加，导致与参考实现的细微偏差。`divisor_override` 在希望"控制除数以匹配某参考实现"时有用。

**实现逻辑与复杂度**　NumPy 风格伪代码（前向）：

```python
def avg_pool2d(x, kH, kW, sH, sW, pH, pW, ceil_mode, count_include_pad, divisor_override):
    N, C, H, W = x.shape
    oH = floor((H + 2*pH - kH - 1) / sH) + 1          # 此处 dilation=1（avg 无 dilation）
    oW = floor((W + 2*pW - kW - 1) / sW) + 1
    out = empty(N, C, oH, oW, dtype=x.dtype)
    for n, c, oh, ow:
        s = 0.0
        cnt = 0
        for kh in range(kH):
            for kw in range(kW):
                ih = oh*sH - pH + kh
                iw = ow*sW - pW + kw
                if 0 <= ih < H and 0 <= iw < W:        # 真实输入元素
                    s += x[n, c, ih, iw]
                    cnt += 1
                elif count_include_pad:                 # padding 区，加 0 但计 1
                    cnt += 1
        denom = divisor_override if divisor_override is not None \
                else (kH*kW if count_include_pad else cnt)
        out[n,c,oh,ow] = s / denom
    return out
```
时间复杂度 `O(N·C·oH·oW·kH·kW)`，空间 `O(N·C·oH·oW)`（分配新输出张量，非零拷贝）。注意 avg pool **没有 dilation 参数**（与 max pool 不同），窗口始终是密集的 `kH×kW`。

**边界与陷阱**
- **padding 补 0 而非 -inf**：与 max pooling 的 padding 语义完全不同。误以为"padding 不参与"会导致 `count_include_pad=True` 时数值偏低。
- **ceil_mode 的尾部窗口**：与 max pooling 同样的几何规则；ceil_mode=True 时尾部窗口可能部分或全部落在 padding 区，`count_include_pad=False` 下分母可能为 0（PyTorch 在此情况会让分母 = 整个窗口大小 `kH*kW` 以避免除零，但这与文档表述有微妙出入——以实际核行为为准）。
- **`count_include_pad=False` 的除零**：理论上某窗口若完全在 padding 区，`cnt=0`；实现通常以 `kH*kW` 兜底。自定义后端移植时务必复测此边界。
- **stride 空默认**：同 max，`stride=[]` ⇒ `stride = kernel_size`。
- **反向**：`avg_pool2d_backward` 把 `grad_output` 按 `1/denom` 均分回窗口内每个位置（padding 区的梯度丢弃）。它**不需要前向的 indices**——这是 avg 与 max 反向的根本差异。
- **非连续/类型提升**：同 max pooling；avg 涉及累加，低精度路径需注意累加精度。

**Inductor 视角**　**template / fallback**——与 max pooling 同理，走专用 template 或 `aten` native。不可规约为通用 reduction（空间几何 + padding 分母规则非规整）。累加路径在 NPU 上需注意低精度累加坑（详见第 0 章 §0.5 类型提升）。

---

### `aten._adaptive_avg_pool2d(self, output_size) -> Tensor` — 自适应平均池化（指定输出尺寸，内部反推 stride/窗口）

**作用与语义**　输入 `(N, C, H, W)`（或 3D），用户**不指定 kernel/stride**，而是直接给出**期望的输出空间尺寸** `output_size=(oH, oW)`。算子自动决定每个输出位置对应输入的哪一段区间并取平均，使输出恰好为 `(N, C, oH, oW)`。最常见用法是 `output_size=(1,1)` —— 即 **global average pooling (GAP)**，把整张特征图压成单个向量，是分类头标准组件。

**示例**　用户只给输出尺寸 `(1,1)`，算子内部反推窗口为整张 4×4、取全局平均（GAP）——`mean(0..15) = 7.5`：

```python
>>> x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
>>> torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
tensor([[[[7.5000]]]])
```

**为什么叫"自适应"**：因为不同的输出位置对应的输入区间**长度可能不同**。当输入尺寸不能被输出尺寸整除时，PyTorch 用一组公式为每个输出位置 `a`（`a ∈ [0, oH)`）计算输入的起止下标 `[start(a), end(a))`：

```
start(a) = floor( a      * in_size / out_size )
end(a)   = floor((a + 1) * in_size / out_size + (out_size-1)/out_size )   # 见下方等价形式
```
等价的、PyTorch 实现实际使用的形式（整数运算）：
```
start(a) = floor( a      * in_size / out_size )
end(a)   = floor((a+1)   * in_size + out_size - 1 ) // out_size          # = floor(((a+1)*in_size + out_size-1)/out_size)
```
因此每个输出位置的窗口大小 `end(a)-start(a)` 可以**逐位置变化**——这就是"adaptive"：不靠固定 kernel+stride 的几何滑动，而是靠**按比例切分输入**自适应地决定窗口边界。当 `in_size % out_size == 0`（整除）时，所有窗口等长，退化为普通的 `avg_pool2d`（PyTorch 的 decomposition 正是检测到整除就直接调 `avg_pool2d` 走快路径）。

**为什么需要这个算子 / 数值与精度**　设计动机：让用户**只关心输出要多大**，不必手算 stride/kernel 去凑目标分辨率。这在网络结构里极为方便——例如把任意输入分辨率的特征图都池化到固定 `(1,1)`（GAP）或 `(7,7)`（对齐下游分类/检测头），无需关心输入 H/W 是多少。数值上与 avg pooling 一致（取平均），精度陷阱同 avg pooling（低精度累加、padding 此处不存在因为自适应池化**没有 padding 参数**——它直接切分真实输入）。

**实现逻辑与复杂度**　基于 PyTorch `decompositions.py` 的实际实现（NumPy 风格化简）：

```python
def _adaptive_avg_pool2d(x, oH, oW):
    N, C, H, W = x.shape
    # 快路径：整除则退化为普通 avg_pool2d
    if H % oH == 0 and W % oW == 0:
        sH, sW = H // oH, W // oW
        kH, kW = H - (oH-1)*sH, W - (oW-1)*sW
        return avg_pool2d(x, (kH,kW), (sH,sW))
    # 慢路径：逐位置自适应窗口
    out = empty(N, C, oH, oW)
    for oh in range(oH):
        i0 = (oh   * H) // oH                      # start
        i1 = ((oh+1)*H + oH - 1) // oH            # end  (≈ ceil((oh+1)*H/oH))
        for ow in range(oW):
            j0 = (ow   * W) // oW
            j1 = ((ow+1)*W + oW - 1) // oW
            out[:,:,oh,ow] = mean(x[:,:, i0:i1, j0:j1], dim=(2,3))
    return out
```
实际核实现把 `start/end` 预计算成索引张量，再用 gather + 规约完成，避免 Python 双循环。时间复杂度 `O(N·C·H·W)`（每个输入元素恰好被一个窗口覆盖一次），空间 `O(N·C·oH·oW)`（新输出）。**注意**：自适应池化无 padding、无 dilation、无 count_include_pad、无 ceil_mode——参数极简，因为窗口边界完全由 `start/end` 公式决定，用户无需也无法干预。

**边界与陷阱**
- **窗口长度不均**：`in_size % out_size != 0` 时不同输出位置的窗口大小差 1（如 `in=7, out=2` ⇒ 窗口为 `[0,3)` 长度 3 和 `[3,7)` 长度 4）。这是"自适应"的精髓，但也意味着用户无法预测每个窗口多大——若下游逻辑依赖固定窗口大小会出错。
- **整除快路径**：`H%oH==0 && W%oW==0` 时等价于普通 avg_pool2d，PyTorch 直接调它（更快）。非整除时走自适应慢路径，两者数值一致但实现不同。
- **GAP 特例**：`output_size=(1,1)` 时整个 H×W 求平均，等价于 `x.mean(dim=(2,3))`。`output_size=(1,1)` 是最常见用法。
- **输入必须非零**：输入 H/W 不能为 0（实现显式检查），否则报错。
- **无反向 indices**：同 avg_pool2d，反向 `_adaptive_avg_pool2d_backward` 均分梯度，无需前向额外信息。
- **3D 变体**：`_adaptive_avg_pool3d` 公式完全同构，只是多一维。

**Inductor 视角**　**template / fallback**——自适应窗口边界使它无法套用通用 reduction 模板（窗口逐位置变化），走专用 template 或 `aten` native。整除快路径会复用 avg_pool2d 的核。GAP 场景在 inductor 里有时可被识别为沿 H/W 的 reduction 并融合，但通用自适应路径仍是 template/fallback。

## 本章小结

池化的核心是"**滑动局部窗口 + 固定规则聚合**"：max pooling 取最大、avg pooling 取平均、adaptive avg pooling 按比例自适应切分窗口取平均。三者共享卷积的滑窗几何（输出尺寸公式、ceil_mode、padding），但聚合规则固定（无权重），因此实现更简单。两条贯穿全章的要点：① **max pooling 前向必须返回 `indices`**，因为反向要把梯度精确路由回取 max 的位置（avg 系列均分梯度，不需要）；② **avg pooling 的分母由 `count_include_pad`/`divisor_override` 控制**，padding 补 0 是否计入分母直接影响边缘数值——这是 avg 与 max 在数值上的根本差异。自适应池化的精髓是"**用户给输出尺寸、内部反推窗口**"，当输入不能被输出整除时窗口长度逐位置变化，整除时退化为普通 avg pooling。

下一章第 15 章进入归一化（`native_layer_norm` / `native_group_norm` / `batch_norm`），它们同样涉及"沿某维聚合 + 减均值除标准差"的数值计算，重点将是**数值稳定性（layer norm 的方差累加与 epsilon）**与**规约轴几何**，与本章的滑窗几何形成对照。

---

[上一章 第 13 章 卷积](13-convolution.md) 　|　 [下一章 第 15 章 归一化](15-normalization.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
