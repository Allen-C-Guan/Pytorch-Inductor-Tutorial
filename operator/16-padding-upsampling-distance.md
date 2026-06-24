# 第 16 章 填充 / 上采样 / 采样

填充、上采样、采样这三类算子共同回答同一个问题：**"当我需要的输出位置，在输入张量上没有对应的格子时，怎么取值？"** 它们都是**位置变化**算子——`out[i,j]` 不再简单对应 `in[i,j]`，而是要算"输出坐标在输入上的源坐标"，再从源坐标取数。区别只在"源坐标"怎么定义：

- **填充 (padding)**：源坐标 = 输出坐标减去 pad 偏移；落在输入外的部分按三种**边界扩展规则**（常数 / 反射 / 复制）补值。这是卷积（第 13 章）必备的前置步骤——没有 `pad`，stride>1 或 kernel>1 的卷积会丢边界。
- **上采样 (upsample)**：给定输出尺寸，把每个输出像素按比例映射回输入坐标，再按**插值规则**（最近邻 / 双线性）取值。是分割/U-Net、超分、GAN 上采样的主力。
- **采样 (grid_sampler_2d)**：源坐标**不是按比例算的**，而是另一个张量 `grid` 直接给出的（归一化坐标 -1~1）。这让"去输入哪里取"本身**可学习**——是空间变换网络 (STN)、可变形卷积、部分 attention 位置编码 (positional embedding 的 RoIAlign/DeformAttn 变体) 的底层原语。

从数学视角，填充是边界条件的特例，上采样是插值，grid_sampler 是"坐标驱动的可微查找表"。三者都比第 14 章池化更"轻"（没有可学习参数、没有最大值挑选），但比逐元素算子复杂——核心难点全在**坐标映射公式**与**边界/对齐约定**。本章是 Part III 的收尾章，重点放在数学公式与数值/精度。

> 公共概念：**广播 (broadcast)** 和 **类型提升 (type promotion)** 详见第 0 章 §0.4 / §0.5；core ATen 的 **SSA / 函数式契约**（这些算子全部产出新张量、不改输入）详见第 0 章 §0.7。

---

## 本章速查（Tier C）

填充族打包讲解（常数 / 反射 / 复制三种边界扩展）。同类合并，逐个签名见附录 A。

| 算子 | 功能一句话 | 形状 / 风险 | inductor 归类 |
|---|---|---|---|
| `constant_pad_nd(input, pad, value=0)` | 在任意维两端填**常量** `value` | `pad` 是从最后一维起、`(左,右)` 成对的反序列；偶数长度，负值=裁剪 | fallback（无统一模板） |
| `reflection_pad2d(input, pad)` | 在 H/W 两维填**镜像反射**（不含边界样本） | `pad=[l,r,t,b]`；输入对应维长度需 ≥2，否则报错 | fallback |
| `replication_pad2d(input, pad)` | 在 H/W 两维填**边缘复制**（重复边界样本） | `pad=[l,r,t,b]`；任何长度都合法 | fallback |

> 1D/3D 变体（`reflection_pad1d` / `reflection_pad3d` / `replication_pad3d`）与上述同构，附录 A 标 ○，本章不展开。`functional_pad`（`F.pad`）是复合算子（→ 分发到这三个 + circular），不是 core 基础算子。

---

## 填充族的统一语义（三种边界扩展对比）

三个 pad 算子的**前半段完全相同**：给定 `pad`（从最后一维起、按 `(左,右)` 成对给出），先求出输出形状——每一对 `(p_l, p_r)` 把该维长度从 `L` 变成 `L + p_l + p_r`（负 pad 即裁剪）。不同只在**落在输入范围之外的下标**取什么值。设输入某维下标域为 `[0, L)`：

| 边界规则 | 下标 `k` 落在 `[-p_l, L+p_r)` 外/内时 | 等价镜像函数 `mirror(k)` |
|---|---|---|
| **constant** | 落在 `[-p_l, L+p_r)` 之外 → 填 `value`；之内 → `in[k]` | `k ∈ [0,L) ? in[k] : value` |
| **reflection** | 把 `in` 内部当镜子：`-1→1, -2→2, ..., L→L-2`（**不重复边界样本**） | 反射周期 `2(L-1)`，`mirror(k) = L-1 - |L-1 - ((k) mod 2(L-1))|` |
| **replication** | 把**边界样本**当镜子：`-1→0, -2→0, ..., L→L-1`（**重复边界样本**） | `mirror(k) = clamp(k, 0, L-1)` |

```python
# 统一伪代码：input shape [N, C, H, W]，对 H 维 pad
def pad_dim(input, p_l, p_r, rule, value=0):
    L = input.shape[H]
    H_out = L + p_l + p_r
    out = empty(shape[:-2] + (H_out,) + shape[-1:])   # 必分配新张量
    for k in range(H_out):                              # 输出下标
        src = k - p_l                                   # 源下标（去偏移）
        if 0 <= src < L:
            out[k] = input[src]
        elif rule == 'constant':
            out[k] = value
        elif rule == 'replication':
            out[k] = input[clamp(src, 0, L-1)]
        elif rule == 'reflection':                      # 周期 2(L-1)，不碰边界
            m = src % (2*(L-1));  m = min(m, 2*(L-1)-m)
            out[k] = input[m]
    return out
```

- **时间** O(numel) 全量写一次；**空间** O(numel) 新分配（填充必拷贝，绝不零拷贝）。
- **`constant_pad_nd` 是唯一 N 维通用的**——`pad` 长度可以是 `2*ndim`（从最后一维起成对），覆盖任意维；反射/复制只有 1D/2D/3D 定版。
- **负 pad = 裁剪**：`constant_pad_nd(x, [0,0,1,-1])` 等价于先在 H 顶 pad 1、底裁 1（右负值即裁掉）。所有规则都支持，但 reflection/replication 在裁剪场景用得少。
- **反射 vs 复制的选择**：卷积里 `reflection` 视觉更自然（边界不出现重复条纹），但要求该维 `L ≥ 2`，否则 `2*(L-1)=0` 周期无定义 → 报错。`replication` 无此限制。
- **Inductor 视角（填充族）**：**fallback**。pad 的访存模式（输出按镜像下标跳读输入）既非逐元素规整，也无 matmul/卷积的稠密结构，inductor 不做模板，落到 eager/native 后端核。卷积图里若 pad 紧贴 conv，常被 conv 后端的内置 padding 吸收掉（见第 13 章），否则单独发射。

---

## 深入算子（Tier D）

### `aten.upsample_nearest2d(input, output_size, scales_h=None, scales_w=None)` — 最近邻上采样

**作用与语义**　把 `[N, C, H, W]` 的输入按 `output_size=[H', W']` 放大到 `[N, C, H', W']`。每个输出像素 `(i, j)` 映射回输入坐标，取**最近的那一个**输入像素的值，不做任何加权：

$$
\text{src}_h = \text{scale}_h \cdot (i + 0.5) - 0.5,\qquad \text{scale}_h = H / H'
$$

$$
\text{out}[n,c,i,j] = \text{in}[n,c,\;\lfloor \text{src}_h \rceil,\;\lfloor \text{src}_w \rceil]
$$

其中 `⌊·⌉` 是"就近取整"（PyTorch 用 `floor(src+0.5)` 实现对称舍入）。`(i+0.5)` 的偏移是**像素中心约定**——把像素看成单位面积的中心而非左上角，保证缩放对称。输入→输出：通道不变，空间维按 `output_size` 给定。返回新张量，无别名。

**示例**　2×2 放大到 4×4：每个输入像素被复制成一个 2×2 块（最近邻特有的"块状锯齿"）：

```python
>>> x = torch.tensor([[[[1.,2.],[3.,4.]]]])                              # (1,1,2,2)
>>> torch.nn.functional.interpolate(x, size=(4,4), mode="nearest")       # 落 aten.upsample_nearest2d
tensor([[[[1., 1., 2., 2.],
          [1., 1., 2., 2.],
          [3., 3., 4., 4.],
          [3., 3., 4., 4.]]]])
```

**为什么需要这个算子 / 数值与精度**　上采样是"低成本放大"。最近邻无计算开销（只查表），但输出有明显**块状锯齿 (blockiness)**——多个相邻输出像素映射到同一输入像素，值完全相同。适合需要硬边缘的场景（分割掩码、最近邻探查），或作为 GAN/diffusion 里"先放大再卷积平滑"的廉价第一步。数值上无精度陷阱（纯索引、不混合），但**反向不可微的隐含约定**：梯度走"最近邻索引"传回（即 `grad` 累加到被选中的源像素），导数处处为 0 或未定义——PyTorch 用 STE 式累加处理。

**实现逻辑与复杂度**

```python
def upsample_nearest2d(input, output_size, scales_h, scales_w):
    N, C, H, W = input.shape
    H2, W2 = output_size
    sh, sw = (H / H2, W / W2) if scales is None else (scales_h, scales_w)
    out = empty(N, C, H2, W2)
    for i in range(H2):
        src_h = int(floor(sh * (i + 0.5) - 0.5 + 1e-9))   # 防浮点抖动
        src_h = clip(src_h, 0, H-1)
        for j in range(W2):
            src_w = int(floor(sw * (j + 0.5) - 0.5 + 1e-9))
            out[:, :, i, j] = input[:, :, src_h, clip(src_w, 0, W-1)]
    return out
```

- **时间** O(N·C·H'·W')，每像素 O(1) 索引；**空间** O(输出 numel) 必分配。
- `scales_*` 为 `None` 时由 `output_size` 反推（`H/H'`），显式给 `scales` 用于反向图保持比例一致。
- `1e-9` 抖动补偿：浮点 `sh` 可能使本应取整到 `k` 的坐标落成 `k-ε`，加偏移避免。

**边界与陷阱**
- **`scales` 与 `output_size` 冲突**：两者都给且比例不一致时，行为以 `output_size` 为准（scales 仅影响"理论比例"，不改变 `H'`），易困惑。
- **反向梯度**：因最近邻是"选一个"，反向把输出梯度**累加**到对应源像素，多个输出像素映射到同一源时会累加多次。
- **非连续输入**：核按 stride 寻址，支持任意 stride；但连续输入缓存更友好。
- **dtype**：索引计算用浮点 `sh`，但取值是直接拷贝输入 dtype，不提升。

**Inductor 视角**　**template / fallback**。upsample 是固定的"按比例跳读"模板，inductor 对 nearest 有专门的 template 调度（仿射索引 + gather），不进通用 pointwise；NPU 后端若无对应模板则 fallback 到 native 核。

---

### `aten.upsample_bilinear2d(input, output_size, align_corners, scales_h=None, scales_w=None)` — 双线性插值上采样

**作用与语义**　同样把 `[N, C, H, W] → [N, C, H', W']`，但每个输出像素的源坐标落在输入的**非整数位置**，由周围 4 个输入像素**按距离加权**混合：

$$
\text{src}_h = \begin{cases}
\text{scale}_h \cdot (i + 0.5) - 0.5 & \text{if } \text{align\_corners}=\text{False}\\
i \cdot \dfrac{H-1}{H'-1} & \text{if } \text{align\_corners}=\text{True}
\end{cases}
$$

$$
\text{out}[n,c,i,j] = \sum_{a\in\{0,1\}}\sum_{b\in\{0,1\}} w_{ab}\cdot \text{in}[n,c,\,h_0{+}a,\,w_0{+}b]
$$

其中 `h0 = floor(src_h)`，权重 `w_h = src_h - h0`（一维线性权重），二维权重 `w_{ab} = w_h^a \cdot w_w^b`（可分离：先 H 再 W）。**`align_corners` 改变坐标映射**：`True` 时两端角点严格对齐（`src(0)=0, src(H'-1)=H-1`），`False` 时按像素中心比例缩放（两端各留半个像素的"外推"空间）。

**示例**　2×2 放大到 3×3（`align_corners=True`，角点严格对齐）：中心像素 = 四角平均 `(1+2+3+4)/4 = 2.5`：

```python
>>> x = torch.tensor([[[[1.,2.],[3.,4.]]]])                  # (1,1,2,2)
>>> torch.nn.functional.interpolate(x, size=(3,3), mode="bilinear", align_corners=True)
tensor([[[[1.0000, 1.5000, 2.0000],
          [2.0000, 2.5000, 3.0000],
          [3.0000, 3.5000, 4.0000]]]])
```

**为什么需要这个算子 / 数值与精度**　双线性产出的图像**平滑无锯齿**，是分割/超分/GAN/diffusion 的默认上采样。代价是每输出像素 4 次取值 + 3 次乘加。数值上有两个真实陷阱：

1. **`align_corners` 的语义差异**：`True` 时 `H'=1` 会除以 0（`H'-1=0`），需特判（PyTorch 退化成单点采样）。同一模型混用两种约定会导致**像素错位 0.5**——预训练权重迁移时是高频 bug。
2. **fp16/bf16 累加精度**：4 项加权和中权重是 `[0,1]` 内小浮点，bf16 下累加误差可见，建议在 fp32 下算权重再转回。inductor 在 NPU 上常自动插入精度提升。

**实现逻辑与复杂度**

```python
def upsample_bilinear2d(input, output_size, align_corners, scales_h, scales_w):
    N, C, H, W = input.shape
    H2, W2 = output_size
    sh = (H-1)/(H2-1) if align_corners else H/H2
    out = empty(N, C, H2, W2)
    for i in range(H2):
        if align_corners:
            src_h = i * sh
        else:
            src_h = sh * (i + 0.5) - 0.5
        h0 = int(floor(src_h)); wh = src_h - h0
        h0 = clip(h0, 0, H-1); h1 = clip(h0+1, 0, H-1)   # 越界钳到边界
        for j in range(W2):
            # 同理算 src_w, w0, ww
            for a in (0,1):
                for b in (0,1):
                    out[:, :, i, j] += (wh if a else 1-wh) * (ww if b else 1-ww) \
                                       * input[:, :, h0+a, w0+b]
    return out
```

- **时间** O(N·C·H'·W')，每像素 4 次取值 + 3 MAD，比最近邻重约 4-8 倍；**空间** O(输出 numel) 必分配。
- **可分离性**：二维双线性 = 先对 H 维做一维线性、再对 W 维做一维线性，两个 pass 可合并成一个，但权重可预计算。
- **边界钳制 (clamp)**：`src` 可能略小于 0 或大于 `H-1`（`align_corners=False` 时首尾像素的外推），统一钳到 `[0, H-1]`——等价于**边缘复制外推**。

**边界与陷阱**
- **`align_corners=True` 且 `output_size` 含 1**：除零，需特判为取角点。
- **`align_corners` 不可由 scales 推断**：必须显式传，且**反向要传同一个值**，否则前反向坐标不一致 → 数值错。
- **负 `scales`**：非法，但某些路径不报错而产出 NaN，建议上层校验。
- **`scales` vs `output_size`**：同 nearest，两者都给时以 `output_size` 为准。
- **梯度**：双线性处处可微，反向梯度按权重 `w_{ab}` 分发回 4 个源像素，数值稳定。

**Inductor 视角**　**template / fallback**。双线性是固定的 4-tap 仿射采样模板，inductor 有专门 template；bf16 路径会插入 fp32 权重计算。NPU 后端若无模板则 fallback。

---

### `aten.grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners)` — 按坐标网格采样

**作用与语义**　这是三类里**最灵活、也最"反直觉"的一个**。`input` 形状 `[N, C, H, W]`，`grid` 形状 `[N, H_out, W_out, 2]`——`grid[n, i, j]` 直接给出输出像素 `(n, i, j)` 应该去 `input` 的**哪个归一化坐标**取值，坐标在 `[-1, 1]` 区间（`-1`=首像素、`+1`=末像素，`align_corners=True` 时精确对齐角点，`False` 时对应像素中心边界）。输出 `[N, C, H_out, W_out]`：

$$
(x, y) = \text{grid}[n, i, j] \quad\text{（x=列方向、y=行方向）}
$$

$$
x_{\text{pix}} = \begin{cases} \dfrac{(x+1)(W-1)}{2} & \text{align\_corners=True}\\[4pt] \dfrac{x \cdot W}{2} - 0.5 & \text{align\_corners=False}\end{cases}
$$

然后按 `interpolation_mode` 取值：`0`=bilinear（4 邻域加权，同上采样双线性）、`1`=nearest、`2`=bicubic。落在 `[0, W)×[0, H)` 之外的源坐标，按 `padding_mode` 处理：`0`=zeros（填 0）、`1`=border（钳到边界，等价 replication）、`2`=reflection（镜像反射，等价 reflection pad）。返回新张量，无别名。

**示例**　`grid=[[-1,-1],[1,1]]`（`align_corners=True`、bilinear）分别去输入的**左上角 `(0,0)`** 与**右下角 `(2,2)`** 取值——注意 `grid` 末维是 `(x=列, y=行)`，与 `(行,列)` 直觉相反：

```python
>>> img  = torch.tensor([[[[0.,1.,2.],[3.,4.,5.],[6.,7.,8.]]]])   # (1,1,3,3)
>>> grid = torch.tensor([[ [[-1.,-1.],[1.,1.]] ]])                 # (1,1,2,2): 末维 (x=列,y=行)
>>> torch.nn.functional.grid_sample(img, grid, mode="bilinear",
...                                  padding_mode="zeros", align_corners=True)
tensor([[[[0., 8.]]]])     # (1,1,1,2): [-1,-1]->img[0,0]=0 ; [1,1]->img[2,2]=8
```

```
img (1,1,3,3)        grid (1,1,2,2), 末维=(x列,y行), 归一化[-1,1]
[[0 1 2]              [[-1,-1]    -> 像素(0,0) -> img[0,0]=0
 [3 4 5]   align=T     [ 1, 1]]   -> 像素(2,2) -> img[2,2]=8     out(1,1,1,2)=[[0, 8]]
 [6 7 8]]             bilinear
```

**为什么需要这个算子 / 数值与精度**　grid_sampler 的革命性在于：**采样位置本身是可学习张量**。这把"固定的坐标映射"（上采样、仿射变换）泛化成"任意的、数据相关的空间重映射"。核心应用：

- **空间变换网络 (STN)**：`grid` 由一个本地化网络预测，实现可学习的仿射/透视变换。
- **可变形卷积 (Deformable Conv)**：`grid` = 标准卷积网格 + 学习到的偏移，采样后再加权求和。
- **RoIAlign / 注意力位置编码**：在检测、ViT 变种里按连续坐标取特征，避免两次量化误差。
- **光流 / 视差 warp**：`grid` = 基准网格 + 光流场，做帧间对齐。

它**对 `grid` 可微**（梯度通过坐标传回，即双线性权重的偏导），是端到端训练空间变换的关键。数值陷阱集中在：

1. **`align_corners` 与归一化坐标的耦合**：`align_corners=True` 时 `-1` 精确指第 0 个像素、`+1` 指第 `W-1` 个；`False` 时 `-1` 指第 0 个像素**左边界外**（即像素中心外推半格）。`grid` 若来自不同约定的预训练模型，会整体错位。
2. **bicubic 模式的振铃**：bicubic 用 4×4=16 邻域 + 三次样条权重，在锐边处有**过冲/欠冲 (ringing)**，可能超出输入值域，bf16 下更明显；安全起见生产多用 bilinear。
3. **`padding_mode=reflection` 的周期**：与 reflection_pad 同公式，要求坐标偏离不超出 `[0, L)` 的反射范围，超出会反复反射（设计如此，但易被误用）。
4. **NaN 传播**：`grid` 含 NaN/Inf 会使所有模式产出 NaN，且梯度也是 NaN——上层常需 `where(isfinite(grid), grid, 0)` 兜底。

**实现逻辑与复杂度**

```python
def grid_sampler_2d(input, grid, interp, pad, align_corners):
    N, C, H, W = input.shape
    _, H2, W2, _ = grid.shape
    out = empty(N, C, H2, W2)
    for n in range(N):
        for i in range(H2):
            for j in range(W2):
                gx, gy = grid[n, i, j]                      # 归一化坐标
                px = denorm(gx, W, align_corners)           # → 像素坐标
                py = denorm(gy, H, align_corners)
                if interp == 0:  # bilinear
                    x0, y0 = floor(px), floor(py); wx, wy = px-x0, py-y0
                    out[n,:,i,j] = sum( w_ab * input_padded(n, x0+a, y0+b) )  # a,b∈{0,1}
                elif interp == 1:  # nearest
                    out[n,:,i,j] = input_padded(n, round(px), round(py))
                # bicubic: 4×4 邻域同理
    return out

def input_padded(n, x, y, pad):                # padding_mode 处理越界
    if pad == 0:   return input[n,:,y,x] if in_range else 0          # zeros
    if pad == 1:   return input[n,:, clip(y,0,H-1), clip(x,0,W-1)]  # border
    if pad == 2:   return input[n,:, reflect(y,H), reflect(x,W)]    # reflection
```

- **时间**：nearest O(N·C·H'·W')；bilinear 同阶但每像素 4 取值；bicubic 16 取值，约 4 倍于 bilinear。**空间** O(输出 numel) 必分配，无别名。
- **梯度**：对 `input` 的梯度按插值权重分发；对 `grid` 的梯度是**坐标处的双线性权重的偏导**（`d(out)/d(px) = input 差分 × 权重`），这是可微采样的数学核心。
- **零拷贝**：无，必分配。

**边界与陷阱**
- **`grid` 的最后一维顺序是 `(x, y)` 即 `(W方向, H方向)`**，与直觉的 `(row, col)` 相反，是高频 bug。
- **`grid` 值域**：归一化坐标理论 `[-1,1]`，但超出也合法（按 padding_mode 处理），不报错。
- **`interpolation_mode`/`padding_mode` 是 int 枚举**（0/1/2），不是字符串——从 `F.grid_sample` 进来时会被转成枚举，但在 ATen 图里直接是整数，调试时易看错。
- **5D 变体 `grid_sampler_3d`** 不在 core 161（附录 A 未列），3D 采样走单独路径。
- **反向**：`grid_sampler_2d_backward` 是 core 算子但本章标 ○，其数值与 forward 的 `align_corners` 必须严格一致，否则梯度坐标错位。

**Inductor 视角**　**fallback（template 边缘）**。grid_sampler 的访存完全由 `grid` 数据驱动（非仿射、无法静态求地址），inductor 无通用模板，落到 native 后端核（GPU 上是专门的 texture-based 核，NPU 上是 CANN 的 grid_sample 算子）。在可变形卷积/STN 计算图里，它通常**不被融合**，作为独立节点发射。

---

## 本章小结

填充、上采样、grid_sampler 是"输出坐标映射回输入取值"的三个层次：**填充**是规则的边界扩展（常数/反射/复制三种镜像函数），**上采样**是按固定比例的插值（最近邻/双线性），**grid_sampler** 是按可学习坐标的任意采样——灵活度递增，可微性与表达能力也递增。数值陷阱的核心始终是两件事：**像素中心约定**（`align_corners` 与 `(i+0.5)` 偏移）和**边界/越界处理**（三种 padding_mode）。理解了坐标映射公式，这三类算子的全部行为都可推导。inductor 对它们多以 fallback / 专门 template 处理，融合机会有限。

> 附录 A 标 ○ 的冷门原语：`_cdist_forward` / `_pdist_forward`（成对欧氏距离，全矩阵 / 下三角）、`_fft_r2c`（实→复 FFT，core 里唯一的谱算子）——它们是特定领域（度量学习、频域卷积）的底层原语，公式直接、用面窄，本书不展开。

至此 Part III（模型层计算原语）完结。全书正文（第 0–16 章）到此结束，附录 B（复合→基础分解 cookbook）与附录 C（inductor 降级速查）供查阅。

---

[上一章 第 15 章 归一化](15-normalization.md) 　|　 [附录 B 复合→基础分解](appendix-b-decompositions.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
