# 第 15 章 归一化（Normalization）

归一化（normalization）是深度网络里与卷积、池化并列的第三大类"层计算原语"。它的共同骨架只有一句话：**沿某几个维对激活值减均值、除标准差、再仿射变换**——即把每个"归一化单元"的输出分布强行拉到接近零均值、单位方差，然后用可学的 `gamma`（缩放）和 `beta`（平移）恢复表达能力。三者的差异**只在"沿哪些维聚合统计量"**：LayerNorm 沿特征维（最后一维）、GroupNorm 把通道分组后沿组内通道+空间维、BatchNorm 沿批+空间维（保留通道为统计轴）。这一条维选择决定了统计量的粒度，也决定了算子在 batch 大小、模型领域（NLP vs CV）和后端实现上的根本差异。

在模型里归一化承担三个角色：①**稳定训练**——抑制内部协变量偏移（internal covariate shift），让各层输入分布不剧烈漂移，从而可用更大学习率、收敛更快；②**分布对齐**——把激活值压到与后续权重/激活函数匹配的数值范围（避免饱和、避免梯度爆炸/消失）；③**尺度无关性**——使模型对输入尺度变化不敏感。Transformer（ViT/GPT/BERT）几乎全部用 LayerNorm，CNN（ResNet/EfficientNet）几乎全部用 BatchNorm，GroupNorm 是 BatchNorm 在小 batch（检测/分割/GAN）场景的替代。一句话：**没有归一化，现代深层网络几乎无法训练**。

本章三个算子都是 Part III 的**模型层计算原语**：数学上它们都是"减均值除标准差"的 trivial 变体，但**数值稳定性要求极高**（方差累加的精度、epsilon 的位置、bf16 下的累加），inductor 因此**不**把它们分解成 pointwise + reduction 后再融合，而是走专门的 **template**（如 `mkldnn`/cuDNN 的 fused LN/GN/BN 核）或降级到 `aten` native **fallback**——因为融合核能在一个 pass 里用高精度累加器算完 mean+var+rstd+affine，避免中间结果落盘。本章重点因此放在**数学公式与数值/精度**，"为什么需要"从简（上面已交代）。反向变体（`native_layer_norm_backward` / `native_group_norm_backward`）也是 core 基础算子——归一化的反向非平凡（需要对 mean/rstd 求导），属基础原语，本章不展开，附录 A 标注。

> 本章属 Part III。广播与类型提升详见第 0 章 §0.4 / §0.5；SSA 契约与张量基底详见 §0.7 与第 0 章全章。

## 本章速查（Tier C）

本章无 Tier C 算子——三个归一化算子全部 Tier D 深入讲解（见下）。它们在数值上非平凡（方差累加、epsilon 位置、bf16 精度），且是 Part III 的旗舰算子，必须全展开。同族的 `instance_norm`（InstanceNorm，即逐样本的 GroupNorm with `group=C`）、`batch_norm`（封装算子，按 `training` flag 分发到 `_native_batch_norm_legit` / `_native_batch_norm_legit_no_training`）属复合/封装变体，附录 A 标注，本章正文不展开。

## 深入算子（Tier D）

### `aten.native_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5) -> (Tensor, Tensor, Tensor)` — 沿最后一维（特征维）做层归一化，返回 (output, mean, rstd)

**作用与语义**　对输入张量的**最后一维**（若干维，由 `normalized_shape` 指定）做归一化。设 `normalized_shape = [D]`（最常见，即最后一维 D），输入 `input` 形状 `(..., D)`，则对每个"行向量"（最后一维的 D 个元素）独立计算均值 `mean` 和方差 `var`，归一化后做仿射。数学定义（对每个归一化单元，下标 `i` 遍历 `normalized_shape` 内的所有元素）：

```
mean = (1/N) * Σ x_i                                  # N = prod(normalized_shape)，即归一化单元内元素数
var  = (1/N) * Σ (x_i - mean)^2                       # 注意：有偏方差（除以 N，不是 N-1）
rstd = 1 / sqrt(var + eps)                            # eps 加在方差根号**里**（见"数值与精度"）
y_i  = (x_i - mean) * rstd * gamma_i + beta_i         # gamma/beta 形状 = normalized_shape，沿归一化维广播
```

输出形状与 `input` 完全相同（同 dtype）。**返回三元组** `(output, mean, rstd)`：`mean`/`rstd` 形状为 `input.shape[:-len(normalized_shape)]`（即"归一化单元的批次维"），它们是反向所需的前向统计量，必须随图保存。当 `normalized_shape` 跨多维（如 `[H, W]` 用于图像）时，归一化单元是这若干维的笛卡尔积。

**示例**　每行被独立归一化成均值 0、方差 1（这里无 `weight`/`bias` 即无仿射）。注意 `F.layer_norm` 只返回 `output`，而裸 `aten.native_layer_norm` 返回 `(output, mean, rstd)` 三元组：

```python
>>> import torch.nn.functional as F
>>> x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
>>> F.layer_norm(x, normalized_shape=[3])   # 等价 aten.native_layer_norm 的 output
tensor([[-1.2247,  0.0000,  1.2247],
        [-1.2247,  0.0000,  1.2247]])
>>> out, mean, rstd = torch.ops.aten.native_layer_norm(x, [3], None, None, 1e-5)
>>> mean, rstd                              # 每行一组统计量，供反向用
tensor([[2.],
        [5.]]), tensor([[1.2247],
        [1.2247]])
```

**为什么需要这个算子 / 数值与精度**　**为什么需要**：见章首导言——稳定训练、抑制内部协变量偏移、把每个 token/样本的特征向量分布对齐到零均值单位方差。Transformer 里每个 token 的 hidden state 经 LN 后分布稳定，注意力/FFN 的输入尺度可预测。**数值与精度（本章重点）**：

- **eps 的位置**：标准定义是 `rstd = 1 / sqrt(var + eps)`，即 **eps 加在方差根号里**（PyTorch 采用此式）。另一种写法是 `y = (x - mean) / (sqrt(var) + eps)`——eps 加在标准差外面。两者数值不等价：根号内的 eps 在 `var` 很小时（如 `var=1e-6`）主导，效果是"把 rstd 截断到 `1/sqrt(eps)`"，避免除零；根号外的 eps 在 `var` 极小时会让分母 ≈ `eps`（可能很大或很小）。**移植到自定义后端时必须确认 eps 位置一致**，否则与参考实现对不上。PyTorch/CUDA/mkldnn 一律是**根号内**。
- **bf16 下的方差累加精度**：方差 = `Σ x² − (Σx)²/N`（两遍式）或 `Σ(x−mean)²`（一遍式）。bf16 下直接累加 `Σ x²` 会**严重丢精度**（bf16 仅 7 位尾数，大值平方后小数部分全丢）。PyTorch 的 LN 核**内部用 float32 累加器**：先把 bf16 输入转成 float32 算 mean/var/rstd，最后再转回 bf16 输出。inductor 的 fused LN template 必须照此实现，否则 bf16 训练 loss 不收敛或数值发散。**这是 LN 在 NPU 后端移植的头号坑**。
- **有偏 vs 无偏方差**：归一化用的是**有偏方差**（除以 `N`，而非 `N-1`）。这是约定（与 `var(unbiased=False)` 一致），不是 bug。
- **NaN 传播**：若归一化单元内**所有元素相同**（如全 0 或全相等），`var = 0`，`rstd = 1/sqrt(eps)` 是有限值，不会 NaN；但若单元内含 `±inf` 或 `NaN`，mean/var 会变 NaN 并扩散到整个单元输出。bf16 下 `inf*0`（来自 `gamma=0` 且 `rstd` 因 `var` 溢出为 inf）会产生 NaN。

**实现逻辑与复杂度**　NumPy 风格伪代码（前向，单 pass，float32 累加）：

```python
def native_layer_norm(x, normalized_shape, gamma, beta, eps):
    D = prod(normalized_shape)               # 归一化单元内元素数 N
    batch_shape = x.shape[:-len(normalized_shape)]
    x2d = x.reshape(-1, D)                   # (B, D)，B = prod(batch_shape)
    # —— 高精度累加路径（核内用 float32，即使 x 是 bf16）——
    x2d_f32 = x2d.to(float32)
    mean = x2d_f32.mean(dim=1, keepdim=True)              # (B,1)
    var  = ((x2d_f32 - mean)**2).mean(dim=1, keepdim=True)  # (B,1)，有偏
    rstd = 1.0 / (var + eps).sqrt()                        # (B,1)
    y2d  = (x2d_f32 - mean) * rstd                         # (B,D)，先归一化
    if gamma is not None:
        y2d = y2d * gamma.to(float32)                      # 仿射缩放（gamma 沿 D 广播）
    if beta is not None:
        y2d = y2d + beta.to(float32)                       # 仿射平移
    y = y2d.to(x.dtype).reshape(x.shape)
    mean = mean.reshape(batch_shape)                       # 返回给反向
    rstd = rstd.reshape(batch_shape)
    return y, mean, rstd
```

时间复杂度 `O(numel(x))`（两遍输入：一遍算 mean/Σx²，一遍算 y；融合核可一遍算 mean+var 再一遍算 y），空间 `O(numel(x) + 2*prod(batch_shape))`（输出 + mean + rstd，**分配新张量**，非零拷贝）。`mean`/`rstd` 相对输入很小，开销可忽略。高效的融合核（如 `mkldnn`/cuDNN/Triton LN template）把 mean→var→rstd→affive 全部塞进一个 kernel，避免中间张量落盘——这正是 inductor 走 template 而非分解的根本原因。

**边界与陷阱**
- **normalized_shape 必须匹配输入尾部**：`input.shape[-len(normalized_shape):] == normalized_shape`，否则报错。`normalized_shape` 可以是 list/tuple/torch.Size，**不能是空**（至少一维）。
- **weight/bias 可选但形状要匹配**：若提供，形状必须等于 `normalized_shape`；`None` 表示不仿射（单位变换）。无 weight/bias 的 LN 用于某些归一化后接线性层的场景。
- **eps 的位置（移植坑）**：根号内 vs 根号外，详见"数值与精度"。NPU 自定义核移植时这是头号对齐项。
- **bf16/hf16 累加精度**：必须 float32 内部累加，否则大 D（如 4096/8192）下 mean/var 严重失真。详见"数值与精度"。
- **非连续输入**：算子接受任意 stride（沿最后一维归一，内部按 stride 寻址），但极度非连续可能触发隐式 contiguous。LN 的归一化维是最后一维（最内维），通常连续，性能最佳。
- **返回值顺序**：`(output, mean, rstd)`——**不是** `(output, mean, var)`。反向直接用 `rstd`（已含 eps），无需重新开方。误把 `rstd` 当 `var` 用会出错。
- **dtype 一致性**：`weight`/`bias` 的 dtype 应与 `input` 一致；不一致时行为依后端而定（部分核会做隐式提升）。
- **空归一化单元**：`D=0`（`normalized_shape` 乘积为 0）会触发除零，PyTorch 抛错。

**Inductor 视角**　**template / fallback**——LN 是数值敏感算子（需要 float32 累加 mean/var、正确的 eps 位置、affine 融合），inductor **不**把它分解成 `mean` + `var` + pointwise 后融合，而是走专用 fused LN **template**（`mkldnn`/cuDNN/Triton LN kernel）或降级到 `aten` native **fallback**。分解路径会丢失累加精度控制和融合机会，导致 bf16 训练数值发散——这是 LN 必须走 template 的根本原因。

---

### `aten.native_group_norm(input, weight=None, bias=None, N, C, HxW, group, eps=1e-5) -> (Tensor, Tensor, Tensor)` — 把通道分组，组内（通道+空间）做归一化

**作用与语义**　输入约定为 4D `(N, C, H, W)`（或对应 N-D），把 `C` 个通道**均匀分成 `group` 组**（要求 `C % group == 0`，每组 `Cg = C/group` 个通道），对每个样本 `n` 的每个组 `g`，在 **`(Cg, H, W)` 这 `Cg*H*W` 个元素**上计算 mean/var/rstd 并归一化，再做沿通道维 `C` 的仿射（`gamma`/`beta` 形状 `(C,)`，每通道一个）。注意 `N`/`C`/`HxW` 是**显式传入的标量**（`HxW = prod(空间维)`），不是从 input 推断——这是为了支持非 4D（如 5D 视频）的通用形状。数学定义（对样本 `n`、组 `g`，下标 `(c,h,w)` 遍历该组的 `Cg*H*W` 个元素）：

```
G = group; Cg = C / G
mean[n,g] = (1/(Cg*HxW)) * Σ_{c∈group g, h, w} x[n,c,h,w]
var[n,g]  = (1/(Cg*HxW)) * Σ (x[n,c,h,w] - mean[n,g])^2            # 有偏方差
rstd[n,g] = 1 / sqrt(var[n,g] + eps)                               # eps 加在方差根号**里**
y[n,c,h,w] = (x[n,c,h,w] - mean[n, c//Cg]) * rstd[n, c//Cg] * gamma[c] + beta[c]
```

输出形状与 `input` 相同 `(N, C, H, W)`，同 dtype。**返回三元组** `(output, mean, rstd)`：`mean`/`rstd` 形状 `(N, group)`——每个样本每组一个统计量（共 `N*group` 个标量），是反向所需。

**示例**　`(N=1, C=4, HxW=4)` 输入、`group=2`（每组 2 个通道），组内 `2*4=8` 个元素被归一化成均值 0、方差 1。`F.group_norm` 只返回 `output`，裸 `aten.native_group_norm` 返回 `(output, mean, rstd)`，其中 `mean`/`rstd` 形状为 `(N, group)=(1,2)`：

```python
>>> xg = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2)
>>> F.group_norm(xg, num_groups=2)          # 等价 aten.native_group_norm 的 output
tensor([[[[-1.5275, -1.0911],
          [-0.6547, -0.2182]],

         [[ 0.2182,  0.6547],
          [ 1.0911,  1.5275]],

         [[-1.5275, -1.0911],
          [-0.6547, -0.2182]],

         [[ 0.2182,  0.6547],
          [ 1.0911,  1.5275]]]])
>>> outg, meang, rstdg = torch.ops.aten.native_group_norm(xg, None, None, 1, 4, 4, 2, 1e-5)
>>> meang, rstdg                           # 形状 (1,2)，每组一对统计量
tensor([[ 4.5000, 12.5000]]), tensor([[0.4364, 0.4364]])
```

**为什么需要这个算子 / 数值与精度**　**为什么需要**：GroupNorm 是 BatchNorm 在**小 batch** 场景的替代。BatchNorm 的统计量沿 `(N, H, W)` 聚合（每个通道一组统计），当 batch 极小（如检测里 `N=1`/`N=2`）时统计量噪声极大、训练不稳。GroupNorm 把统计范围改为**样本内**（`Cg*H*W`，与 batch 无关），因此对小 batch/单样本推理鲁棒；同时分组保留了通道间的部分独立性（与 LayerNorm 把所有通道一起归一不同）。广泛用于检测（Detectron）、分割、GAN、扩散模型（Diffusion U-Net 用 GroupNorm）。**数值与精度**：

- **eps 位置与累加精度**：与 LayerNorm 完全相同——eps 加在方差根号**里**，bf16 下必须 float32 内部累加 mean/var。GroupNorm 的归一化单元 `Cg*H*W` 可能很大（如 `32*32*32 = 32768`），累加精度比 LN 更关键。
- **group 的选择**：`group=1` 等价于 **InstanceNorm 的反例**（实为"全通道一起归一"，接近 LayerNorm 对通道维的做法）；`group=C` 等价于 **InstanceNorm**（每通道独立归一）。典型 `group=32`。group 越大，统计单元越小、归一化越细；group 越小，统计越平滑。
- **`C % group != 0`**：直接报错（无法均匀分组）。
- **仿射沿通道维**：`gamma`/`beta` 形状 `(C,)`，与 BatchNorm 一致（每通道一个），**不是**每组一个。即同一组内的不同通道有不同的 affine 参数，但共享同一对 mean/rstd。

**实现逻辑与复杂度**　NumPy 风格伪代码（前向，float32 累加）：

```python
def native_group_norm(x, gamma, beta, N, C, HxW, group, eps):
    Cg = C // group
    x4d = x.reshape(N, group, Cg, HxW)               # 把 (C) 拆成 (group, Cg) 方便组内归约
    x4d_f32 = x4d.to(float32)
    mean = x4d_f32.mean(dim=(2,3))                   # (N, group, 1, 1) -> 沿 (Cg, HxW) 归约
    var  = ((x4d_f32 - mean)**2).mean(dim=(2,3))     # (N, group, 1, 1)
    rstd = 1.0 / (var + eps).sqrt()                  # (N, group, 1, 1)
    y4d  = (x4d_f32 - mean) * rstd                   # 归一化
    if gamma is not None:
        g = gamma.reshape(1, group, Cg, 1).to(float32)
        y4d = y4d * g                                # 沿通道仿射（gamma 需 reshape 对齐分组）
    if beta is not None:
        b = beta.reshape(1, group, Cg, 1).to(float32)
        y4d = y4d + b
    y = y4d.to(x.dtype).reshape(x.shape)
    mean = mean.reshape(N, group)                    # 返回给反向
    rstd = rstd.reshape(N, group)
    return y, mean, rstd
```

时间复杂度 `O(numel(x))`，空间 `O(numel(x) + 2*N*group)`（输出 + mean + rstd，**分配新张量**）。`mean`/`rstd` 共 `2*N*group` 个标量，相对输入很小。GroupNorm 的 reshape `(N,C,H,W)→(N,group,Cg,HxW)` 把不连续的通道（同一组内 `Cg` 个通道在内存里是连续的）规整化，使组内归约可以一次完成——这是高效核的关键。

**边界与陷阱**
- **`C % group != 0`**：报错（`C` 必须能被 `group` 整除）。
- **`group` 范围**：`1 <= group <= C`。`group=C` 退化为 InstanceNorm，`group=1` 全通道一起归一。
- **N/C/HxW 显式传入**：签名要求显式传 `N`/`C`/`HxW` 三个标量，而非从 input 推断。这是为了支持非 4D 输入（如 5D 视频 `(N,C,D,H,W)`，此时 `HxW = D*H*W`）。若传入值与 input 实际形状不符会得到错误结果（部分核不校验）。
- **eps 位置、bf16 累加精度**：同 LayerNorm——根号内 eps、float32 内部累加。
- **仿射维度**：`weight`/`bias` 沿**通道维** `C`（不是组），形状必须 `(C,)`。误传 `(group,)` 会报错。
- **非连续输入**：reshape `(N,C,H,W)→(N,group,Cg,HxW)` 在 C 是连续维时是零拷贝 view；若输入极度非连续（如 transpose 后）可能触发拷贝。
- **返回值 `(output, mean, rstd)`**：`mean`/`rstd` 形状 `(N, group)`，不是 `(N, C)`（与 BatchNorm 的 `(C,)` 不同）。

**Inductor 视角**　**template / fallback**——同 LayerNorm，GroupNorm 数值敏感（float32 累加、eps 位置、组内规约 + 通道仿射的 reshape 几何），走专用 fused GN **template** 或 `aten` native **fallback**。分组 reshape `(N,C,H,W)→(N,group,Cg,HxW)` 使它无法被当作沿单一轴的 reduction 融合，必须走专门核。

---

### `aten._native_batch_norm_legit(input, weight=None, bias=None, running_mean, running_var, training, momentum, eps) -> (Tensor, Tensor, Tensor)` — 批归一化（沿 N,H,W 归一、保留通道为统计轴），含 running 统计的正式路径

**作用与语义**　输入约定为 4D `(N, C, H, W)`（或 2D `(N, C)`）。**沿 `(N, H, W)` 维对每个通道 `c` 独立计算 mean/var**（即每个通道一组统计量，共 `C` 组），归一化后做沿通道 `C` 的仿射。与 LayerNorm/GroupNorm 的根本差异：**统计轴是 batch + 空间维，通道是被保留的"特征轴"**——这意味着每个通道有自己的统计量。数学定义（对通道 `c`，下标 `(n,h,w)` 遍历 `N*H*W` 个元素）：

```
# —— 训练（training=True）：从当前 batch 算统计 ——
mean[c] = (1/(N*H*W)) * Σ_{n,h,w} x[n,c,h,w]
var[c]  = (1/(N*H*W)) * Σ (x[n,c,h,w] - mean[c])^2            # 有偏方差
# —— 推理（training=False）：直接用 running 统计 ——
mean[c] = running_mean[c];  var[c] = running_var[c]           # 不重新计算
# —— 归一化（两种模式共用）——
rstd[c] = 1 / sqrt(var[c] + eps)                              # eps 加在方差根号**里**
y[n,c,h,w] = (x[n,c,h,w] - mean[c]) * rstd[c] * gamma[c] + beta[c]
# —— 训练时额外更新 running 统计（EMA）——
running_mean = (1 - momentum) * running_mean + momentum * mean   # momentum 是新值的权重
running_var  = (1 - momentum) * running_var  + momentum * unbiased_var   # 注意：running_var 用无偏方差
```

输出形状与 `input` 相同。**返回三元组** `(output, save_mean, save_invstd)`：`save_mean`/`save_invstd` 形状 `(C,)`，是反向所需的前向统计量（`save_invstd` 即 `rstd`，已含 eps）。**注意 running 统计的更新是 in-place**（`running_mean`/`running_var` 带 `a!`/`b!` 别名标注，被原地修改）。

**示例**　`(N=1, C=2, H=2, W=1)` 输入、`training=True`，按当前 batch 沿 `(N,H,W)` 统计并把每通道归一化。注意 `running_var` 用**无偏方差**做 EMA（`0.9*1+0.1*0.5=0.95`），而归一化本身用**有偏方差**——印证"数值与精度"里的混用陷阱。`F.batch_norm` 只返回 `output`，裸 `aten._native_batch_norm_legit` 返回 `(output, save_mean, save_invstd)` 供反向用：

```python
>>> xb = torch.arange(4, dtype=torch.float32).reshape(1, 2, 2, 1)
>>> rm, rv = torch.zeros(2), torch.ones(2)
>>> F.batch_norm(xb, rm, rv, training=True, momentum=0.1, eps=1e-5)   # 等价 output
tensor([[-1.0000,  1.0000],
        [-1.0000,  1.0000]])
>>> rm, rv                                  # in-place EMA 更新后
(tensor([0.0500, 0.2500]), tensor([0.9500, 0.9500]))
>>> outb, save_mean, save_invstd = torch.ops.aten._native_batch_norm_legit(
...     xb, None, None, torch.zeros(2), torch.ones(2), True, 0.1, 1e-5)
>>> save_mean, save_invstd                  # 逐通道 batch 统计，供反向
(tensor([0.5000, 2.5000]), tensor([2.0000, 2.0000]))
```

**`_legit` 后缀的含义**：PyTorch 的 batch_norm 有一族变体，后缀区分"统计路径"：
- **`_native_batch_norm_legit`**——**正式训练路径**：含 `running_mean`/`running_var` 输入（带 `a!`/`b!` in-place 别名），训练时从 batch 算统计并 EMA 更新 running；推理时用 running。这是"legit（合法/正式）"的命名由来——它拥有完整的、带 running buffer 的 BN 语义。
- **`_native_batch_norm_legit_no_training`**——**推理变体**：`training=False`，直接用 running 统计，不更新、不重新计算。用于推理/消融（ablation）。
- **`_native_batch_norm_legit_functional`**——functional 变体，running 更新以**返回值**形式给出（非 in-place）。
- **`_native_batch_norm_legit.no_stats`**——无 running buffer 的变体（无统计输入，纯从 batch 算）。

上层封装算子 `aten.batch_norm` 根据 `training` flag 把调用分发到 `_legit`（训练）或 `_legit_no_training`（推理）；`_native_batch_norm_legit` 是其中的"正式训练"基算子。core 基础算子清单收录的是 `_native_batch_norm_legit`（训练路径），其余变体附录 A 标注。

**为什么需要这个算子 / 数值与精度**　**为什么需要**：BatchNorm 是 CNN 训练的基石（ResNet 及之后几乎所有 CNN）。它的核心洞见：沿 `(N,H,W)` 聚合统计（每通道一组），**把 batch 内的统计信息注入归一化**——这相当于在层间引入轻量噪声/正则化（不同 batch 的统计略有不同），有隐式正则效果，提升泛化。同时 running 统计让**推理时分布稳定**（不依赖 batch）。`momentum` 控制 EMA 平滑强度。**数值与精度**：

- **eps 位置**：`rstd = 1/sqrt(var + eps)`，eps 加在方差根号**里**，同 LN/GN。移植坑同前。
- **bf16/hf16 累加精度**：归一化单元 `N*H*W` 可能很大（如 `32*224*224 ≈ 1.6M`），bf16 直接累加 `Σ x²` 会**完全失真**。BN 核内部必须 float32 累加。这是 BN 在低精度训练的头号坑。
- **有偏 vs 无偏的混用（重要陷阱）**：归一化用**有偏方差**（除以 `N*H*W`），但 **running_var 的更新用无偏方差**（除以 `N*H*W - 1`）。两者在同一算子里混用——归一化 `y` 用有偏，EMA 更新 running 用无偏。这是 PyTorch 的刻意设计（running 用于推理，无偏估计更准），但极易在移植时弄错。
- **`momentum` 的语义**：PyTorch 的 momentum 是"**新值的权重**"——`running = (1-m)*running + m*new`，典型 `m=0.1`。这与某些框架（如 TF，momentum 是旧值的权重）**方向相反**，移植时务必确认。
- **推理时 batch=1**：`training=False` 时直接用 running，与 batch 无关，所以推理 batch=1 没问题。但若误用 `training=True` 且 `N=1`，则每个通道的 var 在 `(1,H,W)` 上算，统计噪声大（这正是 GroupNorm 想解决的）。
- **NaN 传播**：若某通道在 batch 内全相等，`var=0`，`rstd=1/sqrt(eps)` 有限（不 NaN）；若含 inf/NaN 则扩散。

**实现逻辑与复杂度**　NumPy 风格伪代码（前向，训练路径，float32 累加）：

```python
def _native_batch_norm_legit(x, gamma, beta, running_mean, running_var,
                             training, momentum, eps):
    N, C, H, W = x.shape
    xf = x.to(float32)
    if training:
        # 沿 (N,H,W) 归约，保留 C —— 即 transpose 使 C 为最外维再归约
        x_t = xf.permute(1,0,2,3).reshape(C, -1)        # (C, N*H*W)
        mean = x_t.mean(dim=1)                          # (C,)
        var  = ((x_t - mean[:,None])**2).mean(dim=1)    # (C,)，有偏
        unbiased_var = var * (N*H*W) / (N*H*W - 1)      # 无偏，用于 running 更新
        # EMA 更新 running（in-place）
        running_mean.copy_((1-momentum)*running_mean + momentum*mean)
        running_var.copy_ ((1-momentum)*running_var  + momentum*unbiased_var)
    else:
        mean = running_mean                             # (C,)，直接用
        var  = running_var
    rstd = 1.0 / (var + eps).sqrt()                     # (C,)
    y = (xf - mean[None,:,None,None]) * rstd[None,:,None,None]
    if gamma is not None: y = y * gamma.to(float32)[None,:,None,None]
    if beta  is not None: y = y + beta .to(float32)[None,:,None,None]
    y = y.to(x.dtype)
    return y, mean, rstd                                # save_mean, save_invstd
```

时间复杂度 `O(numel(x))`（训练时两遍：一遍算 mean/Σx²，一遍算 y；推理时一遍），空间 `O(numel(x) + 2*C)`（输出 + save_mean + save_invstd，**分配新张量**；running 是 in-place 更新不额外分配）。高效核（cuDNN/`mkldnn`）把 mean→var→rstd→affine→running 更新融成一个 kernel。训练路径比推理多一次"沿 (N,H,W) 规约"（transpose + reduce），是 BN 训练慢于推理的主因。

**边界与陷阱**
- **running_mean/running_var 是 in-place 输入**：签名带 `a!`/`b!`，算子会**原地修改**它们（训练时 EMA 更新）。若调用方不希望被改，需先 clone。inductor 的 SSA 契约（详见第 0 章 §0.7）要求对别名/in-place 输入显式标注。
- **weight/bias 可选**：`None` 表示不仿射。若提供，形状必须 `(C,)`。
- **running_mean/running_var 形状**：必须 `(C,)`，与输入通道数一致。推理时若 running 未初始化（全 0/全 1）会得到错误结果。
- **有偏/无偏混用**：归一化用有偏，running 更新用无偏。移植坑，详见"数值与精度"。
- **momentum 方向**：PyTorch momentum 是新值权重（`m=0.1` 常见），与 TF 相反。移植坑。
- **training=True 且 N=1**：统计噪声大，BN 退化（这是 GroupNorm 存在的理由）。检测/分割小 batch 场景避免用 BN。
- **eps 位置、bf16 累加精度**：同 LN/GN——根号内 eps、float32 内部累加。
- **`_legit` vs 其他变体**：详见上面"`_legit` 后缀的含义"。封装算子 `batch_norm` 按 `training` 分发。
- **3D 变体**：`_native_batch_norm_legit` 对 5D 输入 `(N,C,D,H,W)` 同样适用（沿 `(N,D,H,W)` 归约），称 BatchNorm3D。同一算子，形状自适应。

**Inductor 视角**　**template / fallback**——BN 训练路径数值最敏感（沿大维 `(N,H,W)` 规约 + 有偏/无偏混用 + EMA 更新 + in-place running），inductor 走专用 fused BN **template**（cuDNN/`mkldnn`/Triton BN kernel）或 `aten` native **fallback**。in-place 的 running 更新使它**不能**被当作纯 pointwise+reduction 融合（有副作用），且统计轴 `(N,H,W)` 是非连续的（需 transpose），进一步排除通用 reduction 模板。推理路径（`_no_training`）较简单但仍走 template 以保证 eps/累加精度一致。

## 本章小结

归一化的核心是"**沿某几维减均值、除标准差、再仿射**"，三者的唯一差异是**沿哪些维聚合统计量**：LayerNorm 沿特征维（最后一维，每 token/样本独立）、GroupNorm 把通道分组后沿组内通道+空间维（与 batch 无关，适合小 batch）、BatchNorm 沿批+空间维（保留通道为统计轴，注入 batch 内统计作为隐式正则）。两条贯穿全章的要点：① **数值稳定性是归一化的生命线**——eps 一律加在方差根号**里**（`rstd = 1/sqrt(var+eps)`），bf16/hf16 下**必须用 float32 内部累加器**算 mean/var，否则大归一化单元（BN 的 `N*H*W`、GN 的 `Cg*H*W`、LN 的大 `D`）下累加严重失真，这是 NPU 后端移植的头号坑；② **BatchNorm 的有偏/无偏混用与 momentum 方向**是移植时最易踩的坑——归一化用有偏方差、running 更新用无偏方差、momentum 是新值权重（与 TF 相反）。三者都因数值敏感而走 inductor 的 **template/fallback**（而非分解成 pointwise+reduction 融合）。`_native_batch_norm_legit` 的 `_legit` 后缀标记"含 running buffer 的正式训练路径"，与 `_no_training`（推理）、`_functional`（返回式更新）、`.no_stats`（无统计）变体区分。

下一章第 16 章进入 padding / upsampling / 距离，它们与归一化同样涉及"沿特定维的几何变换与数值计算"，但重点转为**边缘填充策略**与**插值核**，数值上更平凡（无方差累加的精度压力）。

---

[上一章 第 14 章 池化](14-pooling.md) 　|　 [下一章 第 16 章 Padding / Upsampling / 距离](16-padding-upsampling-distance.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)
