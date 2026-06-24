# 第 11 章 创建与填充（Creation & Filling）

创建类算子是模型图的"入口":它们不消费任何输入张量(或只消费形状/标量),凭空产生一块填充好数据的张量。在 ATen 的 161 个 core 算子里,这一类算子有个反直觉的特点——**单看功能描述("返回一个填充了某值的张量")几乎无法理解它在工程上为何存在**。empty 为什么不初始化?full 为什么不叫 fill_new?rand 和 randn 的区别为什么重要?这些问题只有回到 inductor 的代码生成视角(分配策略、确定性、RNG decomposition)才能讲清楚,这也是本书 Part II 的重心。

本章先给速查表覆盖 6 个基础创建算子,再深入 4 个最有讲究的算子(arange / rand / randn / randperm)。读者会注意到 zeros / ones / eye / linspace / randint 这些"看似基础"的算子不在本章——它们在 ATen 里是复合算子,最终分解到 full / arange / fill,详见附录 B。

## 本章速查（Tier C）

| 算子 | 功能一句话 | 形状/风险 | inductor 归类 |
|---|---|---|---|
| `aten.empty` | 分配未初始化张量,只定形状 | 极快(不填值),但内存是垃圾值,用前必须覆盖 | fallback / template(分配原语) |
| `aten.empty_strided` | 同 empty,但可显式指定 stride 与 memory_format | 非连续/转置视图的基础分配,跨步越界风险 | fallback / template |
| `aten.full` | 分配张量并用标量值填充每个元素 | 新分配,受类型提升约束 | pointwise(常量广播) |
| `aten.full_like` | 按输入张量的形状/dtype/device 生成 full | 依赖输入张量属性,内存格式可指定 | pointwise |
| `aten.scalar_tensor` | 把 Python 标量包成 0 维张量 | 0 维,主要用于把数值"张量化"喂给图 | fallback(常量物化) |
| `aten.fill` | 把已有张量原地填成某标量值 | **原地修改**(非函数式),SSA 中需特殊处理 | pointwise(原地 kernel) |

> 随机类核心算子 `rand` / `randn` / `randperm` 见下文 Tier D 深入;`arange` 作为最复杂的确定序列生成器亦在 Tier D。

## 深入算子（Tier D）

### `aten.arange(start=0, end, step=1, *, dtype=None, ...)` — 生成等差序列张量

**作用与语义**　返回一个 1 维张量,元素为 `start, start+step, start+2*step, ...`,序列在到达或超过 `end` 时停止(不含 end)。数学上:

```
out[i] = start + i * step,   i = 0, 1, ..., n-1
n = max(0, ceil((end - start) / step))   // 当 end>start 且 step>0
```

输出形状恒为 1 维 `[n]`。返回值是新分配张量,不别名任何输入。

**示例**　`arange(start, end, step)` 生成不含 `end` 的等差序列,整数参数下 dtype 留在 `int64`:

```python
>>> torch.arange(0, 10, 2)     # start=0, end=10, step=2 -> [0,2,4,6,8]
tensor([0, 2, 4, 6, 8])
```

**为什么需要这个算子**　arange 解决的是"**按规则生成整数/浮点下标序列**"这一高频结构需求。在模型域里它的典型用途几乎全是和"位置/索引"绑定:
- **位置编码**:Transformer 里 `pos_id = arange(seq_len)` 作为绝对位置,与频率向量做外积生成 sinusoidal / RoPE。
- **掩码生成**:causal mask 的下三角靠 `arange(n)[:,None] > arange(n)[None,:]` 构造。
- **下标/gather 索引**:为 gather/scatter 生成 `batch_idx = arange(B)`。
- **区间采样**:数值方法里 `arange(0, T, dt)` 作时间轴。

与相邻算子的边界:`linspace` 是"按数量生成等距点"(指定 num),`arange` 是"按步长生成等距点"(指定 step);`torch.range`(已废弃,包含 end)不要和它混。randint 则是随机整数,与 arange 的确定性正交。

**实现逻辑与复杂度**

```python
def arange(start, end, step=1, dtype=None, device=None):
    n = ceil((end - start) / step)
    if n <= 0: return empty(0, dtype=dtype, device=device)
    out = empty(n, dtype=dtype, device=device)
    # 标准实现:pointwise kernel,每个线程算 i*step+start
    for i in parallel:
        out[i] = start + i * step
    return out
```

时间 O(n),空间 O(n),需要 1 次新分配。无可避免地要写一整块显存——它是"必须落盘的全生成",不像 empty 可以零成本。

**边界与陷阱**
- **dtype 推断**:这是 arange 最经典的坑。当 dtype 未指定时,推断规则是:若三个参数都是整数(包括 `arange(end)` 里只有整数 end),dtype = int64;若有任一是浮点,dtype = float32。这导致 `arange(0, 1, 0.1)` 是 float32,但 `arange(10)` 是 int64——混用进同一张量会触发类型提升。PyTorch 2.x 后引入 `arange.start` / `arange.end` 变体以缓解歧义。
- **浮点 step 粯误差**:`arange(0, 1, 0.1)` 由于 0.1 不可精确表示,长度可能比直觉多一个或少一个(ceil 会放大误差)。精确点数请用 linspace。
- **空序列**:`end <= start`(step>0)时返回长度 0 张量,不是错误。
- **负 step**:支持倒序 `arange(10, 0, -1)`,此时 end<start 才有元素。
- **类型提升不生效**:arange 的 dtype 完全由上述推断规则决定,不会做参数间提升——显式传 dtype 才可控。

**Inductor 视角**　pointwise(常量生成 kernel),通常被识别为简单的 affine 表达式 `i*step+start`,可融合进下游 pointwise;某些后端会用 template 直接生成连续填充 kernel。

---

### `aten.rand(size, *, generator=None, dtype=None, device=None)` — 生成 [0,1) 均匀分布张量

**作用与语义**　返回形状为 `size` 的张量,每个元素独立采样自 **[0, 1) 上的均匀分布**。返回新分配张量,不别名输入。可选 `generator` 参数指定独立 RNG(用于可复现/隔离随机性)。

```
out[i] ~ Uniform(0, 1)   独立同分布
```

**示例**　已设种子(`torch.manual_seed(0)`),实际为随机——`rand(size)` 每个元素 `∈[0, 1)` 均匀分布:

```python
>>> torch.manual_seed(0)            # 设种子可复现,实际为随机
>>> torch.rand(2, 3)                # 每个元素独立采样自 [0,1)
tensor([[0.4963, 0.7682, 0.0885],
        [0.1320, 0.3074, 0.6341]])
```

**为什么需要这个算子**　rand 解决的是"**产生受控的、可复现的随机浮点张量**"这一需求,贯穿训练与推理:
- **Dropout**:经典实现 `mask = (rand(x.shape) > p) / (1-p)`,用 rand 生成伯努利掩码。
- **数据增强**:Cutout / Mixup 的随机区域由 rand 选坐标。
- **采样/Gumbel-softmax**:`gumbel = -log(-log(rand(shape)))` 把均匀转成 Gumbel 噪声。
- **初始化**:某些自定义初始化直接用 uniform_(0, std)。

为什么需要专门的 rand 而不是"先用 randint 再除":(1) 数值正确性——直接用底层 RNG 的 [0,1) 输出避免整数范围/除法精度问题;(2) **确定性/可复现**——通过 generator 与 seed,rand 能保证跨设备、跨进程生成相同序列,这是分布式训练 checkpoint 的硬要求;(3) 性能——单条 rand kernel 比"randint + div"两步快且不分配中间张量。

与相邻算子的边界:randn 采样正态而非均匀;randint 采样离散整数区间;randperm 生成排列(无重复)。

**实现逻辑与复杂度**

```python
def rand(size, generator=None, dtype=float32, device=None):
    out = empty(size, dtype=dtype, device=device)
    # 每个 element 从 generator 消耗若干 bit,做均匀映射
    for i in parallel:
        raw = generator.random64()          # 64 bit
        out[i] = (raw >> 11) * (1.0 / (1<<53))   # [0,1),53 位精度
    return out
```

时间 O(N),空间 O(N)(N = prod(size))。无零拷贝变体,必须分配。

**边界与陷阱**
- **nondeterministic**:默认(无 generator)rand 消费全局默认 RNG,两次运行结果不同。**这会让 inductor graph retrace 失效、让测试不稳定**——调试时务必加 `torch.manual_seed`。
- **dtype 默认 float32**(不是 float64),与 numpy 不同,易混淆。
- **半精度 rand**:float16/bfloat16 的 rand 精度有限,统计上偏离均匀更明显;某些后端仍先算 float32 再转。
- **generator 跨设备**:CPU generator 不能直接给 CUDA/NPU rand 用,需 Philox 子 RNG。
- **空 size**:`rand(0)` 返回空张量,不报错,但会 advance RNG 状态(取决于实现)。

**Inductor 视角**　属 nondeterministic pointwise,inductor 对 rand/randn 有专门的 **RNG decomposition**:把 `aten.rand` 分解为"读取 Philox 计数器 + pointwise 均匀变换",使其能与下游 pointwise(如 dropout 的 mask 计算)融合,避免单独发射一个随机 kernel。这是 inductor 里少数显式处理随机性的算子之一。

---

### `aten.randn(size, *, generator=None, dtype=None, device=None)` — 生成标准正态分布张量

**作用与语义**　返回形状为 `size` 的张量,每个元素独立采样自 **标准正态分布 N(0,1)**(均值 0、方差 1)。返回新分配张量,可选 generator。

```
out[i] ~ Normal(0, 1)   独立同分布
```

**示例**　已设种子(`torch.manual_seed(0)`),实际为随机——`randn(size)` 每个元素为标准正态 `N(0,1)`,有负有正:

```python
>>> torch.manual_seed(0)            # 设种子可复现,实际为随机
>>> torch.randn(2, 3)               # 标准正态 -> 有负有正
tensor([[ 1.5410, -0.2934, -2.1788],
        [ 0.5684, -1.0845, -1.3986]])
```

**为什么需要这个算子**　randn 解决的是"**产生高斯随机张量**"的需求,这是深度学习里权重初始化与噪声注入的物理基础:
- **权重初始化**:Xavier / He / Kaiming 初始化的核心都是 `w = randn(shape) * gain`,randn 提供高斯基底。
- **激活噪声**:扩散模型(ddpm)里每步加 `eps = randn(shape) * sigma` 作为扩散噪声。
- **VAE 重参数化**:`z = mu + sigma * randn(shape)`。
- **对抗样本**:FGSM 等方法里用 randn 作随机扰动起点。
- **训练扰动**:梯度噪声、label smoothing 的随机项。

为什么模型初始化偏好正态而非均匀:中心极限定理下,高维求和(如 matmul 的加权和)趋于正态,初始化用正态能让激活方差更可控(见 Xavier/He 推导)。这也解释了 randn 比 uniform-init 更主流。

与相邻算子的边界:randn 仅生成 N(0,1);若需 N(mu,sigma) 用 `randn*sigma+mu`(复合);`normal_` 是原地版(→ 分解成 randn + add)。与 rand 的区别仅在分布形状,确定性/seed 语义完全一致。

**实现逻辑与复杂度**

```python
def randn(size, generator=None, dtype=float32, device=None):
    out = empty(size, dtype=dtype, device=device)
    for i in parallel:
        u1, u2 = generator.uniform01(), generator.uniform01()
        # Box-Muller 变换:均匀 -> 正态
        out[i] = sqrt(-2 * log(u1)) * cos(2*pi*u2)
    return out
```

实际后端多用 Box-Muller 或极坐标法(Marsaglia)将均匀 RNG 输出转为正态。时间 O(N),空间 O(N),需分配。每个元素消耗 2 个均匀样本(Box-Muller),RNG 消耗速度是 rand 的约 2 倍。

**边界与陷阱**
- **统计正确性**:float16 randn 在尾部精度极差(|x|>4 时),且极端值会 overflow 成 inf。训练大模型常用 float32 生成再转 fp16。
- **nondeterministic**:同 rand,需 generator 或 seed 保证可复现。
- **Box-Muller 的 NaN**:`u1=0` 时 `log(0)=-inf`,实现必须保证 u1>0(通常用 `1 - rand` 或屏蔽 0)。
- **跨后端一致性**:CPU 与 CUDA 的正态变换算法可能不同,**同 seed 跨设备结果不同**——这是分布式/迁移测试的经典坑。
- **dtype 默认 float32**,同 rand。

**Inductor 视角**　nondeterministic pointwise,走与 rand 相同的 RNG decomposition 路径:Philox 计数器 → 均匀 → Box-Muller pointwise,可与下游乘加(如初始化的 `*gain`)融合。inductor 显式建模 RNG 状态张量以保证确定性。

---

### `aten.randperm(n, *, generator=None, dtype=int64, device=None)` — 生成随机排列

**作用与语义**　返回长度为 `n` 的 1 维张量,内容是 `[0, 1, ..., n-1]` 的一个**随机排列**(无重复、全覆盖)。可选 generator,dtype 默认 int64(也接受 int32)。

```
out 是 {0,1,...,n-1} 的一个均匀随机置换
```

**示例**　已设种子(`torch.manual_seed(0)`),实际为随机——`randperm(n)` 返回 `0..n-1` 的一个随机排列(无重复、全覆盖):

```python
>>> torch.manual_seed(0)            # 设种子可复现,实际为随机
>>> torch.randperm(5)               # 0..4 的一个随机排列
tensor([4, 0, 1, 3, 2])
```

**为什么需要这个算子**　randperm 解决的是"**打乱有序下标**"的需求,核心场景是训练循环的 shuffle:
- **DataLoader shuffle**:`perm = randperm(N); batch = data[perm]`——这是 epoch 级打乱的标准实现。
- **负采样/对比学习**:从候选池随机抽无重复下标。
- **k-fold 交叉验证**:划分 fold 时打乱样本。
- **强化学习**:experience replay buffer 的随机抽样。

为什么不能用 `randint(0, n, size)` 替代:randint 有重复,而 shuffle 要求"每个样本恰好出现一次"。为什么不用 Python `random.shuffle`:它是原地且单线程 CPU,无法走张量/RNG decomposition,也无法在 NPU 上跑。

与相邻算子的边界:randperm 输出是 int64 下标(可作 advanced index);randint 是可重复随机整数;multinomial 是加权抽样(可放回/不放回),与 randperm 的"等概率全排列"不同。

**实现逻辑与复杂度**

```python
def randperm(n, generator=None, dtype=int64, device=None):
    out = arange(0, n, dtype=dtype, device=device)   # [0,1,...,n-1]
    # Fisher-Yates (Knuth) shuffle
    for i in range(n-1, 0, -1):
        j = generator.randint(0, i+1)                # [0, i]
        out[i], out[j] = out[j], out[i]
    return out
```

时间 O(n),空间 O(n)。注意 Fisher-Yates 有**串行数据依赖**(每步的 j 依赖当前状态),这是 randperm 与 rand/randn 的本质区别——后者可完全并行,前者不能。这使得 randperm 在 GPU/NPU 上很难高效,常退化为分段并行或 host 端实现。

**边界与陷阱**
- **串行性**:大 n 时 GPU 上的 randperm 是性能黑洞,很多框架把 shuffle 放在 CPU host。inductor 一般不把 randperm 融进 kernel。
- **nondeterministic**:同 rand 系列,需 generator/seed。
- **dtype 限制**:只支持 int64 / int32,传 float 会报错。
- **n=0/1**:合法,返回空或单元素。
- **内存**:需 O(n) int64 暂存,大 vocab 打乱(如 50k token)需注意显存。
- **跨后端不一致**:不同后端的 shuffle 算法实现不同,**同 seed 跨设备结果不同**,比 rand/randn 更严重(因算法本身实现多样)。

**Inductor 视角**　fallback / host(非 pointwise,因有串行依赖),通常不被 inductor 融合,而是作为 graph 里的一个未融合节点直接调用后端 op。这是创建类里唯一无法归入 pointwise/reduction 的算子。

---

## 本章小结

创建与填充算子看似简单("造一个张量"),但工程上每个都有讲究:empty 用"不初始化"换速度(代价是必须手动覆盖);full/empty_strided 是 zeros/ones/linspace/transpose 等复合算子的分解终点;随机算子(rand/randn/randperm)引入了 nondeterminism,而 inductor 通过 RNG decomposition 把均匀/正态变换融进 pointwise 以兼顾确定性与性能,randperm 则因串行依赖成为唯一例外。理解这一层,读者才能在 trace/decode inductor graph 时正确处理随机性、初始化与索引生成。

下一章《第 12 章 内存布局与 dtype》将深入 stride / memory_format / 类型提升等支撑所有算子的底层机制——本章反复出现的"非连续""dtype 推断"都将在那里给出统一答案。

---

[上一章 第 10 章](10-sort-topk.md) 　|　 [下一章 第 12 章](12-memory-layout-dtype.md) 　|　 [第 0 章 张量基底](00-tensor-substrate.md) 　|　 [README 索引](README.md)

> 公共概念:广播 (broadcast)、类型提升 (type promotion)、SSA 契约详见第 0 章 §0.4 / §0.5 / §0.7。复合算子(zeros / ones / eye / linspace / randint / normal_ / uniform_)的分解链路详见附录 B。
