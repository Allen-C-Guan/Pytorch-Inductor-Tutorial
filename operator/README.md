# ATen 基础算子讲解书（给 inductor 开发者的培训教材）

> 这是一本面向**参与 torch-inductor（含 torch-npu `_inductor_new` 后端）开发人员**的基础算子功能手册。目标只有一个：**系统性、全面地理解 PyTorch 基础算子到底在做什么**——尤其是那些"看不出在干什么、为什么需要它"的**操作类算子**。
>
> 本书**不讲算子开发**（不教你写新算子），只讲**算子功能**：数学公式 / 操作逻辑 / 实现复杂度 / 在模型里的角色。

---

## 序言：为什么需要这本书

inductor 的工作是把一个 PyTorch 计算图（由 `aten.*` 算子组成）编译成后端可执行的核（NPU 上是 Triton→CANN）。要做好这件事，第一步是**看懂图里每个算子在干什么**。但现实是：

- **数学类算子**（`add`/`mul`/`matmul`…）好懂——它们就是数学公式的代码翻译；
- **操作类算子**（`as_strided`/`gather`/`scatter`/`expand`/`sort`…）难懂——它们不在"算数学"，而在"搬运/重排/索引"数据，从功能描述很难直觉地理解"它为什么存在、模型哪里需要它"。

本书的重心因此放在**操作类算子**（Part II），每个都深讲"**它在干什么 / 功能是什么 / 为什么需要它**"。数学类算子（Part I）则更紧凑，重点放在数学本身和数值精度。

## 怎么读

- **新手**：先读 [第 0 章 张量基底](00-tensor-substrate.md)（shape/stride/广播/类型提升 + 全书分类轴），再按章号顺序读。
- **查某个算子**：跳到下方[附录 A 全 161 算子速查](#附录-a-core-aten-全-161-算子速查表)，找到它属于哪一章。
- **inductor 开发者**：每章末尾有"inductor 视角"，附录 C 有归类速查（pointwise / reduction / template / fallback）。

---

## 范围规则（先说清楚什么算"基础算子"）

> **基础算子 = Core ATen Ops = `torch.export(..., aten_graph=True)` 后图里能看到的、不可再分解的算子集 = `native_functions.yaml` 里带 `core` 标签的算子（共 161 个基名 / 190 个重载）。**

这条规则杜绝两类 scope 蔓延：

1. **"为什么没有 `silu`？"** —— `silu` 是复合算子，分解成 `sigmoid` + `mul`，两者都在第 1/6 章。`silu` 本身只在[附录 B](appendix-b-decompositions.md)给出分解式。
2. **"`transpose`/`reshape`/`contiguous`/`to`/`zeros`/`linspace`/`eye` 呢？"** —— 这些也是复合算子（`transpose`→`permute`、`reshape`→`view`+`copy`、`contiguous`→`clone`、`to`→`_to_copy`、`zeros`/`linspace`/`eye`→`full`/`arange`）。本书主章节只讲它们分解**到**的那些基础算子；复合算子的分解式统一在附录 B。

主章节**精选约 100 个最主流的** core 算子做讲解；其余约 60 个冷门算子（`fft_r2c`/`col2im`/`_local_scalar_dense`/`_pdist_forward`/3D 池化变体/`reflection_pad1d` 等）只在[附录 A](#附录-a-core-aten-全-161-算子速查表)一行带过。

---

## 分层契约（为什么 `cos` 一行、`cumsum` 两页）

为兼顾"全面"与"不臃肿"，每个算子分两层之一，**序言预先声明**：

- **Tier D（详解）**——固定模板全展开。分配给：①所有**操作类（Part II）算子**（用户强调的重点）；②数学/数值非平凡的算子（`_softmax`/`var`/`cumsum`/`mm`/`gelu`/`native_layer_norm`/`convolution`/`div`/`pow`/`clamp`/`where`/`erf` 等）。
- **Tier C（速查）**——本章开头一张速查表带过（算子 | 功能一句话 | 形状/风险 | inductor 归类）。分配给：公式一目了然的（`abs`/`neg`/`add`/`cos`/`sin`/`exp`/`log`/`relu`/`eq`/`lt`/`logical_*`/`bitwise_*` 等），且**同类打包**（三角族、六个比较算子、logical/bitwise 各一组）。

---

## 分类法：两大轴

详见 [第 0 章 0.6 节](00-tensor-substrate.md#06-全书的核心分类轴位置保持-vs-位置变化)。本质是 **位置保持 (position-preserving) vs 位置变化 (position-changing)**：

| 轴 | 内涵 | 形状 | 重心 | inductor 归口 | 本书 |
|---|---|---|---|---|---|
| **Part I 数学语义类** | `out[i]` 的读地址**闭式仿射、与数据值无关、不重解释存储**（逐元素 1:1 / 规约 N:1 / mm 行·列块） | 平凡可推 | 数学/数值 | pointwise/reduction/template | 第 1–6 章 |
| **Part II 张量操作类** | `out[i]` 依赖 `in` 别处地址 / 重解释存储 | 结构逻辑为主 | **为什么存在 + 操作逻辑** | 多为 fallback/scatter/gather | 第 7–12 章（**重心**）|
| **Part III 模型层计算原语** | 重计算，inductor 用 template | 专门规则 | 数学 + im2col/数值稳定性 | template/fallback | 第 13–16 章 |

---

## 章节地图

### 前置
- [第 0 章 张量基底](00-tensor-substrate.md) — shape/dtype/stride/contiguity/广播/类型提升/SSA 契约

### 专题
- [stride 的本质](stride-essence.md) — 一块一维内存如何假装成任意多维张量；讲透 stride「能做 / 不能做」的精确边界（第 0 / 7 章的深度补充）

### Part I　数学语义类（位置保持）
| 章 | 标题 | 核心算子 |
|---|---|---|
| 1 | [逐元素算术](01-elementwise-arithmetic.md) | add/sub/mul/div/pow/clamp/exp/log/… |
| 2 | [三角与超越函数](02-transcendental.md) | sin/cos/atan2/sigmoid/erf/… |
| 3 | [比较与布尔/位运算](03-comparison-boolean.md) | eq/lt/where/logical_*/bitwise_*/isinf/isnan |
| 4 | [规约](04-reductions.md) | sum/mean/var/amax/argmax/cumsum/… |
| 5 | [线性代数核心](05-linear-algebra-core.md) | mm/bmm/addmm（matmul 旗舰） |
| 6 | [激活函数](06-activations.md) | relu/gelu/sigmoid/_softmax/native_dropout |

### Part II　张量操作/结构类（位置变化，全书重心）
| 章 | 标题 | 核心算子 |
|---|---|---|
| 7 | [形状与视图](07-shape-and-view.md) | as_strided/view/expand/permute/select/slice |
| 8 | [拼接与切分](08-concat-split.md) | cat/repeat/split_with_sizes |
| 9 | [索引家族（最难）](09-indexing-family.md) | gather/scatter/index_put/embedding/nonzero |
| 10 | [排序与选取](10-sort-topk.md) | sort/topk |
| 11 | [创建与填充](11-creation-filling.md) | empty/full/arange/rand/randn |
| 12 | [内存布局与类型转换](12-memory-layout-dtype.md) | clone/copy/_to_copy/resize_ |

### Part III　模型层计算原语
| 章 | 标题 | 核心算子 |
|---|---|---|
| 13 | [卷积](13-convolution.md) | convolution（+im2col/col2im） |
| 14 | [池化](14-pooling.md) | max_pool2d/avg_pool2d/_adaptive_avg_pool2d |
| 15 | [归一化](15-normalization.md) | native_layer_norm/native_group_norm/batch_norm |
| 16 | [填充/上采样/采样](16-padding-upsampling-distance.md) | constant_pad_nd/upsample_*/grid_sampler_2d |

### 附录
- [附录 B 复合→基础分解 cookbook](appendix-b-decompositions.md) — silu/gelu/softmax/bce/mse/std/stack/… 怎么分解成基础算子
- [附录 C inductor 降级速查](appendix-c-inductor-cheatsheet.md) — 每类算子走 pointwise/reduction/template/fallback 的哪条路（torch 2.7.1）

---

## 单算子讲解模板（所有 Tier D 算子一致）

```
## aten.<op>(签名) — 一句话功能
作用与语义　　数学公式 / 操作定义；输入→输出形状；返回值/别名
示例　极简的「输入→输出」实例（`>>>` REPL + 真实 torch 输出，值经 `docs/docs/aten/_examples/verify_chNN.py` 实跑）；
       难点结构算子（as_strided/gather/scatter/conv/grid_sampler 等）在 REPL 后再加一张「输入张量→输出张量」标注图
为什么需要这个算子 / 数值与精度　（自适应二选一）
  · 操作类：解决什么结构变换 + 模型域用途 + 与相邻算子的边界
  · 数学类：数值稳定性 / 精度陷阱
实现逻辑与复杂度　伪代码（NumPy 风格）+ 时间/空间复杂度 + 是否零拷贝
边界与陷阱　负索引/空输入/accumulate/非连续/类型提升/NaN
Inductor 视角（一行）　pointwise/reduction/template/fallback + 为什么
```

---

## 附录 A：Core ATen 全 161 算子速查表

> 标注：**★主章节详解** = 该章有完整 Tier D 条目；**△速查** = 该章速查表一行；**○仅此列出** = 冷门，正文不展开。所有算子均属 core-aten 基础集（带 `core` 标签）。

### A.0 查询/标量抽取（→ 第 0/12 章）
| 算子 | 功能 | 讲解 |
|---|---|---|
| `sym_size` / `sym_stride` / `sym_numel` / `sym_storage_offset` | 读 shape/stride/元素数/偏移（不计算） | 第 0 章 |
| `_local_scalar_dense` | 0 维张量 → Python 标量 | ○ |

### A.1 逐元素算术（→ [第 1 章](01-elementwise-arithmetic.md)）
| 算子 | 功能 | |
|---|---|---|
| `add` `sub` `mul` `neg` `abs` `sign` `reciprocal` | 加减乘除/取反/绝对值/符号/倒数 | △ |
| `div` `pow` `fmod` `remainder` `clamp` | 除/幂/取余（两版）/区间裁剪 | ★ |
| `sqrt` `rsqrt` `exp` `expm1` `log` `log2` `log10` `log1p` | 指数/对数族 | △ |
| `maximum` `minimum` | 逐元素取大/取小 | △ |
| `floor` `ceil` `round` `trunc` | 取整族 | △ |

### A.2 三角与超越（→ [第 2 章](02-transcendental.md)）
| 算子 | 功能 | |
|---|---|---|
| `sin` `cos` `tan` `sinh` `cosh` `asin` `acos` `atan` `asinh` `acosh` `atanh` | 三角/双曲族 | △ |
| `atan2` | 两参反正切（象限正确） | ★ |
| `sigmoid` `erf` | 逻辑斯蒂/误差函数 | ★ |

### A.3 比较与布尔/位运算（→ [第 3 章](03-comparison-boolean.md)）
| 算子 | 功能 | |
|---|---|---|
| `eq` `ne` `lt` `le` `gt` `ge` | 六个逐元素比较（返回 bool） | △ |
| `where` | 按掩码逐元素选择 | ★ |
| `logical_and` `logical_or` `logical_not` `logical_xor` | 布尔逻辑（操作 bool） | △ |
| `bitwise_and` `bitwise_or` `bitwise_not` `bitwise_xor` | 按位逻辑（操作整数） | △ |
| `isinf` `isnan` | inf/NaN 判定 | △ |

### A.4 规约（→ [第 4 章](04-reductions.md)）
| 算子 | 功能 | |
|---|---|---|
| `sum` `prod` `mean` `amax` `amin` `max` `min` `any` | 沿某维聚合 | △/★ |
| `var` | 方差（数值/无偏估计） | ★ |
| `argmax` `argmin` | 极值所在下标 | ★ |
| `cumsum` | 前缀和（scan 类） | ★ |
> 注：`cumprod`/`cummax`/`cummin`/`logcumsumexp` 不在 core 161（scan 家族复合），第 4 章以 `cumsum` 为代表并点名家族。

### A.5 线性代数核心（→ [第 5 章](05-linear-algebra-core.md)）
| 算子 | 功能 | |
|---|---|---|
| `mm` `bmm` `addmm` | 矩阵乘 / 批矩阵乘 / αAB+βC | ★ |
> 注：`matmul`/`mv`/`dot`/`outer`/`einsum`/`linear` 都是复合（→ mm/bmm），附录 B 给分解式。

### A.6 激活函数（→ [第 6 章](06-activations.md)）
| 算子 | 功能 | |
|---|---|---|
| `relu` `leaky_relu` `hardtanh` `tanh` | 分段线性/双曲正切 | △ |
| `gelu` | 高斯误差线性单元 | ★ |
| `sigmoid` | （见 A.2） | ★ |
| `_softmax` `_log_softmax` | 数值稳定的归一化指数 | ★ |
| `native_dropout` | 随机置零 + 缩放（训练） | ★ |

### A.7 形状与视图（→ [第 7 章](07-shape-and-view.md)）
| 算子 | 功能 | |
|---|---|---|
| `as_strided` | 直接指定 (offset,shape,stride) 的视图（**底层原语**） | ★ |
| `view` `expand` `permute` `squeeze` `unsqueeze` `select` `slice` | 改元数据/零拷贝视图 | ★ |
| `flip` `diagonal` | 翻转/取对角 | △ |
> 注：`transpose`/`reshape`/`flatten`/`broadcast_to` 是复合（→ permute / view+copy / expand）。

### A.8 拼接与切分（→ [第 8 章](08-concat-split.md)）
| 算子 | 功能 | |
|---|---|---|
| `cat` | 沿某维拼接 | ★ |
| `split_with_sizes` | 按尺寸切分 | ★ |
| `repeat` | 按各维重复（**分配内存**） | ★ |
> 注：`stack`/`chunk`/`unbind` 是复合（→ unsqueeze+cat / split / slice+squeeze）。

### A.9 索引家族（→ [第 9 章](09-indexing-family.md)）
| 算子 | 功能 | |
|---|---|---|
| `gather` `scatter` `scatter_add` `scatter_reduce` | 按 index 张量收集/散布（**轴语义是难点**） | ★ |
| `index` `index_select` `index_put` | 高级索引（index_put 支持 accumulate） | ★ |
| `masked_scatter` `nonzero` | 掩码散布 / 非零位置 | ★ |
| `embedding` | 按 id 查表（gather 的特化） | ★ |
| `_embedding_bag` | 分段查表+规约（sum/mean/max） | ○ |
| `embedding_dense_backward` | embedding 的稠密反向 | ○ |

### A.10 排序与选取（→ [第 10 章](10-sort-topk.md)）
| 算子 | 功能 | |
|---|---|---|
| `sort` | 排序，返回 (values, indices) | ★ |
| `topk` | 取前 k 大/小 | ★ |
> 注：`argsort` 复合（→ sort 的 indices）。

### A.11 创建与填充（→ [第 11 章](11-creation-filling.md)）
| 算子 | 功能 | |
|---|---|---|
| `empty` `empty_strided` | 未初始化张量（后者指定 stride） | △ |
| `full` `full_like` `scalar_tensor` | 填充常量 | △ |
| `arange` | 等差数列 | ★ |
| `rand` `randn` `randperm` | 均匀/正态/随机排列 | ★/△ |
| `fill` | 原地填充 | △ |
> 注：`zeros`/`ones`/`eye`/`linspace`/`randint`/`*_like` 多为复合（→ full/arange）。

### A.12 内存布局与类型转换（→ [第 12 章](12-memory-layout-dtype.md)）
| 算子 | 功能 | |
|---|---|---|
| `clone` `copy` | 显式拷贝（含跨设备） | ★ |
| `_to_copy` | dtype/device 转换的底层拷贝 | ★ |
| `resize_` `alias` | 原地改形 / 恒等别名（**SSA 逃生舱**，见第 0 章） | △ |
> 注：`to`/`contiguous`/`float()` 等是复合（→ _to_copy / clone）。

### A.13 卷积（→ [第 13 章](13-convolution.md)）
| 算子 | 功能 | |
|---|---|---|
| `convolution` | 通用卷积（stride/pad/dilation/groups/transposed/output_padding） | ★ |
| `convolution_backward` | 卷积反向 | ○ |
| `col2im` | 列还原成图（im2col 的逆，降级可懂性关键） | ○ |

### A.14 池化（→ [第 14 章](14-pooling.md)）
| 算子 | 功能 | |
|---|---|---|
| `max_pool2d_with_indices` | 最大池化（带 argmax，反向用） | ★ |
| `avg_pool2d` `avg_pool2d_backward` | 平均池化 | ★/○ |
| `_adaptive_avg_pool2d` `_adaptive_avg_pool2d_backward` | 自适应平均池化 | ★/○ |
| `max_pool3d_with_indices` `avg_pool3d` `_adaptive_avg_pool3d` `adaptive_avg_pool1d` `avg_pool1d` | 3D/1D 变体（与 2D 同构） | ○ |

### A.15 归一化（→ [第 15 章](15-normalization.md)）
| 算子 | 功能 | |
|---|---|---|
| `native_layer_norm` `native_layer_norm_backward` | 层归一化（数值稳定 + eps） | ★ |
| `native_group_norm` `native_group_norm_backward` | 组归一化 | ★ |
| `_native_batch_norm_legit` `_native_batch_norm_legit_no_training` | 批归一化（`_legit`=含 running stats 的"正式"路径） | ★ |

### A.16 填充/上采样/采样（→ [第 16 章](16-padding-upsampling-distance.md)）
| 算子 | 功能 | |
|---|---|---|
| `constant_pad_nd` `reflection_pad2d` `replication_pad2d` | 常数/反射/复制填充 | △ |
| `reflection_pad1d` `reflection_pad3d` `replication_pad3d` | 1D/3D 填充变体 | ○ |
| `upsample_nearest2d` `upsample_bilinear2d` | 最近邻/双线性上采样 | ★ |
| `grid_sampler_2d` | 双线性网格采样（空间变换/attention 用） | ★ |
| `_cdist_forward` `_pdist_forward` | 成对距离（全/下三角） | ○ |
| `_fft_r2c` | 实→复 FFT（core 里唯一的谱算子） | ○ |

---

*本书验证基准：torch 2.7.1 + torch-npu 2.7.1.post4。算子归类（core-aten）以 `torchgen/.../native_functions.yaml` 的 `core` 标签为准；inductor 降级以 `torch/_inductor/lowering.py` 为准。*
