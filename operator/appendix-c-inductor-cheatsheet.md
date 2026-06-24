# 附录 C inductor 降级速查（torch 2.7.1）

> 本附录面向 **inductor 开发者**。当你想知道"某个 `aten.*` 算子在 inductor 里会被编译成什么样的核"时，这张速查表给出答案。前置阅读：[README](README.md)、[附录 B（decomposition）](appendix-b-decompositions.md)。

## 导言

inductor 的核心数据结构是 `lowerings: Dict[OpOverload, Callable]`（定义于 `torch/_inductor/lowering.py`）。这个字典把每个 `aten.*` overload 映射成一个**返回 IR 节点的 Python 函数**。后端调度器（scheduler）拿到 IR 节点后，再把它们拼成 Triton 核。

`lowerings` 不是铁板一块——按**生成代码的方式**，所有注册项可归为四类。看到一个不认识的算子，先问"它走哪条路径？"，问题就解决了一半。

### 总览表

| 类别 | 注册机制（lowering.py 内） | 生成的 IR / 核 | 数量级（torch 2.7.1） | 融合性 |
|---|---|---|---|---|
| **pointwise** | `register_pointwise` / 直接 `register_lowering` + `make_pointwise` | `ir.Pointwise` → 融合的 Triton elementwise 核 | ~67 处 `register_pointwise`（合计约 76 处含别名） | 极强：多个 pointwise 会被融合进同一个核 |
| **reduction** | `make_reduction(...)` + `register_lowering` | `ir.Reduction`（含 scan） | 9 处 `make_reduction` 调用 + 若干直连 | 受限：reduction 是融合"汇点"（sink） |
| **template** | `kernel/mm.py`、`kernel/bmm.py` 内 `@L.register_lowering` + `autotune_select_algorithm` + `TritonTemplate` | `ir.Choice` → 选定的 Triton mm/bmm 模板 | mm / bmm / addmm / baddbmm / scaled_mm / convolution / flex_attention | 弱：模板核自成一体，几乎不参与融合 |
| **fallback** | `make_fallback(op)` | `ir.FallbackKernel` → 退回 eager `aten`（NPU 上即 ACLNN） | **112 行**调用 / **92 个**不同算子 | 无：fallback 是**融合边界** |

> 数字来源：`grep -c "make_fallback(" lowering.py` = 112；`grep -oE "make_fallback\([a-z_.]+" | sort -u | wc -l` = 92；`grep -c register_pointwise` = 67；`grep -c "make_reduction("` = 9。

---

## 四类降级路径

### 1. pointwise —— 逐元素算子 → 融合的 elementwise 核

**含义**：输出每个元素只依赖输入同位置元素，无跨点数据依赖。inductor 把一串 pointwise 算子**整体融合**进一个 Triton 核（fused pointwise），这是 inductor 性能的最大来源。

**生成方式**：`register_pointwise(op, type_promotion_kind=...)` 把算子注册成 `make_pointwise(...)` 返回的 handler，handler 输出 `ir.Pointwise` 节点。调度器的 fusion pass 会把若干连续 `ir.Pointwise` 合成一个核。

**典型算子**（torch 2.7.1）：

| 子族 | 算子 |
|---|---|
| 算术 | `add` / `sub` / `mul` / `div` / `neg` / `rsqrt` / `sqrt` / `pow` / `fmod` / `remainder` |
| 逐元素数学 | `exp` / `log` / `log1p` / `exp2` / `sin` / `cos` / `tanh` / `sigmoid` / `erf` / `erfinv` / `reciprocal` / `abs` / `clamp` |
| 选择 / 极值 | `where` / `maximum` / `minimum` / `clamp_min` / `clamp_max` |
| 比较 | `eq` / `ne` / `lt` / `le` / `gt` / `ge` |
| logical / bitwise | `logical_and` / `logical_or` / `logical_not` / `bitwise_and` / `bitwise_or` / `bitwise_not` / `__lshift__` / `__rshift__` |
| 类型 / 取整 | `to` / `floor` / `ceil` / `round` / `trunc` / `frac` / `copysign` |

**为什么这一类**：pointwise 是**可融合的原子单位**。inductor 的 fusion 启发式只对 pointwise 自由组合；reduction / template / fallback 都会切断融合链。

### 2. reduction —— 沿维聚合 → `ir.Reduction` 核

**含义**：沿某一（或若干）维做聚合，输出形状比输入小。含**规约**（sum/mean/max）和**扫描**（cumsum/cumprod）。

**生成方式**：
- 规约：`fn = make_reduction("sum", override_return_dtype=...)` 然后 `register_lowering(aten.sum)(fn)`，handler 输出 `ir.Reduction`。
- 扫描：`register_lowering(aten.cumsum)` 内部根据 dtype / 输入布局决定走 Triton scan 核还是 `fallback_handler`（见 lowering.py:6181-6254）。

**典型算子**（torch 2.7.1）：

| 算子 | 注册位置（lowering.py） | 备注 |
|---|---|---|
| `sum` | `make_reduction("sum")` (6177) | `prims.sum` 共享 |
| `prod` | `make_reduction("prod")` (6305) | override dtype |
| `mean` | `register_lowering(aten.mean)` (5788) | |
| `amax` / `amin` | `make_reduction("max")` / `"min"` (6338-6339) | |
| `argmax` / `argmin` | `make_reduction("argmax"/"argmin", override_return_dtype=int64)` (6340-6344) | |
| `any` | `make_reduction("any")` (6312) | bool 规约 |
| `xor_sum` | `make_reduction("xor_sum")` (6337) | |
| `var_mean` | `register_lowering(aten.var_mean)` (5917) | |
| `cumsum` / `cumprod` / `logcumsumexp` | `register_lowering` + `fallback_handler` (6181-6254) | scan；非连续/特殊 dtype 走 fallback |

**为什么这一类**：reduction 是 fusion 的**汇点（sink）**——pointwise 可以喂给 reduction、也可以接在 reduction 后面，但两个 reduction 一般不融合。把它独立成一类，是因为 Triton reduction 核的 tile 化与 elementwise 完全不同。

### 3. template —— 矩阵乘 / 卷积 → 选模板 + autotune

**含义**：GEMM、卷积这类"算法空间巨大"的算子，inductor 不走通用 pointwise/reduction 路径，而是**枚举一组 Triton 模板（choice）**，经 `select_algorithm.autotune_select_algorithm` 实测挑选最优配置。

**生成方式**：注册不在 `lowering.py` 主体，而在 `torch/_inductor/kernel/mm.py`、`kernel/bmm.py`：
- `@L.register_lowering(aten.mm)` → `mm()` handler：枚举 `mm_template`（`TritonTemplate`）+ 可选 CUTLASS / CPP-gemm / CK-gemm 候选 → `autotune_select_algorithm("mm", choices, ...)` 返回 `ir.Choice` → 选定一个 `ExternKernel` / Triton 模板。
- `@L.register_lowering(aten.bmm)`（bmm.py:132）同构；`addmm` / `baddbmm` 复用 mm 的 `mm_plus_mm` / `baddbmm` 路径。
- `convolution` 走 `kernel/mm.py` 的卷积模板分支（CPU 上常退回 oneDNN extern kernel）。
- `scaled_mm` 在 `kernel/mm_scaled.py`，`flex_attention` / `flex_decoding` 在各自模块。

**典型算子**：`mm`、`bmm`、`addmm`、`baddbmm`、`scaled_mm`（fp8/int8 量化）、`convolution`、`flex_attention`、`flex_decoding`。

> 在 lowering.py 里，这些算子出现在 `add_needs_realized_inputs([... bmm, convolution, mm ...])`（227-232 行）——意思是**输入必须先 realize（物化成实际 buffer）**，因为模板核需要确定的 stride / layout。

**为什么这一类**：GEMM 的性能由 tile / 流水 / 分块逐层决定，pointwise fusion 帮不上忙。把它从通用路径拆出来，才能集中做 autotuning。

### 4. fallback —— inductor 原生降不了 → 退回 eager（ACLNN）

**含义**：inductor **没有**为该算子写 Triton 核、也没有合适的 template。`make_fallback(op)` 把它包成 `FallbackKernel` IR 节点——运行时直接调用 eager `aten`（CPU 上是 ATen，**NPU 上是 ACLNN**）。

**生成方式**：
```python
make_fallback(aten.sort)                  # 默认 warn=True，会在日志里告警
make_fallback(aten._addmm_activation, warn=False)   # 静默
make_fallback(aten.convolution_backward, constrain_to_fx_strides)  # 带布局约束
```

**典型算子**（torch 2.7.1，共 92 个唯一算子 / 112 行调用，含 overload 变体）：

| 族 | 算子 |
|---|---|
| 排序 / top-k | `sort`、`sort.stable`、`topk`、`mode` |
| 直方图 | `histc`、`bucketize`、`bincount` |
| embedding | `embedding`、`embedding_bag`（多 overload） |
| 距离 | `_cdist_forward`、`_cdist_backward` |
| 卷积反传 | `convolution_backward`、`_slow_conv*d` 系列、`miopen_*` |
| 激活 fused-matmul | `_addmm_activation`（warn=False） |
| 线性代数 | `linalg_qr`、`linalg_lu`、`linalg_cholesky_ex`、`linalg_solve_triangular`、`linalg_householder_product`、`linalg_inv_ex`、`linalg_pinv`、`linalg_matrix_exp` 等（约 15 个） |
| 稀疏 / 其它 | `*_embedding_dense_backward`、`_embedding_bag.*`、部分 scatter/gather 特化 |

**为什么这一类**：这些算子要么 (a) 算法复杂度超出 Triton 表达力（QR 分解、LU），(b) 数值/排序语义难 tile 化（sort、topk、histc），或 (c) eager 已有高度优化实现且重写收益低（部分 linalg）。

> **开发者警告**：fallback 是**融合边界**。一个 fallback 调用会把前后两段原本可融合的 pointwise 切断成两个核。生产中 `make_fallback(..., warn=True)` 会在 inductor 日志里打印 `Found a fallback path: aten.xxx`——撞到它通常意味着**优化机会丢失**，值得调查能否分解（decompose，见 [附录 B](appendix-b-decompositions.md)）成 pointwise/reduction。

---

## NPU（`_inductor_new`）特有覆写

NPU 后端的策略是：**复用 GPU 的 `lowerings` 字典**（~99% 算子行为相同），只在少数几处做"外科手术式"覆写。覆写集中在 `torch_npu/_inductor_new/__init__.py`，由 `register_backend()` 在 `TORCHINDUCTOR_NPU_BACKEND=new` 时调用。

### 覆写一览

| 覆写类型 | 函数 | 目标算子 | 原因 | 处理方式 |
|---|---|---|---|---|
| **decomposition 替换** | `_register_npu_decompositions` | `gelu` | core 的 erf 分解（`0.5x(1+erf(x/√2))`）在 float32 下与 CANN native gelu 偏差 ~4.7e-4，超默认容差 | 改注册成 **tanh 近似**（`0.5x(1+tanh(0.7978(x+0.0447x³)))`），与 NPU native gelu 在 ULP 内一致 |
| **decomposition 新增** | 同上 | `erfc` | Triton-Ascend 无原生 erfc | 注册为 `1 - erf(x)` |
| **静默 fallback** | `_register_npu_fallbacks` | `special_bessel_j0/j1/y0/y1` | Triton-Ascend 不支持 | `make_fallback(op, warn=False)` → ACLNN |
| **静默 fallback** | 同上 | `isinf` | CANN `IsFinite` TBE 核在 ±inf 输入崩溃 | ACLNN |
| **静默 fallback** | 同上 | `any`（bool 规约） | bishengir-compile 在 bool 经 `persistent_reduction` 时生成非法 CCU 指令（vector core 异常 507035） | ACLNN |
| **静默 fallback** | 同上 | `cumprod` | `split_scan` 生成跨位宽指针 bitcast（`*u8 → *u64`），被 `BitcastCanonicalizer` 拒绝 | ACLNN |
| **静默 fallback** | 同上 | `matmul_backward` | NPU 的 PrivateUse1 dispatch 让 `aten.matmul` 优先于 CIA 分解（matmul→mm），autograd 走 `MatmulBackward0`；core 无 `matmul_backward` 分解 | ACLNN（注意：前向 `matmul` 本身仍可被 CIA 分解成 mm，走 template） |
| **静默 fallback** | 同上 | `npu::l1_loss_backward` | `CustomFunctions.cpp` 把 autograd 劫持到 `npu::l1_loss_backward`，无分解；前向 `l1_loss` 是 CIA，make_fx 时已分解成 `sub→abs→mean`，正常编译 | ACLNN（仅反向） |
| **lowering 补丁** | `_patch_sort_lowering` | `sort` / `sort.stable` | NPU 的 `aten.sort` 对非连续输入也返回**连续**输出，但 core `meta_sort` 用 `empty_like(self)`（保留输入 stride），导致 `assert_size_stride` 失败 | 包装原 lowering：**非连续输入先 `clone` + realize + freeze_layout** 再调原 handler |

### 三类覆写的入口

```python
# torch_npu/_inductor_new/__init__.py:register_backend()
_register_npu_decompositions()   # gelu → tanh, erfc → 1-erf
_register_npu_fallbacks()         # 7 个静默 fallback（warn=False）
_patch_sort_lowering()            # sort 非连续输入预 clone
```

### 关键认知

- NPU 与 GPU **共享同一份 `lowerings`**：pointwise / reduction / template 三类路径完全复用，不重写。
- 上述 7 个 fallback **不是** GPU `lowering.py` 已有的 fallback，而是 NPU **额外**加的——因为这些算子在 GPU 上能被 Triton/CUTLASS 处理，在 Triton-Ascend/CANN 上不行。
- `warn=False` 是有意的：这些都是已知且无法短期修复的底层限制，告警只会污染日志。新增此类 fallback 时，请在代码注释里写清**复现的 test case 名**（如 `test_any_npu`、`test_split_cumprod_npu`、`test_dropout3_npu`、`test_inductor_sequence_nr`），便于将来重新评估是否还需要。

---

## 维护说明

**验证版本**：torch 2.7.1 · torch-npu 2.7.1.post4 · Triton 3.2.0 · aarch64 Ascend NPU。

**fallback 归类会随版本变化**。复核当前数字：

```bash
f=<venv>/lib/python3.11/site-packages/torch/_inductor/lowering.py

# fallback：调用行数 / 不同算子数
grep -c "make_fallback(" "$f"
grep -oE "make_fallback\([a-z_.]+" "$f" | sort -u | wc -l

# pointwise / reduction 注册数
grep -c "register_pointwise" "$f"
grep -c "make_reduction(" "$f"

# 列出所有 fallback 算子（含行号，便于审计）
grep -nE "make_fallback\(" "$f"

# 确认 mm/bmm 走 template（不在 lowering.py 主体）
grep -rnE "register_lowering\(aten\.(mm|bmm|addmm)\)" \
    <venv>/lib/python3.11/site-packages/torch/_inductor/kernel/
```

**NPU 覆写复核**：

```bash
f=<pytorch>/torch_npu/_inductor_new/__init__.py
grep -nE "make_fallback|register_decomposition|_register_npu_|_patch_sort" "$f"
```

**相关文档**：[README](README.md) · [附录 B：decomposition](appendix-b-decompositions.md)
