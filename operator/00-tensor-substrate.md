# 第 0 章 张量基底（Tensor Substrate）

> 这一章**不是讲某个算子**，而是讲所有算子赖以存在的**公共基底**：张量到底是什么、它的形状与内存如何对应、广播与类型提升这两条"隐形规则"如何运行。后面每一章都会反复回链到这里——尤其是 *broadcast* 和 *类型提升*。如果你只能读一章，读这章。

---

## 0.1 张量是什么：`(storage, offset, shape, stride)` 四元组

一个 PyTorch 张量，本质上不是"一块数据"，而是对一段**一维连续存储 (storage)** 的一种**观察方式 (view)**。它由四元组唯一决定：

```
tensor = ( storage   : 一维连续内存（真正的数据）
         , offset    : 起始偏移（从 storage 第几个元素开始读）
         , shape     : 每个维度的长度          [d0, d1, ..., dn-1]
         , stride    : 沿每个维度前进 1 时，storage 下标增加多少 [s0, s1, ..., sn-1]
         )
```

**关键洞察**：`shape` 和 `stride` 只是"如何从一维 storage 里取数"的坐标变换。任意一个多维下标 `(i0, i1, ..., in-1)` 映射到 storage 的下标就是：

```
storage_index = offset + i0*s0 + i1*s1 + ... + in-1*sn-1
```

这一条公式是**第 7 章（形状与视图）所有算子的总根**。`view` / `reshape` / `permute` / `transpose` / `slice` / `expand` / `as_strided` 做的事情，本质上都只是修改 `offset / shape / stride` 这三个**元数据**，而不动 `storage`——这就是为什么它们能"零拷贝"。

> **为什么要这样设计？** 因为"改元数据"比"搬数据"便宜几个数量级。一个 `[1024, 1024]` 矩阵转置，如果真去搬内存是 O(N)；而 `transpose` 只是把 stride 从 `[1024,1]` 改成 `[1,1024]`，O(1)。代价是：转置后的张量**不再连续**（见 0.3），某些算子会因此变慢或需要先 `contiguous()`。

### 形状/stride 查询 API（`sym_*`）

core aten 提供 4 个"只读查询"算子，它们**不计算、不搬数据**，只把上述元数据吐出来：

| 算子 | 返回 | 用途 |
|---|---|---|
| `aten.sym_size(input, dim)` | 该维长度 | 等价 `t.size(dim)` |
| `aten.sym_stride(input, dim)` | 该维 stride | 等价 `t.stride(dim)` |
| `aten.sym_numel(input)` | 元素总数 = ∏shape | 等价 `t.numel()` |
| `aten.sym_storage_offset(input)` | 起始偏移 | 内部/编译器用得多 |

外加 `aten._local_scalar_dense`：把一个 0 维（标量）张量取出成 Python 标量。这 5 个都是"从张量里**抽取信息**"而非"计算"，属于操作类的轻量末端。

---

## 0.2 数据类型（dtype）与设备（device）

- **dtype**：每个元素占几位、如何解释（`float32` / `float16` / `bfloat16` / `int64` / `bool` / ...）。它决定**精度**与**内存占用**。`storage` 里存的只是裸比特，dtype 是解读规则。
- **device**：这段 storage 在哪块硬件上（`cpu` / `npu:0` / `cuda:0`）。跨设备运算需要先搬运（`_to_copy` / `to`，见第 12 章）。

本书绝大多数算子的"功能"与 dtype/device 无关（数学是数学），但**精度**与**性能**强相关——这也是为什么第 5 章矩阵乘、第 15 章归一化要专门讲数值稳定性。

---

## 0.3 连续性（contiguity）

一个张量"连续 (contiguous)"，指它的 stride 恰好满足**行主序 (row-major / C-order)**：从最后一维往前，`stride[k] = stride[k+1] * shape[k+1]`，且 `stride[最后一维]=1`。

- `torch.zeros(2,3)` → shape `[2,3]`, stride `[3,1]` → **连续**。
- 对它做 `.t()`（转置）→ shape `[3,2]`, stride `[1,3]` → **不连续**（行主序要求最后一维 stride=1，现在是 3）。

**为什么连续性重要**：
1. 很多算子在连续张量上有专门的快速路径（按内存顺序线性扫描，缓存友好）。
2. `view`（第 7 章）**只接受连续张量**——因为它假设改 shape 时能直接套行主序 stride，不连续时会报错；这时要用 `reshape`（复合，内部视情况 `view` 或 `contiguous()+view`）。
3. `contiguous()`（第 12 章）会**真正复制**一份数据，把不连续张量重排成连续——这是"花钱买简单"。

> 一句话：**`view` 要求连续且零拷贝；`reshape` 不要求连续、可能拷贝；`contiguous()` 必拷贝。** 三者关系见第 7 章。

---

## 0.4 广播（Broadcasting）——最常被误解的"隐形规则"

**广播不是算子，是一条规则**：当两个形状不同的张量做逐元素运算（加减乘除、比较…）时，如何把它们"对齐"成可计算的形状，而**不真正复制数据**。

### 规则（三步）

1. 从**最右**（最低维）开始对齐两边的 shape，短的在左边补 1。
2. 某维上，若两者相同，取该值；若其中一个为 1，扩展成另一个；**若都非 1 且不相同，报错**。
3. 广播的实质是：被扩展的维度上 **stride = 0**——读到的永远是同一个元素，**零分配**。

### 例子

| A shape | B shape | 对齐后 | 合法？ |
|---|---|---|---|
| `[3]` | `[2,3]` | `[2,3]` | ✓（A 左补 1 → `[1,3]` → 广播成 `[2,3]`） |
| `[4,1]` | `[1,5]` | `[4,5]` | ✓（双向广播） |
| `[2,3]` | `[3,2]` | — | ✗（最后一维 3≠2 且都非 1） |

### 与算子的关系

- "广播"作为**规则**，内嵌在所有逐元素算子和某些规约里（见第 1/3/4 章）。
- "广播"作为**显式动作**有两个算子：`expand`（零拷贝，靠 stride=0）和 `broadcast_to`（第 7 章）。**`expand` 不分配内存、`repeat` 分配内存**——这是第 7 章的重点对比。
- 伪代码（广播下的逐元素加法）：

  ```python
  def add_broadcast(A, B):
      out_shape = broadcast(A.shape, B.shape)          # 规则对齐
      out = empty(out_shape)
      for idx in all_indices(out_shape):
          out[idx] = A[broadcast_idx(idx, A.shape)]    # 越界维度靠 stride=0 取同一元素
                + B[broadcast_idx(idx, B.shape)]
      return out
  ```

> **inductor 视角**：广播在编译期被静态求解成 stride，pointwise 算子生成的 Triton 核直接用这些 stride 寻址，运行期无额外开销。所以"广播"对 inductor 开发者基本是**编译期元数据问题**，不是运行期问题。

---

## 0.5 类型提升（Type Promotion）——另一条"隐形规则"

两个不同 dtype 的张量运算时，结果用哪种 dtype？规则可简化为两层：

1. **类别优先级**：`bool < 整数 < 浮点 < 复数`（类别高的吃掉低的）。
2. **同类内**：取能容纳两者的那个（`int32` + `int64` → `int64`；`float16` + `float32` → `float32`）。

| 左 \ 右 | bool | int64 | float32 | float64 |
|---|---|---|---|---|
| bool | bool | int64 | float32 | float64 |
| int64 | int64 | int64 | float32 | float64 |
| float32 | float32 | float32 | float32 | float64 |

经典坑：
- `int64` 张量 `/` `int64` 张量 → **不会**变成浮点！`aten.div` 有 `rounding_mode`，默认"真除"会先提升到浮点，但 `aten.floor_divide`/`aten.remainder` 留在整数域——这是第 1 章 `div`/`fmod`/`remainder` 详解的动因。
- `aten.pow` 的指数是负数/分数时，整数底会被提升到浮点，否则报错或截断。
- `scatter_reduce(..., accumulate=True)` 的累加发生在**提升后的 dtype** 上，常导致溢出（见第 9 章）。

> **inductor 视角**：类型提升在 tracing 时就被求解成显式的 `prims.convert_element_type`（dtype cast）插入，pointwise 核里的 dtype 因此是确定的。但**数值精度**（尤其是 bfloat16 的累加）是 inductor 在 NPU 上反复踩坑的地方，见第 4/5/15 章。

---

## 0.6 全书的核心分类轴：位置保持 vs 位置变化

读后面所有章节前，请记住这条判据——它同时决定了**算子的难度**、**是否产生别名 (aliasing)**、以及 **inductor 如何降级它**：

| | **位置保持 (position-preserving)** | **位置变化 (position-changing)** |
|---|---|---|
| 定义 | `out[i]` 读哪些输入地址，由**仅依赖 shape/stride/dim 的闭式表达式**决定（**不含任何输入元素的值**），且**不重解释存储** | `out[i]` 读的地址**取决于某个输入的值**（如 `gather` 的 `index`），或用新 `(offset,stride)` **重解释同一段存储**（视图类） |
| 形状 | 平凡可推（同形或广播） | 形状/stride 逻辑才是主角 |
| 数学 | 数学公式本身是全部内容 | 数学通常平凡（拷贝/比较/恒等），**结构变换**才是内容 |
| 别名风险 | 无（产出新张量） | 高（视图类可能共享 storage） |
| inductor 归口 | pointwise / reduction / template(mm) | 多为 fallback 或 scatter/gather |
| 对应本书 | **Part I**（数学语义类）+ Part III（计算原语） | **Part II**（张量操作/结构类，全书重心） |

> **两个容易混的轴，务必分开**（这是理解"位置保持"的关键）：
>
> 1. **基数轴（cardinality，正交于位置轴）**——输出元素个数相对输入：1:1（逐元素）/ N:1（规约）/ 1:N（广播、`expand`）。**这三种全部是位置保持**，只是形状变化方向不同。"输出变小"（规约的 N:1）**绝不**意味着位置变化。
> 2. **位置轴（即上表）**——`out[i]` 的地址映射规不规律、会不会别名。
>
> **判别口诀**：把 `out[i]` 要读的每个输入地址写成下标表达式。若它是**闭式仿射、且不含任何输入"值"**（逐元素 `in[i]`、广播的 stride=0、规约的正则块 `in[i,k]` 都满足）→ **位置保持**。若地址里出现了**输入的值**（`gather`/`scatter`/`index` 的 `index[...]`），或用一组**任意新 stride 重解释同一段 storage**（`as_strided`/`view`/`permute`）→ **位置变化**。
>
> **杀手级对照**：`argmax` 的**输出**是 index，但那 index 是被**算出来**的结果，地址映射仍是"扫 `in[i,:]` 这个正则块"→ **位置保持**（它是规约）；`gather` 把 index 当**输入**去**驱动寻址** → **位置变化**。判别不看"index 出现的位置"，只看"index 是结果、还是驱动地址的源"。

**为什么 Part II 是重心**（用户要求 5）：位置保持算子就是"数学公式翻译成代码"，会公式就会用；位置变化算子（`as_strided`、`gather`、`scatter`、`expand`、`sort`…）的**功能**和**存在的理由**很难从直觉猜出来——"它到底在干什么？为什么模型需要它？" 这正是本书要重点回答的。

---

## 0.7 一处必须说明的"契约裂缝"：SSA、`resize_`、`alias`

Core ATen 整体遵循 **SSA（静态单赋值）/函数式**契约：算子不修改输入，只产出新张量。这让编译器（inductor）可以自由地重排、融合、消除中间量。

但 core 里有两个**例外**，是低层的"逃生舱"：

- **`aten.resize_(input, shape)`**：原地改变张量形状。它**违反**函数式契约（修改输入）。存在是因为某些形状推导的底层管道需要它；functionalization（函数化 pass）通常会把 `resize_` 改写掉，使编译图恢复纯函数式。
- **`aten.alias(input)`**：恒等返回输入（零操作）。存在是 SSA 框架里"给一个张量起别名/占位"的概念锚点，不搬数据。

请把它们理解为**框架内部管道用的低层算子**，普通模型代码很少直接写它们。本书在第 12 章简要带过，其余章节默认"算子是纯函数、产出新张量"——除了第 7 章视图类算子会**别名共享 storage**（只读共享，不修改）。

---

## 小结

读到这里，你应该已经掌握：

1. 张量 = 对一维 storage 的 `(offset, shape, stride)` 观察；**改元数据 = 零拷贝**。
2. 广播（stride=0 的隐形对齐）和类型提升（类别优先级）是两条贯穿全书的隐形规则。
3. 全书按 **位置保持（数学类）/ 位置变化（操作类）** 分两大轴，Part II 是重心。
4. core aten 整体是函数式的，只有 `resize_`/`alias` 是逃生舱例外。

接下来：[第 1 章 逐元素算术](01-elementwise-arithmetic.md)，或先看 [README 索引与全书地图](README.md)。
