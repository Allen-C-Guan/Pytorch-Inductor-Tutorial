# Chapter 1 附录

> 本文档为《Triton Kernel 全流程揭秘：从编译到运行》的附属资料，按章节索引，持续补充。

---

<a id="编译各阶段-ir-产物详解"></a>

## 编译各阶段 IR 产物详解

配合正文「Compile 层级」的五阶段编译流程，以同一个 vector add kernel 为例，系统展示 Python 源码 → Python AST → TTIR → TTGIR → LLIR → PTX → cubin 每个阶段的产物。

### 完整管线总览

```
Python 源码 → Python AST → TTIR → TTGIR → LLIR → PTX → cubin
```

### 起点：Python 源码

以一个简单的向量加法 kernel 作为贯穿示例：

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### Stage 0: Python AST

`ast.parse(src)` 之后得到标准 Python 抽象语法树，尚未进入 Triton 领域。`CodeGenerator`（`ast.NodeVisitor` 子类）将遍历这些节点并逐个翻译为 TTIR。

```python
Module(
  body=[
    FunctionDef(
      name='add_kernel',
      args=arguments(
        args=[
          arg(arg='x_ptr'),
          arg(arg='y_ptr'),
          arg(arg='output_ptr'),
          arg(arg='n_elements'),
          arg(arg='BLOCK_SIZE', annotation=Attribute(value=Name(id='tl'), attr='constexpr')),
        ],
        ...
      ),
      body=[
        Assign(targets=[Name(id='pid')], value=Call(func=Attribute(value=Name(id='tl'), attr='program_id'), keywords=[keyword(arg='axis', value=Constant(value=0))])),
        Assign(targets=[Name(id='block_start')], value=BinOp(left=Name(id='pid'), op=Mult(), right=Name(id='BLOCK_SIZE'))),
        Assign(targets=[Name(id='offsets')], value=BinOp(left=Name(id='block_start'), op=Add(), right=Call(func=Attribute(value=Name(id='tl'), attr='arange'))))),
        Assign(targets=[Name(id='mask')], value=Compare(left=Name(id='offsets'), ops=[Lt()], comparators=[Name(id='n_elements')])),
        Assign(targets=[Name(id='x')], value=Call(func=Attribute(value=Name(id='tl'), attr='load'))),
        Assign(targets=[Name(id='y')], value=Call(func=Attribute(value=Name(id='tl'), attr='load'))),
        Assign(targets=[Name(id='output')], value=BinOp(left=Name(id='x'), op=Add(), right=Name(id='y'))),
        Expr(value=Call(func=Attribute(value=Name(id='tl'), attr='store'))),
      ],
    )
  ]
)
```

**特征：** 全是纯 Python 结构——`Assign`、`BinOp`、`Call`、`Compare`。所有 Triton 语义（`tl.load`、`tl.program_id` 等）仍以 `Call(func=Attribute(...))` 的 AST 节点形式存在，尚未解析。

### Stage 1: TTIR（Triton IR）

`ast_to_ttir()` 的输出。**算子级 IR**，与硬件无关。所有类型都是 `tensor<N x dtype>`，有 `tt.load`/`tt.store`/`tt.dot` 等高级算子，但**没有线程/block/warp 等 GPU 硬件概念**。

```mlir
module {
  tt.func public @add_kernel(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg3: i32
  ) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32                    // pid = tl.program_id(0)
    %1 = arith.muli %0, %c1024_i32 : i32               // block_start = pid * BLOCK_SIZE
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>  // tl.arange(...)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>         // 广播标量→张量
    %4 = arith.addi %3, %2 : tensor<1024xi32>          // offsets = block_start + arange
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>      // 广播 n_elements
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>     // mask = offsets < n_elements
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>  // x_ptr + offsets
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>    // tl.load(..., mask=mask)
    // ...对称的 y_ptr load...
    %13 = arith.addf %9, %12 : tensor<1024xf32>        // output = x + y
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>  // tl.store(..., mask=mask)
    tt.return
  }
}
```

**关键翻译对照：**

| Python / Triton DSL | TTIR |
|---------------------|------|
| `tl.program_id(0)` | `tt.get_program_id x : i32` |
| `tl.arange(0, 1024)` | `tt.make_range {start=0, end=1024} : tensor<1024xi32>` |
| `pid * BLOCK_SIZE` | `arith.muli %pid, %c1024_i32` |
| `offsets < n_elements` | `arith.cmpi slt, %4, %5` |
| `tl.load(ptr, mask=mask)` | `tt.load %ptr, %mask` |
| `tl.store(ptr, val, mask=mask)` | `tt.store %ptr, %val, %mask` |
| `x + y` | `arith.addf %x, %y` |
| scalar → broadcast | `tt.splat %scalar -> tensor<...>` |
| ptr + offset | `tt.addptr %ptr, %offset` |

### Stage 2: TTGIR（Triton GPU IR）

`make_ttgir()` 的输出。在 TTIR 的基础上引入 **GPU 硬件抽象**：数据分布 layout（`#ttg.blocked`）、warp 数、CTA 数等。算子结构几乎不变，但每个 `tensor` 类型多了一个 layout 标注。

```mlir
#blocked = #ttg.blocked<{          // ← 数据分布策略：
  sizePerThread = [4],              //    每个线程算 4 个元素
  threadsPerWarp = [32],            //    每个 warp 32 线程
  warpsPerCTA = [4],                //    每个 CTA 4 个 warp
  order = [0]                       //    沿 dim=0 离散分布
}>

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,       // ← 硬件映射参数
  ttg.target = "cuda:90",
  "ttg.threads-per-warp" = 32 : i32
} {
  tt.func public @add_kernel(...) {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32}
      : tensor<1024xi32, #blocked>   // ← 每个 tensor 都带上了 #blocked layout
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32, #blocked>
    // ... load/store 全部标注了 tensor<...x..., #blocked> ...
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
```

**TTIR → TTGIR 的核心变化：**

| 变化点 | 说明 |
|--------|------|
| `#ttg.blocked<...>` | 每个 tensor 增加了数据分布描述——规定了 1024 个元素如何分散到 4×32=128 条物理线程上 |
| `ttg.num-warps` / `ttg.num-ctas` | module 属性新增 GPU 硬件配置参数 |
| `ttg.target` | 标记目标硬件架构（如 `cuda:90`） |
| `ttg.*` ops | 引入 GPU 专有算子：`ttg.barrier`（同步）、`ttg.warp_specialize`（warp 特化）等 |

### Stage 3: LLIR（MLIR LLVM Dialect）

`make_llir()` 的输出。高级算子（`tt.load`、`arith.addf` 等）被 lowered 成 LLVM 风格的指令——指针操作、向量化、地址计算全部显式化。

```llvm
llvm.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>,
                       %arg2: !llvm.ptr<1>, %arg3: i32) {
  %c9_i32 = llvm.mlir.constant(9 : i32) : i32                      // 常量
  %pid = nvvm.read.ptx.sreg.ctaid.x : i32                          // cta id（硬件寄存器读）
  %block_start = llvm.mul %pid, %c9_i32 : i32                      // 整数乘法
  %offsets = llvm.add %block_start, %c9_i32 : i32                  // 整数加法
  %mask = llvm.icmp "slt" %offsets, %arg3 : i32                    // 比较
  %x_ptr = llvm.getelementptr %arg0[%block_start]                  // 指针偏移
             : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
  %x_vec = llvm.load %x_ptr : !llvm.ptr<1> -> vector<4xf32>        // 向量化加载
  %x_scalar = llvm.extractelement %x_vec[%c0_i32 : i32]
             : vector<4xf32>                                        // 从向量中提取标量
  %y_scalar = ...                                                   // 对称
  %output = llvm.fadd %x_scalar, %y_scalar : f32                   // 浮点加
  llvm.store %output_val, %out_ptr : !llvm.ptr<1>, f32             // 存储
  llvm.return
}
```

**TTGIR → LLIR 的核心变化：**

| 变化点 | 说明 |
|--------|------|
| `tensor<1024xf32, #blocked>` → `vector<4xf32>` | 按 blocked layout 拆成每线程的向量 |
| `tt.load` / `tt.store` → `llvm.load` / `llvm.store` | 带上显式的 `getelementptr` 地址计算 |
| `tt.get_program_id` → `nvvm.read.ptx.sreg.ctaid.x` | 从 IR 算子降为直接读 GPU 硬件寄存器 |
| `#blocked` layout 消失 | 数据分布已在 lowering 过程中"物理化"为向量操作 |

### Stage 4: PTX（Parallel Thread Execution）

`make_ptx()` 的输出。MLIR LLVM dialect 翻译为 NVIDIA GPU 的**中间汇编语言**（文本形式，非二进制）。

```ptx
.visible .entry add_kernel(
    .param .u64 add_kernel_param_0,     // x_ptr
    .param .u64 add_kernel_param_1,     // y_ptr
    .param .u64 add_kernel_param_2,     // output_ptr
    .param .u32 add_kernel_param_3      // n_elements
) {
    .reg .pred  %p<4>;                  // 谓词寄存器
    .reg .b32   %r<20>;                 // 32-bit 通用寄存器
    .reg .f32   %f<8>;                  // 32-bit 浮点寄存器
    .reg .b64   %rd<8>;                 // 64-bit 地址寄存器

    ld.param.u64    %rd1, [add_kernel_param_0];   // 加载参数
    ld.param.u64    %rd2, [add_kernel_param_1];
    ld.param.u64    %rd3, [add_kernel_param_2];
    ld.param.u32    %r1, [add_kernel_param_3];

    mov.u32     %r2, %tid.x;            // 线程 id
    mov.u32     %r3, %ctaid.x;          // block id
    mov.u32     %r4, %ntid.x;           // block 内线程数

    mad.lo.s32  %r5, %r3, %r4, %r2;    // 全局线程索引
    setp.lt.s32 %p1, %r5, %r1;         // mask = tid < n_elements

    shl.b32     %r6, %r5, 2;           // 偏移量 = tid * 4 (sizeof float)
    add.s64     %rd4, %rd1, %r6;       // x_addr = x_ptr + offset
    add.s64     %rd5, %rd2, %r6;       // y_addr = y_ptr + offset
    add.s64     %rd6, %rd3, %r6;       // out_addr = output_ptr + offset

    ld.global.f32   %f1, [%rd4], %p1;  // 条件加载 x
    ld.global.f32   %f2, [%rd5], %p1;  // 条件加载 y
    add.f32         %f3, %f1, %f2;     // output = x + y
    st.global.f32   [%rd6], %f3, %p1;  // 条件存储

    ret;
}
```

**LLIR → PTX 的核心变化：**

| 变化点 | 说明 |
|--------|------|
| `llvm.func` → `.visible .entry` | PTX 入口函数声明 |
| `llvm.load` → `ld.global.f32` | 全局内存加载指令 |
| `llvm.store` → `st.global.f32` | 全局内存存储指令 |
| `llvm.fadd` → `add.f32` | 浮点加法指令 |
| `llvm.icmp` → `setp.lt.s32` | 比较 → 谓词寄存器 |
| mask → `%p1`（谓词化执行） | `ld.global.f32 ..., %p1` 表示条件执行 |
| 引入寄存器声明 | `.reg .pred`, `.reg .b32`, `.reg .f32`, `.reg .b64` |

### Stage 5: cubin（CUDA Binary）

`make_cubin()` 的输出。PTX 通过 NVIDIA 的 `ptxas` 汇编器翻译为 GPU 可执行的**二进制代码**（ELF 格式），无法直接阅读：

```
\x7fELF\x02\x01\x01\x03\x00\x00\x00...（二进制数据流）
```

可通过 `cuobjdump -sass` 反汇编为 SASS（机器码级指令）查看最终硬件指令。

### 总结对照表

| 阶段 | 产物 | 产出函数 | 核心特征 |
|------|------|----------|----------|
| **Python 源码** | `str` | 用户手写 | `@triton.jit`, `tl.load`, `tl.program_id` |
| **Python AST** | `ast.AST` | `ast.parse(src)` | `FunctionDef`, `Assign`, `Call` —— 纯 Python 树 |
| **TTIR** | MLIR text | `ast_to_ttir()` | `tt.load`, `tt.store`, `tensor<N x dtype>` —— 硬件无关算子 IR |
| **TTGIR** | MLIR text | `make_ttgir()` | 同上 + `#ttg.blocked<...>` layout + `ttg.num-warps` —— 加入 GPU 数据分布 |
| **LLIR** | MLIR text | `make_llir()` | `llvm.load`, `llvm.getelementptr`, `vector<4xf32>` —— 显式指针/向量 |
| **PTX** | text | `make_ptx()` | `ld.global.f32`, `setp.lt.s32`, `mad.lo.s32` —— NVIDIA GPU 汇编 |
| **cubin** | bytes | `make_cubin()` | ELF 二进制 —— 直接喂给 CUDA driver 执行 |

---

<a id="stages-抽象与-stage-注册语义解析"></a>

## stages 抽象与 stage 注册语义解析

### stages 是什么抽象？

先看抽象基类的契约（`triton/python/triton/backends/compiler.py`，`add_stages` 方法的 docstring）：

```
Populates `stages` dictionary with entries of the form:
ir_name [str] => Function[(src: str, metadata: dict) -> str|bytes]
Stages will be run sequentially (in insertion order) and can
communicate using `metadata`.
All stages are expected to return a `str` object, except for the
last stage which returns a `bytes` object for execution by the launcher.
```

`stages` 是一个 **`dict[str, Callable[[str, dict], str|bytes]]`** ——从 IR 扩展名到编译函数的映射。它在 `add_stages()` 调用前是空 `dict()`，调用后被后端填入编译管线。

它的抽象含义是：**有序的编译阶段注册表**。每个 key 是阶段的输出文件名后缀（如 `"ttir"`），value 是执行该阶段转换的可调用对象。Python dict 自 3.7 起保证插入顺序，因此 `stages` 天然就是有序的 pipeline。

消费端在 `triton/python/triton/compiler/compiler.py` 的 `compile()` 函数中：

```
for ext, compile_ir in list(stages.items())[first_stage:]:
    next_module = compile_ir(module, metadata)   // 执行这一阶段的转换
    module = next_module                          // 输出变下一阶段的输入
```

每一轮迭代：当前 IR (`module`) 被 `compile_ir` 转换 → 产生下一级 IR → 赋值给 `module` → 进入下一轮。形成一条链：`ttir → ttgir → llir → ptx → cubin`。

---

### `stages["..."] =` 是什么语义？

**注册/安装一个编译阶段。**

```
stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options)
```

这一行的语义是：**往编译管线里插入一个名为 `"llir"` 的阶段，该阶段的转换逻辑由 `self.make_llir` 完成**。

- `"llir"` 是阶段名，也是输出文件的后缀（`.llir`），用于缓存文件的命名和 `first_stage` 的定位
- `lambda src, metadata: ...` 是该阶段的转换函数，签名必须匹配 `(str, dict) -> str|bytes`

---

### `self.make_so(src, metadata, options)` 的功能是什么？

这是 CPU 后端**最后一个阶段**的转换函数。类比 AMD 后端的 `make_hsaco`：

```
def make_hsaco(src, metadata, options):
    hsaco = amd.assemble_amdgcn(src, options.arch, target_features)
    // ...链接...
    return ret  // bytes
```

`make_so` 的功能：**接收 LLVM IR 文本（或汇编文本），调用 LLVM 工具链将其编译为 `.so` 共享库的二进制 bytes**。这是从文本 IR 到可执行二进制码的最后一步——之后这个 bytes 会通过 `ctypes.cdll.LoadLibrary` 加载为 Python 可调用的函数。

注意抽象方法的注释：*"All stages are expected to return a `str` object, except for the last stage which returns a `bytes` object"* —— `make_so` 返回的就是 `bytes`。
