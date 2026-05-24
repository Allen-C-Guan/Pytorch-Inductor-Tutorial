# Chapter 3 附录

> 本文档为《Triton Compiler Core：Dialect 与 Pass Pipeline》的附属资料，按章节索引，持续补充。

---

## CoreIR 详解：Triton IR 到标准 MLIR 方言的中间驿站

### 它是什么

"CoreIR" 不是 Triton 源码中一个正式命名的 IR，而是对编译管线中某一阶段的称呼——指的是 TTIR 经过 `_ttir_to_coreir` 转换后、尚未进行硬件特化 lowering 之前的那个"中间态 IR"。

从 `_ttir_to_coreir` 的代码可以看出，它做了一件事：

```
tt.dot, tt.reduce, tt.load, tt.addptr ...
          │
          │ --triton-to-core-dialects
          ▼
linalg.matmul, linalg.generic, scf.for, arith.addf, memref.load ...
```

即把 Triton 自定义方言的操作（`tt.xxx`）**全部替换为标准 MLIR 方言**——主要是 `linalg`（线性代数）、`arith`（算术）、`scf`（结构化控制流）、`tensor`/`memref`（内存抽象）。

命名逻辑：这些标准 MLIR 方言（linalg、scf、arith、memref）是 MLIR 生态的"**core** dialects"，所以转换结果被称为"CoreIR"。

### 一个具体的例子

以文章中的 add_kernel 为例。这是输入 **TTIR**：

```
// TTIR: Triton 自定义语义
%2 = tt.make_range {end = 1024, start = 0} : tensor<1024xi32>
%9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
%13 = arith.addf %9, %12 : tensor<1024xf32>
tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
```

经过 `_ttir_to_coreir` 后会变成类似这样的 **CoreIR**：

```
// CoreIR: 纯标准 MLIR 方言，无 tt.* 操作
scf.for %i = %c0 to %c1024 step %c1 {
  %cond = arith.cmpi slt, %i, %arg3 : i32
  %ptr = memref.load %base[%i] : memref<1024x!llvm.ptr>
  %val = memref.load %ptr[] : memref<f32>       // was: tt.load
  %result = arith.addf %val, %other : f32
  // ...mask 变成了 scf.if 分支
  memref.store %result, %out_ptr[] : memref<f32> // was: tt.store
}
```

关键变化：

- `tt.make_range` + `tt.splat` + `tt.addptr`（pointer arithmetic with mask）→ 展开为显式 `scf.for` 循环
- `tt.load` / `tt.store` → `memref.load` / `memref.store`
- `tt.get_program_id` → 留在 TTIR 到 TTGIR 层处理，或者变成循环的 iv 边界

### CoreIR 的主要功能：Triton 世界与标准 MLIR 世界之间的桥

| | TTIR | CoreIR | LLVM IR |
|---|---|---|---|
| 方言 | `tt.*` 自定义 | `linalg`, `scf`, `arith`, `memref` | `llvm.*` |
| 抽象层级 | 张量+块并行 | 循环+缓冲区 | 基本块+寄存器 |
| 可做的优化 | Triton 专属 | **复用 MLIR 标准 Pass** | LLVM 后端 Pass |

CoreIR 存在的意义是**让 Triton 编译器能吃上 MLIR 生态的"标准餐"**——`linalg-tiling`、`one-shot-bufferize`、`cse`、`canonicalize` 这些 Pass 是 MLIR 社区维护的通用优化，任何降到 CoreIR 级别的编译器都能直接使用。如果 Triton 直接从 TTIR 跳到 LLVM IR，这些通用优化要么无法执行，要么需要 Triton 自己重复实现一套。

### TTIR → CoreIR → LLVM IR 的完整 lowering 链

```
TTIR (tt.*)
  │
  │ _ttir_to_coreir: --triton-to-core-dialects
  │                  将 tt.dot/tt.load/tt.reduce 等映射到 linalg/scf/arith/memref
  ▼
CoreIR (linalg.*, scf.*, arith.*, memref.*)
  │
  │ 通用优化: --linalg-tiling, --one-shot-bufferize, --cse, --canonicalize
  │
  ▼
CoreIR (已优化)
  │
  │ _coreir_to_llir: --convert-linalg-to-affine-loops → --lower-affine
  │                  → --convert-scf-to-cf → --convert-*-to-llvm
  ▼
LLVM IR (llvm.*)
```

到 `_coreir_to_llir` 阶段，CoreIR 再经过 `--convert-linalg-to-affine-loops` → `--lower-affine` → `--convert-scf-to-cf` → `--convert-*-to-llvm` 这一串 Pass，最终生成 LLVM IR。

**简言之：CoreIR = TTIR 丢掉了 Triton 自定义语义后、穿上标准 MLIR 方言外衣的中间形态。它是一个"标准 MLIR 化"的驿站，让 Triton 可以合法地复用 MLIR 社区的全部优化基础设施。**

---

## MLIR 是如何被使用的：从 IR 到 Pass 替换的完整链路

### 起点：输入的 IR

```
func.func @main(%a: i32, %b: i32) -> i32 {
  %0 = "calc.add"(%a, %b) : (i32, i32) -> i32
  return %0 : i32
}
```

内存中是一棵 Operation 树：

```
ModuleOp
 └── FuncOp (@main)
      └── Block
           ├── %a: i32  (BlockArgument)
           ├── %b: i32  (BlockArgument)
           ├── AddOp  (%0 = calc.add %a, %b)   ← 这就是 AddOp 实例
           └── func::ReturnOp
```

`AddOp` 实例是 `mlir::Operation` 的匿名子类，存有：
- 2 个 operands：指向 `%a` 和 `%b` 的 `Value` 句柄
- 1 个 result：`%0`，类型为 i32
- operation name 字段：`"calc.add"` 字符串

### 第一步：Pass 启动，框架遍历 IR

当你运行 `triton-opt --convert-calculator-to-arith input.mlir`，MLIR Pass 框架做以下事情：

```
PassManager::run(ModuleOp)
  │
  └── ConvertCalculatorToArithPass::runOnOperation()
        │
        ├── ConversionTarget::addLegalDialect<arith::ArithDialect>()
        │     "转换完成后，arith.addi 是合法的"
        │
        ├── ConversionTarget::addIllegalDialect<CalculatorDialect>()
        │     "转换完成后，calc.* 必须全部消失"
        │
        ├── RewritePatternSet::add<AddOpConversion, SubOpConversion, ...>()
        │     注册四个转换规则
        │
        └── applyPartialConversion(module, target, patterns)
              │
              └── 遍历 module 中的每一个 Operation：
                    │
                    │ 当遍历到 AddOp 时：
                    ▼
```

### 第二步：模式匹配——"这个 Op 我认识"

框架对每个 Operation 依次询问所有注册的 Pattern：

```cpp
// applyPartialConversion 内部逻辑（简化）：
for (Operation &op : module.getOps()) {
    for (auto &pattern : patterns) {
        // 尝试将 op 匹配到 pattern 的模板参数 <AddOp>
        if (auto addOp = dyn_cast<AddOp>(op)) {
            // ↑ dyn_cast<AddOp> 检查 operation name == "calc.add"
            // 匹配成功！得到类型安全的 AddOp 句柄
            LogicalResult result = pattern->matchAndRewrite(addOp, rewriter);
            if (succeeded(result)) break;  // 已替换，处理下一个 op
        }
    }
}
```

这里的 `dyn_cast<AddOp>` 不依赖 RTTI，而是检查 Operation 内部存储的 `operation name` 字符串和 dialect 注册表——这是 MLIR 自己的高效类型分发机制，一切由 TableGen 自动生成的注册代码支撑。

### 第三步：matchAndRewrite——替换发生

匹配到 AddOp 后，调用 `AddOpConversion::matchAndRewrite`：

```cpp
LogicalResult AddOpConversion::matchAndRewrite(
    AddOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {

    // 1. 通过自动生成的 getter 取出 operands（不需要知道 index）
    Value lhs = adaptor.getLhs();   // → %a
    Value rhs = adaptor.getRhs();   // → %b

    // 2. 创建目标方言的等价操作，替换原 op
    rewriter.replaceOpWithNewOp<arith::AddIOp>(
        op,      // 要被替换的 calc.add
        lhs,     // 操作数原样传递
        rhs
    );

    return success();
}
```

`replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs)` 做三件事：

1. **创建** `arith::AddIOp`（同样是 .td 生成的类，它的 `build()` 自动推导 result type = lhs.getType() = i32）
2. **替换所有使用** `%0`（原 calc.add 的 result）的地方，全部指向新的 arith.addi 的 result。这里 `return %0` 的 operands 自动更新
3. **删除** 旧的 AddOp

### 终点：输出 IR

```
func.func @main(%a: i32, %b: i32) -> i32 {
  %0 = arith.addi %a, %b : i32
  return %0 : i32
}
```

内存中的树变化：

```
Before:                     After:
AddOp (%0 = calc.add)       AddIOp (%0 = arith.addi)
  operand[0] → %a             operand[0] → %a
  operand[1] → %b             operand[1] → %b
  result[0] → %0              result[0] → %0
                              ↑ ReturnOp 的 operands 自动重新指向
```

### 完整链路一张图

```
input.mlir                    addOp::matchAndRewrite           output.mlir
──────────────────────────────────────────────────────────────────────────
"calc.add"(%a,%b)  ──匹配──▶  AddOpConversion  ──重写──▶  arith.addi %a,%b
      │                            │                              │
      │  operation name            │  adaptor.getLhs() → %a       │  operation name
      │  == "calc.add"             │  adaptor.getRhs() → %b       │  == "arith.addi"
      │                            │                              │
      │  自动生成的访问器           │  手写的重写逻辑               │  自动生成的 build()
      │  getLhs()/getRhs()        │  replaceOpWithNewOp()        │  自动推导 result type
      └───────────────────────────┴──────────────────────────────┘
                .td 生成                                  .td 生成
```

**关键洞察：开头和结尾的 Op 都是 .td 生成的代码，中间手写的 matchAndRewrite 只有 3 行**。这就是 MLIR 的工程效率——定义 IR 有 TableGen 生成样板代码，写 Pass 只需要写"匹配什么、替换成什么"的核心逻辑。
