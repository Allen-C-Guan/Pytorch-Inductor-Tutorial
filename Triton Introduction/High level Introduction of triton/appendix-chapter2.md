# Chapter 2 附录

> 本文档为《Triton 编译器架构全景与多后端适配》的附属资料，按章节索引，持续补充。

---

## MLIR 多层 IR 实例：同一个 GEMV 在不同抽象层级的表示

### 第一层：Linalg — 最高抽象，硬件无关

```
func.func @gemv(%A: memref<64x128xf32>, %x: memref<128xf32>, %y: memref<64xf32>) {
  linalg.matvec
    ins(%A, %x : memref<64x128xf32>, memref<128xf32>)
    outs(%y : memref<64xf32>)
  return
}
```

这一层只说"这里有一个矩阵乘向量运算"，**不规定循环顺序、不规定内存布局、不规定并行策略**。编译器可以在此层做：

- **算子融合**：如果 `y` 后面接了 `linalg.add`，可以直接合并为一个 `linalg.matvec + add` 的通用收缩操作
- **布局推断**：根据 `A` 的数据布局自动决定遍历顺序

### 第二层：Affine — 显式循环 + 可分析的访存模式

第一层 lowering 后，`matvec` 被展开为带 affine access pattern 的循环：

```
func.func @gemv(%A: memref<64x128xf32>, %x: memref<128xf32>, %y: memref<64xf32>) {
  affine.for %i = 0 to 64 {
    %sum = affine.for %j = 0 to 128 iter_args(%acc = 0.0f32) -> f32 {
      %a = affine.load %A[%i, %j] : memref<64x128xf32>
      %xj = affine.load %x[%j] : memref<128xf32>
      %mul = arith.mulf %a, %xj : f32
      %new_acc = arith.addf %acc, %mul : f32
      affine.yield %new_acc : f32
    }
    affine.store %sum, %y[%i] : memref<64xf32>
  }
  return
}
```

关键变化：

- 循环边界已显式化，但索引表达式 `%A[%i, %j]`、`%x[%j]` 是 **affine 形式**（`idx = base + stride * iv`），编译器可以精确分析依赖关系
- 此层可做**循环变换**：tiling（分块）、interchange（交换内外层）、fusion（合并相邻循环）。这些都是纯粹数学变换，**不依赖任何硬件信息**

### 第三层：SCF + MemRef — 结构化控制流，告别 affine 约束

进一步 lowering，affine 循环变成通用 `scf.for`，affine access 降为显式指针算术：

```
func.func @gemv(%A: memref<64x128xf32>, %x: memref<128xf32>, %y: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c128 = arith.constant 128 : index
  scf.for %i = %c0 to %c64 step %c1 {
    %sum = memref.alloca() : memref<1xf32>
    scf.for %j = %c0 to %c128 step %c1 {
      %a = memref.load %A[%i, %j] : memref<64x128xf32>
      %xj = memref.load %x[%j] : memref<128xf32>
      %mul = arith.mulf %a, %xj : f32
      %old = memref.load %sum[] : memref<1xf32>
      %new = arith.addf %old, %mul : f32
      memref.store %new, %sum[] : memref<1xf32>
    }
    %result = memref.load %sum[] : memref<1xf32>
    memref.store %result, %y[%i] : memref<64xf32>
  }
  return
}
```

这一层引入了 **alloca**（栈分配），为后续 register / memory 分配做准备。此层可做**硬件相关的优化**：

- 将 `scf.for` 映射到 SIMD 向量化
- 决定哪些 alloca 提升为寄存器

### 第四层：LLVM Dialect — 贴近机器

最终 lowering 到 LLVM IR：

```
llvm.func @gemv(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr) {
  llvm.br ^bb1(0 : i64)
^bb1(%i: i64):
  %cond = llvm.icmp "slt" %i, %c64 : i64
  llvm.cond_br %cond, ^bb2, ^bb6
^bb2:
  llvm.br ^bb3(0.0 : f32, 0 : i64)
^bb3(%acc: f32, %j: i64):
  %inner_cond = llvm.icmp "slt" %j, %c128 : i64
  llvm.cond_br %inner_cond, ^bb4, ^bb5
^bb4:
  %a_ptr = llvm.getelementptr %arg0[%i, %j] : (!llvm.ptr, i64, i64) -> !llvm.ptr
  %a = llvm.load %a_ptr : !llvm.ptr -> f32
  %x_ptr = llvm.getelementptr %arg1[%j] : (!llvm.ptr, i64) -> !llvm.ptr
  %xj = llvm.load %x_ptr : !llvm.ptr -> f32
  %mul = llvm.fmul %a, %xj : f32
  %new_acc = llvm.fadd %acc, %mul : f32
  %next_j = llvm.add %j, %c1 : i64
  llvm.br ^bb3(%new_acc, %next_j : f32, i64)
^bb5:
  %y_ptr = llvm.getelementptr %arg2[%i] : (!llvm.ptr, i64) -> !llvm.ptr
  llvm.store %acc, %y_ptr : f32, !llvm.ptr
  %next_i = llvm.add %i, %c1 : i64
  llvm.br ^bb1(%next_i : i64)
^bb6:
  llvm.return
}
```

这一层已经是 LLVM 能直接处理的 IR，basic block + phi node + `getelementptr`。后续交给 LLVM 后端生成 x86/ARM/RISC-V 机器码。

### 这四层解释了文中的核心论断

#### "多个抽象层级 IR 共存"

上面四层 IR **同时存在于同一个 MLIR module 中**是完全合法的。比如 `linalg.matvec` 和邻居的 `scf.for` 可以共存，彼此通过 `func.call` 互相调用。MLIR 不要求整个模块处于统一抽象级别。这使得：

- 编译器可以**局部 lowering**：只对热点函数降级，其他函数保持高层表示
- 不同 dialect 的 pass 可以**混合编排**：`affine-loop-fusion` 和 `linalg-fuse-elementwise-ops` 在一次 pipeline 中先后运行

#### "显式、可控的 lowering 路径"

从 linalg → affine → scf → llvm，每一步都有**确定的 pass** 执行转换：

```
linalg.matvec ──[linalg-to-affine-loops]──▶ affine.for
affine.for    ──[affine-to-scf]────────────▶ scf.for
scf.for       ──[scf-to-cf]────────────────▶ cf.br + basic blocks
cf.br         ──[cf-to-llvm]───────────────▶ llvm.br + phi
```

这和 TVM 的"从 Relay IR 直接到 TIR"的单步 lowering 不同——MLIR 的 lowering 是**可分段、可组合、可插入自定义 pass 的**。你可以选择在 affine 层插入一个自定义的 `affine-loop-tile` pass，而不影响其上下游。

#### "在较高抽象层做硬件无关优化"

以 **算子融合** 为例，在 linalg 层它是一个简单的 pattern match：

```
// Before: two separate ops
%0 = linalg.matvec ins(%A, %x) outs(%tmp)
%1 = linalg.add ins(%tmp, %b) outs(%y)

// After fusion (one generic op)
%1 = linalg.generic {
  indexing_maps = [affine_map<(i,j) -> (i,j)>,   // A
                   affine_map<(i,j) -> (j)>,      // x
                   affine_map<(i,j) -> (i)>,      // b
                   affine_map<(i,j) -> (i)>],     // y
  iterator_types = ["parallel", "reduction"]
} ins(%A, %x, %b) outs(%y) {
^bb(%a: f32, %xj: f32, %b_i: f32, %y_i: f32):
  %mul = arith.mulf %a, %xj : f32
  %sum = arith.addf %mul, %b_i : f32
  linalg.yield %sum : f32
}
```

这次融合**完全不需要知道目标 CPU 是 x86 还是 ARM**。融合收益（减少一次内存遍历）是纯数学/访存层面的判断。等 lowering 到 `scf` 层时，循环结构已经固定，再做融合就难得多。

每层有自己的优化窗口：linalg 做算子融合，affine 做循环变换，scf 做向量化，llvm 做指令选择。分层之后，每一层关心的问题更少、规则更简单、优化更彻底。
