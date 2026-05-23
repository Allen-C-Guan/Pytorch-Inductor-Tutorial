# 附录 C：GPU 体系结构速查

本附录提供本书所需的 GPU（图形处理器）体系结构基础知识，涵盖 NVIDIA GPU、AMD GPU 和华为 Ascend NPU。重点介绍与 Triton 编译器设计密切相关的硬件概念和性能参数。

---

## C.1 NVIDIA GPU 体系结构

### C.1.1 核心概念

NVIDIA GPU 采用 **SIMT（Single Instruction, Multiple Thread）** 执行模型。以下是关键层级概念：

| 概念 | 英文全称 | CUDA 术语 | 说明 |
|------|---------|----------|------|
| **线程** | Thread | Thread | 最基本的执行单元，拥有独立的寄存器文件和程序计数器 |
| **线程束** | Warp | Warp | 32 个线程为一组，以锁步（lockstep）方式执行同一条指令。Warp 是 GPU 的调度和执行基本单元 |
| **线程块** | Thread Block | CTA (Cooperative Thread Array) | 一组 warp 的集合（通常 1-32 个 warp），共享 Shared Memory。同一 CTA 内的线程可同步（`__syncthreads` / `ttg.barrier`） |
| **网格** | Grid | Grid | 一个 kernel launch 中所有 CTA 的集合，每个 CTA 有独立的坐标（`program_id`） |
| **流式多处理器** | Streaming Multiprocessor (SM) | SM | GPU 的核心计算单元。一个 SM 包含多个 Warp Scheduler、执行单元（CUDA Core / Tensor Core）、Register File、Shared Memory / L1 Cache |
| **线程束调度器** | Warp Scheduler | Warp Scheduler | 每个 SM 包含 4 个 Warp Scheduler（Ampere 起），每个周期从就绪 warp 中选择一条指令发射 |

### C.1.2 内存层次

```
┌────────────────────────────────────────────────────────────┐
│                    HBM（高带宽内存 / 全局内存）               │
│            容量: 40GB (A100) / 80GB (H100)                  │
│            带宽: ~2 TB/s (A100) / ~3.35 TB/s (H100)         │
│            延迟: ~250-800 cycles                             │
├────────────────────────────────────────────────────────────┤
│                       L2 Cache                               │
│            容量: 40 MB (A100) / 50 MB (H100)                │
│            延迟: ~200 cycles                                 │
├──────────────────────────┬─────────────────────────────────┤
│   SM 内部                │                                 │
│  ┌─────────────────────┐ │                                 │
│  │  Shared Memory /    │ │  Per-SM:                        │
│  │  L1 Data Cache      │ │  192 KB (V100) /                │
│  │  (可编程 on-chip)   │ │  256 KB (A100, 可配分) /       │
│  │  延迟: ~20-30 cycles │ │  256 KB (H100)                 │
│  └─────────────────────┘ │                                 │
│  ┌─────────────────────┐ │                                 │
│  │  Register File       │ │  Per-SM: 256 KB (V100/A100/H100)│
│  │  延迟: ~0 cycles     │ │  每线程最多 255 个 32-bit 寄存器│
│  └─────────────────────┘ │                                 │
│  ┌─────────────────────┐ │                                 │
│  │  Constant Cache /   │ │  只读缓存                        │
│  │  Texture Cache      │ │                                 │
│  └─────────────────────┘ │                                 │
└──────────────────────────┴─────────────────────────────────┘
```

### C.1.3 Triton 的内存空间映射

Triton 的 Memory Space 属性与 GPU 硬件的对应关系：

| Triton Memory Space | 硬件对应 | 访问范围 | 典型延迟 |
|--------------------|---------|---------|---------|
| Global Memory | HBM (DRAM) | 所有 CTA | ~250-800 cycles |
| Shared Memory (`#ttg.shared_memory`) | On-chip SRAM | 同一 CTA 内所有线程 | ~20-30 cycles |
| Register | 每线程寄存器文件 | 单线程 | ~0 cycles (立即数) |

### C.1.4 Tensor Core（张量核心）

Tensor Core 是 NVIDIA GPU 中专用于矩阵乘加运算（D = A x B + C）的硬件单元。

| GPU 架构 | Tensor Core 代数 | 精度支持 | Triton MMA 版本 |
|---------|-----------------|---------|----------------|
| Volta (V100) | 第1代 | FP16×FP16→FP32/FP16 | MMAv1 |
| Turing (T4) | 第2代 | + INT8, INT4 | MMAv1 |
| Ampere (A100) | 第3代 | + TF32, BF16, FP64 | MMAv2 (mma.sync) |
| Hopper (H100) | 第4代 | + FP8, WGMMA 指令 | MMAv3 (wgmma) |

Triton 中 Tensor Core 使用方式：
- **MMAv1/MMAv2**：操作数需要通过 `ttg.local_load` 从 Shared Memory 加载，结果在 `nvidia_mma` encoding 的 Distributed Tensor 中
- **MMAv3 (wgmma)**：直接通过 `NVMMASharedEncoding` 编码的 Shared Memory 描述符作为输入（异步 warpgroup 矩阵乘累加）

---

## C.2 关键性能参数速查

### C.2.1 数据中心 GPU

| 参数 | NVIDIA A100 (Ampere) | NVIDIA H100 (Hopper) | AMD MI250X (CDNA2) | AMD MI300X (CDNA3) |
|------|---------------------|---------------------|--------------------|--------------------|
| **制造工艺** | TSMC 7nm | TSMC 4nm | TSMC 6nm | TSMC 5nm |
| **SM / CU 数量** | 108 SM | 132 SM | 220 CU (2 dies) | 304 CU (8 dies) |
| **CUDA Core / Stream Processor** | 6,912 FP32 | 14,592 FP32 + 14,592 FP64 | 14,080 FP32 | 19,456 FP32 (per die) |
| **Tensor Core / Matrix Core** | 432 (第3代) | 528 (第4代) | 880 MFMA | 1,216 MFMA |
| **HBM 容量** | 40GB / 80GB | 80GB | 128GB | 192GB |
| **HBM 带宽** | 1,555 GB/s (40GB) / 2,039 GB/s (80GB) | 3,352 GB/s | 3,277 GB/s (total) | 5,300 GB/s |
| **L2 Cache** | 40 MB | 50 MB | 8 MB (per die) | 256 MB (Infinity Cache) |
| **Shared Memory / SM** | 可配分 164 KB (max 164 KB shared) | 256 KB (max 228 KB shared) | 64 KB (per CU) | 64 KB (per CU) |
| **寄存器 / SM** | 256 KB (最大 64K × 32-bit) | 256 KB | unknown | unknown |
| **最大 Warp / SM** | 64 (2,048 threads) | 64 (2,048 threads) | 64 waves | 64 waves |
| **Warp Size** | 32 | 32 | 64 (Wavefront) | 64 (Wavefront) |
| **TDP** | 400W | 700W | 560W | 750W |
| **FP16 (非稀疏) TFLOPS** | 312 | 990 | 384 | 1,307 |
| **TF32 TFLOPS** | 156 | 495 | N/A | N/A |
| **FP64 TFLOPS** | 19.5 | 60 | 48 | 163 |

### C.2.2 延迟参考值 (A100)

| 操作 | 近似延迟 (cycles) |
|------|------------------|
| 寄存器访问 | ~0 |
| Shared Memory 访问 (无 bank conflict) | ~20-30 |
| L1 Cache 命中 | ~30-40 |
| L2 Cache 命中 | ~200 |
| HBM 访问 | ~250-800 |
| Warp Shuffle (线程间寄存器交换) | ~5-10 |
| Barrier 同步 (`__syncthreads`) | ~20-30 |
| Tensor Core MMA (m16n8k16, FP16) | ~50-100 |

---

## C.3 AMD GPU 体系结构概述

### C.3.1 核心概念对比

AMD GPU 采用类似的并行计算模型，但术语和硬件组织有所不同：

| 概念 | NVIDIA 术语 | AMD 术语 | 说明 |
|------|-----------|---------|------|
| 最小执行单元 | Thread | Work-Item | 一个线程/work-item |
| 线程束 | Warp (32 threads) | Wavefront (64 work-items) | AMD 的 wavefront 大小为 64 |
| 线程块 | CTA / Thread Block | Work-Group | 一组 wavefront 的集合 |
| 计算单元 | SM | Compute Unit (CU) | GPU 的核心计算单元 |
| 调度器 | Warp Scheduler | Wave Scheduler | 每个 CU 有 4 个调度器（RDNA3） |

### C.3.2 AMD Matrix Core

AMD 的矩阵加速硬件有两种：

1. **MFMA**（Matrix-Fused Multiply-Add）：CDNA 架构（MI200/MI300 系列），对应 NVIDIA Tensor Core
2. **WMMA**（Wave Matrix Multiply-Accumulate）：RDNA 架构（Radeon 消费级 GPU）

Triton 通过 `AMDMfmaEncodingAttr` 和 `AMDWmmaEncodingAttr` 这两个 Layout 编码支持 AMD GPU。

---

## C.4 Ascend NPU 体系结构概述

### C.4.1 达芬奇架构（Da Vinci Architecture）

华为昇腾（Ascend）NPU 采用自研的达芬奇架构，专为 AI 计算设计。

| 概念 | 说明 |
|------|------|
| **AI Core** | Ascend NPU 的核心计算单元，等价于 NVIDIA SM |
| **AI CPU** | 通用计算单元，处理标量和控制流 |
| **Cube Unit** | 矩阵乘法加速单元（类似 Tensor Core），支持 FP16、INT8、INT4 |
| **Vector Unit** | 向量运算单元，支持逐元素操作 |
| **Scalar Unit** | 标量运算与控制流 |
| **L1 Buffer** | 类似 Shared Memory 的片上缓存 |
| **HBM** | 高带宽内存（如 64GB on Ascend 910B） |

### C.4.2 Ascend 内存层次

```
HBM (64GB, ~1.2 TB/s) → L2 Cache → L1 Buffer (per AI Core) → Register
```

### C.4.3 Triton 的 Ascend 支持

Triton 通过 `triton-ascend` 后端插件支持 Ascend NPU，定义 Ascend 专属的 Dialect 和 Layout。其编译管线为：

```
Triton DSL → TTIR → TTGIR → Ascend LLVM IR → Ascend C (AscendCL) → 二进制
```

---

## C.5 CUDA 编程模型速查

### C.5.1 线程层次

```
                    Grid
                      │
       ┌──────────────┼──────────────┐
       │              │              │
    CTA[0,0]      CTA[0,1]      CTA[1,0]     ...
       │
  ┌────┴────┐
  │         │
Warp 0    Warp 1   ...  Warp N
  │
  ├── Thread 0
  ├── Thread 1
  ├── ...
  └── Thread 31
```

### C.5.2 内置变量

| CUDA 内置变量 | Triton 对应 | 含义 |
|-------------|-----------|------|
| `threadIdx.x/y/z` | 隐式（Layout 编码） | 线程在 CTA 内的坐标 |
| `blockIdx.x/y/z` | `tl.program_id(axis)` | CTA 在 Grid 中的坐标 |
| `blockDim.x/y/z` | `tl.num_programs(axis)` | Grid 中 CTA 的数量 |
| `gridDim.x/y/z` | 由 kernel launch 参数决定 | CTA 的总数 |
| `warpSize` | 32 (NVIDIA) / 64 (AMD) | warp/wavefront 大小 |

### C.5.3 Triton vs. CUDA 编程模型

| 特性 | CUDA C++ | Triton |
|------|---------|--------|
| **编程抽象** | 以 Thread 为基本单元 | 以 Tile（数据块）为基本单元 |
| **并行表达** | 显式 threadIdx, blockIdx | 隐式通过 Layout 编码，编译器自动将 tile 映射到线程 |
| **内存管理** | 显式 `__shared__`、`cudaMalloc` | 隐式通过 `tl.load/tl.store` 和 Layout 转换 |
| **同步** | 显式 `__syncthreads()` | 通过 `ttg.barrier` 或编译器自动生成 |
| **Tensor Core** | 显式 inline PTX 或 wmma API | 隐式通过 `tt.dot` + MMA Layout |
| **优化粒度** | 手工调优（tiling, swizzle, register usage） | 编译期自动优化（Coalescing, Pipelining, Warp Specialization）+ Autotuner |

### C.5.4 性能关键因素

| 因素 | 限制 | 优化手段 |
|------|------|---------|
| **全局内存带宽** | HBM 带宽有限 | 合并访问（Coalescing）、数据预取 |
| **Shared Memory 带宽** | Bank conflict | Swizzle 模式、Padded Shared Layout |
| **寄存器压力** | 每 SM 寄存器数量固定 | 减少活跃变量、循环展开平衡 |
| **Occupancy** | 每 SM 可调度 warp 数量上限 | 调整 num_warps、寄存器使用 |
| **Tensor Core 利用率** | MMA 指令需要特定数据布局 | `nvidia_mma` / `dot_op` Layout |
| **延迟隐藏** | Warp 调度器在 warp 间切换 | 软件流水线、Warp Specialization |
