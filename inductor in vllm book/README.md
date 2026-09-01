# 《vLLM 中的 Inductor》— 面向 Inductor 开发者的 vLLM 编译栈全景教材

> 返回：[← PyTorch 编译器技术栈总览](../README.md)
>
> **读者契约**:假设你懂 dynamo / AOTAutograd / Inductor 流水线、做过简单的 Inductor 开发;**不假设任何 vLLM 知识**(第 2 章兜底)。
> **事实基准**:vLLM **0.27.1** 源码(`venv_vllm/lib/python3.11/site-packages/vllm/`);官方文档仅作叙事背景,凡不一致处以源码为准。

**核心叙事主线**:decode 是 CPU/launch bound → 必须消灭 Python 开销 → 代价是编译 → 编译的代价是启动时间 → 所以全部编译前置 + 缓存 → 所以敢丢 guards、敢一次 trace、敢去泛函化——**所有机制都是这条因果链上的必然产物**。

## 章节目录

### Part I 背景与全景
| 章 | 标题 | 难度 | 状态 |
|---|---|---|---|
| [1](01-为何推理服务需要torch-compile.md) | 为何推理服务需要 torch.compile:launch-bound 的 decode | ★★ | 🏁 终审定稿 |
| [2](02-vLLM-V1引擎最小背景.md) | vLLM V1 引擎最小背景:编译所服务的运行时 | ★★ | 🏁 终审定稿 |
| [3](03-编译栈全景架构.md) | vLLM 编译栈全景:一张架构图读懂全书 | ★★★ | 🏁 终审定稿 |

### Part II 配置与前端
| 章 | 标题 | 难度 | 状态 |
|---|---|---|---|
| [4](04-配置体系与优化级别.md) | 配置体系:CompilationMode、CUDAGraphMode 与 -O 优化级别 | ★★★ | 🏁 终审定稿 |
| [5](05-Dynamo前端与一次捕获.md) | Dynamo 前端:@support_torch_compile 与「一次捕获」哲学 | ★★★★ | 🏁 终审定稿 |

### Part III 中端:图的切分与算子边界
| 章 | 标题 | 难度 | 状态 |
|---|---|---|---|
| [6](06-Piecewise切分.md) | Piecewise 切分:为什么 attention 是图的边界 | ★★★ | 🏁 终审定稿 |
| [7](07-自定义算子边界与vllm-ir.md) | 自定义算子边界:unified_attention、vllm_ir 与 activation donation | ★★★★ | 🏁 终审定稿 |

### Part IV 后端:逐 piece 的 Inductor 编译
| 章 | 标题 | 难度 | 状态 |
|---|---|---|---|
| [8](08-VllmBackend总编排.md) | VllmBackend:编译流水线的总指挥 | ★★★★ | 🏁 终审定稿 |
| [9](09-逐range编译与Inductor对接.md) | 逐 range 编译:PiecewiseBackend 与 Inductor 的对接 | ★★★★★ | 🏁 终审定稿 |
| [10](10-Pass流水线与算子融合.md) | 图变换流水线:PostGradPassManager 与自定义 fusion | ★★★★ | 🏁 终审定稿 |

### Part V 产物与运行时
| 章 | 标题 | 难度 | 状态 |
|---|---|---|---|
| [11](11-代码生成与可序列化产物.md) | 代码生成:execution_fn、liveness 与可序列化产物 | ★★★★ | 🏁 终审定稿 |
| [12](12-CUDA-Graph与运行时分派.md) | CUDA Graph:capture/replay、FULL vs PIECEWISE 与运行时分派 | ★★★★★ | 🏁 终审定稿 |
| [13](13-编译缓存与AOT.md) | 编译缓存与 AOT:compile-once 的工程闭环 | ★★★★ | 🏁 终审定稿 |

### Part VI 哲学与实践
| 章 | 标题 | 难度 | 状态 |
|---|---|---|---|
| [14](14-设计哲学总结.md) | 设计哲学:把机制上升为思想 | ★★★ | 🏁 终审定稿 |
| [15](15-动手实践与调试.md) | 动手实践与调试 | ★★★★ | 🏁 终审定稿 |

### 前言与附录
| 文件 | 内容 |
|---|---|
| [00-前言与阅读指南](00-前言与阅读指南.md) | 前言 + 三条阅读路径 + 全书总览大图 |
| [附录 A 术语表中英对照](appendix-A-术语表中英对照.md) | 99 条术语,按主题分组 |
| [附录 B 图例与写作规范](appendix-B-图例与写作规范.md) | 图例约定、章节模板、写作规范 |
| [附录 C 源码锚点索引](appendix-C-源码锚点索引.md) | 全书 file:line 汇总(版本升级时批量校对) |

## 阅读路径

- **速览**(约 2 小时,建立心智模型):第 1 → 3 → 14 章
- **主线**(教材式):第 1~15 章顺序读
- **深潜**(要改代码的读者):第 3 章后按 Part III → IV → V 逐层下钻,最后 15 章上手

## 图例速查

蓝=vLLM 代码 · 橙=torch/Inductor 上游 · 绿=编译产物/缓存 · 灰=外部系统 · 红=monkey-patch/危险点
实线=编译期数据流 · 虚线=运行时调用流 · 粗线=本章主线索 —— 详见[附录 B](appendix-B-图例与写作规范.md)

---

*状态图例:⏳ 未写 · ✍️ 草稿 · 🔍 审校中 · ✅ 已写并审校 · 🏁 终审定稿*

**进度**:15/15 章 + 前言 + 附录 A/B/C 均已通过逐锚点事实审校与全书一致性终审,定稿。
