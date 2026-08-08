# Inductor 编译阶段骨架分析

> 返回：[← Inductor 教程导航](../README.md)
>
> 按 Inductor 编译管线的主要阶段逐步拆解，聚焦每个阶段的输入输出与核心逻辑。适合快速了解编译管线的各阶段边界和数据流转。

## 文件导航

| 文件 | 阶段 | 核心内容 |
|------|------|---------|
| [inductor_overview.md](inductor_overview.md) | 全景 | 从文件结构角度建立全局地图，理解 `torch/_inductor/` 下各文件的职责与协作关系 |
| [phase1_global_view.md](phase1_global_view.md) | Phase 1 | 数据流全景：Dynamo → FX Graph → Lowering → Scheduling → Codegen |
| [phase2_fx_optimization.md](phase2_fx_optimization.md) | Phase 2 | FX 图优化 |
| [phase3_lowering.md](phase3_lowering.md) | Phase 3 | Lowering：FX Graph → Inductor IR |
| [phase4_scheduling_fusion.md](phase4_scheduling_fusion.md) | Phase 4 | 调度与融合 |
| [phase5_codegen.md](phase5_codegen.md) | Phase 5 | 代码生成 |

## 推荐阅读顺序

先读 `inductor_overview.md` 建立全局地图，再按 Phase 1 → 5 顺序阅读，每个 phase 自包含，可按需跳转。
