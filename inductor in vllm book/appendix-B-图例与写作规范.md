---
part: 附录
title: 附录 B 图例与写作规范
status: done
difficulty: ★
---

# 附录 B 图例与写作规范

> 本附录是全书写作规范的唯一权威(source of truth)。每章写作与审校均以本附录为准。

## B.1 章节模板(五段式,强制)

每章严格按以下结构组织(节号用 `N.0`~`N.4`,N 为章号):

```markdown
---
chapter: N
part: <I~VI>
title: <章标题>
status: draft | review | done
difficulty: ★~★★★★★
depends: [章号列表]
---
# 第 N 章 <标题>

<!-- 6 行 ASCII 位置导航图(见 B.4) -->

## N.0 本章俯瞰
<!-- 300~500 字。以一个问题开场,第一段末尾给一句答案预告。
     说清本章解决什么问题、为什么存在。不使用本章新术语。 -->

## N.1 最小前置知识
<!-- 只列本章用到的 3~5 个概念,每个 2~3 句,给跳转链接。
     原则:够用就行——理解本章所需的最小集合,不多不少。 -->

## N.2 设计思想与设计哲学
<!-- 先讲"为什么这样设计",再讲怎么做。标注【哲学时刻】的段落
     是全书七条命题(见第 14 章)在本书的落点。 -->

## N.3 具体流程(流水线视角)
<!-- 主体。每个机制必须写成"流水线产物"句式(见 B.2)。
     配 Mermaid 正式图 + 源码片段 + 一个简单具体的例子。 -->

## N.4 本章小结
<!-- 3~5 条要点(每条 ≤25 字)+ 一句"本机制服务于哪条哲学命题"
     + 下一章预告 2 句。 -->
```

## B.2 流水线产物句式(强制)

每个编译机制的讲解必须回答三件事:

> **上游给我什么**(产物类型 + 来源 `file:line`)→ **我做什么加工**(算法/变换)→ **我产出什么**(产物类型 + 去向 `file:line`)

示例:
> `PiecewiseCompileInterpreter` 从 `split_graph` 拿到 `split_gm`(`vllm/compilation/backends.py:1179`,产物:带 `submod_N` 子模块的 GraphModule)→ 逐个子模块构造 `PiecewiseBackend` 并全量预编译(`backends.py:1222`)→ 产出的编译产物**替换** `split_gm` 的 submodule 属性(`backends.py:766`),供后续 codegen 直接引用。

## B.3 代码与锚点规范

- **源码锚点双保险**:`路径:行号`(`符号名`)。行号随版本漂移,符号名可重定位。全书基线 **vLLM 0.27.1**。
- 代码片段:每章 ≤3 段,每段 ≤15 行;片段上方标注 `# vllm/compilation/backends.py:1020-1046 (v0.27.1)`;只贴"承重墙"代码(承载设计决定的那几行),其余用自然语言或伪代码。
- 术语:类名/函数名/字段/env var/文件名一律英文原文;首次出现加粗并给中文释义(如 **guard(守卫条件)**),此后只用英文。dynamo、fake tensor、functionalization、max_autotune、capture/replay 等通用词直接用英文。
- 章间引用:`[第 9 章](09-逐range编译与Inductor对接.md)` 或"(详见第 5 章)";禁止"如前所述"式无锚点回指。
- 手推示例若非真实 dump,须标注「示意图」;来自真实 dump 的标注「真实产物摘录」。

## B.4 ASCII 位置导航图(每章开头,统一模板)

全书 15 章按编译栈阶段排成 6 行,当前章以 `◄==` 追加标注。模板(以第 9 章为例,其余章只挪动 `◄==`):

```text
I   背景:   [1 为何compile] → [2 V1引擎] → [3 编译栈全景]
II  前端:   [4 配置体系] → [5 Dynamo一次捕获]
III 中端:   [6 Piecewise切分] → [7 算子边界与vllm_ir]
IV  后端:   [8 VllmBackend] → [9 逐range编译]◄== → [10 Pass与融合]
V   运行时: [11 代码生成] → [12 CUDA Graph] → [13 缓存与AOT]
VI  收官:   [14 设计哲学] → [15 动手实践]
```

- ASCII 图行宽硬上限 **78 列**;只用于局部示意(单机制输入→变换→输出、导航图),复杂架构一律 Mermaid。

## B.5 Mermaid 图规范

### 图型选择(一一对应,不混用)

| 图型 | 用途 | 约束 |
|---|---|---|
| `flowchart` | 架构/分层/数据流 | 分层图一律 `TB`,流水线图一律 `LR` |
| `sequenceDiagram` | 有明确参与方的控制流 | 参与方 ≤6 |
| `classDiagram` | 核心类关系 | 只画本章涉及的 3~7 个类,禁止全量 |
| `stateDiagram-v2` | 生命周期/状态迁移 | capture→replay、缓存命中、range 匹配等 |

### 颜色语义(全书统一,经 `class` + `style` 声明)

| 颜色 | 含义 | Mermaid class 名 |
|---|---|---|
| 蓝 `#3B82F6` | vLLM 自身代码 | `vllm` |
| 橙 `#F59E0B` | torch / Inductor / dynamo 上游 | `torch` |
| 绿 `#10B981` | 编译产物 / 缓存 / 生成的代码 | `artifact` |
| 灰 `#6B7280` | 外部系统(CUDA / NCCL / 文件系统) | `ext` |
| 红 `#EF4444` | monkey-patch / 危险点 / 陷阱 | `danger` |

每个 flowchart/class 图开头声明一次(可直接复制):

```
classDef vllm fill:#DBEAFE,stroke:#3B82F6,color:#1E3A8A
classDef torch fill:#FEF3C7,stroke:#F59E0B,color:#78350F
classDef artifact fill:#D1FAE5,stroke:#10B981,color:#064E3B
classDef ext fill:#E5E7EB,stroke:#6B7280,color:#374151
classDef danger fill:#FEE2E2,stroke:#EF4444,color:#7F1D1D
```

### 线型语义

- 实线 `-->`:编译期数据流(产物传递)
- 虚线 `-.->`:运行时调用流
- 粗线 `==>`:本章主线索

### 节点形状语义

| 形状 | 语法 | 含义 |
|---|---|---|
| 圆角矩形 | `A[ ]` | 模块 / 类 / 函数 |
| 菱形 | `A{ }` | 决策 / 分支 |
| 圆柱 | `A[( )]` | 缓存 / 存储 |
| 体育场形 | `A([ ])` | 入口 / 出口 |

### 渲染兼容硬规则(违反即渲染失败)

1. 节点文本**禁用**半角括号 `()`、冒号 `:`、双引号 `"`——用全角 `()`、中文冒号或空格替代
2. `sequenceDiagram` 的别名只用 ASCII 字母数字
3. `classDiagram` 的关系符号与标签间必须有空格:`VllmBackend *-- CompilerManager`
4. 图内文本优先中文(读者友好),类名/函数名保持英文

### 数量要求

每章 ≥1 张正式 Mermaid 图 + ≥1 张 ASCII 小图(或第二张 Mermaid);第 3、12、15 章可 3 张。

## B.6 字数与难度

- 每章正文(不含 YAML/代码块/图)3000~5000 字;第 9、12 章允许至 5000+
- 难度标注:★(通识)~★★★★★(需反复研读)
- 前言约 1500 字,阅读指南约 1200 字

## B.7 章末格式

每章 N.4 之后可附:

```markdown
### 本章源码索引
| 锚点 | 内容 |
|---|---|

### 参考资料
1. [vLLM design doc — torch.compile](https://docs.vllm.ai/en/latest/design/torch_compile/) — 已与 0.27.1 源码核对一致 / 仅作背景
```

引用标注规则:每条官方引用注明「已与 0.27.1 源码核对一致」或「仅作背景叙事」。
