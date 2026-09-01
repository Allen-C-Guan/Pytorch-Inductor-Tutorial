---
part: 附录
title: 附录 C 源码锚点索引
status: done
difficulty: ★
---

# 附录 C 源码锚点索引

> 全书所有 `文件:行号` 锚点的汇总索引,基线 **vLLM 0.27.1**(`venv_vllm/lib/python3.11/site-packages/vllm/`)。
> 用途:**版本升级时批量校对**——升级后按本表逐文件核对行号漂移;行号失效时以各章标注的符号名(双锚点)重定位。
> 生成方式:程序化抽取各章正文与章末源码索引,路径经源码树存在性校验后去重合并;"引用章"为出现该锚点的章节。

## 使用方法

1. 升级 vLLM 后,按本表逐文件核对行号;命中→仍准;偏移→grep 符号名重定位,更新正文与本表;
2. 符号消失→机制已演进,回"引用章"修订对应叙述;
3. torch/ 与 depyf/ 路径锚点对应同一 venv 内的对应包安装。

## `depyf/explain/enable_debugging.py`

| 锚点 | 引用章 |
|---|---|
| `depyf/explain/enable_debugging.py:139-142` | 15 |

## `vllm/__init__.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/__init__.py:14` | 07 |

## `vllm/compilation/backends.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/backends.py:59-93` | 11 |
| `vllm/compilation/backends.py:59-94` | 08 |
| `vllm/compilation/backends.py:78-91` | 08 |
| `vllm/compilation/backends.py:84-86` | 11 |
| `vllm/compilation/backends.py:88-89` | 11 |
| `vllm/compilation/backends.py:96` | 08, 09, 14 |
| `vllm/compilation/backends.py:96-122` | 08 |
| `vllm/compilation/backends.py:97-99` | 11 |
| `vllm/compilation/backends.py:101-121` | 08 |
| `vllm/compilation/backends.py:114-121` | 14 |
| `vllm/compilation/backends.py:124` | 08 |
| `vllm/compilation/backends.py:124-404` | 08 |
| `vllm/compilation/backends.py:140` | 08 |
| `vllm/compilation/backends.py:149-161` | 08, 09 |
| `vllm/compilation/backends.py:150` | 10 |
| `vllm/compilation/backends.py:150-154` | 10 |
| `vllm/compilation/backends.py:163-214` | 13 |
| `vllm/compilation/backends.py:176-178` | 13 |
| `vllm/compilation/backends.py:191` | 08 |
| `vllm/compilation/backends.py:215-222` | 13 |
| `vllm/compilation/backends.py:223-261` | 13 |
| `vllm/compilation/backends.py:263-399` | 08, 09 |
| `vllm/compilation/backends.py:264` | 03 |
| `vllm/compilation/backends.py:280-297` | 13, 15 |
| `vllm/compilation/backends.py:329-350` | 08, 14 |
| `vllm/compilation/backends.py:346` | 08 |
| `vllm/compilation/backends.py:369-397` | 13, 15 |
| `vllm/compilation/backends.py:370-373` | 13 |
| `vllm/compilation/backends.py:370-374` | 08 |
| `vllm/compilation/backends.py:406` | 06 |
| `vllm/compilation/backends.py:414` | 06 |
| `vllm/compilation/backends.py:437` | 06 |
| `vllm/compilation/backends.py:479` | 06 |
| `vllm/compilation/backends.py:482-483` | 06 |
| `vllm/compilation/backends.py:553` | 06 |
| `vllm/compilation/backends.py:562-576` | 06 |
| `vllm/compilation/backends.py:570-576` | 06 |
| `vllm/compilation/backends.py:578` | 06 |
| `vllm/compilation/backends.py:578-591` | 06 |
| `vllm/compilation/backends.py:599-608` | 06 |
| `vllm/compilation/backends.py:602-608` | 06 |
| `vllm/compilation/backends.py:610-625` | 06 |
| `vllm/compilation/backends.py:627` | 06 |
| `vllm/compilation/backends.py:633` | 06, 12 |
| `vllm/compilation/backends.py:633-684` | 06, 08, 12 |
| `vllm/compilation/backends.py:655-659` | 12 |
| `vllm/compilation/backends.py:667-683` | 14 |
| `vllm/compilation/backends.py:675-683` | 14 |
| `vllm/compilation/backends.py:687` | 06 |
| `vllm/compilation/backends.py:693-696` | 06 |
| `vllm/compilation/backends.py:730` | 06 |
| `vllm/compilation/backends.py:742-764` | 09 |
| `vllm/compilation/backends.py:742-772` | 06 |
| `vllm/compilation/backends.py:746-748` | 09 |
| `vllm/compilation/backends.py:766` | 03, 06, 11 |
| `vllm/compilation/backends.py:785` | 08 |
| `vllm/compilation/backends.py:785-802` | 08 |
| `vllm/compilation/backends.py:805` | 08 |
| `vllm/compilation/backends.py:833-870` | 08 |
| `vllm/compilation/backends.py:865-867` | 10 |
| `vllm/compilation/backends.py:868` | 09 |
| `vllm/compilation/backends.py:872` | 11 |
| `vllm/compilation/backends.py:934` | 03, 07, 10 |
| `vllm/compilation/backends.py:934-953` | 08 |
| `vllm/compilation/backends.py:934-971` | 08, 14 |
| `vllm/compilation/backends.py:939` | 07 |
| `vllm/compilation/backends.py:955-969` | 10 |
| `vllm/compilation/backends.py:971` | 10 |
| `vllm/compilation/backends.py:973` | 08 |
| `vllm/compilation/backends.py:973-1017` | 15 |
| `vllm/compilation/backends.py:1020` | 03, 08 |
| `vllm/compilation/backends.py:1020-1339` | 08 |
| `vllm/compilation/backends.py:1027` | 15 |
| `vllm/compilation/backends.py:1031-1055` | 13 |
| `vllm/compilation/backends.py:1031-1057` | 13, 15 |
| `vllm/compilation/backends.py:1031-1079` | 03, 08, 14 |
| `vllm/compilation/backends.py:1034` | 04, 13 |
| `vllm/compilation/backends.py:1038-1041` | 15 |
| `vllm/compilation/backends.py:1058-1079` | 13 |
| `vllm/compilation/backends.py:1063-1079` | 13 |
| `vllm/compilation/backends.py:1066` | 13 |
| `vllm/compilation/backends.py:1081-1101` | 13 |
| `vllm/compilation/backends.py:1085-1101` | 08 |
| `vllm/compilation/backends.py:1092-1097` | 15 |
| `vllm/compilation/backends.py:1094-1097` | 15 |
| `vllm/compilation/backends.py:1105-1112` | 15 |
| `vllm/compilation/backends.py:1121-1134` | 13, 14, 15 |
| `vllm/compilation/backends.py:1149-1164` | 08 |
| `vllm/compilation/backends.py:1150-1164` | 15 |
| `vllm/compilation/backends.py:1155-1158` | 13 |
| `vllm/compilation/backends.py:1168` | 08 |
| `vllm/compilation/backends.py:1171` | 03, 08, 10 |
| `vllm/compilation/backends.py:1173-1177` | 06 |
| `vllm/compilation/backends.py:1173-1179` | 06, 08 |
| `vllm/compilation/backends.py:1179` | 03, 06, 07, 08 |
| `vllm/compilation/backends.py:1181-1185` | 06 |
| `vllm/compilation/backends.py:1189-1191` | 15 |
| `vllm/compilation/backends.py:1194-1199` | 15 |
| `vllm/compilation/backends.py:1201-1228` | 08 |
| `vllm/compilation/backends.py:1202-1206` | 06 |
| `vllm/compilation/backends.py:1202-1216` | 06 |
| `vllm/compilation/backends.py:1208-1216` | 06 |
| `vllm/compilation/backends.py:1222` | 03, 06, 08, 09, 14 |
| `vllm/compilation/backends.py:1231-1234` | 15 |
| `vllm/compilation/backends.py:1240-1257` | 05 |
| `vllm/compilation/backends.py:1240-1272` | 08 |
| `vllm/compilation/backends.py:1259` | 06 |
| `vllm/compilation/backends.py:1259-1272` | 13, 15 |
| `vllm/compilation/backends.py:1275-1277` | 11 |
| `vllm/compilation/backends.py:1279` | 03, 11 |
| `vllm/compilation/backends.py:1279-1289` | 03, 08, 11 |
| `vllm/compilation/backends.py:1279-1305` | 08 |
| `vllm/compilation/backends.py:1280-1282` | 06, 11 |
| `vllm/compilation/backends.py:1280-1286` | 06, 11 |
| `vllm/compilation/backends.py:1283-1286` | 11 |
| `vllm/compilation/backends.py:1287-1289` | 11 |
| `vllm/compilation/backends.py:1291-1294` | 11 |
| `vllm/compilation/backends.py:1291-1305` | 11 |
| `vllm/compilation/backends.py:1291-1339` | 11 |
| `vllm/compilation/backends.py:1295-1305` | 11 |
| `vllm/compilation/backends.py:1307-1317` | 11 |
| `vllm/compilation/backends.py:1312-1326` | 08 |
| `vllm/compilation/backends.py:1322-1326` | 11 |
| `vllm/compilation/backends.py:1322-1339` | 11 |

## `vllm/compilation/breakable_cudagraph.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/breakable_cudagraph.py:59-117` | 12 |
| `vllm/compilation/breakable_cudagraph.py:125-230` | 12 |
| `vllm/compilation/breakable_cudagraph.py:246` | 12 |

## `vllm/compilation/caching.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/caching.py:37` | 11 |
| `vllm/compilation/caching.py:37-151` | 11, 13 |
| `vllm/compilation/caching.py:59-83` | 11, 13 |
| `vllm/compilation/caching.py:119-140` | 11 |
| `vllm/compilation/caching.py:154-163` | 11 |
| `vllm/compilation/caching.py:166` | 08, 13 |
| `vllm/compilation/caching.py:166-408` | 11 |
| `vllm/compilation/caching.py:178-214` | 11 |
| `vllm/compilation/caching.py:210-214` | 11 |
| `vllm/compilation/caching.py:216-217` | 11 |
| `vllm/compilation/caching.py:219-242` | 11 |
| `vllm/compilation/caching.py:228-233` | 11 |
| `vllm/compilation/caching.py:234-235` | 11 |
| `vllm/compilation/caching.py:251-297` | 11 |
| `vllm/compilation/caching.py:255-259` | 11 |
| `vllm/compilation/caching.py:260-267` | 11 |
| `vllm/compilation/caching.py:269-283` | 11 |
| `vllm/compilation/caching.py:288-296` | 13 |
| `vllm/compilation/caching.py:299-373` | 11, 13 |
| `vllm/compilation/caching.py:319-342` | 11 |
| `vllm/compilation/caching.py:344-373` | 11 |
| `vllm/compilation/caching.py:349-353` | 11 |
| `vllm/compilation/caching.py:360-369` | 11, 13 |
| `vllm/compilation/caching.py:375-401` | 11, 13 |
| `vllm/compilation/caching.py:411-562` | 11 |
| `vllm/compilation/caching.py:429-437` | 11 |
| `vllm/compilation/caching.py:500-509` | 11 |
| `vllm/compilation/caching.py:513-519` | 11 |
| `vllm/compilation/caching.py:528-534` | 11 |
| `vllm/compilation/caching.py:543-553` | 11 |
| `vllm/compilation/caching.py:565-581` | 13 |

## `vllm/compilation/codegen.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/codegen.py:21-127` | 11, 14 |
| `vllm/compilation/codegen.py:42-55` | 11 |
| `vllm/compilation/codegen.py:61-62` | 11 |
| `vllm/compilation/codegen.py:64-90` | 11 |
| `vllm/compilation/codegen.py:78-87` | 11 |
| `vllm/compilation/codegen.py:89` | 11 |
| `vllm/compilation/codegen.py:93-97` | 11 |
| `vllm/compilation/codegen.py:99-104` | 11 |
| `vllm/compilation/codegen.py:106-109` | 11 |
| `vllm/compilation/codegen.py:114-117` | 11 |
| `vllm/compilation/codegen.py:130-161` | 11 |
| `vllm/compilation/codegen.py:131` | 03 |
| `vllm/compilation/codegen.py:161` | 11 |
| `vllm/compilation/codegen.py:164-204` | 11 |
| `vllm/compilation/codegen.py:165` | 08 |
| `vllm/compilation/codegen.py:185-192` | 11, 15 |
| `vllm/compilation/codegen.py:193-204` | 11 |
| `vllm/compilation/codegen.py:198-202` | 11 |
| `vllm/compilation/codegen.py:207-235` | 11 |
| `vllm/compilation/codegen.py:228-235` | 11 |

## `vllm/compilation/compiler_interface.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/compiler_interface.py:27` | 03, 08 |
| `vllm/compilation/compiler_interface.py:27-114` | 08 |
| `vllm/compilation/compiler_interface.py:117-156` | 09 |
| `vllm/compilation/compiler_interface.py:170-190` | 13 |
| `vllm/compilation/compiler_interface.py:193-207` | 08, 13, 15 |
| `vllm/compilation/compiler_interface.py:251-446` | 09, 14 |
| `vllm/compilation/compiler_interface.py:280-414` | 09 |
| `vllm/compilation/compiler_interface.py:289` | 13 |
| `vllm/compilation/compiler_interface.py:390-414` | 13 |
| `vllm/compilation/compiler_interface.py:399-413` | 13 |
| `vllm/compilation/compiler_interface.py:416-446` | 13 |
| `vllm/compilation/compiler_interface.py:431` | 13 |
| `vllm/compilation/compiler_interface.py:449` | 03 |
| `vllm/compilation/compiler_interface.py:463-480` | 09, 13, 14 |
| `vllm/compilation/compiler_interface.py:472-480` | 13 |
| `vllm/compilation/compiler_interface.py:482-658` | 09 |
| `vllm/compilation/compiler_interface.py:500` | 13 |
| `vllm/compilation/compiler_interface.py:606-642` | 09 |
| `vllm/compilation/compiler_interface.py:645-654` | 13, 15 |
| `vllm/compilation/compiler_interface.py:660-727` | 09, 13 |
| `vllm/compilation/compiler_interface.py:753-760` | 09 |
| `vllm/compilation/compiler_interface.py:768-793` | 09 |
| `vllm/compilation/compiler_interface.py:796-810` | 09 |

## `vllm/compilation/counter.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/counter.py:13-41` | 08, 13, 15 |
| `vllm/compilation/counter.py:46-55` | 08 |
| `vllm/compilation/counter.py:58` | 08 |

## `vllm/compilation/cuda_graph.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/cuda_graph.py:32-37` | 12, 15 |
| `vllm/compilation/cuda_graph.py:40-124` | 12, 15 |
| `vllm/compilation/cuda_graph.py:43-49` | 15 |
| `vllm/compilation/cuda_graph.py:71-118` | 15 |
| `vllm/compilation/cuda_graph.py:127-135` | 12 |
| `vllm/compilation/cuda_graph.py:138-142` | 12 |
| `vllm/compilation/cuda_graph.py:145` | 03, 12 |
| `vllm/compilation/cuda_graph.py:161-167` | 12 |
| `vllm/compilation/cuda_graph.py:170-176` | 12 |
| `vllm/compilation/cuda_graph.py:200` | 12 |
| `vllm/compilation/cuda_graph.py:207` | 12 |
| `vllm/compilation/cuda_graph.py:211` | 12 |
| `vllm/compilation/cuda_graph.py:233-256` | 12 |
| `vllm/compilation/cuda_graph.py:233-263` | 12 |
| `vllm/compilation/cuda_graph.py:265-344` | 12 |
| `vllm/compilation/cuda_graph.py:303-332` | 12 |
| `vllm/compilation/cuda_graph.py:304` | 12 |
| `vllm/compilation/cuda_graph.py:325-332` | 12 |
| `vllm/compilation/cuda_graph.py:346-361` | 12, 15 |

## `vllm/compilation/decorators.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/decorators.py:36` | 05 |
| `vllm/compilation/decorators.py:48` | 05 |
| `vllm/compilation/decorators.py:53` | 05 |
| `vllm/compilation/decorators.py:58` | 05 |
| `vllm/compilation/decorators.py:85-125` | 05 |
| `vllm/compilation/decorators.py:118` | 03, 05 |
| `vllm/compilation/decorators.py:201-231` | 05 |
| `vllm/compilation/decorators.py:206` | 05 |
| `vllm/compilation/decorators.py:211-219` | 05 |
| `vllm/compilation/decorators.py:255-262` | 13 |
| `vllm/compilation/decorators.py:265-281` | 05, 13 |
| `vllm/compilation/decorators.py:284-328` | 13 |
| `vllm/compilation/decorators.py:299` | 13 |
| `vllm/compilation/decorators.py:315-328` | 13 |
| `vllm/compilation/decorators.py:331` | 03, 05 |
| `vllm/compilation/decorators.py:340-351` | 05 |
| `vllm/compilation/decorators.py:353-412` | 05 |
| `vllm/compilation/decorators.py:390-395` | 03, 05 |
| `vllm/compilation/decorators.py:406` | 05 |
| `vllm/compilation/decorators.py:414-500` | 05 |
| `vllm/compilation/decorators.py:448-450` | 05 |
| `vllm/compilation/decorators.py:502` | 03, 05 |
| `vllm/compilation/decorators.py:502-520` | 05 |
| `vllm/compilation/decorators.py:525-574` | 13 |
| `vllm/compilation/decorators.py:552-558` | 13 |
| `vllm/compilation/decorators.py:570` | 03 |
| `vllm/compilation/decorators.py:585-604` | 05 |
| `vllm/compilation/decorators.py:592` | 05 |
| `vllm/compilation/decorators.py:604-615` | 13 |
| `vllm/compilation/decorators.py:610-615` | 05 |
| `vllm/compilation/decorators.py:621-624` | 05 |
| `vllm/compilation/decorators.py:621-682` | 05 |
| `vllm/compilation/decorators.py:631-633` | 05 |
| `vllm/compilation/decorators.py:688-717` | 13 |
| `vllm/compilation/decorators.py:701-706` | 13 |
| `vllm/compilation/decorators.py:719` | 05 |
| `vllm/compilation/decorators.py:724-780` | 12 |
| `vllm/compilation/decorators.py:729-737` | 12 |

## `vllm/compilation/monitor.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/monitor.py:14` | 08 |
| `vllm/compilation/monitor.py:18-60` | 15 |
| `vllm/compilation/monitor.py:20` | 15 |
| `vllm/compilation/monitor.py:29` | 08 |
| `vllm/compilation/monitor.py:31-40` | 15 |
| `vllm/compilation/monitor.py:62-84` | 03, 08, 14 |
| `vllm/compilation/monitor.py:63` | 05 |
| `vllm/compilation/monitor.py:63-80` | 05 |
| `vllm/compilation/monitor.py:63-84` | 15 |
| `vllm/compilation/monitor.py:81-84` | 15 |
| `vllm/compilation/monitor.py:87-104` | 08, 15 |
| `vllm/compilation/monitor.py:87-105` | 12 |
| `vllm/compilation/monitor.py:90-100` | 15 |
| `vllm/compilation/monitor.py:90-99` | 08, 12, 15 |

## `vllm/compilation/partition_rules.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/partition_rules.py:14` | 06 |
| `vllm/compilation/partition_rules.py:21-38` | 06 |
| `vllm/compilation/partition_rules.py:41` | 06 |
| `vllm/compilation/partition_rules.py:41-74` | 12 |

## `vllm/compilation/passes/fusion/act_quant_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/act_quant_fusion.py:283` | 10 |

## `vllm/compilation/passes/fusion/add_rms_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/add_rms_fusion.py:140` | 10 |
| `vllm/compilation/passes/fusion/add_rms_fusion.py:152` | 10 |

## `vllm/compilation/passes/fusion/allreduce_rms_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:992` | 10 |
| `vllm/compilation/passes/fusion/allreduce_rms_fusion.py:1134` | 10 |

## `vllm/compilation/passes/fusion/attn_quant_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/attn_quant_fusion.py:362` | 10 |

## `vllm/compilation/passes/fusion/collective_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/collective_fusion.py:900` | 10 |
| `vllm/compilation/passes/fusion/collective_fusion.py:976-983` | 10 |

## `vllm/compilation/passes/fusion/matcher_utils.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/matcher_utils.py:52-83` | 10 |

## `vllm/compilation/passes/fusion/mla_attn_quant_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/mla_attn_quant_fusion.py:574` | 10 |

## `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py:240` | 10 |

## `vllm/compilation/passes/fusion/qk_norm_rope_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/qk_norm_rope_fusion.py:197` | 10 |

## `vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py:415` | 10 |
| `vllm/compilation/passes/fusion/qk_norm_rope_kvcache_fusion.py:488` | 10 |

## `vllm/compilation/passes/fusion/rms_quant_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/rms_quant_fusion.py:47-60` | 10 |
| `vllm/compilation/passes/fusion/rms_quant_fusion.py:118-138` | 15 |
| `vllm/compilation/passes/fusion/rms_quant_fusion.py:395` | 10 |
| `vllm/compilation/passes/fusion/rms_quant_fusion.py:423-449` | 10 |
| `vllm/compilation/passes/fusion/rms_quant_fusion.py:613` | 10 |
| `vllm/compilation/passes/fusion/rms_quant_fusion.py:613-671` | 10 |

## `vllm/compilation/passes/fusion/rope_kvcache_fusion.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/rope_kvcache_fusion.py:412` | 10 |
| `vllm/compilation/passes/fusion/rope_kvcache_fusion.py:461-465` | 10 |

## `vllm/compilation/passes/fusion/sequence_parallelism.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/fusion/sequence_parallelism.py:498` | 10 |
| `vllm/compilation/passes/fusion/sequence_parallelism.py:592-618` | 10 |

## `vllm/compilation/passes/inductor_pass.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/inductor_pass.py:29-37` | 07 |
| `vllm/compilation/passes/inductor_pass.py:29-54` | 10 |
| `vllm/compilation/passes/inductor_pass.py:44` | 09 |
| `vllm/compilation/passes/inductor_pass.py:57-97` | 10 |
| `vllm/compilation/passes/inductor_pass.py:72-79` | 10, 13, 14 |
| `vllm/compilation/passes/inductor_pass.py:110-111` | 10 |
| `vllm/compilation/passes/inductor_pass.py:133-146` | 10 |

## `vllm/compilation/passes/ir/clone_elimination.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/ir/clone_elimination.py:72` | 07, 14 |

## `vllm/compilation/passes/ir/inplace_functionalization.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/ir/inplace_functionalization.py:21` | 07 |

## `vllm/compilation/passes/ir/lowering_pass.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/ir/lowering_pass.py:25` | 07 |

## `vllm/compilation/passes/pass_manager.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/pass_manager.py:91` | 08, 10 |
| `vllm/compilation/passes/pass_manager.py:98-103` | 10 |
| `vllm/compilation/passes/pass_manager.py:109-141` | 10, 15 |
| `vllm/compilation/passes/pass_manager.py:111-139` | 15 |
| `vllm/compilation/passes/pass_manager.py:113-141` | 10 |
| `vllm/compilation/passes/pass_manager.py:119` | 15 |
| `vllm/compilation/passes/pass_manager.py:123-138` | 07 |
| `vllm/compilation/passes/pass_manager.py:141` | 15 |
| `vllm/compilation/passes/pass_manager.py:143-224` | 10 |
| `vllm/compilation/passes/pass_manager.py:146-167` | 10 |
| `vllm/compilation/passes/pass_manager.py:230-252` | 10 |

## `vllm/compilation/passes/utility/fix_functionalization.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/utility/fix_functionalization.py:19` | 10, 14 |
| `vllm/compilation/passes/utility/fix_functionalization.py:48-232` | 10 |

## `vllm/compilation/passes/utility/noop_elimination.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/utility/noop_elimination.py:18` | 10 |

## `vllm/compilation/passes/utility/post_cleanup.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/utility/post_cleanup.py:8` | 10 |

## `vllm/compilation/passes/vllm_inductor_pass.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/passes/vllm_inductor_pass.py:60-79` | 10 |
| `vllm/compilation/passes/vllm_inductor_pass.py:74-80` | 15 |
| `vllm/compilation/passes/vllm_inductor_pass.py:90-92` | 15 |
| `vllm/compilation/passes/vllm_inductor_pass.py:95-190` | 10 |
| `vllm/compilation/passes/vllm_inductor_pass.py:104` | 10, 14, 15 |
| `vllm/compilation/passes/vllm_inductor_pass.py:121-123` | 15 |
| `vllm/compilation/passes/vllm_inductor_pass.py:125-190` | 15 |
| `vllm/compilation/passes/vllm_inductor_pass.py:197-249` | 10 |
| `vllm/compilation/passes/vllm_inductor_pass.py:230-249` | 14 |
| `vllm/compilation/passes/vllm_inductor_pass.py:296-335` | 10 |
| `vllm/compilation/passes/vllm_inductor_pass.py:308-317` | 10 |
| `vllm/compilation/passes/vllm_inductor_pass.py:319-323` | 10 |

## `vllm/compilation/piecewise_backend.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/piecewise_backend.py:26` | 09 |
| `vllm/compilation/piecewise_backend.py:37-76` | 09 |
| `vllm/compilation/piecewise_backend.py:61-75` | 09 |
| `vllm/compilation/piecewise_backend.py:80` | 09 |
| `vllm/compilation/piecewise_backend.py:86` | 03, 06, 09 |
| `vllm/compilation/piecewise_backend.py:108-116` | 09 |
| `vllm/compilation/piecewise_backend.py:131-132` | 06 |
| `vllm/compilation/piecewise_backend.py:137-149` | 09 |
| `vllm/compilation/piecewise_backend.py:137-184` | 09 |
| `vllm/compilation/piecewise_backend.py:165-184` | 09 |
| `vllm/compilation/piecewise_backend.py:189-190` | 03, 08, 14 |
| `vllm/compilation/piecewise_backend.py:190` | 03 |
| `vllm/compilation/piecewise_backend.py:194-207` | 09 |
| `vllm/compilation/piecewise_backend.py:209-243` | 09, 11 |
| `vllm/compilation/piecewise_backend.py:245-277` | 09, 14 |
| `vllm/compilation/piecewise_backend.py:279-304` | 15 |
| `vllm/compilation/piecewise_backend.py:319-341` | 09 |
| `vllm/compilation/piecewise_backend.py:343-356` | 09 |
| `vllm/compilation/piecewise_backend.py:358-380` | 09 |

## `vllm/compilation/wrapper.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/compilation/wrapper.py:26-44` | 05 |
| `vllm/compilation/wrapper.py:47` | 03, 05 |
| `vllm/compilation/wrapper.py:72-160` | 03, 05 |
| `vllm/compilation/wrapper.py:83-103` | 05 |
| `vllm/compilation/wrapper.py:105-125` | 05 |
| `vllm/compilation/wrapper.py:120-122` | 03, 14 |
| `vllm/compilation/wrapper.py:127` | 05 |
| `vllm/compilation/wrapper.py:148` | 03 |
| `vllm/compilation/wrapper.py:148-154` | 03, 05, 14 |
| `vllm/compilation/wrapper.py:156-159` | 05 |
| `vllm/compilation/wrapper.py:171` | 03 |
| `vllm/compilation/wrapper.py:171-201` | 05 |
| `vllm/compilation/wrapper.py:192-197` | 15 |
| `vllm/compilation/wrapper.py:195` | 03 |
| `vllm/compilation/wrapper.py:206-208` | 05 |
| `vllm/compilation/wrapper.py:210-264` | 05, 15 |
| `vllm/compilation/wrapper.py:232` | 13 |
| `vllm/compilation/wrapper.py:232-248` | 15 |
| `vllm/compilation/wrapper.py:250-264` | 15 |
| `vllm/compilation/wrapper.py:273-290` | 05, 14 |
| `vllm/compilation/wrapper.py:293` | 05 |
| `vllm/compilation/wrapper.py:293-346` | 05 |

## `vllm/config/compilation.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/config/compilation.py:37` | 04 |
| `vllm/config/compilation.py:53` | 04 |
| `vllm/config/compilation.py:53-103` | 04, 12 |
| `vllm/config/compilation.py:59-69` | 04 |
| `vllm/config/compilation.py:71-97` | 04 |
| `vllm/config/compilation.py:92-97` | 04 |
| `vllm/config/compilation.py:107` | 04 |
| `vllm/config/compilation.py:334-351` | 05 |
| `vllm/config/compilation.py:339-342` | 05 |
| `vllm/config/compilation.py:349-351` | 05 |
| `vllm/config/compilation.py:355` | 04, 05 |
| `vllm/config/compilation.py:355-394` | 05 |
| `vllm/config/compilation.py:368` | 05 |
| `vllm/config/compilation.py:398` | 04 |
| `vllm/config/compilation.py:461` | 15 |
| `vllm/config/compilation.py:461-463` | 15 |
| `vllm/config/compilation.py:495-507` | 15 |
| `vllm/config/compilation.py:517` | 06 |
| `vllm/config/compilation.py:575` | 04, 09 |
| `vllm/config/compilation.py:595` | 04 |
| `vllm/config/compilation.py:607` | 04 |
| `vllm/config/compilation.py:607-638` | 01 |
| `vllm/config/compilation.py:607-652` | 04 |
| `vllm/config/compilation.py:607-659` | 12 |
| `vllm/config/compilation.py:630-632` | 01 |
| `vllm/config/compilation.py:634-638` | 04 |
| `vllm/config/compilation.py:764-778` | 04 |
| `vllm/config/compilation.py:780-811` | 04 |
| `vllm/config/compilation.py:914-916` | 04 |
| `vllm/config/compilation.py:918-935` | 15 |
| `vllm/config/compilation.py:1062` | 03 |
| `vllm/config/compilation.py:1062-1104` | 03 |
| `vllm/config/compilation.py:1106` | 04 |
| `vllm/config/compilation.py:1106-1131` | 04, 09 |
| `vllm/config/compilation.py:1133` | 04, 06, 14 |
| `vllm/config/compilation.py:1133-1248` | 04 |
| `vllm/config/compilation.py:1147-1155` | 01, 12 |
| `vllm/config/compilation.py:1157-1161` | 04, 06 |
| `vllm/config/compilation.py:1368` | 04, 12 |
| `vllm/config/compilation.py:1368-1516` | 04 |
| `vllm/config/compilation.py:1518` | 04 |
| `vllm/config/compilation.py:1518-1563` | 04 |
| `vllm/config/compilation.py:1565` | 09 |
| `vllm/config/compilation.py:1565-1570` | 04 |

## `vllm/config/model.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/config/model.py:235` | 04 |

## `vllm/config/observability.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/config/observability.py:56` | 12, 15 |

## `vllm/config/utils.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/config/utils.py:387` | 09 |

## `vllm/config/vllm.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/config/vllm.py:104` | 04 |
| `vllm/config/vllm.py:104-116` | 01 |
| `vllm/config/vllm.py:115-116` | 04 |
| `vllm/config/vllm.py:121-128` | 04 |
| `vllm/config/vllm.py:131` | 04 |
| `vllm/config/vllm.py:131-175` | 04 |
| `vllm/config/vllm.py:142` | 04 |
| `vllm/config/vllm.py:155-175` | 04 |
| `vllm/config/vllm.py:229-320` | 04 |
| `vllm/config/vllm.py:322` | 04 |
| `vllm/config/vllm.py:377` | 04 |
| `vllm/config/vllm.py:409` | 04 |
| `vllm/config/vllm.py:811-824` | 04 |
| `vllm/config/vllm.py:819-824` | 04 |
| `vllm/config/vllm.py:826-853` | 04 |
| `vllm/config/vllm.py:972` | 04 |
| `vllm/config/vllm.py:1193-1199` | 04, 15 |
| `vllm/config/vllm.py:1201-1241` | 04 |
| `vllm/config/vllm.py:1211-1241` | 12 |
| `vllm/config/vllm.py:1270` | 04 |
| `vllm/config/vllm.py:1272-1276` | 04 |
| `vllm/config/vllm.py:1285-1292` | 04, 14 |
| `vllm/config/vllm.py:1299-1300` | 04 |
| `vllm/config/vllm.py:1310-1321` | 04 |
| `vllm/config/vllm.py:1375-1422` | 04 |
| `vllm/config/vllm.py:1425-1432` | 04, 15 |
| `vllm/config/vllm.py:1432` | 12 |
| `vllm/config/vllm.py:1682-1690` | 15 |
| `vllm/config/vllm.py:1778` | 04 |
| `vllm/config/vllm.py:1778-1943` | 04 |
| `vllm/config/vllm.py:1945` | 04, 09 |
| `vllm/config/vllm.py:1945-2053` | 04, 09 |
| `vllm/config/vllm.py:2117-2127` | 15 |

## `vllm/distributed/device_communicators/pynccl_allocator.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/distributed/device_communicators/pynccl_allocator.py:63-65` | 12 |
| `vllm/distributed/device_communicators/pynccl_allocator.py:169-198` | 12 |

## `vllm/engine/arg_utils.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/engine/arg_utils.py:451-456` | 04 |
| `vllm/engine/arg_utils.py:669` | 04 |
| `vllm/engine/arg_utils.py:864` | 04 |
| `vllm/engine/arg_utils.py:1437-1440` | 15 |
| `vllm/engine/arg_utils.py:1438` | 12 |
| `vllm/engine/arg_utils.py:1537-1543` | 04 |
| `vllm/engine/arg_utils.py:1598` | 04 |
| `vllm/engine/arg_utils.py:1613` | 04 |

## `vllm/env_override.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/env_override.py:793-858` | 07 |

## `vllm/envs.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/envs.py:34` | 13, 15 |
| `vllm/envs.py:114` | 13 |
| `vllm/envs.py:168` | 09 |
| `vllm/envs.py:766-767` | 05 |
| `vllm/envs.py:824` | 15 |
| `vllm/envs.py:854` | 13 |
| `vllm/envs.py:1899-1906` | 09 |
| `vllm/envs.py:2131` | 13 |
| `vllm/envs.py:2131-2261` | 13 |

## `vllm/forward_context.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/forward_context.py:29-58` | 12 |
| `vllm/forward_context.py:131` | 12 |
| `vllm/forward_context.py:131-149` | 12 |
| `vllm/forward_context.py:132` | 02 |
| `vllm/forward_context.py:148` | 02 |
| `vllm/forward_context.py:149` | 02 |
| `vllm/forward_context.py:161-185` | 01 |
| `vllm/forward_context.py:196-209` | 05 |
| `vllm/forward_context.py:199` | 02 |
| `vllm/forward_context.py:260` | 02 |
| `vllm/forward_context.py:312-316` | 12 |

## `vllm/ir/op.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/ir/op.py:21` | 07 |
| `vllm/ir/op.py:155` | 07 |
| `vllm/ir/op.py:219` | 07 |
| `vllm/ir/op.py:327` | 07, 14 |
| `vllm/ir/op.py:415` | 14 |
| `vllm/ir/op.py:481` | 07 |
| `vllm/ir/op.py:514-539` | 07 |
| `vllm/ir/op.py:542` | 07 |

## `vllm/ir/ops/layernorm.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/ir/ops/layernorm.py:9` | 07 |
| `vllm/ir/ops/layernorm.py:43` | 07 |

## `vllm/model_executor/layers/attention/attention.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/model_executor/layers/attention/attention.py:488` | 07 |
| `vllm/model_executor/layers/attention/attention.py:514-524` | 01 |
| `vllm/model_executor/layers/attention/attention.py:531` | 07 |
| `vllm/model_executor/layers/attention/attention.py:531-541` | 07 |
| `vllm/model_executor/layers/attention/attention.py:533-541` | 01 |
| `vllm/model_executor/layers/attention/attention.py:534-535` | 07 |
| `vllm/model_executor/layers/attention/attention.py:775` | 07 |
| `vllm/model_executor/layers/attention/attention.py:775-798` | 07 |
| `vllm/model_executor/layers/attention/attention.py:817-846` | 07 |
| `vllm/model_executor/layers/attention/attention.py:849-859` | 07, 14 |
| `vllm/model_executor/layers/attention/attention.py:862-867` | 07 |

## `vllm/model_executor/layers/layernorm.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/model_executor/layers/layernorm.py:88` | 07 |

## `vllm/model_executor/models/jamba.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/model_executor/models/jamba.py:302` | 03 |

## `vllm/model_executor/models/llama.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/model_executor/models/llama.py:115-119` | 01 |
| `vllm/model_executor/models/llama.py:116` | 01 |
| `vllm/model_executor/models/llama.py:117` | 01 |
| `vllm/model_executor/models/llama.py:118` | 01 |
| `vllm/model_executor/models/llama.py:221-231` | 01 |
| `vllm/model_executor/models/llama.py:226` | 01 |
| `vllm/model_executor/models/llama.py:227` | 01 |
| `vllm/model_executor/models/llama.py:228` | 01 |
| `vllm/model_executor/models/llama.py:229` | 01 |
| `vllm/model_executor/models/llama.py:230` | 01 |
| `vllm/model_executor/models/llama.py:305-308` | 01 |
| `vllm/model_executor/models/llama.py:310` | 03 |
| `vllm/model_executor/models/llama.py:310-327` | 01 |
| `vllm/model_executor/models/llama.py:319` | 01 |
| `vllm/model_executor/models/llama.py:325` | 01 |
| `vllm/model_executor/models/llama.py:334-343` | 01, 03, 05, 14 |
| `vllm/model_executor/models/llama.py:400-407` | 05 |

## `vllm/model_executor/models/olmoe.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/model_executor/models/olmoe.py:266` | 03 |

## `vllm/model_executor/models/qwen3_next.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/model_executor/models/qwen3_next.py:589` | 03 |

## `vllm/platforms/cpu.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/platforms/cpu.py:182` | 04 |

## `vllm/platforms/interface.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/platforms/interface.py:177` | 08, 14 |
| `vllm/platforms/interface.py:248` | 14 |
| `vllm/platforms/interface.py:1152` | 12 |
| `vllm/platforms/interface.py:1152-1159` | 12 |

## `vllm/utils/argparse_utils.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/utils/argparse_utils.py:322-333` | 04 |
| `vllm/utils/argparse_utils.py:337-427` | 04 |

## `vllm/utils/jit_monitor.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/utils/jit_monitor.py:129` | 09 |

## `vllm/utils/torch_utils.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/utils/torch_utils.py:901` | 07 |
| `vllm/utils/torch_utils.py:901-913` | 01 |
| `vllm/utils/torch_utils.py:933-939` | 07 |

## `vllm/v1/attention/backend.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/attention/backend.py:606` | 04 |
| `vllm/v1/attention/backend.py:606-620` | 04 |

## `vllm/v1/attention/backends/flash_attn.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/attention/backends/flash_attn.py:243-258` | 02 |
| `vllm/v1/attention/backends/flash_attn.py:294-295` | 02 |

## `vllm/v1/core/sched/output.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/core/sched/output.py:193` | 02 |
| `vllm/v1/core/sched/output.py:193-208` | 02 |
| `vllm/v1/core/sched/output.py:205` | 02 |
| `vllm/v1/core/sched/output.py:208` | 02 |

## `vllm/v1/core/sched/scheduler.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/core/sched/scheduler.py:439` | 02 |
| `vllm/v1/core/sched/scheduler.py:439-459` | 02 |
| `vllm/v1/core/sched/scheduler.py:441-450` | 02 |
| `vllm/v1/core/sched/scheduler.py:459` | 02 |
| `vllm/v1/core/sched/scheduler.py:483-485` | 02 |

## `vllm/v1/cudagraph_dispatcher.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/cudagraph_dispatcher.py:15` | 03, 12 |
| `vllm/v1/cudagraph_dispatcher.py:23-24` | 12 |
| `vllm/v1/cudagraph_dispatcher.py:72-109` | 12, 14 |
| `vllm/v1/cudagraph_dispatcher.py:82-91` | 12 |
| `vllm/v1/cudagraph_dispatcher.py:111-130` | 12 |
| `vllm/v1/cudagraph_dispatcher.py:132-156` | 12 |
| `vllm/v1/cudagraph_dispatcher.py:166-233` | 12 |
| `vllm/v1/cudagraph_dispatcher.py:235-324` | 12 |
| `vllm/v1/cudagraph_dispatcher.py:326` | 03 |
| `vllm/v1/cudagraph_dispatcher.py:326-350` | 12 |

## `vllm/v1/engine/async_llm.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/engine/async_llm.py:544` | 02 |
| `vllm/v1/engine/async_llm.py:579` | 02 |
| `vllm/v1/engine/async_llm.py:676` | 02 |

## `vllm/v1/engine/core.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/engine/core.py:103` | 02 |
| `vllm/v1/engine/core.py:584` | 02, 03 |
| `vllm/v1/engine/core.py:584-614` | 02, 03 |
| `vllm/v1/engine/core.py:603-604` | 02 |
| `vllm/v1/engine/core.py:1008` | 02 |
| `vllm/v1/engine/core.py:1378` | 02 |
| `vllm/v1/engine/core.py:1420` | 02 |

## `vllm/v1/metrics/loggers.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/metrics/loggers.py:119-123` | 12 |

## `vllm/v1/worker/gpu_input_batch.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/worker/gpu_input_batch.py:92` | 02 |

## `vllm/v1/worker/gpu_model_runner.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/worker/gpu_model_runner.py:764` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:1192` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:1217` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:1960` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:1960-2079` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:1981-1989` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:2019-2023` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:2023` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:2073-2079` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:2239` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:2338-2340` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:3545` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:3545-3690` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:3562-3564` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:3653` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:3879-3909` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:3912-3930` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:3932` | 02, 12 |
| `vllm/v1/worker/gpu_model_runner.py:3932-4044` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:3975-3990` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:3977` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:3983` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:4029-4036` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:4166` | 01, 02, 03 |
| `vllm/v1/worker/gpu_model_runner.py:4202` | 01 |
| `vllm/v1/worker/gpu_model_runner.py:4265-4299` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:4271` | 01 |
| `vllm/v1/worker/gpu_model_runner.py:4318` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:4378-4392` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:4401-4403` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:4432-4449` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:4432-4456` | 01, 02, 12 |
| `vllm/v1/worker/gpu_model_runner.py:4433-4456` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:4484-4485` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:4553` | 02 |
| `vllm/v1/worker/gpu_model_runner.py:5303` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:5435-5445` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:5449-5478` | 03, 12 |
| `vllm/v1/worker/gpu_model_runner.py:5463-5469` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:5830` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:5943-5944` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:6077-6080` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:6145-6156` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:6156` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:6538-6548` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:6558-6562` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:6814` | 03, 12 |
| `vllm/v1/worker/gpu_model_runner.py:6814-6918` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:6829-6831` | 03, 12 |
| `vllm/v1/worker/gpu_model_runner.py:6876-6884` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:6876-6885` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:6900` | 02, 03, 12, 14, 15 |
| `vllm/v1/worker/gpu_model_runner.py:6911-6917` | 15 |
| `vllm/v1/worker/gpu_model_runner.py:6913-6915` | 15 |
| `vllm/v1/worker/gpu_model_runner.py:6920-6966` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:6968` | 03 |
| `vllm/v1/worker/gpu_model_runner.py:6968-7018` | 12 |
| `vllm/v1/worker/gpu_model_runner.py:7115` | 12 |

## `vllm/v1/worker/gpu_worker.py`

| 锚点 | 引用章 |
|---|---|
| `vllm/v1/worker/gpu_worker.py:435` | 03 |
| `vllm/v1/worker/gpu_worker.py:679` | 00 |
| `vllm/v1/worker/gpu_worker.py:679-718` | 03 |
| `vllm/v1/worker/gpu_worker.py:686` | 03 |
| `vllm/v1/worker/gpu_worker.py:686-708` | 03 |
| `vllm/v1/worker/gpu_worker.py:706-708` | 03 |
| `vllm/v1/worker/gpu_worker.py:713` | 03 |
| `vllm/v1/worker/gpu_worker.py:716-717` | 04, 15 |
| `vllm/v1/worker/gpu_worker.py:827-830` | 09 |

---

共 70 个文件、649 个唯一锚点。