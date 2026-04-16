"""
【教学用例】数据流加工全景演示

阶段：Phase 1 — 建立全局观
对应文档：Introduction/phase1_global_view.md 第一节"数据流加工全景"

核心概念：演示 Inductor 编译管线的四道工序
  工序 0：Dynamo 追踪 → FX GraphModule（原料准备）
  工序 1：FX 前处理（fake_tensor_prop + 算子分解 + 模式匹配）
  工序 2：IR 构建与翻译（GraphLowering.run → call_function × N）
  工序 3：调度与融合（Scheduler：依赖分析 + 贪心融合 + 内存规划）
  工序 4：代码生成（Scheduler.codegen → Triton/C++ kernel + Python wrapper）

教学目标：
  1. 观察每一道工序的"加工前"和"加工后"产品状态
  2. 理解数据形态在管线中的演变：Python → FX → IR → SchedulerNode → Code
  3. 通过 IDE 断点深入理解每个阶段的关键变量

运行方法：
  conda run -n pt-inductor python "my pytorch tutorial/code/src/data_flow_panorama.py"

查看详细日志（可选）：
  TORCH_LOGS="+inductor" conda run -n pt-inductor python "my pytorch tutorial/code/src/data_flow_panorama.py"
"""

import os
import glob
import tempfile

import torch



# ============================================================================
# 原料：教学模型
# ============================================================================

def teaching_model(x: torch.Tensor) -> torch.Tensor:
    """教学模型：log2(x) * 2 + 1

    选择理由（与论文 Figure 2/3 一致）：
    - torch.log2 会被分解为 log * (1/ln2) → 展示"分解"过程
    - 分解后的多个 pointwise 算子可被融合为 1 个 kernel → 展示"融合"
    - 足够简单可读，足够完整地覆盖管线主要分支
    """
    return torch.log2(x) * 2 + 1


# ============================================================================
# 辅助函数
# ============================================================================

def print_fx_graph(gm, title):
    """打印 FX Graph 的节点表"""
    nodes = list(gm.graph.nodes)
    print(f"\n  ┌─ {title} ─────────────────────────────────────────")
    print(f"  │ 节点总数: {len(nodes)}\n  │")
    print(f"  │ {'op':<10} {'name':<15} {'target':<25} {'args'}")
    print(f"  │ {'─'*10} {'─'*15} {'─'*25} {'─'*20}")
    for node in nodes:
        args_str = str(node.args)[:35] if node.args else ""
        target_str = str(node.target)[:25]
        print(f"  │ {node.op:<10} {node.name:<15} {target_str:<25} {args_str}")
    print(f"  └─────────────────────────────────────────────────\n")


def print_section(stage, title, doc_ref, body):
    """统一格式打印每个工序"""
    print(f"\n{'═' * 70}")
    print(f"  【{stage}】{title}")
    print(f"  对应文档：{doc_ref}")
    print(f"{'═' * 70}")
    print(body)


# ============================================================================
# 断点教学指南（核心教学价值）
# ============================================================================

BREAKPOINT_GUIDE = r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🔴 IDE 断点教学指南                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  在 IDE 中设置断点，然后调试运行本文件。                                     ║
║  以下每个断点对应一道工序，包含：断点位置、加工前后对比、关注变量。          ║
║                                                                              ║
║  ── 完整调用链 ───────────────────────────────────────────────────────────  ║
║   compile_fx.py:787      compile_fx_inner()                                 ║
║   → compile_fx.py:1236   codegen_and_compile()                              ║
║     → graph.py:1049      GraphLowering.run()              ← 工序 2         ║
║       → graph.py:1319    call_function() × N              ← 工序 2 子步骤  ║
║     → scheduler.py:3078  Scheduler.__init__()             ← 工序 3         ║
║     → scheduler.py:7324  Scheduler.codegen()              ← 工序 4         ║
║                                                                              ║
║  ── 各工序断点详情 ──────────────────────────────────────────────────────  ║
║                                                                              ║
║  工序 1：FX 前处理                                                           ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ 断点：torch/_inductor/compile_fx.py:1236                           │   ║
║  │       _InProcessFxCompile.codegen_and_compile()                    │   ║
║  │ 加工前：self.gm — 原始 FX Graph（placeholder/log2/mul/add/output）  │   ║
║  │ 加工后：self.gm — 标准化 FX Graph（shape 已传播、算子已分解）       │   ║
║  │ 关注：self.gm.graph.nodes 数量变化                                  │   ║
║  │        node.meta['val'] — FakeTensor（形状/dtype 信息）            │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  工序 2：IR 构建与翻译（GraphLowering.run）                                  ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ 断点：torch/_inductor/graph.py:1049  GraphLowering.run()           │   ║
║  │ 加工前：self.graph.buffers = {} （空）                              │   ║
║  │         self.graph.operations = {} （空）                           │   ║
║  │ 加工后：self.graph.buffers → {buf0: Buffer, buf1: Buffer, ...}     │   ║
║  │         self.graph.operations → 对应的 IR 操作                      │   ║
║  │ 关注：self.graph.buffers — 数据存储视图（仓储物流清单）             │   ║
║  │       self.graph.operations — 计算逻辑视图（车间施工图纸）          │   ║
║  │       每个 buffer.data — Pointwise? Reduction? ExternKernel?       │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  工序 2 子步骤：call_function 处理单个 FX Node                               ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ 断点：torch/_inductor/graph.py:1319  GraphLowering.call_function() │   ║
║  │ 加工前：target = "aten.log" / "aten.mul" / "aten.add"              │   ║
║  │ 加工后：result = TensorBox(StorageBox(Pointwise(...)))             │   ║
║  │ 关注：target — 当前翻译的算子名                                     │   ║
║  │       result — 生成的 IR 节点类型                                   │   ║
║  │       result.data.data.inner_fn — define-by-run IR 核心！           │   ║
║  │         ↑ 这就是论文中的闭包函数，替换 V.ops 即可改变行为           │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  工序 3：调度与融合                                                          ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ 断点：torch/_inductor/scheduler.py:3078  Scheduler.__init__()      │   ║
║  │ 加工前：nodes — 独立 IR 节点（3~5 个小节点）                       │   ║
║  │ 加工后：self.nodes — 融合后 FusedSchedulerNode（1~2 个）            │   ║
║  │ 关注：self.nodes — 融合前后数量对比（最直观的变化！）               │   ║
║  │       self.dependency_data — 依赖图（MemoryDep/StarDep）            │   ║
║  │       node.node_group — 哪些原始节点被融合在一起                    │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  工序 4：代码生成                                                            ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ 断点：torch/_inductor/scheduler.py:7324  Scheduler.codegen()       │   ║
║  │ 加工前：只有 IR 描述（inner_fn 闭包），没有实际代码                 │   ║
║  │ 加工后：生成 .py（Triton）或 .cpp（C++）文件                        │   ║
║  │ 关注：V.graph.wrapper_code — 生成的 Python wrapper 代码             │   ║
║  │       self._grouped_nodes — 待代码生成的节点分组                    │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     Inductor 数据流加工全景 — 教学演示                         ║")
    print("║     对应文档：Introduction/phase1_global_view.md 第一节        ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ================================================================
    # 【原料准备】用户 Python 代码 + 输入张量
    # ================================================================
    print(f"\n{'═' * 70}")
    print("  【原料】用户 Python 代码")
    print(f"{'═' * 70}")

    x = torch.rand(4, 4) * 5 + 0.1  # 正数，避免 log2 负数产生 nan
    print(f"""
  函数定义：
    def model(x):
        return torch.log2(x) * 2 + 1

  输入张量：
    shape={list(x.shape)}, dtype={x.dtype}, device={x.device}
    值域: [{x.min():.2f}, {x.max():.2f}]（正数，避免 log2 产生 nan）

  Eager 模式结果（基准）：
    {teaching_model(x)[0, :4].tolist()}""")

    # ================================================================
    # 【工序 0】Dynamo 追踪 → FX GraphModule
    # ================================================================
    print_section(
        "工序 0", "Dynamo 追踪 → FX GraphModule",
        "数据流加工全景 → '原料准备'",
        """
  加工过程：
    Python 字节码分析 → 构建 FX Graph
    - torch.log2(x) → call_function 节点 (target=aten.log2)
    - * 2           → call_function 节点 (target=aten.mul)
    - + 1           → call_function 节点 (target=aten.add)
    - x             → placeholder 节点
    - 返回值        → output 节点""")

    try:
        exported = torch._dynamo.export(teaching_model)(x)
        gm = exported.graph_module
        print_fx_graph(gm, "产品：Dynamo 追踪产出的 FX Graph")
    except Exception as e:
        print(f"  ⚠ dynamo.export 失败（不影响后续编译）: {e}")

    # ================================================================
    # 【工序 1】FX 前处理
    # ================================================================
    print_section(
        "工序 1", "FX 前处理",
        "数据流加工全景 → '第一道工序'",
        """
  加工过程（在 compile_fx.py:1236 codegen_and_compile() 内部）：

  1. fake_tensor_prop — 用 FakeTensor 传播形状信息
     加工前：FX 节点缺少精确的 shape/dtype
     加工后：每个节点的 meta['val'] 包含 FakeTensor(shape, dtype)

  2. _recursive_post_grad_passes — 算子分解 + 模式匹配
     加工前：torch.log2 是一个大算子（1 个节点）
     加工后：被分解为 aten.log + aten.mul（× 1/ln2 常量）→ 节点数增加
     ⭐ 这是"两阶段优化哲学"的第一阶段铺垫：分解创造碎片，Inlining/Fusion 重新粘合

  3. view_to_reshape — 统一 view 操作（本例不涉及）

  🔴 断点：torch/_inductor/compile_fx.py:1236
           _InProcessFxCompile.codegen_and_compile()
     观察入口处的 self.gm — 标准化后的 FX GraphModule""")

    # ================================================================
    # 【工序 2】IR 构建与翻译
    # ================================================================
    print_section(
        "工序 2", "IR 构建与翻译（GraphLowering.run）",
        "数据流加工全景 → '第二道工序'",
        """
  加工过程（graph.py:1049 GraphLowering.run()）：
    遍历 FX Graph 的每个节点，调用 call_function() 翻译为 IR

    以 log2(x) * 2 + 1 为例（分解后），翻译过程：

    FX Node                →  IR Node
    ─────────────────────────────────────────────────────────────
    placeholder(x)         →  TensorBox(输入 buffer)
    aten.log(x)            →  TensorBox(StorageBox(Pointwise(ops.load+ops.log)))
    aten.mul(log, 1/ln2)   →  TensorBox(StorageBox(Pointwise(ops.mul)))
    aten.mul(result, 2)    →  TensorBox(StorageBox(Pointwise(ops.mul)))
    aten.add(result, 1)    →  TensorBox(StorageBox(Pointwise(ops.add)))

    ⭐ 关键事件：Inlining 发生在此阶段！
    - 多个 pointwise 的 inner_fn 被复制到消费者中
    - 最终可能只有 1 个大的 inner_fn 包含 load→log→mul→mul→add 全部操作
    - 中间的 StorageBox 被标记为 Unrealized（不分配内存）
    - 这就是为什么 V.ops 这么重要：inner_fn 中的 ops.load/ops.log
      不是真正计算，而是通过 V.ops handler 分发到不同行为

  🔴 断点 1：torch/_inductor/graph.py:1049  GraphLowering.run()
     加工前：self.graph.buffers = {} (空)
     加工后：self.graph.buffers 包含所有 Buffer（buf0, buf1...）

  🔴 断点 2：torch/_inductor/graph.py:1319  GraphLowering.call_function()
     加工前：target = "aten.log" / "aten.mul" / "aten.add"
     加工后：result = TensorBox(StorageBox(Pointwise(...)))
     ⭐ 关注：result.data.data.inner_fn — define-by-run IR 的核心闭包！""")

    # ================================================================
    # 【工序 3】调度与融合
    # ================================================================
    print_section(
        "工序 3", "调度与融合（Scheduler）",
        "数据流加工全景 → '第三道工序'",
        """
  加工过程（scheduler.py:3078 Scheduler.__init__()）：

  1. 依赖分析 — extract_read_writes() 通过 V.ops RecordLoadStore handler
     提取每个节点的内存读写模式，构建 MemoryDep 依赖图
     加工前：IR 节点只有计算逻辑（inner_fn），没有依赖关系
     加工后：构建出完整的依赖图（谁读谁的 buffer、谁写谁的 buffer）

  2. 贪心融合 — 最多 10 轮迭代，将可融合的节点合并
     加工前：3~5 个独立 IR 节点
     加工后：1 个 FusedSchedulerNode（本例全部是 pointwise，全部可融合）
     ⭐ 这是"两阶段优化哲学"的第二阶段：Fusion 把碎片粘合回来

  3. 内存规划 — 计算 buffer 生命周期，决定复用策略
     Unrealized 的 buffer 不分配内存，只为最终输出分配

  🔴 断点：torch/_inductor/scheduler.py:3078  Scheduler.__init__()
     加工前：传入的 operations 为多个独立 IR 节点
     加工后：self.nodes 为融合后的 FusedSchedulerNode
     ⭐ 对比 self.nodes 的长度变化——这是融合效果最直观的体现""")

    # ================================================================
    # 【工序 4】代码生成
    # ================================================================
    print_section(
        "工序 4", "代码生成（Codegen）",
        "数据流加工全景 → '第四道工序'",
        """
  加工过程（scheduler.py:7324 Scheduler.codegen()）：

  1. Kernel 代码生成 — 为每个 FusedSchedulerNode 生成后端代码
     CPU → C++/OpenMP 代码（codegen/cpp.py:2000 CppKernel）
     GPU → Triton 代码（codegen/triton.py:2767 TritonKernel）
     本例（CPU）：所有操作融合为 1 个 C++ kernel

  2. Wrapper 代码生成 — 生成 Python 调用入口
     包含 torch.empty() 内存分配 + kernel 调用 + CSE 优化

  3. 编译为共享库（.so），动态加载

  🔴 断点：torch/_inductor/scheduler.py:7324  Scheduler.codegen()
     加工前：只有 IR 描述（inner_fn 闭包）
     加工后：V.graph.wrapper_code 包含完整 Python wrapper
     ⭐ 关注生成的 kernel 代码——它就是 inner_fn 经过 V.ops 翻译后的产物""")

    # ================================================================
    # 触发编译（上面的所有工序在此一次完成）
    # ================================================================
    print(f"\n{'═' * 70}")
    print("  🚀 开始编译（上面的所有工序在此发生）...")
    print(f"{'═' * 70}")

    compiled_model = torch.compile(teaching_model, backend="inductor", fullgraph=True)
    result = compiled_model(x)

    print(f"""
  ✅ 编译完成！结果验证：
    Eager:    {teaching_model(x)[0, :4].tolist()}
    Compiled: {result[0, :4].tolist()}
    数值一致: {torch.allclose(result, teaching_model(x))}""")

    # ================================================================
    # 【最终产品】查看生成的代码
    # ================================================================
    print(f"\n{'═' * 70}")
    print("  【最终产品】生成的代码文件")
    print(f"{'═' * 70}")

    cache_dirs = sorted(
        glob.glob(os.path.join(tempfile.gettempdir(), "torchinductor_*")),
        key=os.path.getmtime,
        reverse=True,
    )

    if cache_dirs:
        cache_dir = cache_dirs[0]
        print(f"\n  代码缓存目录: {cache_dir}")
        for pattern, desc in [("*.cpp", "C++ kernel"), ("*.py", "Wrapper"), ("*.so", "编译产物")]:
            files = sorted(glob.glob(os.path.join(cache_dir, "**", pattern), recursive=True),
                           key=os.path.getmtime, reverse=True)
            if files:
                print(f"\n  {desc}（最新 3 个）:")
                for f in files[:3]:
                    print(f"    {os.path.relpath(f, cache_dir)} ({os.path.getsize(f)} bytes)")
    else:
        print("  未找到缓存目录（可能已被清理）")

    print("""
  💡 查看完整生成代码：
    TORCH_LOGS="+inductor" conda run -n pt-inductor python "my pytorch tutorial/code/src/data_flow_panorama.py"
""")

    # ================================================================
    # 断点教学指南
    # ================================================================
    print(BREAKPOINT_GUIDE)


if __name__ == "__main__":
    main()
