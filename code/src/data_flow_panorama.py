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

教学模型：
  def teaching_model(a, b, c):
      x = a + b       # pointwise: add
      y = x * c       # pointwise: mul
      z = torch.relu(y)  # pointwise: relu
      return x, y, z
  纯 pointwise 链（add → mul → relu），三个输出均被返回。
  全部算子都是 ATen 基础算子，不涉及分解；全部可融合为 1 个 kernel。

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

def teaching_model(a, b, c):
    # 同样构造 3 个相互依赖但都被强制 return 的节点
    x = a + b
    y = x * c
    z = torch.relu(y)
    return x, y, z




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
║  │ 加工前：self.gm — 原始 FX Graph（placeholder/add/mul/relu/output）  │   ║
║  │ 加工后：self.gm — 标准化 FX Graph（shape 已传播）                   │   ║
║  │ 关注：self.gm.graph.nodes — 本例全为基础算子，节点数不变           │   ║
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
║  │       self.graph.operations — 计算逻辑视图（车间施工图纸）            ║   ║
║  │       每个 buffer.data — 全部为 Pointwise（本例无 Reduction 等）    │   ║
║  └──────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  工序 2 子步骤：call_function 处理单个 FX Node                               ║
║  ┌──────────────────────────────────────────────────────────────────────┐   ║
║  │ 断点：torch/_inductor/graph.py:1319  GraphLowering.call_function() │   ║
║  │ 加工前：target = "aten.add" / "aten.mul" / "aten.relu"             │   ║
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
║  │ 加工前：nodes — 独立 IR 节点（3~4 个 pointwise 节点）              │   ║
║  │ 加工后：self.nodes — 融合后 FusedSchedulerNode（1 个）              │   ║
║  │ 关注：self.nodes — 融合前后数量对比（最直观的变化！）               │   ║
║  │       self.dependency_data — 依赖图（MemoryDep）                    │   ║
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

    N = 1024
    a = torch.randn(N)
    b = torch.randn(N)
    c = torch.randn(N)
    print(f"""
  函数定义：
    def teaching_model(a, b, c):
        x = a + b          # pointwise: add
        y = x * c          # pointwise: mul
        z = torch.relu(y)  # pointwise: relu
        return x, y, z

  输入张量：
    a: shape=[{N}], dtype={a.dtype}, device={a.device}
    b: shape=[{N}], dtype={b.dtype}, device={b.device}
    c: shape=[{N}], dtype={c.dtype}, device={c.device}

  Eager 模式结果（基准）：
    x = a + b:  [{", ".join(f"{v:.4f}" for v in (a + b)[:5])}, ...]
    y = x * c:  [{", ".join(f"{v:.4f}" for v in ((a + b) * c)[:5])}, ...]
    z = relu(y): [{", ".join(f"{v:.4f}" for v in torch.relu((a + b) * c)[:5])}, ...]""")

    # ================================================================
    # 【工序 0】Dynamo 追踪 → FX GraphModule
    # ================================================================
    print_section(
        "工序 0", "Dynamo 追踪 → FX GraphModule",
        "数据流加工全景 → '原料准备'",
        """
  加工过程：
    Python 字节码分析 → 构建 FX Graph
    - a, b, c         → placeholder 节点 (3 个输入)
    - x = a + b       → call_function 节点 (target=aten.add)
    - y = x * c       → call_function 节点 (target=aten.mul)
    - z = relu(y)     → call_function 节点 (target=aten.relu)
    - return x, y, z  → output 节点 (3 个输出)
    本例全部为基础 ATen 算子，Dynamo 直接映射，无需分解。""")

    try:
        exported = torch._dynamo.export(teaching_model)(a, b, c)
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

  2. aot_autograd 算子分解 — 将复合算子拆为 ATen 基础算子
     本例的 add/mul/relu 已是基础算子，无需分解，节点数不变
     （对比旧模型：gelu 会被分解为 mul + erf + add + mul 等多个基础算子）

  3. _recursive_post_grad_passes — 后处理优化（模式匹配、常量折叠等）

  4. view_to_reshape — 统一 view 操作（本例不涉及）

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

    本例全部为 Pointwise 算子，翻译过程：

    FX Node                       →  IR Node
    ─────────────────────────────────────────────────────────────────────────────
    placeholder(a)                →  TensorBox(输入 buffer a)
    placeholder(b)                →  TensorBox(输入 buffer b)
    placeholder(c)                →  TensorBox(输入 buffer c)
    aten.add(a, b)                →  TensorBox(StorageBox(Pointwise(ops.add)))
    aten.mul(x, c)                →  TensorBox(StorageBox(Pointwise(ops.mul)))
    aten.relu(y)                  →  TensorBox(StorageBox(Pointwise(ops.relu)))
    output: (x, y, z)            →  三个输出 buffer

    ⭐ 关键事件：Inlining 发生在此阶段！
    - x 被 y 使用，y 被 z 使用 → 形成依赖链 add → mul → relu
    - 多个 pointwise 的 inner_fn 被复制到消费者中
    - 最终 inner_fn 融合为 1 个大的闭包（load_a + load_b → add → load_c → mul → relu）
    - 由于 x 和 y 也被返回，它们对应的 StorageBox 不会被标记为 Unrealized
    - 这就是为什么 V.ops 这么重要：inner_fn 中的 ops.load/ops.add/ops.mul/ops.relu
      不是真正计算，而是通过 V.ops handler 分发到不同行为

  🔴 断点 1：torch/_inductor/graph.py:1049  GraphLowering.run()
     加工前：self.graph.buffers = {} (空)
     加工后：self.graph.buffers 包含所有 Buffer

  🔴 断点 2：torch/_inductor/graph.py:1319  GraphLowering.call_function()
     加工前：target = "aten.add" / "aten.mul" / "aten.relu"
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
     加工后：构建出完整的依赖图
       add 读 a, b → 写 x
       mul 读 x, c → 写 y
       relu 读 y   → 写 z

  2. 贪心融合 — 最多 10 轮迭代，将可融合的节点合并
     加工前：3 个独立 pointwise IR 节点
     加工后：1 个 FusedSchedulerNode（全部 pointwise 融合为 1 个 kernel）
     ⭐ 这是"两阶段优化哲学"的第二阶段：Fusion 把碎片粘合回来

  3. 内存规划 — 计算 buffer 生命周期，决定复用策略
     三个输出（x, y, z）都需要分配内存，因为全部被返回

  🔴 断点：torch/_inductor/scheduler.py:3078  Scheduler.__init__()
     加工前：传入的 operations 为 3 个独立 IR 节点
     加工后：self.nodes 为 1 个 FusedSchedulerNode
     ⭐ 对比 self.nodes 的长度变化——从 3 变为 1，融合效果最直观的体现""")

    # ================================================================
    # 【工序 4】代码生成
    # ================================================================
    print_section(
        "工序 4", "代码生成（Codegen）",
        "数据流加工全景 → '第四道工序'",
        """
  加工过程（scheduler.py:7324 Scheduler.codegen()）：

  1. Kernel 代码生成 — 为 FusedSchedulerNode 生成后端代码
     CPU → C++/OpenMP 代码（codegen/cpp.py:2000 CppKernel）
     GPU → Triton 代码（codegen/triton.py:2767 TritonKernel）
     本例（CPU）：融合后的 add + mul + relu 生成 1 个 C++ kernel

  2. Wrapper 代码生成 — 生成 Python 调用入口
     包含 torch.empty() 内存分配（3 个输出 buffer）+ kernel 调用 + CSE 优化

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
    result = compiled_model(a, b, c)

    eager_x, eager_y, eager_z = teaching_model(a, b, c)
    print(f"""
  ✅ 编译完成！结果验证：
    Eager x:    [{", ".join(f"{v:.4f}" for v in eager_x[:5])}, ...]
    Compiled x: [{", ".join(f"{v:.4f}" for v in result[0][:5])}, ...]
    数值一致: {torch.allclose(result[0], eager_x) and torch.allclose(result[1], eager_y) and torch.allclose(result[2], eager_z)}""")

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


cpu_fusion_target = torch.compile(teaching_model, backend="inductor", fullgraph=True)

if __name__ == "__main__":
    # 【核心修改】：去除 device="cuda"，让张量默认留在 CPU
    N = 1024
    a = torch.randn(N)
    b = torch.randn(N)
    c = torch.randn(N)

    print("开始进行 CPU 编译与执行...")
    out = cpu_fusion_target(a, b, c)
    print("执行完毕！请去 torch_compile_debug 文件夹查看生成的 C++ 源码。")
