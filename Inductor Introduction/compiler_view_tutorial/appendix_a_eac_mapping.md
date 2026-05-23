# 附录 A：Engineering a Compiler 章节完整映射

> 本附录建立了 *Engineering a Compiler* (Keith D. Cooper & Linda Torczon, 3rd Edition) 全部 13 章与 PyTorch Inductor 源码之间的详细映射。每一条目列出 EaC 章节的核心概念、Inductor 中的对应模块与源文件、本书的讨论章节，以及具体的代码示例。

---

## 符号约定

| 缩写 | 含义 |
|------|------|
| **EaC** | *Engineering a Compiler*, 3rd Edition |
| **本书章节** | 本教程的 12 章正文 |
| 源文件路径 | 相对于 PyTorch 仓库根目录 |

---

## 总览表

| EaC 章节 | 主题 | Inductor 模块 | 本书章节 |
|----------|------|--------------|---------|
| Ch.1 | 编译器概览 | 全栈 | 第 1 章 |
| Ch.2 | 词法分析 | Dynamo 字节码拦截 | 第 2 章 |
| Ch.3 | 语法分析 | FX Graph 构建 | 第 2 章 |
| Ch.4 | 语义分析与中间表示 | Inductor IR (ir.py) | 第 3 章 |
| Ch.5 | 语法导向翻译 | Lowering (lowering.py) | 第 4 章 |
| Ch.6 | 编译器前端综述 | Dynamo 字节码分析 | 第 2 章 |
| Ch.7 | 后端综述 | Inductor 后端架构 | 第 1、11 章 |
| Ch.8 | 优化简介 | 图优化 passes | 第 5 章 |
| Ch.9 | 循环优化 | Scheduler 融合与 tiling | 第 7 章 |
| Ch.10 | 指令选择 | Codegen (Triton/C++) | 第 8 章 |
| Ch.11 | 指令调度 | Scheduler 节点排序 | 第 6、10 章 |
| Ch.12 | 寄存器分配 | Buffer 管理 / memory planning | 第 9 章 |
| Ch.13 | 后端编译总结 | 端到端 pipeline | 第 11 章 |

---

## 第 1 章：编译器概览 (Overview of Compilation)

### EaC 核心概念

- 编译器的定义：将源语言程序翻译为目标语言程序，同时保持语义不变
- 三阶段模型：前端 (Frontend) → 中间表示 (IR) → 后端 (Backend)
- 关注点分离 (Separation of Concerns)：前端只关心源语言，后端只关心目标机器
- 编译器的工程复杂性：多 pass、多模块、多阶段

### Inductor 对应模块

| Inductor 组件 | 源文件 | 职责 |
|---------------|--------|------|
| Dynamo (前端) | `torch/_dynamo/` | 拦截 Python 字节码，追踪 tensor 操作 |
| FX (中间表示) | `torch/fx/` | 提供基于 Python 的计算图 IR |
| Inductor (后端) | `torch/_inductor/` | 优化、调度、代码生成 |

### 本书章节

**第 1 章：编译器设计导论与 Inductor 全景** — 完整覆盖 EaC Ch.1 的三阶段模型，并映射到 Dynamo → FX → Inductor 编译栈。

### 具体代码示例

```python
# torch/_inductor/compile_fx.py — compile_fx() 函数是三阶段模型的编排入口
def compile_fx(
    model: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    ...
) -> AotCompiledModule:
    # 阶段 1：前端已完成（Dynamo 已将 Python 代码转换为 FX Graph）
    # 阶段 2 + 3：Inductor 编译（IR 构建 → 优化 → 调度 → codegen）
    return aot_module_simplified(model, example_inputs, ...)
```

`torch.compile()` 本身体现关注点分离：用户代码不因编译而改变，Dynamo、FX、Inductor 各自独立工作。

---

## 第 2 章：词法分析 (Scanning)

### EaC 核心概念

- 正则表达式与有限自动机 (Finite Automaton)
- 词法单元 (Token) 的识别与分类
- 字符流 → 词法单元流的转换
- 关键字、标识符、常量、运算符的识别
- 扫描器的生成器 (Scanner Generator)

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| 字节码拦截器 | `torch/_dynamo/bytecode_transformation.py` | 将 CPython 字节码视为"词法单元" |
| 字节码分析 | `torch/_dynamo/bytecode_analysis.py` | 分析字节码序列的结构 |
| Frame 转换 | `torch/_dynamo/convert_frame.py` | 拦截函数调用帧，启动追踪 |

### 本书章节

**第 2 章：Python 字节码追踪与 FX Graph 构建** — 第 2.2 节讨论 Dynamo 如何拦截和解释 CPython 字节码。

### 具体代码示例

```python
# torch/_dynamo/bytecode_transformation.py
# Dynamo 修改 CPython 的 frame 字节码，插入追踪钩子
# 类比传统编译器的 scanner generator：Dynamo 不生成词法分析器，
# 而是直接操作 CPython 已经生成的字节码（已经完成词法分析）

# torch/_dynamo/convert_frame.py — 拦截函数调用
def convert_frame_assert(fn, compiler_fn):
    """将 Python 函数的执行帧转发给编译器"""
    # 类比：scanner 入口，将源代码（字节码）转换为可分析的形式
```

**类比说明**：EaC 的词法分析器从字符流中识别 Token；Dynamo 从 CPython 字节码中识别 tensor 操作和 Python 控制流。CPython 自身已经完成了"词法分析"（源码 → 字节码），Dynamo 在字节码层面进行二次扫描。

---

## 第 3 章：语法分析 (Parsing)

### EaC 核心概念

- 上下文无关文法 (Context-Free Grammar, CFG)
- 自顶向下分析 (LL) 与自底向上分析 (LR)
- 抽象语法树 (Abstract Syntax Tree, AST)
- 歧义消除与错误恢复
- 递归下降分析器 (Recursive Descent Parser)

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| FX Graph 构建 | `torch/fx/graph.py` | 构建 DAG（类比 AST） |
| FX Node | `torch/fx/node.py` | 图节点（类比 AST 节点） |
| Dynamo 追踪器 | `torch/_dynamo/output_graph.py` | 将字节码操作组织为图结构 |
| FX Interpreter | `torch/fx/interpreter.py` | 图的遍历与解释 |

### 本书章节

**第 2 章：Python 字节码追踪与 FX Graph 构建** — 第 2.3-2.4 节讨论 FX Graph 的构建过程。

### 具体代码示例

```python
# torch/fx/graph.py — Graph 类管理节点集合和边关系
class Graph:
    """FX 计算图——Inductor 编译栈的"AST" """
    def placeholder(self, name): ...     # 输入节点
    def call_function(self, target, args, kwargs): ...  # 函数调用节点
    def call_module(self, target, args, kwargs): ...    # 模块调用节点
    def output(self, result): ...        # 输出节点
    def lint(self): ...                  # 图的合法性检查（类比语法检查）

# torch/fx/node.py — Node 类
class Node:
    """FX 图中的一个节点——类比 AST 中的语法节点"""
    op: str          # "placeholder", "call_function", "call_module", "output"
    target: Callable # 实际调用的函数或模块
    args: Tuple      # 位置参数（定义数据依赖边）
    users: Dict      # 谁使用了我（反向引用）
```

**类比说明**：传统编译器通过语法分析器从 Token 流构建 AST；Dynamo 通过字节码追踪从 Python 执行中构建 FX Graph。FX Graph 的 `placeholder`、`call_function`、`output` 节点类型类比 AST 中的变量声明、函数调用和返回语句。

---

## 第 4 章：语义分析与中间表示 (Semantic Analysis and Intermediate Representations)

### EaC 核心概念

- 中间表示 (IR) 的层次与设计：高层 IR vs 低层 IR
- 静态单赋值形式 (Static Single Assignment, SSA)
- 基本块 (Basic Block) 与控制流图 (Control Flow Graph, CFG)
- 值编号 (Value Numbering)
- 类型系统与类型检查

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| Inductor IR 定义 | `torch/_inductor/ir.py` | 多层次 IR 设计 |
| TensorBox | `ir.py` 中 `class TensorBox` | SSA 式的张量抽象（不可变语义） |
| StorageBox | `ir.py` 中 `class StorageBox` | 存储层的包装（管理 buffer 生命周期） |
| Buffer | `ir.py` 中 `class Buffer` | 已实现 (realized) 的计算结果，类比基本块中的变量 |
| Pointwise / Reduction | `ir.py` 中 `class Pointwise` / `class Reduction` | IR 操作节点类型 |
| LoopBody | `torch/_inductor/loop_body.py` | 循环体的 IR，内含基本块式的指令序列 |
| 大小变量 | `torch/_inductor/sizevars.py` | 符号化的维度表达式管理 |

### 本书章节

**第 3 章：Inductor 中间表示设计** — 完整覆盖，详细讨论 IR 层次结构、SSA 式设计、基本块抽象。

### 具体代码示例

```python
# torch/_inductor/ir.py — IR 层次结构
class IRNode:
    """IR 节点基类——所有 IR 节点的公共接口"""
    def get_name(self): ...
    def realize(self): ...  # 强制实现（类比 SSA 中的 materialization）

class TensorBox(MutableBox):
    """张量盒子——提供类似 SSA 的不可变接口
    外层包装，跟踪 view 操作（如 reshape、permute），
    内部指向 StorageBox，而 StorageBox 指向实际的 IR 数据"""
    pass

class StorageBox(MutableBox):
    """存储盒子——管理底层 IR 数据的生命周期
    一个 StorageBox 可能被多个 TensorBox 共享（view 关系）"""
    pass

class Buffer(IRNode, CodegenSymbol):
    """已实现的计算结果——对应一个需要分配内存的张量
    包含 name（唯一标识）、data（计算逻辑的 IR 节点）、layout（内存布局）"""
    pass

class Pointwise(Loops):
    """逐元素计算的 IR 节点——循环体的每个元素独立计算"""
    # ranges: 循环范围（类比基本块中的迭代空间）
    # body: LoopBody（类比基本块中的指令序列）

class Reduction(Loops):
    """归约操作的 IR 节点——包含归约维度和归约操作"""
    # ranges: 正常循环范围
    # reduction_ranges: 归约维度范围
```

---

## 第 5 章：语法导向翻译 (Syntax-Directed Translation)

### EaC 核心概念

- 翻译方案 (Translation Scheme)
- 属性文法 (Attribute Grammar)：综合属性 (Synthesized) 与继承属性 (Inherited)
- 自顶向下翻译与自底向上翻译
- 模式驱动的翻译规则
- 翻译中的类型转换与隐式规则

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| Lowering 注册表 | `torch/_inductor/lowering.py` | 翻译规则的映射表 |
| GraphLowering | `torch/_inductor/graph.py` | 编排翻译过程 |
| 分解规则 | `torch/_inductor/decomposition.py` | 将复杂算子分解为基本算子 |
| 模式匹配器 | `torch/_inductor/pattern_matcher.py` | 模式驱动的翻译 |
| FX Passes | `torch/_inductor/fx_passes/` | 图级别的翻译变换 |

### 本书章节

**第 4 章：Lowering——从 FX Graph 到 Inductor IR** — 完整覆盖翻译机制。

### 具体代码示例

```python
# torch/_inductor/lowering.py — 翻译规则注册
# 每个注册调用相当于一条翻译规则（类比 EaC 的语义动作）

lowerings: Dict[Callable, Callable] = {}  # 全局翻译规则表

def register_lowering(aten_op, decomp_fn=None):
    """注册一个算子的 lowering 规则——类比翻译方案中的一条产生式"""
    def decorator(fn):
        lowerings[aten_op] = fn  # 将 FX 算子映射到 IR 构建函数
        return fn
    return decorator

# 示例：torch.add 的翻译规则
@register_lowering(aten.add)
def add(lowering_fn, input, other, alpha=1):
    """将 FX 的 aten.add 翻译为 Inductor 的 Pointwise IR"""
    # 类比：语义动作 { val = new_tmp(); emit(IR_ADD, val, input, other) }
    return make_pointwise(
        lambda input, other: input + alpha * other
    )(input, other)

# torch/_inductor/graph.py — GraphLowering 编排翻译
class GraphLowering(torch.fx.Interpreter):
    """遍历 FX Graph，对每个节点调用对应的 lowering 规则"""
    def placeholder(self, target, args, kwargs):
        # 创建输入 TensorBox（继承属性：从输入推导的 shape/dtype）
        ...

    def call_function(self, target, args, kwargs):
        # 查找翻译规则（综合属性：根据 target 和 args 生成 IR 节点）
        return lowerings[target](*args, **kwargs)

    def output(self, target, args, kwargs):
        # 收集输出，realize 所有未实现的 buffer
        ...
```

---

## 第 6 章：编译器前端综述 (The Frontend)

### EaC 核心概念

- 前端的职责：词法分析 → 语法分析 → 语义分析 → IR 生成
- 符号表管理 (Symbol Table)
- 错误检测与恢复
- 前端与后端的接口
- 中间表示的选择

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| Dynamo 追踪入口 | `torch/_dynamo/eval_frame.py` | 前端入口点 |
| Guard 管理 | `torch/_dynamo/guards/` | 符号表/假设管理 |
| Graph break 处理 | `torch/_dynamo/resume_execution.py` | 错误恢复（类比前端的容错机制） |
| 字节码解释器 | `torch/_dynamo/symbolic_convert.py` | 核心追踪逻辑 |
| 假设管理 | `torch/_dynamo/variables/` | 追踪变量的类型和值 |

### 本书章节

**第 2 章：Python 字节码追踪与 FX Graph 构建** — 第 2.5 节讨论 Guard 和 graph break。

### 具体代码示例

```python
# torch/_dynamo/eval_frame.py — 前端入口
def _compile(frame, compiler_fn, ...):
    """拦截 Python 函数调用帧——编译器前端的入口点"""
    # 1. 检查是否可以编译（类比前端的有效性检查）
    # 2. 启动字节码追踪（类比前端的词法+语法分析）
    # 3. 生成 Guard 条件（类比前端的符号表约束）
    # 4. 产出 FX Graph（类比前端输出 IR）
    ...

# torch/_dynamo/guards/ — Guard 管理（类比符号表中的约束）
# Guard 记录编译时的假设条件：
#   "tensor_size == [32, 512]"
#   "dtype == torch.float32"
#   "device == 'cuda:0'"
# 运行时检查这些条件——如果违反，回退到 eager mode（类比前端的错误恢复）
```

---

## 第 7 章：后端综述 (The Backend)

### EaC 核心概念

- 后端的职责：指令选择、寄存器分配、指令调度
- 指令集架构 (ISA) 与目标机器描述
- 编译器后端的组织：三阶段后端模型
- 代码生成策略
- 多目标支持

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| 编译入口 | `torch/_inductor/compile_fx.py` | 后端编排 |
| 调度器 | `torch/_inductor/scheduler.py` | 融合+排序（对应后端三阶段中的调度） |
| Triton codegen | `torch/_inductor/codegen/triton.py` | GPU 指令选择 |
| C++ codegen | `torch/_inductor/codegen/cpp.py` | CPU 指令选择 |
| 内存规划 | `torch/_inductor/codegen/memory_planning.py` | 寄存器/内存分配 |
| 输出封装 | `torch/_inductor/output_code.py` | 最终可执行代码的封装 |

### 本书章节

**第 1 章**（第 4 节，后端架构概览）和 **第 11 章：端到端编译流程回顾**（后端阶段的完整串联）。

### 具体代码示例

```python
# torch/_inductor/compile_fx.py — 后端入口
def compile_fx(model, example_inputs, ...):
    """Inductor 后端的入口函数——编排三阶段后端"""
    # 后端三阶段（EaC Ch.7）：
    #   1. 指令选择 → codegen/triton.py 或 codegen/cpp.py
    #   2. 寄存器分配 → codegen/memory_planning.py
    #   3. 指令调度 → scheduler.py
    ...

# torch/_inductor/codegen/triton.py — GPU 目标（类比特定 ISA 的后端）
class TritonKernel:
    """GPU 后端的代码生成器——将 IR 翻译为 Triton kernel"""
    ...

# torch/_inductor/codegen/cpp.py — CPU 目标（类比另一 ISA 的后端）
class CppKernel:
    """CPU 后端的代码生成器——将 IR 翻译为 C++ 循环"""
    ...
```

---

## 第 8 章：优化简介 (Introduction to Optimization)

### EaC 核心概念

- 优化的定义：在不改变语义的前提下，找到等价但更高效的程序形式
- 常量折叠 (Constant Folding) 与常量传播 (Constant Propagation)
- 公共子表达式消除 (Common Subexpression Elimination, CSE)
- 死代码消除 (Dead Code Elimination, DCE)
- 值编号 (Value Numbering)
- 优化的分类：局部 (Local) vs 全局 (Global) vs 过程间 (Interprocedural)
- 优化的安全性 (Safety) 与收益性 (Profitability)

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| CSE | `torch/_inductor/ir.py` 中 `cse` 相关方法 | 公共子表达式消除 |
| 常量折叠 | `torch/_inductor/constant_folding.py` | 常量折叠 |
| DCE | `torch/_inductor/fx_passes/` 中各 pass | 死代码消除 |
| 后置梯度 pass | `torch/_inductor/fx_passes/post_grad.py` | 反向传播后的优化 |
| 预梯度 pass | `torch/_inductor/fx_passes/pre_grad.py` | 反向传播前的优化 |
| Reinplace | `torch/_inductor/fx_passes/reinplace.py` | 就地化优化（减少内存分配） |
| 索引优化 | `torch/_inductor/optimize_indexing.py` | 索引表达式简化 |

### 本书章节

**第 5 章：图优化** — 完整覆盖各类优化 pass。

### 具体代码示例

```python
# torch/_inductor/constant_folding.py — 常量折叠
# 在编译时计算可确定的常量表达式
class ConstantFolder:
    """遍历 FX Graph，识别可折叠的常量表达式"""
    ...

# torch/_inductor/ir.py — CSE 实现（值编号）
# 在 ir.py 中，通过缓存已计算的 IR 表达式来消除重复计算
# 类比 EaC 中的值编号算法：
#   hash_table = {}
#   for expr in expressions:
#       if expr.key in hash_table:
#           replace expr with hash_table[expr.key]  # CSE
#       else:
#           hash_table[expr.key] = expr

# torch/_inductor/fx_passes/reinplace.py — 就地化
# 将 out-of-place 操作转换为 in-place 操作，减少内存分配
# 例如：aten.add → aten.add_ (节省一次 buffer 分配)
```

---

## 第 9 章：循环优化 (Loop Optimizations)

### EaC 核心概念

- 循环不变量外提 (Loop-Invariant Code Motion, LICM)
- 循环融合 (Loop Fusion) 与循环分裂 (Loop Fission)
- 循环展开 (Loop Unrolling)
- 循环嵌套的交换、反转与倾斜 (Interchange, Reversal, Skewing)
- 循环分块 (Tiling / Blocking)
- 数据局部性优化 (Data Locality)
- 向量化 (Vectorization) 与 SIMD

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| 融合决策 | `torch/_inductor/scheduler.py` | 循环融合的核心逻辑 |
| FusedSchedulerNode | `scheduler.py` 中 `class FusedSchedulerNode` | 融合后的节点 |
| Tiling 策略 | `torch/_inductor/tiling_utils.py` | 分块策略 |
| Triton kernel 生成 | `torch/_inductor/codegen/triton.py` | 向量化与并行化 |
| CPU SIMD | `torch/_inductor/codegen/simd.py` | CPU 向量化 |
| 融合区域分析 | `torch/_inductor/fx_passes/fusion_regions.py` | 融合候选区域识别 |

### 本书章节

**第 7 章：融合策略与循环优化** — 完整覆盖，是全书最核心的后端优化章节。

### 具体代码示例

```python
# torch/_inductor/scheduler.py — 循环融合
class Scheduler:
    def _fuse_nodes(self):
        """循环融合的决策算法
        遍历所有 SchedulerNode，根据依赖关系和代价模型
        决定哪些节点应该融合为一个 kernel"""
        ...

    def _can_fuse(self, node1, node2):
        """判断两个节点是否可以融合
        条件：依赖关系兼容、没有写-写冲突、内存带宽收益"""
        ...

# torch/_inductor/codegen/triton.py — GPU 向量化与并行
# Triton kernel 中的 BLOCK_SIZE 参数对应 tiling 的块大小
# 每个 program instance 处理一个 tile——类比循环分块
class TritonKernel:
    def _generate_kernel(self, ...):
        # 生成形如下列的 Triton 代码：
        # @triton.jit
        # def kernel(ptr, BLOCK_SIZE: tl.constexpr):
        #     offsets = tl.arange(0, BLOCK_SIZE)  # -- tiling
        #     data = tl.load(ptr + offsets)        # -- 向量化内存访问
        #     ...
```

---

## 第 10 章：指令选择 (Instruction Selection)

### EaC 核心概念

- 指令选择的定义：将 IR 操作映射为目标机器的指令
- 树模式匹配 (Tree-Pattern Matching)
- DAG 模式匹配
- 覆盖 (Covering) 与最优覆盖
- 指令选择中的代价模型
- 基于语法的指令选择 (Grammar-Based Selection)

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| Triton codegen | `torch/_inductor/codegen/triton.py` | GPU 指令选择 |
| C++ codegen | `torch/_inductor/codegen/cpp.py` | CPU 指令选择 |
| GEMM 模板 | `torch/_inductor/codegen/cpp_gemm_template.py` | 矩阵乘法的专用指令选择 |
| Triton kernel 模板 | `torch/_inductor/codegen/triton.py` | Triton kernel 模式匹配 |
| 算法选择 | `torch/_inductor/select_algorithm.py` | 选择最优算法实现 |
| 外部 kernel | `torch/_inductor/codegen/custom_extern_kernel_codegen.py` | 调用外部优化的 kernel |
| CUTLASS 集成 | `torch/_inductor/codegen/cutlass/` | 使用 CUTLASS 库的 GEMM 实现 |

### 本书章节

**第 8 章：指令选择与代码生成** — 完整覆盖。

### 具体代码示例

```python
# torch/_inductor/codegen/triton.py — GPU 指令选择
# 将 Inductor IR 中的操作"翻译"为 Triton 内置操作
# 类比 EaC 中的树模式匹配：
#
#   IR: Pointwise(op=mul, inputs=[a, b])
#   →  Triton: tl.load(a_ptr + offsets) * tl.load(b_ptr + offsets)
#
#   IR: Reduction(op=sum, axis=0)
#   →  Triton: tl.sum(data, axis=0)

# torch/_inductor/select_algorithm.py — 算法选择
# 对于矩阵乘法等操作，选择最优实现：
#   - Triton kernel（通用实现）
#   - CUTLASS kernel（高度优化的 GEMM）
#   - cuBLAS（供应商库）
# 类比指令选择中的代价模型：不同"指令"有不同的代价

# torch/_inductor/codegen/cpp.py — CPU 指令选择
# 将 IR 操作映射为 C++ intrinsics：
#   IR: Pointwise(op=add) → C++: _mm256_add_ps (AVX2)
#   IR: Pointwise(op=mul) → C++: _mm256_mul_ps (AVX2)
```

---

## 第 11 章：指令调度 (Instruction Scheduling)

### EaC 核心概念

- 指令调度的定义：确定操作的最优执行顺序
- 列表调度 (List Scheduling)
- 关键路径 (Critical Path) 分析
- 前向调度与后向调度
- 软件流水线 (Software Pipelining)
- 调度中的资源约束
- 依赖图的构建与分析

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| 调度器 | `torch/_inductor/scheduler.py` | 节点排序的核心逻辑 |
| 依赖分析 | `torch/_inductor/dependencies.py` | 依赖图构建 |
| MemoryDep | `dependencies.py` 中 `class MemoryDep` | 内存依赖（读-写、写-读、写-写） |
| StarDep / WeakDep | `dependencies.py` | 特殊依赖类型 |
| 关键路径计算 | `scheduler.py` 中相关逻辑 | 调度优先级计算 |
| CUDA Graph | `torch/_inductor/cudagraph_trees.py` | 调度与 CUDA graph 集成 |

### 本书章节

**第 6 章：依赖分析与调度前置**（依赖图构建）和 **第 10 章：指令调度**（调度算法）。

### 具体代码示例

```python
# torch/_inductor/dependencies.py — 依赖图构建
class MemoryDep(Dep):
    """内存依赖——记录两个节点之间的内存访问冲突
    类比 EaC 中的依赖边：true dependence (RAW), anti dependence (WAR),
    output dependence (WAW)"""
    def __init__(self, name, index, size):
        self.name = name     # buffer 名称
        self.index = index   # 访问的索引范围
        self.size = size     # 访问的大小

# torch/_inductor/scheduler.py — 调度算法
class Scheduler:
    def _schedule_nodes(self):
        """节点排序——类比列表调度算法
        1. 构建依赖图
        2. 计算关键路径长度
        3. 按优先级排序（关键路径优先、融合组优先）
        4. 生成最终执行顺序"""
        ...

    def _create_iteration_order(self):
        """确定迭代顺序——前向调度或后向调度"""
        ...
```

---

## 第 12 章：寄存器分配 (Register Allocation)

### EaC 核心概念

- 寄存器分配的定义：将无限虚拟寄存器映射到有限物理寄存器
- 图着色算法 (Graph Coloring)
- 活跃性分析 (Liveness Analysis)
- 干涉图 (Interference Graph)
- 寄存器合并 (Coalescing)
- 寄存器溢出 (Spilling)
- 寄存器重命名

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| Buffer 管理 | `torch/_inductor/ir.py` 中 `Buffer` 类 | 虚拟"寄存器"（每个 buffer 对应一个虚拟寄存器） |
| 内存规划 | `torch/_inductor/codegen/memory_planning.py` | 内存分配与复用（类比寄存器分配） |
| Memory donation | `torch/_inductor/memory.py` | 内存捐赠/复用（类比寄存器合并） |
| Buffer 名称管理 | `torch/_inductor/utils.py` | 唯一命名（类比虚拟寄存器编号） |
| 输出代码 | `torch/_inductor/output_code.py` | 最终封装中的 buffer 生命周期管理 |

### 本书章节

**第 9 章：内存管理与缓冲区分配** — 完整覆盖。

### 具体代码示例

```python
# torch/_inductor/codegen/memory_planning.py — 内存分配
# 类比图着色寄存器分配：
#   Buffer = 虚拟寄存器
#   物理内存 = 物理寄存器
#   Buffer 生命周期 = 活跃区间
#   内存复用 = 寄存器合并
#
# 算法：
#   1. 计算每个 buffer 的生命周期（活跃性分析）
#   2. 将同时活跃的 buffer 映射到不同的内存区域（干涉图着色）
#   3. 合并生命周期不重叠的 buffer（寄存器合并）
#   4. 溢出过大的 buffer 到动态分配区（寄存器溢出）

class MemoryPlanning:
    """管理 buffer 的内存分配——Inductor 版本的寄存器分配"""
    ...

# torch/_inductor/memory.py — Memory donation
# 当一个 buffer 不再需要时，将其内存"捐赠"给后续的 buffer
# 类比寄存器合并：两个不冲突的变量共享同一个寄存器
```

---

## 第 13 章：后端编译总结 (The Back End — A Look Back)

### EaC 核心概念

- 后端三阶段的交互：指令选择 ↔ 寄存器分配 ↔ 指令调度
- 编译器后端的工程组织
- 不同优化之间的交互与冲突
- 端到端的正确性验证
- 编译器测试与质量保证

### Inductor 对应模块

| Inductor 组件 | 源文件 | 对应概念 |
|---------------|--------|---------|
| 编译主流程 | `torch/_inductor/graph.py` | 端到端编排 |
| 编译入口 | `torch/_inductor/compile_fx.py` | 流程协调 |
| AOT Autograd | `torch/_functorch/aot_autograd.py` | 自动微分与编译的集成 |
| 编译缓存 | `torch/_inductor/cache.py` | 编译结果缓存与复用 |
| CUDA Graph 集成 | `torch/_inductor/cudagraph_trees.py` | 运行时优化 |
| 包装器生成 | `torch/_inductor/codegen/wrapper.py` | 最终可执行代码封装 |

### 本书章节

**第 11 章：端到端编译流程回顾** — 完整覆盖，将所有后端阶段串联起来。

### 具体代码示例

```python
# torch/_inductor/graph.py — GraphLowering.compile()
# 展示后端三阶段的完整交互：
class GraphLowering:
    def compile(self):
        """端到端编译流程"""
        # 阶段 1：翻译（EaC Ch.5）
        self._lower_graph()  # FX Graph → Inductor IR

        # 阶段 2：优化（EaC Ch.8）
        self._optimize_ir()  # CSE, DCE, 常量折叠

        # 阶段 3：调度 + 代码生成（EaC Ch.9-12）
        # 3a. 依赖分析（Ch.11）
        # 3b. 融合决策（Ch.9）
        # 3c. 节点排序（Ch.11）
        # 3d. 代码生成（Ch.10）
        # 3e. 内存规划（Ch.12）
        scheduler = Scheduler(self.buffers)
        scheduler.run()  # 执行 3a-3e

        # 阶段 4：封装为可执行模块
        wrapper = WrapperCodeGen()
        wrapper.generate(scheduler.scheduled_nodes)
        return wrapper.compile()
```

---

## 交叉引用表：本书章节 → EaC 章节

下表从本书视角列出每章涉及的 EaC 章节，方便读者按需回溯理论背景。

| 本书章节 | 主要 EaC 章节 | 次要 EaC 章节 | 理论重点 |
|---------|-------------|-------------|---------|
| 第 1 章：编译器设计导论 | Ch.1, Ch.7 | Ch.3 | 三阶段模型、关注点分离 |
| 第 2 章：FX Graph 构建 | Ch.2, Ch.3, Ch.6 | Ch.1 | 词法/语法分析、前端综述 |
| 第 3 章：IR 设计 | Ch.4 | Ch.5 | SSA、CFG、IR 层次 |
| 第 4 章：Lowering | Ch.5 | Ch.4 | 语法导向翻译、属性文法 |
| 第 5 章：图优化 | Ch.8 | — | CSE、DCE、常量折叠 |
| 第 6 章：依赖分析 | Ch.11 | Ch.4 | 依赖图、活跃性分析 |
| 第 7 章：融合与循环优化 | Ch.9 | Ch.10 | 融合、tiling、向量化 |
| 第 8 章：代码生成 | Ch.10 | — | 指令选择、模式匹配 |
| 第 9 章：内存管理 | Ch.12 | Ch.11 | 图着色、活跃性分析 |
| 第 10 章：指令调度 | Ch.11 | Ch.9 | 列表调度、关键路径 |
| 第 11 章：端到端回顾 | Ch.13, Ch.7 | — | 后端组织、三阶段交互 |
| 第 12 章：生态协同 | Ch.1 | Ch.7 | 编译器设计空间 |

---

## 概念密度热力图

下表标出每对 (EaC 章节, Inductor 模块) 的关联强度，`***` 表示强关联，`*` 表示弱关联。

| EaC 章节 \ Inductor 模块 | Dynamo | FX | Lowering | IR | Scheduler | Dependencies | Codegen | Memory Planning |
|--------------------------|--------|-----|----------|-----|-----------|-------------|---------|-----------------|
| Ch.1 编译器概览 | ** | ** | * | * | * | * | * | * |
| Ch.2 词法分析 | *** | * | | | | | | |
| Ch.3 语法分析 | * | *** | | | | | | |
| Ch.4 中间表示 | | * | * | *** | * | | | |
| Ch.5 语法导向翻译 | | * | *** | * | | | | |
| Ch.6 前端综述 | *** | ** | | | | | | |
| Ch.7 后端综述 | | | * | * | ** | * | *** | ** |
| Ch.8 优化简介 | | | * | *** | * | | | |
| Ch.9 循环优化 | | | | * | *** | * | ** | * |
| Ch.10 指令选择 | | | | * | | | *** | |
| Ch.11 指令调度 | | | | | *** | *** | * | * |
| Ch.12 寄存器分配 | | | | * | * | * | | *** |
| Ch.13 后端总结 | | | * | * | ** | * | ** | ** |

---

## 阅读建议

**如果你正在学习 EaC**：按 EaC 章节顺序，参考本附录找到对应的 Inductor 源码，将抽象理论映射到具体实现。

**如果你正在阅读本书**：遇到编译器理论术语时，参考本附录找到对应的 EaC 章节进行深入学习。

**如果你正在调试 Inductor**：根据调试涉及的模块，找到对应的 EaC 章节，理解该模块的设计原理和理论基础。
