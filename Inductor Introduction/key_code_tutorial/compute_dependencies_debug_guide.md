# `Scheduler.compute_dependencies()` 深度讲解

## 设计哲学与编译器对应

**What：** `compute_dependencies()` 构建一张节点间的依赖 DAG，为后续拓扑排序和 fusion 提供正确性保证。

**Why：** 传统编译器中，依赖分析服务于指令调度和循环变换。但 Inductor 面临三个额外复杂性：
1. **别名（Aliasing）**：PyTorch 的 view 语义使多个 buffer 名指向同一块内存
2. **原地修改（Mutation）**：`x.add_(1)` 直接修改输入，打破 SSA 不变性
3. **动态符号（Unbacked SymInt）**：运行时才知道的尺寸，引入隐式依赖

**编译器对应：** Dependence Analysis（Def-Use Chain、RAW/WAR/WAW 三类依赖）

**类比：** 想象一个餐厅后厨。`compute_dependencies` 就是建立一张"工序依赖表"：
- 做酱汁必须等洋葱炒好（True Dependence — RAW）
- 洗锅必须等炒完菜（Anti-Dependence — WAR）
- 两个人用同一个砧板（别名），必须排队
- 某道菜的大小要等顾客下单后才知道（动态符号）

---

## 核心数据结构

### Dep 层次体系

```
Dep (抽象基类)
├── MemoryDep   — 精确的内存区域依赖（name + index + size）
├── StarDep     — 整个 buffer 的依赖（"我要用这个 buffer 的全部"）
└── WeakDep     — 弱依赖（只约束顺序，不约束 buffer 生命周期）
```

```
API: MemoryDep(name, index, var_names, size, mode)
├── 语义：表示对一个 buffer 中特定区域的读写依赖
├── 字段：
│   - name: str — buffer 名（如 "buf0"）
│   - index: sympy.Expr — 访问的索引表达式（如 4*i + j）
│   - var_names: tuple[sympy.Symbol, ...] — 索引中使用的迭代变量
│   - size: tuple[sympy.Expr, ...] — 每个维度的范围
│   - mode: str | None — 存储模式（如 "store"）
├── 用途：精确的逐元素依赖分析，支持 fusion 决策中的重排判断
└── 编译器对应：Array Dependence Analysis
```

```
API: StarDep(name, mode)
├── 语义：对整个 buffer 的依赖（不关心具体哪些元素）
├── 字段：name: str, mode: str | None
├── 用途：mutation 依赖 / unbacked symint 隐式依赖 / 保护输出不被 DCE 消除
├── 不变量：没有精确 index，无法用于 fusion 的重排判断
└── 编译器对应：Coarse-grained Dependence
```

```
API: WeakDep(name, mutating_buf, is_fake)
├── 语义：弱依赖——只需要顺序约束，不延长 buffer 生命周期
├── 字段：
│   - name: str — 被依赖的 buffer 名
│   - mutating_buf: str — 触发此弱依赖的修改操作对应的 buffer
│   - is_fake: bool — 是否是纯控制依赖
├── HOW（核心区别）：
│   - is_fake=False（真弱依赖）：view 与 mutation 共享存储，必须 keep buffer 活着
│   - is_fake=True（假弱依赖）：clone 后 mutation，只需顺序约束，可被 DCE 优化掉
├── 用途：处理 WAR（anti-dependence）场景
└── 编译器对应：Anti-dependence with liveness tracking
```

### NodeUser

```
API: NodeUser(node, can_inplace, is_weak)
├── 语义：记录"谁使用了一个 buffer"
├── 字段：
│   - node: BaseSchedulerNode | OutputNode — 使用者节点
│   - can_inplace: bool — 是否可以原地操作
│   - is_weak: bool — 是否是弱使用者（只约束顺序）
├── 用途：
│   - can_inplace=True → fusion 时可以做 in-place 优化
│   - is_weak=True → DCE 时可以忽略
└── 存储位置：SchedulerBuffer.users: list[NodeUser]
```

### DedupList

```
API: DedupList(items, membership)
├── 语义：去重列表——append 时自动去重，保留 list 语义（可边遍历边 append）
├── HOW：
│   - items: list[_T] — 实际元素列表
│   - membership: OrderedSet[_T] — O(1) 去重检查
│   - append(x): 已存在则跳过，否则加入 items + membership
│   - __add__: 合并两个 DedupList，去重后返回新对象
├── Why 不用 OrderedSet：
│   主循环中边遍历 name_to_users 边 append，Python 不允许迭代时修改 dict/set。
│   DedupList 用 items 列表保持迭代稳定性，用 membership 保证去重。
└── 编译器对应：SSA 构建中的 work-list 数据结构
```

---

## 全局数据流图

```
[ 编译分析期 | Dependency Analysis Phase ]
(静态分析，扫描节点，确立图纸与等待名单)

(1) 基础设施搭建
    │  └── 数据流 0 (容器初始化)：创建 name_to_users: defaultdict[str, DedupList[NodeUser]]
    ▼
【 name_to_users (空的 defaultdict) 】
    │
    │
(2) 别名合并
    │  └── 数据流 3 (别名穿透)：提取 buf1.get_aliases()，让 name_to_users 中多个 key 指向同一 DedupList 实例
    ▼
【 name_to_users (别名已合并，多个 key 共享 DedupList 指针) 】
    │
    │
(3) Unbacked Symint 注册
    │  ├── 数据流 4a (输入符号)：遍历 V.graph.graph_inputs，提取 free_symbols → unbacked_symbol_to_origin_node[s] = None
    │  └── 数据流 4b (节点符号)：遍历 nodes，提取 node.node.get_unbacked_symbol_defs() → unbacked_symbol_to_origin_node[s] = node_name
    ▼
【 unbacked_symbol_to_origin_node (符号→源节点映射) 】
    │
    │
(4) 主循环：逐节点建立依赖边
    │  ├── 数据流 4 (符号锁死)：提取 node.node.get_free_symbol_uses()，查符号来源，生成 StarDep → node.unmet_dependencies
    │  ├── 数据流 2 (防脏读锁)：提取欲覆写的 alt_name，查阅 name_to_users[alt_name] 拿老读者名单，生成 WeakDep
    │  ├── 数据流 1 (常规读取)：遍历 node.read_writes.reads，将 Node 追加注册到 name_to_users[read.name]
    │  └── 数据流 6 (版本映射)：更新 mutation_renames[alt_name] = buf_name，维护重命名链
    ▼
【 name_to_users (全局字典，填充完成) & Node.read_writes (含额外依赖锁) & mutation_renames (版本链) 】
    │
    │  ========== (分析期结束：资产转移、副作用上报与销毁) ==========
    ▼
(5) 保护性依赖
    │  ├── 数据流 5a (输出保护)：V.graph.get_output_names() → add_user(buf_name, OutputNode)
    │  ├── 数据流 5b (符号保护)：unbacked symint 的定义节点 → add_user(buf_name, OutputNode)
    │  └── 数据流 5c (输入变异保护)：mutation_renames 中的输入名 → add_user + V.graph.mutated_inputs.add(name)
    ▼
【 name_to_users (含 OutputNode 保护) & V.graph.mutated_inputs (被修改的输入集合) 】
    │
    │
(6) 终局数据落地
    │  ├── 动作 A (名单挂载)：name_to_users[buf_name].items → buf.set_users()，随后 name_to_users 被销毁
    │  └── 动作 B (捐赠转移)：name_to_donated_buffer 同样执行 set_users
    ▼
【 Buffer.users (出度名单确立) & V.graph.mutated_inputs (向外层引擎通报完成) 】
```

---

## Pass 1：粗粒度功能拆解

整个 `compute_dependencies()`（`scheduler.py:3476-3745`）分为 **6 个阶段**：

| 阶段 | 行范围 | 功能 | 一句话总结 |
|------|--------|------|------------|
| **阶段 1：基础设施** | 3482-3516 | 定义 DedupList，初始化 name_to_users | 建造依赖收集器 |
| **阶段 2：别名合并** | 3518-3548 | 让别名 buffer 共享同一个用户列表 | 处理 view 语义 |
| **阶段 3：Unbacked Symint 注册** | 3550-3600 | 定义 rename/add_user，注册 unbacked symint 源 | 处理隐式依赖 |
| **阶段 4：主循环** | 3602-3688 | 遍历每个节点，建立所有类型的依赖边 | **核心**：构建依赖 DAG |
| **阶段 5：保护性依赖** | 3690-3725 | 保护输出/mutation/unbacked 不被 DCE | 防止优化过度 |
| **阶段 6：结果写入** | 3727-3745 | 将 name_to_users 写入各 buffer | 持久化依赖信息 |

---

## Pass 2：逐行详解

### 阶段 1：基础设施 (3482-3516)

```python
# ========== 阶段 1: 基础设施 ==========
# ┌─ 设计哲学：为后续所有阶段准备核心数据结构
# └─ 核心不变量：name_to_users 是整个函数的"一张大表"

    class DedupList(Generic[_T]):
        # 见上文"核心数据结构"部分
        # 局部类，仅在 compute_dependencies 内部可见
        # Why 局部类：不需要在外部复用，保持最小作用域
```

```python
    name_to_users: defaultdict[str, DedupList[NodeUser]] = collections.defaultdict(
        DedupList
    )
```

```
# [CORE] ★★★ 核心数据结构 ★★★
#
# 🔴 断点：scheduler.py:3514
#
# What：buffer 名 → 使用该 buffer 的节点列表
# How (defaultdict)：访问不存在的 key 时自动调用 DedupList() 创建空实例
# Why defaultdict：比 dict 方便，不需要检查 key 是否存在
#
# 生命周期：
#   阶段 2: 别名合并让多个 key 指向同一 DedupList 实例
#   阶段 4: add_user() 向其中添加条目
#   阶段 6: .items 写入 SchedulerBuffer.users，随后被 GC 销毁
#
# 示例（加工完成后）：
#   name_to_users = {
#     "x":    DedupList([NodeUser(node=buf0_node)]),
#     "buf0": DedupList([NodeUser(node=buf1_node)]),
#     "buf1": DedupList([NodeUser(node=OUTPUT)]),
#   }
```

---

### 阶段 2：别名合并 (3518-3548)

```
数据流 3 (别名穿透)：
  来源：node.get_outputs() → buf1.get_aliases()
  加工：让 name_to_users 中多个 key 指向同一个 DedupList 实例
  去向：name_to_users[alias_a] is name_to_users[alias_b]（同一 Python 对象）
```

```python
# ========== 阶段 2: 别名合并 ==========
# ┌─ 设计哲学：view 操作使多个 buffer 名指向同一块内存。
# │   如果 foo 是 bar 的 view，foo 的用户也应被记录为 bar 的用户。
# │   否则 DCE 可能错误消除 bar，导致 foo 读到无效数据。
# └─ 编译器对应：Alias Analysis + SSA 中的 φ 节点合并
#
# 核心技巧：Python 对象别名
#   让 name_to_users["foo"] 和 name_to_users["bar"] 指向同一 DedupList
#   向 foo 的用户列表添加时，bar 的也自动更新

    for node in self.nodes:
        for buf1 in node.get_outputs():
            buf1_name = buf1.get_name()
```

```
# [CORE] 遍历每个节点的每个输出 buffer
# buf1: SchedulerBuffer, buf1_name: str
```

```python
            if (
                isinstance(buf1.node.layout, ir.NoneLayout)
                and len(buf1.get_aliases()) > 1
            ):
                continue
```

```
# [AUX] 边界情况：auto functionized ops 返回 None 且 mutate 多个输入
#   这些 buffer 之间不一定互为别名，不能让它们共享 user list
```

```python
            for buf2_name in buf1.get_aliases():
                if buf1_name in name_to_users and buf2_name in name_to_users:
                    # merge the two
                    list1 = name_to_users[buf1_name]
                    list2 = name_to_users[buf2_name]
                    combined = list1 + list2
                    for key in name_to_users:
                        if (
                            name_to_users[key] is list1
                            or name_to_users[key] is list2
                        ):
                            name_to_users[key] = combined
                elif buf1_name in name_to_users:
                    name_to_users[buf2_name] = name_to_users[buf1_name]
                else:
                    name_to_users[buf1_name] = name_to_users[buf2_name]
```

```
# [CORE] ★★★ 别名合并的三种情况 ★★★
#
# 🔴 断点：scheduler.py:3534
#
# What：让互为别名的 buffer 共享同一个 user list
#
# How（三种情况）：
#   场景 1（两个都有独立 list）：创建合并 DedupList = list1 + list2，
#     遍历 name_to_users 更新所有指向 list1/list2 的 key
#     Why 遍历更新：可能还有其他 key 也指向同一 list（传递别名）
#
#   场景 2（只有 buf1 有 list）：buf2 指向 buf1 的 list（Python 引用赋值）
#     效果：后续向 buf2 添加用户，实际添加到 buf1 的 list
#
#   场景 3（两个都没有）：buf1 指向 buf2（defaultdict 自动创建空 DedupList）
#     效果：后续无论向哪个添加用户，另一个也同步更新
#
# Why Python 对象别名而非合并 dict：
#   零拷贝，O(1) 操作。所有后续 add_user 自动同步。
#
# 关键不变量：合并后 name_to_users[alias_a] is name_to_users[alias_b]
#
# 示例：
#   name_to_users["view_a"] = DedupList([NodeUser(node1)])
#   name_to_users["view_b"] = DedupList([NodeUser(node2)])
#   发现 view_a 和 view_b 互为别名
#   → combined = DedupList([NodeUser(node1), NodeUser(node2)])
#   → name_to_users["view_a"] = combined
#   → name_to_users["view_b"] = combined  (is combined, True)
```

---

### 阶段 3：Unbacked Symint 注册 + 工具函数 (3550-3600)

```
数据流 4a (输入符号)：
  来源：V.graph.graph_inputs.values()
  加工：提取 free_symbols
  去向：unbacked_symbol_to_origin_node[s] = None

数据流 4b (节点符号)：
  来源：node.node.get_unbacked_symbol_defs()
  加工：sorted by name，取第一个定义者
  去向：unbacked_symbol_to_origin_node[s] = node.get_name()
```

```python
# ========== 阶段 3: 工具函数 + Unbacked Symint 注册 ==========
# ┌─ 设计哲学：PyTorch 支持动态形状（运行时才知道的尺寸），
# │   表示为 sympy.Symbol（"unbacked symint"）。
# │   它们不通过 buffer 传递，而是由某个节点"产出"。
# │   因此不能通过 read_writes 追踪，必须单独处理。
# └─ 编译器对应：SSA 中的隐式定义（Implicit Definitions）

    def rename(n: str) -> str:
        if n in self.mutation_renames:
            return rename(self.mutation_renames[n])
        return n
```

```
# [CORE] 递归重命名函数
# What：沿 mutation_renames 链递归查找最终真实 buffer 名
# How：rename("buf2") → rename("buf1") → rename("buf0") → "buf0"
# Why：所有依赖边都应指向最终的真实 buffer 名，而非中间 mutation 产生的临时名
```

```python
    def add_user(
        used_by_name: str,
        user_node: BaseSchedulerNode | OutputNode,
        can_inplace: bool = False,
        is_weak: bool = False,
    ) -> None:
        name_to_users[rename(used_by_name)].append(
            NodeUser(user_node, can_inplace, is_weak)
        )
```

```
# [CORE] ★★★ 依赖边添加的统一入口 ★★★
#
# What：向 name_to_users 追加一条 NodeUser 记录
# How：
#   1. rename(used_by_name) — 将可能被 mutation 重命名的 buffer 名解析为最终真实名
#   2. name_to_users[真实名].append(NodeUser(...))
# Why 统一入口：所有依赖添加都经过 rename，保证指向正确的 buffer
#
# API: NodeUser(user_node, can_inplace, is_weak)
# ├── hash/eq 基于 (node_name, can_inplace, is_weak) 三元组
# └── DedupList 保证同一三元组不会被重复添加
```

```python
    unbacked_symbol_to_origin_node: dict[sympy.Symbol, str | None] = {}

    for val in V.graph.graph_inputs.values():
        if isinstance(val, sympy.Expr):
            for fs in val.free_symbols:
                unbacked_symbol_to_origin_node[fs] = None
        elif isinstance(val, ir.TensorBox):
            sym_size = [s for s in val.get_size() if isinstance(s, sympy.Expr)]
            for s in sym_size:
                for fs in s.free_symbols:
                    unbacked_symbol_to_origin_node[fs] = None
```

```
# [CORE] 注册来自 graph input 的 unbacked symint
#
# How：遍历 graph_inputs：
#   - 值本身就是 sympy 表达式（标量输入）→ 提取 free_symbols
#   - 值是 TensorBox → 提取 size 中的符号表达式 → 提取 free_symbols
#   全部映射到 None（表示"来自输入，不由任何节点定义"）
#
# Why None 而不是输入名：输入不需要被调度，
#   如果创建依赖指向输入，Inductor 会尝试释放输入的 unbacked int，无意义
```

```python
    has_non_input_unbacked_defs = False
    for node in self.nodes:
        assert node.node is not None
        unbacked_symbol_defs = sorted(
            node.node.get_unbacked_symbol_defs(), key=lambda x: x.name
        )
        for s in unbacked_symbol_defs:
            assert isinstance(s, sympy.Symbol)
            has_non_input_unbacked_defs = True
            if s not in unbacked_symbol_to_origin_node:
                unbacked_symbol_to_origin_node[s] = node.get_name()
```

```
# [CORE] 注册由节点定义（产出）的 unbacked symint
#
# How：
#   遍历每个节点：
#     node.node.get_unbacked_symbol_defs() → 该节点产出的 unbacked symbols
#     对每个 symbol：如果尚未注册 → 记录为该节点定义的
#
# Why "第一个定义者"：MultiOutputLayout 的 buffer 可能向多个输出
#   传播同一个 unbacked symint，都声称定义了它。取第一个即可。
#
# 示例：
#   unbacked_symbol_to_origin_node = {
#     s0: None,       # 来自输入
#     s1: "buf0",     # 由 buf0 定义
#     s2: "buf5",     # 由 buf5 定义
#   }
```

---

### 阶段 4：主循环——依赖边建立 (3602-3688)

这是整个函数最核心、最复杂的部分。

```
[ 依赖边建立 | Edge Building Phase ]
(遍历每个节点，按优先级建立 5 类依赖边)

(4a) Unbacked Symint 依赖
     │  └── 数据流 4 (符号锁死)：提取 node.node.get_free_symbol_uses()，查 unbacked_symbol_to_origin_node，生成 StarDep → node.add_fake_dep()
     ▼
【 node.unmet_dependencies 新增 StarDep 】

(4b) Mutation 模式检测
     │  └── 提取 node.read_writes.writes 的唯一 MemoryDep 的 mode
     ▼
【 node_mode (str | None) 】

(4c) ★★★ Mutation 依赖处理 ★★★
     │  ├── 数据流 2a (WAW)：add_user(alt_name, node) + StarDep(alt_name)
     │  └── 数据流 2b (WAR)：遍历 name_to_users[alt_name].items → WeakDep(other_name, is_fake=not is_alias)
     ▼
【 name_to_users 更新 & node.read_writes 新增 StarDep + WeakDep 】

(4d) 额外控制依赖
     │  ├── additional_buffer_deps → WeakDep(is_fake=True)
     │  └── additional_star_deps → StarDep
     ▼

(4e) 普通读依赖
     │  └── 数据流 1 (常规读取)：遍历 node.read_writes.reads（跳过 WeakDep），add_user(read.name, node, can_inplace)
     ▼
【 name_to_users 更新 】

(4f) Mutation 映射更新
     │  └── 数据流 6 (版本映射)：mutation_renames[alt_name] = buf_name, mutation_real_name[buf_name] = 源名
     ▼
【 mutation_renames & mutation_real_name 更新，后续节点可见 】
```

#### 4a：Unbacked Symint 依赖 (3605-3619)

```python
        if has_non_input_unbacked_defs:
            assert node.node is not None
            unbacked_symbol_uses = sorted(
                node.node.get_free_symbol_uses(unbacked_only=True),
                key=lambda x: x.name,
            )
            for s in unbacked_symbol_uses:
                assert s in unbacked_symbol_to_origin_node
                if (r := unbacked_symbol_to_origin_node[s]) is not None:
                    for buf in self.name_to_node[r].get_outputs():
                        node.add_fake_dep(StarDep(buf.get_name()))
```

```
# [CORE] 为使用 unbacked symint 的节点添加隐式依赖
#
# What：当前节点依赖产出 unbacked symint 的源节点
# How：
#   1. node.node.get_free_symbol_uses(unbacked_only=True) → 该节点使用的 unbacked symbols
#   2. 对每个 symbol，查 unbacked_symbol_to_origin_node 拿源节点名 r
#   3. r 不为 None（非输入）→ 对源节点每个输出 buffer 添加 StarDep
#
# Why StarDep：unbacked symint 的值不通过 buffer 传递，
#   但语义上需要源节点先执行。StarDep 表示"整个 buffer 都需要"。
#
# API: node.add_fake_dep(StarDep(buf_name))
# ├── How: 将 StarDep 加入 read_writes.reads
# │   → prune_deps() 检查 buffer 是否在 available_buffer_names 中
# │   → 不在则加入 unmet_dependencies
# └── 效果：unmet_dependencies 包含对源节点的依赖
```

#### 4b：Mutation 模式检测 (3621-3628)

```python
        if (
            len(node.read_writes.writes) == 1
            and (dep := next(iter(node.read_writes.writes)))
            and isinstance(dep, MemoryDep)
        ):
            node_mode = dep.mode
        else:
            node_mode = None
```

```
# [CORE] 检测当前节点的写入模式
# What：提取节点写入的 MemoryDep 的 mode 字段
# How：如果恰好写一个 MemoryDep → 提取 mode，否则 None
# Why：后续创建 StarDep 时携带 mode，供 codegen 阶段使用
```

#### 4c：Mutation 依赖处理 (3630-3662) ★★★最复杂部分★★★

```
数据流 2a (WAW)：
  来源：buf.get_mutations() → alt_name (被修改的原始 buffer 名)
  加工：add_user(alt_name, node) + node.add_fake_dep(StarDep(alt_name))
  去向：name_to_users[alt_name] 追加 NodeUser + node.unmet_dependencies 新增 StarDep

数据流 2b (WAR)：
  来源：name_to_users[alt_name].items (之前的读者列表)
  加工：遍历每个读者 → 每个输出 buffer → 判断 is_alias → 生成 WeakDep
  去向：node.add_fake_dep(WeakDep) + add_user(other_name, node, is_weak=True)
```

```python
        # Handle output mutations
        for buf in node.get_outputs():
            assert len(buf.get_mutations()) <= 1
            for alt_name in buf.get_mutations():
                alt_name = rename(alt_name)
                # this node must run after the prior writer
                add_user(alt_name, node)
                node.add_fake_dep(StarDep(alt_name, mode=node_mode))
```

```
# [CORE] 步骤 1：WAW (Output Dependence)
#
# 🔴 断点：scheduler.py:3631
#
# What：当前节点必须在被修改 buffer 的前一个写入者之后运行
# How：
#   add_user(alt_name, node) → name_to_users["x"].append(NodeUser(node))
#   add_fake_dep(StarDep(alt_name)) → node.unmet_dependencies 增加 StarDep("x")
# Why：因为当前节点写 alt_name，必须等上一个写 alt_name 的节点完成（WAW）
#
# 前置知识：buf.get_mutations() 返回该 buffer 覆写的输入 buffer 名
#   例如 node 执行 x.add_(1) → buf.get_mutations() = ["x"]
#   不变量：一个节点最多 mutate 一个 buffer
```

```python
                for user in name_to_users[alt_name].items:
                    if user.get_name() == node.get_name():
                        continue

                    assert isinstance(user.node, BaseSchedulerNode)
                    for out_buf in user.node.get_outputs():
                        other_name = out_buf.get_name()
                        other_name = rename(other_name)
                        is_alias = alt_name in out_buf.get_aliases()
                        node.add_fake_dep(
                            WeakDep(
                                other_name,
                                mutating_buf=buf.get_name(),
                                is_fake=not is_alias,
                            )
                        )
                        add_user(other_name, node, is_weak=True)
```

```
# [CORE] ★★★ 步骤 2：WAR (Anti-Dependence) ★★★
#
# 🔴 断点：scheduler.py:3639
#
# What：当前节点要修改 x，但之前有其他节点读了 x，必须保证先读完再修改
#
# How：
#   遍历 name_to_users[alt_name] 中所有已有用户（之前的读者）：
#     跳过自己
#     对每个之前读者的每个输出 buffer（other_name）：
#       判断 is_alias → 决定 WeakDep 的 is_fake 标志
#
# ★ is_alias 判断 ★
#
#   is_alias=True（view 关系，共享存储）：
#     WeakDep(is_fake=False) — 真弱依赖
#     Why：view 和原始 buffer 共享内存，mutation 会破坏 view 的数据
#     → view 的 buffer 必须活到 mutation 完成后，不能提前释放
#
#   is_alias=False（clone 关系，独立存储）：
#     WeakDep(is_fake=True) — 假弱依赖
#     Why：clone 已复制数据，mutation 不影响 clone 的结果
#     → clone 的 buffer 可以提前释放，只需顺序约束
#
# 示例（以 x.add_(1) 为例）：
#   之前 node_A 执行 y = x.view(...)  # view（别名）
#   之前 node_B 执行 z = x.clone()   # clone（非别名）
#
#   当前节点 mutate x 时：
#     node_A 输出 "y"：is_alias=True → WeakDep("y", is_fake=False)
#     node_B 输出 "z"：is_alias=False → WeakDep("z", is_fake=True)
```

#### 4d：额外控制依赖 (3664-3672)

```python
            for add_dep in V.graph.additional_buffer_deps[node.get_name()]:
                add_user(add_dep, node, is_weak=True)
                node.add_fake_dep(WeakDep(add_dep, node.get_name(), is_fake=True))

            for add_dep in V.graph.additional_star_deps[node.get_name()]:
                add_user(add_dep, node, is_weak=False)
                node.add_fake_dep(StarDep(add_dep))
```

```
# [CORE] 额外控制依赖
#
# What：外部注入的额外依赖约束
# How：
#   additional_buffer_deps → WeakDep(is_fake=True)，仅约束顺序，不延长 buffer 生命周期
#   additional_star_deps → StarDep，全量 buffer 依赖，不可被 DCE 消除
# Why：例如 collectives 的全局排序约束、跨设备同步依赖
```

#### 4e：普通读依赖 (3674-3677)

```python
            # add normal non-mutation dependencies
            for read in node.read_writes.reads:
                if not isinstance(read, WeakDep):
                    add_user(read.name, node, node.can_inplace(read))
```

```
# [CORE] ★★★ True Dependence（RAW）处理 ★★★
#
# What：为当前节点读过的每个 buffer 注册使用者
# How：
#   遍历 node.read_writes.reads（跳过 WeakDep，已在 mutation 处理中添加）：
#     add_user(read.name, node, can_inplace)
#
#   can_inplace 含义：如果读可以被"原地写回"，后续 fusion 可做 in-place 优化
#
# Why 跳过 WeakDep：WeakDep 是 mutation 添加的顺序约束，不是真正的数据依赖
#
# 示例：
#   node.read_writes.reads = {MemoryDep("buf0", ...), MemoryDep("y", ...)}
#   → add_user("buf0", node, True)   # buf0 可以 in-place（唯一消费者）
#   → add_user("y", node, False)     # y 是外部输入，不能 in-place
```

#### 4f：Mutation 映射更新 (3679-3688)

```python
            node.update_mutated_names(self.mutation_renames)

            for buf in node.get_outputs():
                for alt_name in buf.get_mutations():
                    self.mutation_renames[rename(alt_name)] = buf.get_name()
                    self.mutation_renames[alt_name] = buf.get_name()
                    self.mutation_real_name[buf.get_name()] = (
                        self.mutation_real_name.get(alt_name, alt_name)
                    )
```

```
# [CORE] ★★★ Mutation 重命名链更新 ★★★
#
# What：更新全局 mutation 映射表，使后续节点能看到最新的 buffer 名
# How：
#   1. update_mutated_names(): 更新当前节点内部的 mutation buffer 名
#   2. 更新 Scheduler.mutation_renames：
#     rename(alt_name) → buf_name（解析链后的旧名 → 新名）
#     alt_name → buf_name（原始旧名 → 新名）
#     两层映射确保链式 mutation（A→B→C）能被正确追踪
#   3. 更新 mutation_real_name：
#     buf_name → 沿 rename 链回溯到最终真实名
#
# Why 放在循环末尾：处理完当前节点后，后续节点调用 rename() 才能看到最新映射
#
# 示例：
#   buf1 mutate buf0 → mutation_renames["buf0"] = "buf1", mutation_real_name["buf1"] = "buf0"
#   buf2 mutate buf1 → mutation_renames["buf1"] = "buf2", mutation_real_name["buf2"] = "buf0"
#   → 链式：buf2 的真实名是 buf0
```

---

### 阶段 5：保护性依赖 (3690-3725)

```
数据流 5a (输出保护)：
  来源：V.graph.get_output_names()
  加工：add_user(buf_name, OutputNode(StarDep(buf_name)))
  去向：name_to_users[buf_name] 追加 OutputNode 用户

数据流 5b (符号保护)：
  来源：V.graph.graph_outputs → get_free_symbol_uses() → unbacked_symbol_to_origin_node
  加工：查符号源节点 → add_user(buf_name, OutputNode)
  去向：name_to_users[buf_name] 追加 OutputNode 用户

数据流 5c (输入变异保护)：
  来源：self.mutation_renames
  加工：过滤出 graph_inputs 和 constants 中的名 → add_user + V.graph.mutated_inputs.add(name)
  去向：name_to_users 追加 OutputNode + V.graph.mutated_inputs 集合
```

```python
# ========== 阶段 5: 保护性依赖 ==========
# ┌─ 设计哲学：依赖图构建完毕后，必须确保"重要的东西"不会被后续 DCE 消除
# │   方法：为重要节点添加 OutputNode 作为"虚拟用户"
# └─ OutputNode 是特殊用户，DCE 会跳过有 OutputNode 用户的 buffer

    # make sure outputs aren't dead-code-eliminated
    for buf_name in V.graph.get_output_names():
        log.debug("scheduling output %s", buf_name)
        add_user(buf_name, OutputNode(StarDep(buf_name)))
```

```
# [CORE] 保护 1：图输出不被 DCE 消除
# What：为每个输出 buffer 添加 OutputNode 作为用户
# How：OutputNode.get_name() 返回 "OUTPUT"，DCE 检查时该 buffer 的 users 不为空
# Why：图的输出 buffer 无论是否被其他节点使用，都必须保留
```

```python
    # make sure unbacked symints aren't dead-code-eliminated
    if has_non_input_unbacked_defs:
        for out in V.graph.graph_outputs:
            for s in out.get_free_symbol_uses(unbacked_only=True):
                assert s in unbacked_symbol_to_origin_node
                if r := unbacked_symbol_to_origin_node[s]:
                    for buf_name in self.name_to_node[r].get_buffer_names():
                        add_user(buf_name, OutputNode(StarDep(buf_name)))
```

```
# [CORE] 保护 2：unbacked symint 的定义节点不被 DCE 消除
# What：如果某 unbacked symint 被图输出使用，其定义节点的 buffer 也需保护
# Why：即使定义节点的 buffer 没被直接使用，它产出了 unbacked symint 这个隐式值
```

```python
    # make sure input mutation isn't dead-code-eliminated
    for name in self.mutation_renames:
        if name in V.graph.graph_inputs:
            add_user(name, OutputNode(StarDep(name)))
            V.graph.mutated_inputs.add(name)
        elif name in V.graph.constants:
            add_user(name, OutputNode(StarDep(name)))
```

```
# [CORE] 保护 3：被原地修改的输入不被 DCE 消除
# What：mutation_renames 中的输入名和常量名需要保护
# How：add_user + 将输入名加入 V.graph.mutated_inputs
# Why：如果用户执行 x.add_(1)，x 被原地修改，mutation 节点必须保留
#   V.graph.mutated_inputs 通知 codegen 需要更新输入
```

```python
    inp_names = {name: index for index, name in enumerate(V.graph.graph_inputs.keys())}
    V.graph.mutated_input_idxs = [
        inp_names[name] for name in V.graph.mutated_inputs
    ]
```

```
# [AUX] 计算被修改输入的索引列表，供 codegen 使用
```

---

### 阶段 6：结果写入 (3727-3745)

```
动作 A (名单挂载)：
  来源：name_to_users[buf.get_name()].items
  加工：buf.set_users(items) — 去重合并同一 node 的多个 NodeUser
  去向：SchedulerBuffer.users: list[NodeUser]

动作 B (捐赠转移)：
  来源：name_to_users[name].items
  加工：name_to_donated_buffer[name].set_users(items)
  去向：donated buffer 的 users
```

```python
    # copy users information onto the nodes
    for node in self.nodes:
        for buf in node.get_outputs():
            buf.set_users(name_to_users[buf.get_name()].items)

    for name in self.name_to_donated_buffer:
        self.name_to_donated_buffer[name].set_users(name_to_users[name].items)
```

```
# [CORE] ★★★ 最终结果持久化 ★★★
#
# What：将 name_to_users 中的用户列表挂载到物理内存对象上
# How：
#   遍历所有节点的所有输出 buffer：
#     name_to_users[buf_name].items → buf.set_users(items)
#
# API: SchedulerBuffer.set_users(users: list[NodeUser])
# ├── How: 对 users 做去重合并（同一 node 的多个 NodeUser merge 为一个）
# │   merge 规则：
# │     can_inplace = can_inplace AND can_inplace（两者都同意才能 in-place）
# │     is_weak = is_weak AND is_weak（两者都弱才弱）
# └── 存储位置：SchedulerBuffer.users: list[NodeUser]
#
# Why donated buffer 也需要：它是用户不再使用的输入，可被原地复用，
#   需要 users 信息来决定何时可以释放。
#
# 注意：此步之后 name_to_users 不再被引用，函数结束时被 GC 销毁。
```

---

## 四大数据结构生命周期总览

```
[编译分析期 Dependency Analysis]
1. 提取操作意图 →  【Node.read_writes】 (基石建立)
                        │
                        ▼
2. 扫描全局读写 →  【name_to_users】 (建立反向索引，解决 WAR 冲突，修正 read_writes)
                        │
                        ▼ (分析期结束，资产转移)
3. 挂载到内存   →  【Buffer.users】 (出度名单确立)

==================== (时间边界) ====================

[调度执行期 Scheduling & Codegen]
4. 图纸转锁     →  【Node.read_writes】 初始化给 【Node.unmet_dependencies】
                        │
                        ▼
5. 拓扑弹栈     →   某个 Node 执行完，输出一个 Buffer
                        │
                        ▼
6. 解锁广播     →   遍历该 【Buffer.users】，挨个移除下游节点的 【Node.unmet_dependencies】
                        │
                        ▼
7. 触发执行     →   当某个下游节点的 【Node.unmet_dependencies】 归零，送入 GPU 执行！
```

- `name_to_users`：脚手架，建完楼就拆了
- `read_writes`：设计图，永远不变
- `Buffer.users`：出度名单，运行时驱动解锁
- `unmet_dependencies`：入度倒计时，归零即触发执行

---

## 执行结果示例

```python
import torch

@torch.compile
def f(x, y):
    a = x * 2          # node_a: reads x, writes buf_a
    b = a + y          # node_b: reads buf_a + y, writes buf_b
    y.add_(b)          # node_c: mutates y, reads buf_b + y
    c = a.sum()        # node_d: reads buf_a, writes buf_c
    return c, y
```

```
# => compute_dependencies() 执行过程模拟：
#
# === 阶段 1：基础设施 ===
# name_to_users = defaultdict(DedupList)  # 空
#
# === 阶段 2：别名合并 ===
# 无别名操作（没有 view），name_to_users 仍为空
#
# === 阶段 3：Unbacked Symint ===
# 无动态尺寸，unbacked_symbol_to_origin_node = {}
# has_non_input_unbacked_defs = False
#
# === 阶段 4：主循环 ===
#
# --- node_a (a = x * 2) ---
# 无 mutation，无 unbacked symint
# 普通读依赖：
#   read_writes.reads = {MemoryDep("x", ...)}
#   → add_user("x", node_a, False)
#   → name_to_users["x"] = [NodeUser(node_a, can_inplace=False)]
#
# --- node_b (b = a + y) ---
# 无 mutation
# 普通读依赖：
#   read_writes.reads = {MemoryDep("buf_a", ...), MemoryDep("y", ...)}
#   → add_user("buf_a", node_b, True)   # 可以 in-place（唯一消费者）
#   → add_user("y", node_b, False)
#   → name_to_users["buf_a"] = [NodeUser(node_b, can_inplace=True)]
#   → name_to_users["y"] = [NodeUser(node_b, can_inplace=False)]
#
# --- node_c (y.add_(b)) ---
# ★ 有 mutation：mutate y ★
# buf.get_mutations() = ["y"]
# alt_name = "y"
#
# 步骤 1：WAW
#   add_user("y", node_c)
#   → name_to_users["y"] = [NodeUser(node_b, F), NodeUser(node_c, F)]
#   add_fake_dep(StarDep("y"))
#   → node_c.unmet_dependencies 新增 StarDep("y")
#
# 步骤 2：WAR（遍历 y 的已有用户）
#   用户 node_b 读了 y：
#     out_buf = "buf_b"
#     is_alias = ("y" in buf_b.get_aliases()) → False（不是 view）
#     → WeakDep("buf_b", is_fake=True)
#     → add_user("buf_b", node_c, is_weak=True)
#     含义：node_c 只需等 node_b 完成，但 buf_b 可以提前释放
#
# 普通读依赖：
#   read_writes.reads = {MemoryDep("buf_b", ...), StarDep("y")}
#   → add_user("buf_b", node_c, False)
#   → add_user("y", node_c, False)
#
# mutation_renames 更新：
#   mutation_renames["y"] = "buf_c"
#   mutation_real_name["buf_c"] = "y"
#
# --- node_d (c = a.sum()) ---
# 无 mutation
# 普通读依赖：
#   read_writes.reads = {MemoryDep("buf_a", ...)}
#   → add_user("buf_a", node_d, False)
#   → name_to_users["buf_a"] = [NodeUser(node_b, T), NodeUser(node_d, F)]
#
# === 阶段 5：保护性依赖 ===
# 输出 "buf_c" 和 "y"（被 mutate 的输入）：
#   add_user("buf_c", OutputNode(StarDep("buf_c")))
#   add_user("y", OutputNode(StarDep("y")))
#   V.graph.mutated_inputs = {"y"}
#
# === 阶段 6：结果写入 ===
# 最终 name_to_users：
#   "x"     → [NodeUser(node_a, F, F)]
#   "buf_a" → [NodeUser(node_b, T, F), NodeUser(node_d, F, F)]
#   "y"     → [NodeUser(node_b, F, F), NodeUser(node_c, F, F), NodeUser(OUTPUT)]
#   "buf_b" → [NodeUser(node_c, F, F), NodeUser(node_c, F, T)]  # 普通读+弱依赖
#   "buf_c" → [NodeUser(OUTPUT)]
#
# 依赖图（DAG）：
#   node_a → node_b (buf_a)
#   node_a → node_d (buf_a)
#   node_b → node_c (buf_b)
#   y 的读写 → node_c 必须在 node_b 之后
#
# 拓扑合法序：[node_a, node_b, node_d, node_c] 或 [node_a, node_d, node_b, node_c]
# 注意：node_d 不依赖 node_b 或 node_c，可以与它们并行
```

---

## 断点速查表

| 断点位置 | 阶段 | 观察重点 |
|----------|------|----------|
| `scheduler.py:3514` | 1 | name_to_users 初始状态（空 defaultdict） |
| `scheduler.py:3534` | 2 | 别名合并后多个 key 是否指向同一 DedupList |
| `scheduler.py:3568` | 3 | unbacked_symbol_to_origin_node 填充情况 |
| `scheduler.py:3617` | 4a | unbacked symint 依赖添加后 node.unmet_dependencies 变化 |
| `scheduler.py:3631` | 4c | buf.get_mutations() 返回值（是否有 mutation） |
| `scheduler.py:3639` | 4c | WAR 处理：遍历已有用户时的 is_alias 判断 |
| `scheduler.py:3654` | 4c | is_alias 值——决定 WeakDep 的 is_fake 标志 |
| `scheduler.py:3675` | 4e | 普通 read 依赖添加（True Dependence） |
| `scheduler.py:3684` | 4f | mutation_renames 链更新（每次 mutation 后检查） |
| `scheduler.py:3691` | 5 | 保护性依赖：V.graph.get_output_names() 列表 |
| `scheduler.py:3712` | 5 | mutation 输入保护：self.mutation_renames 中的输入名 |
| `scheduler.py:3730` | 6 | 最终 buf.set_users() 写入后 buf.users 的内容 |

---

## 核心概念速查

| 概念 | 代码中的体现 | 编译器术语 |
|------|-------------|-----------|
| True Dependence | `add_user(read.name, node)` | RAW (Read After Write) |
| Anti-Dependence | `WeakDep(other_name, is_fake=...)` | WAR (Write After Read) |
| Output Dependence | `add_user(alt_name, node)` + StarDep | WAW (Write After Write) |
| 别名分析 | `name_to_users["foo"] is name_to_users["bar"]` | Alias Analysis |
| 隐式依赖 | `StarDep` via unbacked symint | Implicit Dependence |
| 弱依赖 | `WeakDep(is_fake=True)` | Ordering-only Constraint |
| 保护依赖 | `OutputNode` 作为用户 | Live Range Extension |
| 变量版本化 | `mutation_renames` chain | SSA Variable Versioning |
