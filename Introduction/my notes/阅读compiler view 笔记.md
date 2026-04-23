## IR是如何被realize的

为了弥补这一点，并精确对齐源码的递归本质，我们需要加入一个**中间惰性节点**。

假设我们有计算图：$Z = \text{ceil}(\text{sin}(X))$

- **节点 $X$**: 物理显存（已物化，递归基点）。
- **节点 $Y$**: Y = sin(X)。**纯惰性节点（未物化）**，它是诱发递归的关键！
- **节点 $Z$**: Z = ceil(Y)。作为最终输出被强制 `realize()`，它是触发整个崩塌过程的起点。

现在，我们把显微镜对准 `Z.inner_fn` 的触发瞬间，精确跟踪这段**包含完整递归**的数据流：

------

### 源码回忆：递归产生的原动力

在追踪前，请牢记 `loader` 内部触发递归的那句源码：

Python

```
def loader(index):
    if self.is_realized:
        return ops.load(...) # 【递归终止】
    else:
        return self.inner_fn(index) # 【递归发生！直接跳转到自身的 inner_fn】
```

------

### 具象化：函数调用栈的递归下潜与回溯

**【触发原点】**：调度器需要写入 $Z$，主动调用 **`Z.inner_fn(index)`**。

#### 阶段一：递归下潜（深度优先搜索）

- **【Depth 0：Z 的 inner_fn】**
  - `Z.inner_fn` 执行第一行：尝试获取上游 $Y$ 的数据。
  - **调用**：`Y.loader(index)`。
- **【Depth 1：Y 的 loader 触发递归】**
  - 进入 `Y.loader`。
  - **状态判断**：$Y$ 是惰性节点，`Y.is_realized == False`。
  - **动作（核心递归发生）**：`loader` 拒绝返回字符串，而是**直接 `return Y.inner_fn(index)`**。
  - *执行流瞬间被转移到了 $Y$ 的内部！*
- **【Depth 2：Y 的 inner_fn】**
  - `Y.inner_fn` 执行第一行：尝试获取上游 $X$ 的数据。
  - **调用**：`X.loader(index)`。
- **【Depth 3：X 的 loader 触达基点】**
  - 进入 `X.loader`。
  - **状态判断**：$X$ 是物理显存，`X.is_realized == True`。
  - **动作（递归终止）**：不再调用 `inner_fn`，而是调用底层的 `ops.load("X", index)`。
  - CSE 拦截 `ops.load`，在后台写入代码 `tmp0 = tl.load(X + index)`。
  - **基点返回**：返回字符串 `"tmp0"` 给 Depth 2。

------

#### 阶段二：回溯展开与 `fn` 计算（代码生成）

此时，我们到达了调用栈的最深处，开始带着获取到的临时变量 `"tmp0"`，逐层向外执行 `fn`（即 `ops_wrapper` 的代理闭包），并回溯。

- **【Depth 2 回溯：执行 Y 的 fn】**
  - `Y.inner_fn` 拿到了 `loaded_args = ["tmp0"]`。
  - 执行第二行：`return fn_sin("tmp0")`。
  - 路由到底层 `ops.sin("tmp0")`。
  - CSE 拦截，后台写入：`tmp1 = tl.math.sin(tmp0)`。
  - **向上返回**：`Y.inner_fn` 结束，向 Depth 1 返回字符串 `"tmp1"`。
- **【Depth 1 回溯：Y 的 loader 传递结果】**
  - `Y.loader` 本身没有计算逻辑，它仅仅是把 `Y.inner_fn` 返回的 `"tmp1"` 继续向上抛给 Depth 0。
- **【Depth 0 回溯：执行 Z 的 fn】**
  - `Z.inner_fn` 终于等到了 `Y.loader` 的返回值！它拿到了 `loaded_args = ["tmp1"]`。
  - 执行第二行：`return fn_ceil("tmp1")`。
  - 路由到底层 `ops.ceil("tmp1")`。
  - CSE 拦截，后台写入：`tmp2 = tl.math.ceil(tmp1)`。
  - **最终返回**：`Z.inner_fn` 结束，向调度器返回字符串 `"tmp2"`。

------

### 递归思想精炼总结

通过这次增加了一层 Y 的推演，递归的脉络就极其清晰了：

1. **谁在驱动递归？** 是**未被 realize 的 `loader`**。它像一个传送门，当你向它要数据时，它不给你数据，而是把你一脚踢进了它自己的 `inner_fn` 里。
2. **递归何时停止？** 只有当这条链条触碰到一个**被 realize 的 `loader`**（物理显存边界）时，递归才会停止，转而变成底层的 `ops.load`（生成 `"tmp0"`）。
3. **`fn` 扮演了什么角色？** `fn`（数学运算代理）**全都在递归的回溯（归）阶段执行**。伴随着调用栈的层层弹出，`tmp0` 像滚雪球一样，经过 `sin` 变成 `tmp1`，再经过 `ceil` 变成 `tmp2`。

这种设计使得 Inductor 根本不需要去写复杂的 AST 遍历器（Visitor）。**函数的递归调用栈本身，就是遍历 AST 树的天然引擎！** 这正是深度学习编译器前端最核心的函数式编程美学。