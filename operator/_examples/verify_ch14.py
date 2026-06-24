# verify_ch14.py — 实跑第 14 章 Tier D 池化算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch14.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

# 用 1×1×4×4 张量（含小整数 0..15）便于看清 2×2 窗口塌缩。
x = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
print("x =", x)

print("\n===== max_pool2d_with_indices =====")
out, idx = torch.nn.functional.max_pool2d_with_indices(x, kernel_size=2, stride=2)
print("output  =", out)           # 每个窗口最大值
print("indices =", idx)           # 每个输出来源的展平下标（沿 C*H*W 维）

print("\n===== avg_pool2d =====")
print("avg_pool2d(k=2, s=2) =", torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2))

print("\n===== _adaptive_avg_pool2d =====")
print("adaptive_avg_pool2d((1,1)) =", torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)))  # GAP
