# verify_ch04.py — 实跑第 4 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch04.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

print("===== var (correction 对比) =====")
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print("var(a, dim=1, correction=1) =", torch.var(a, dim=1, correction=1))   # 无偏，除以 n-1
print("var(a, dim=1, correction=0) =", torch.var(a, dim=1, correction=0))   # 有偏，除以 n

print("\n===== argmax (返回索引而非值) =====")
b = torch.tensor([[1., 5., 3.]])
print("argmax(b, dim=1) =", torch.argmax(b, dim=1))   # 返回下标 -> 1

print("\n===== argmin (返回最小值索引) =====")
print("argmin(b, dim=1) =", torch.argmin(b, dim=1))

print("\n===== cumsum (前缀和) =====")
c = torch.tensor([1., 2., 3., 4.])
print("cumsum(c, 0) =", torch.cumsum(c, 0))
