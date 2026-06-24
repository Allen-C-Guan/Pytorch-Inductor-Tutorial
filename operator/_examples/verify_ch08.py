# verify_ch08.py — 实跑第 8 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch08.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

print("===== cat =====")
a = torch.tensor([[1, 2]])
b = torch.tensor([[3, 4]])
print("a =", a)
print("b =", b)
print("cat([a,b], dim=0) =", torch.cat([a, b], dim=0))   # 沿第0维拼 -> 2x2
print("cat([a,b], dim=1) =", torch.cat([a, b], dim=1))   # 沿第1维拼 -> 1x4

print("\n===== split_with_sizes =====")
t = torch.arange(10)
parts = torch.split(t, [3, 2, 5])      # 不等分 -> list[Tensor]
for p in parts:
    print(p.shape, p)

print("\n===== repeat =====")
r = torch.tensor([[1, 2]])
print("r =", r)
out = r.repeat(2, 3)
print("r.repeat(2,3) shape =", tuple(out.shape))
print(out)
