# verify_ch01.py — 实跑第 1 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch01.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

print("===== div =====")
a = torch.tensor([7, -7])
b = torch.tensor([2, 2])
print("div(int,int)            =", torch.div(a, b))                       # 真除 -> float
print("div(..., trunc)        =", torch.div(a, b, rounding_mode="trunc"))  # 向0
print("div(..., floor)        =", torch.div(a, b, rounding_mode="floor"))  # 向-∞

print("\n===== pow =====")
base = torch.tensor([2, 3])
print("pow(int, 2)     =", torch.pow(base, 2))      # 非负整指数 -> 留在 int
for e in (-1, -1.0, 0.5):
    try:
        r = torch.pow(base, e)
        print(f"pow(int, {e})  =", r, " dtype=", r.dtype)
    except Exception as ex:
        print(f"pow(int, {e})  -> ERROR: {type(ex).__name__}: {ex}")

print("\n===== fmod / remainder =====")
aa = torch.tensor([-7])
bb = torch.tensor([3])
print("fmod(-7,3)      =", torch.fmod(aa, bb))      # 符号同 a (trunc)
print("remainder(-7,3) =", torch.remainder(aa, bb))  # 符号同 b (floor)

print("\n===== clamp =====")
x = torch.tensor([-1.0, 0.5, 3.0])
print("clamp(x, min=0)        =", torch.clamp(x, min=0.0))
print("clamp(x, 0, 1)         =", torch.clamp(x, 0.0, 1.0))
