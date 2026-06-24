# verify_ch06.py — 实跑第 6 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch06.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

print("===== gelu (approximate 对比) =====")
x = torch.tensor([-1.0, 0.0, 1.0])
print("gelu(x, none)  =", torch.nn.functional.gelu(x, approximate="none"))   # erf 精确版
print("gelu(x, tanh)  =", torch.nn.functional.gelu(x, approximate="tanh"))   # tanh 近似版

print("\n===== _softmax =====")
s_in = torch.tensor([1.0, 2.0, 3.0])
sm = torch._softmax(s_in, dim=0, half_to_float=False)
print("_softmax([1,2,3], dim=0) =", sm)
print("sum over dim=0           =", sm.sum())

print("\n===== _log_softmax =====")
lsm = torch._log_softmax(s_in, dim=0, half_to_float=False)
print("_log_softmax([1,2,3], dim=0) =", lsm)

print("\n===== native_dropout (随机类，已设种子) =====")
torch.manual_seed(0)
out, mask = torch.native_dropout(torch.ones(10), 0.5, True)
print("out  =", out)
print("mask =", mask)
print("存活个数 =", int(mask.sum().item()), " 缩放因子 =", 1.0 / (1.0 - 0.5))
