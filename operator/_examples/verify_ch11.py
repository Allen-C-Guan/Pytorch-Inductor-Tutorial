# verify_ch11.py — 实跑第 11 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch11.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

print("===== arange =====")
print("arange(0, 10, 2) =", torch.arange(0, 10, 2))   # 等差序列 start, end, step

print("\n===== rand =====")
torch.manual_seed(0)                                   # 设种子可复现
print("rand(2, 3) =", torch.rand(2, 3))               # [0,1) 均匀

print("\n===== randn =====")
torch.manual_seed(0)                                   # 设种子可复现
print("randn(2, 3) =", torch.randn(2, 3))             # 标准正态 N(0,1)

print("\n===== randperm =====")
torch.manual_seed(0)                                   # 设种子可复现
print("randperm(5) =", torch.randperm(5))             # 0..n-1 的随机排列
