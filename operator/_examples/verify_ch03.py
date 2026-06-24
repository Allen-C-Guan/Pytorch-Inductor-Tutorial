# verify_ch03.py — 实跑第 3 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch03.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

# --- where: 按条件逐元素选 self(True) 或 other(False)；三路广播 ---
print("===== where =====")
cond = torch.tensor([[True, False],
                     [False, True]])
a    = torch.tensor([[1, 2],
                     [3, 4]])
b    = torch.zeros(2, 2)
print(repr(torch.where(cond, a, b)))   # True 取 a，False 取 b

# --- 展示三路广播：cond 标量、self 行向量、other 列向量 -> 2x2 ---
print("--- 三路广播对齐 ---")
print(repr(torch.where(cond[0:1, :], torch.tensor([[10, 20]]), torch.tensor([[0], [100]]))))
#   cond = [T, F] (1x2)   self = [10,20] (1x2)   other = [[0],[100]] (2x1)
#   广播出 2x2
