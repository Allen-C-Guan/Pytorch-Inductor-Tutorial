import torch

torch.set_printoptions(precision=4, sci_mode=False)

# --- atan2: 两参反正切，象限正确（注意参数顺序 self=y, other=x）---
# 展示向量 (other, self) = (x, y) 的辐角，能区分象限
y = torch.tensor([1., 0., -1.])   # self = 分子 y
x = torch.tensor([0., 1., 0.])    # other = 分母 x
# 对应点 (x,y): (0,1)->π/2,  (1,0)->0,  (0,-1)->-π/2
print("=== atan2 ===")
print(repr(torch.atan2(y, x)))

# 对照：单参 atan(y/x) 会把 (1,1) 与 (-1,-1) 算成同一个比值 1 -> 都 π/4
print("--- 对比 atan(y/x) 丢失象限 ---")
yy = torch.tensor([1., -1.])
xx = torch.tensor([1., -1.])
print(repr(torch.atan2(yy, xx)))   # π/4, -3π/4
print(repr(torch.atan(yy / xx)))   # π/4,  π/4  <- 丢象限

# --- sigmoid: 0->0.5，单调递增 ---
print("=== sigmoid ===")
s = torch.tensor([-1., 0., 1.])
print(repr(torch.sigmoid(s)))

# --- erf: 奇函数 erf(-x)=-erf(x)，gelu 的积分部件 ---
print("=== erf ===")
e = torch.tensor([0., 0.5, -0.5])
print(repr(torch.erf(e)))
