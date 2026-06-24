# verify_ch15.py — 实跑第 15 章 Tier D 归一化算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch15.py
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=4, sci_mode=False)

print("===== native_layer_norm =====")
x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
# 高层等价：F.layer_norm 只返回归一化后的 out
out = F.layer_norm(x, normalized_shape=[3])
print("F.layer_norm(out)        =\n", out)
print("行均值(应≈0)             =", out.mean(dim=1))
print("行方差(应≈1)             =", out.var(dim=1, unbiased=False))
# 裸 aten 调用返回 (out, mean, rstd) 三元组，mean/rstd 供反向用
out2, mean, rstd = torch.ops.aten.native_layer_norm(x, [3], None, None, 1e-5)
print("aten.native_layer_norm mean =", mean)   # 每行均值
print("aten.native_layer_norm rstd =", rstd)   # 每行 1/sqrt(var+eps)

print("\n===== native_group_norm =====")
# N=1, C=4, HxW=2(即 2x2)，group=2 -> 每组 2 个通道
# 用 reshape 避免 deep-nested literal
xg = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2)
outg = F.group_norm(xg, num_groups=2)
print("F.group_norm(out)        =\n", outg)
# 裸 aten 调用：(out, mean, rstd)，mean/rstd 形状 (N, group) = (1,2)
outg2, meang, rstdg = torch.ops.aten.native_group_norm(xg, None, None, 1, 4, 4, 2, 1e-5)
print("aten.native_group_norm mean =", meang)  # 形状 (1,2)
print("aten.native_group_norm rstd =", rstdg)  # 形状 (1,2)

print("\n===== _native_batch_norm_legit =====")
xb = torch.arange(1 * 2 * 2 * 1, dtype=torch.float32).reshape(1, 2, 2, 1)  # (N=1,C=2,H=2,W=1)
running_mean = torch.zeros(2)
running_var = torch.ones(2)
# training=True：按当前 batch 沿 (N,H,W) 统计，并 EMA 更新 running
outb = F.batch_norm(xb, running_mean, running_var, training=True, momentum=0.1, eps=1e-5)
print("F.batch_norm(out)        =\n", outb.reshape(2, 2))
print("running_mean (EMA更新后) =", running_mean)
print("running_var  (EMA更新后) =", running_var)
# 裸 aten 训练路径返回 (out, save_mean, save_invstd)，供反向用
outb2, smean, sinvstd = torch.ops.aten._native_batch_norm_legit(
    xb, None, None, torch.zeros(2), torch.ones(2), True, 0.1, 1e-5)
print("save_mean   =", smean)    # 逐通道 batch 均值
print("save_invstd =", sinvstd)  # 逐通道 1/sqrt(var+eps)
