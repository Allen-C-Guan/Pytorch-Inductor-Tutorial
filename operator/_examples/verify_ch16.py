# verify_ch16.py — 上采样 / grid_sampler 示例实跑
import torch
torch.set_printoptions(precision=4, sci_mode=False)

print("===== upsample_nearest2d (via F.interpolate, mode='nearest') =====")
x = torch.tensor([[[[1., 2.], [3., 4.]]]])   # (1,1,2,2)
n = torch.nn.functional.interpolate(x, size=(4, 4), mode="nearest")
print("2x2 -> 4x4:"); print(n[0, 0])

print("\n===== upsample_bilinear2d (align_corners=True) =====")
b = torch.nn.functional.interpolate(x, size=(3, 3), mode="bilinear", align_corners=True)
print("2x2 -> 3x3:"); print(b[0, 0])

print("\n===== grid_sampler_2d (via F.grid_sample) =====")
img = torch.tensor([[[[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]]])   # (1,1,3,3)
# grid: (N, Hout, Wout, 2), 末维 (x=列, y=行), 归一化 [-1,1]
grid = torch.tensor([[ [[-1., -1.], [1., 1.]] ]])   # (1,1,2,2): 取左上角与右下角
out = torch.nn.functional.grid_sample(img, grid, mode="bilinear",
                                      padding_mode="zeros", align_corners=True)
print("img  ", tuple(img.shape)); print(img[0, 0])
print("grid ", tuple(grid.shape), "=", grid[0, 0])
print("out  ", tuple(out.shape), "=", out[0, 0])
