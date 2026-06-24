# verify_ch07.py — 形状与视图算子示例实跑（统一用 a = arange(12).reshape(3,4) 便于追地址）
import torch
torch.set_printoptions(precision=4, sci_mode=False)

a = torch.arange(12).reshape(3, 4)
print("a ="); print(a)
print("shape", tuple(a.shape), "stride", tuple(a.stride()), "contiguous", a.is_contiguous())

print("\n--- as_strided(a, (4,3), (1,4))  # stride(1,4) 即转置 ---")
print(torch.as_strided(a, (4, 3), (1, 4)))

print("\n--- a.view(2, 6) ---"); print(a.view(2, 6))

print("\n--- expand: a[0:1] 形状 (1,4) -> (3,4) ---")
print(a[0:1].expand(3, 4))

print("\n--- a.permute(1, 0)  # 转置 ---"); print(a.permute(1, 0))

print("\n--- squeeze / unsqueeze ---")
t = torch.arange(3).reshape(1, 3, 1)
print("t.shape", tuple(t.shape), "-> squeeze()", tuple(t.squeeze().shape))
print("a.unsqueeze(0).shape", tuple(a.unsqueeze(0).shape))

print("\n--- a.select(1, 1)  # 取第 1 列 ---"); print(a.select(1, 1))

print("\n--- a[:, ::2]  # 落 aten.slice(dim=1, start=0, end=4, step=2) -> 列 0,2 ---")
print(a[:, ::2])

print("\n--- a.flip(0)  # 翻转行（负 stride） ---"); print(a.flip(0))

print("\n--- a.diagonal()  # 主对角 ---"); print(a.diagonal())
print("--- a.diagonal(offset=1)  # 上对角 ---"); print(a.diagonal(offset=1))
