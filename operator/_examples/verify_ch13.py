# verify_ch13.py — 卷积示例实跑
import torch
torch.set_printoptions(precision=4, sci_mode=False)

print("===== convolution (F.conv2d 落 aten.convolution) =====")
x = torch.tensor([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]])   # (1,1,3,3)
w = torch.tensor([[[[1., 0.], [0., 1.]]]])                          # (1,1,2,2) 取窗口主对角
out = torch.nn.functional.conv2d(x, w, stride=1, padding=0)
print("input ", tuple(x.shape)); print(x[0,0])
print("weight", tuple(w.shape)); print(w[0,0])
print("out   ", tuple(out.shape)); print(out[0,0])
