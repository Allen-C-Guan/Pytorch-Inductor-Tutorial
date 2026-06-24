# verify_ch12.py — 实跑第 12 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch12.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

print("===== clone =====")
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = a.clone()                                         # 深拷贝
print("a            =", a)
print("b = a.clone()=", b)
print("b.data_ptr() =", b.data_ptr(), " a.data_ptr() =", a.data_ptr(),
      " -> 独立:", b.data_ptr() != a.data_ptr())
b[0, 0] = 99                                          # 改 b 不影响 a
print("b[0,0]=99 后 a =", a, " b =", b)

print("\n===== copy_ =====")
dst = torch.zeros(2, 3, dtype=torch.int64)
src = torch.tensor([[10, 20, 30], [40, 50, 60]])
print("dst (copy_ 前) =", dst)
dst.copy_(src)                                        # 原地把 src 拷进 dst
print("dst (copy_ 后) =", dst)

print("\n===== _to_copy (用户层入口 .to(dtype=...)) =====")
i = torch.tensor([1, 2, 3])                           # 整数张量
f = i.to(dtype=torch.float32)                         # 落到 aten._to_copy，拷贝并转 dtype
print("i        =", i, " dtype=", i.dtype)
print("f = i.to =", f, " dtype=", f.dtype)
