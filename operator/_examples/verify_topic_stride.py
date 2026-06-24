"""verify_topic_stride.py — 实跑专题《stride 的本质》里的所有示例。

按 docs/docs/aten/_examples/SPEC.md 的正确性硬约束：输出值必须来自实跑，
绝不手写。运行：source env.sh new && python docs/docs/aten/_examples/verify_topic_stride.py
"""
import torch

torch.set_printoptions(linewidth=120)

print("===== 1. 基底：连续张量 a (2,3) =====")
a = torch.arange(6).reshape(2, 3)
print("a        =", a.tolist())
print("a.shape  =", tuple(a.shape), " a.stride =", a.stride(),
      " a.is_contiguous =", a.is_contiguous())

print("\n===== 2. view 在连续张量上：按行主序重新分组（注意：这不是转置！）=====")
print("a.view(3,2) =", a.view(3, 2).tolist())

print("\n===== 3. 转置 t = a.t()：shape/stride 交换，零拷贝 =====")
t = a.t()
print("t        =", t.tolist())
print("t.shape  =", tuple(t.shape), " t.stride =", t.stride(),
      " t.is_contiguous =", t.is_contiguous())

print("\n===== 4. 对转置后的张量做 view(-1)：必然报错 =====")
try:
    t.view(-1)
except RuntimeError as e:
    print("RuntimeError:", e)

print("\n===== 5. 物理偏移轨迹：把转置张量按行主序读出来 =====")
# t 的寻址：storage_index = 0 + i*1 + j*3   （offset=0, stride=(1,3)）
offsets = [0 + i * 1 + j * 3 for i in range(3) for j in range(2)]
print("逻辑展平读到的值   =", offsets, "  (storage=[0..5]，故值==storage 下标)")
steps = [offsets[k + 1] - offsets[k] for k in range(len(offsets) - 1)]
print("相邻两元素的物理步长 =", steps)

print("\n===== 6. reshape(-1) 不报错：悄悄物化（拷贝）=====")
r = t.reshape(-1)
print("t.reshape(-1) =", r.tolist())
print("r.stride      =", r.stride(), " r.is_contiguous =", r.is_contiguous())
print("r 与 t 共享 storage？", r.storage().data_ptr() == t.storage().data_ptr(),
      "  (False = 发生了拷贝)")

print("\n===== 7. 对照：连续张量上 view(-1) 才是真正零拷贝 =====")
v = a.reshape(-1)
print("a.reshape(-1) =", v.tolist())
print("v 与 a 共享 storage？", v.storage().data_ptr() == a.storage().data_ptr(),
      "  (True = 零拷贝)")

print("\n===== 8. contiguous() 必拷贝：物化后恢复行主序 =====")
c = t.contiguous()
print("t.contiguous() =", c.tolist())
print("c.stride       =", c.stride(), " c.is_contiguous =", c.is_contiguous())

print("\n===== 9. is_contiguous() 的几种典型取值 =====")
print("a.t()                 .is_contiguous =", a.t().is_contiguous())          # False
print("a.permute(1,0)        .is_contiguous =", a.permute(1, 0).is_contiguous())# False
print("a[:, ::2]             .is_contiguous =", a[:, ::2].is_contiguous())      # False
print("a.unsqueeze(0)        .is_contiguous =", a.unsqueeze(0).is_contiguous())# True
print("a.view(6)             .is_contiguous =", a.view(6).is_contiguous())     # True
