import torch

# 第 10 章 排序与选取 —— Tier D 详解条目示例的实跑验证脚本。
# 数值绝不手写；本文档中粘贴的输出全部来自本脚本在
#   source env.sh new && python docs/docs/aten/_examples/verify_ch10.py
# 下的真实打印。

print("=== sort: 默认升序 ===")
v, i = torch.sort(torch.tensor([3., 1., 4., 1., 5.]))
print("values :", v)
print("indices:", i)

print()
print("=== sort: descending=True ===")
vd, id_ = torch.sort(torch.tensor([3., 1., 4., 1., 5.]), descending=True)
print("values :", vd)
print("indices:", id_)

print()
print("=== topk: k=2 (默认 largest=True) ===")
tv, ti = torch.topk(torch.tensor([3., 1., 4., 1., 5.]), k=2)
print("values :", tv)
print("indices:", ti)
