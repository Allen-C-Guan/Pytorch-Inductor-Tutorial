# verify_ch09.py — 索引家族算子示例实跑
import torch
torch.set_printoptions(precision=4, sci_mode=False)

print("--- gather (只读抽取) ---")
inp = torch.tensor([[10, 11, 12], [20, 21, 22]])
idx = torch.tensor([[2, 0], [1, 1]])
print(torch.gather(inp, 1, idx))

print("\n--- scatter (后写覆盖) ---")
s0 = torch.zeros(2, 3, dtype=torch.long)
print(torch.scatter(s0, 1, torch.tensor([[0, 1], [1, 2]]), torch.tensor([[1, 2], [3, 4]])))

print("\n--- scatter_add (同位置累加) ---")
s1 = torch.zeros(2, 2, dtype=torch.long)
print(torch.scatter_add(s1, 1, torch.tensor([[0, 0, 1], [0, 1, 1]]), torch.tensor([[1, 2, 3], [4, 5, 6]])))

print("\n--- scatter_reduce (amax) ---")
s2 = torch.zeros(2, 2)
print(torch.scatter_reduce(s2, 1, torch.tensor([[0, 0], [1, 1]]), torch.tensor([[5., 1.], [3., 8.]]), "amax", include_self=True))

print("\n--- index (高级索引) ---")
a = torch.arange(12).reshape(3, 4)
print(a[torch.tensor([0, 2]), torch.tensor([1, 3])])

print("\n--- index_select ---")
print(a.index_select(1, torch.tensor([0, 2])))

print("\n--- index_put (accumulate 对比) ---")
idx4 = (torch.tensor([0, 0, 1]),)
vals = torch.tensor([1., 2., 3.])
print("accumulate=True :", torch.zeros(3).index_put(idx4, vals, accumulate=True))
print("accumulate=False:", torch.zeros(3).index_put(idx4, vals, accumulate=False))

print("\n--- masked_scatter ---")
ma = torch.zeros(2, 3)
mask = torch.tensor([[True, False, True], [False, True, False]])
print(torch.masked_scatter(ma, mask, torch.tensor([10., 20., 30., 40., 50.])))

print("\n--- nonzero ---")
b = torch.tensor([[1, 0, 2], [0, 3, 0]])
print(b.nonzero())

print("\n--- embedding (gather dim=0 特化) ---")
w = torch.tensor([[10., 11.], [20., 21.], [30., 31.]])
print(torch.nn.functional.embedding(torch.tensor([0, 2, 2, 1]), w))
