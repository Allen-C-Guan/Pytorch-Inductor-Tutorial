# verify_ch05.py — 实跑第 5 章 Tier D 算子示例，取真实输出。
# 运行：source env.sh new && python docs/docs/aten/_examples/verify_ch05.py
import torch

torch.set_printoptions(precision=4, sci_mode=False)

print("===== mm =====")
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])            # 2x3
b = torch.tensor([[7, 8],
                  [9, 10],
                  [11, 12]])             # 3x2
print("a.shape =", tuple(a.shape), " b.shape =", tuple(b.shape))
print("mm(a,b) =")
print(torch.mm(a, b))                    # 2x3 @ 3x2 -> 2x2

print("\n===== bmm =====")
A = torch.tensor([[[1, 2, 3],
                   [4, 5, 6]],           # batch 0: 2x3
                  [[1, 0, 0],
                   [0, 1, 0]]])          # batch 1: 2x3   (B=2)
B = torch.tensor([[[7, 8],
                   [9, 10],
                   [11, 12]],            # batch 0: 3x2
                  [[1, 2],
                   [3, 4],
                   [5, 6]]])             # batch 1: 3x2
print("A.shape =", tuple(A.shape), " B.shape =", tuple(B.shape))
print("bmm(A,B) =")
print(torch.bmm(A, B))                   # (2,2,3) @ (2,3,2) -> (2,2,2)

print("\n===== addmm =====")
input_ = torch.tensor([[10, 0],
                       [0, 10]])         # 明显非零的"被加项" (2x2)
mat1 = torch.tensor([[1, 2],
                     [3, 4]])            # 2x2
mat2 = torch.tensor([[1, 0],
                     [0, 1]])            # 2x2 单位阵 -> mat1@mat2 == mat1
print("input =")
print(input_)
print("mat1 @ mat2 (= mat1) =")
print(mat1 @ mat2)
print("addmm(input, mat1, mat2, beta=2, alpha=10) =")   # 2*input + 10*(mat1@mat2)
print(torch.addmm(input_, mat1, mat2, beta=2, alpha=10))
