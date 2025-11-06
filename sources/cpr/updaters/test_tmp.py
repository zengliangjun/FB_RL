import torch

F = torch.randn((2, 16, 16)) * 4
B = torch.randn((16, 16)) * 4

Ms = torch.matmul(F, B.T)
M = torch.einsum('psd, td -> pst', F, B)

sum = torch.sum(Ms - M)

sum2 = torch.sum(torch.abs(Ms - M))

print(sum, sum2)
print(sum, sum2)
