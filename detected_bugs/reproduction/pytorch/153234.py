import torch
#cuda
tensor1 = torch.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=torch.float32, device='cuda')
tensor2 = torch.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=torch.float32, device='cuda')
result1 = torch.quantile(tensor1, tensor2, dim=0)
print("cuda:", result1)

#cpu
tensor3 = torch.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=torch.float32, device='cpu')
tensor4 = torch.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=torch.float32, device='cpu')
result2 = torch.quantile(tensor3, tensor4, dim=0)
print("cpu:", result2)