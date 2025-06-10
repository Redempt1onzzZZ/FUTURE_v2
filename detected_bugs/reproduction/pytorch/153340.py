import torch
tensor1 = torch.tensor([float('nan')], dtype=torch.float32, device='cpu')
result1 = torch.quantize_per_tensor(input=tensor1,scale=1.0,zero_point=0,dtype=torch.quint8)
print(result1)
tensor2 = torch.tensor([float('nan')], dtype=torch.float32, device='cuda')
result2 = torch.quantize_per_tensor(input=tensor2,scale=1.0,zero_point=0,dtype=torch.quint8)
print(result2)