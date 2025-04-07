import torch
import numpy as np

input_tensor=torch.tensor(np.array([[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5]], dtype=np.float32))
input_tensor2=torch.tensor(np.array([float('inf'), float('nan'), 5], dtype=np.float32))

cpu = torch.orgqr(input_tensor,input_tensor2)
print(cpu)

input_tensor = input_tensor.cuda()
input_tensor2 = input_tensor2.cuda()
gpu = torch.orgqr(input_tensor,input_tensor2)
print(gpu)

# tensor([[-inf, nan, nan],
#         [-inf, nan, nan],
#         [-inf, nan, nan]])
# tensor([[nan, nan, nan],
#         [nan, nan, nan],
#         [nan, nan, nan]], device='cuda:0')