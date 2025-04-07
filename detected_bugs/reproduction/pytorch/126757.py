import torch
import numpy as np

input_tensor=torch.tensor(np.array([[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5]], dtype=np.float32))

cpu = torch.unique(input_tensor)
print(cpu)

input_tensor = input_tensor.cuda()
gpu = torch.unique(input_tensor)
print(gpu)

# tensor([5., nan, nan, nan, inf])
# tensor([5., inf, nan, nan, nan], device='cuda:0')