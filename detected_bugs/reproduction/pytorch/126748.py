import torch
import numpy as np

input_tensor=torch.tensor(np.array([[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5]], dtype=np.float32))

cpu = torch.geqrf(input_tensor)
print(cpu)

input_tensor = input_tensor.cuda()
gpu = torch.geqrf(input_tensor)
print(gpu)

# torch.return_types.geqrf(
# a=tensor([[nan, nan, nan],
#         [nan, nan, nan],
#         [nan, nan, nan]]),
# tau=tensor([nan, nan, 0.]))
# torch.return_types.geqrf(
# a=tensor([[nan, nan, nan],
#         [nan, nan, nan],
#         [nan, nan, nan]], device='cuda:0'),
# tau=tensor([nan, 0., 0.], device='cuda:0'))