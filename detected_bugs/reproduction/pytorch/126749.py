import torch
import numpy as np

input_tensor=torch.tensor(np.array([[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5],[float('inf'), float('nan'), 5]], dtype=np.float32))

cpu = torch.matrix_exp(input_tensor)
print(cpu)

input_tensor = input_tensor.cuda()
gpu = torch.matrix_exp(input_tensor)
print(gpu)

# tensor([[-2.1411e+09,  4.5621e-41, -1.6782e-04],
#         [ 3.0795e-41, -7.8090e+11,  4.5619e-41],
#         [-6.1933e-08,  3.0795e-41, -1.8737e+14]])
# tensor([[0., 0., 0.],
#         [0., 0., 0.],
#         [0., 0., 0.]], device='cuda:0')