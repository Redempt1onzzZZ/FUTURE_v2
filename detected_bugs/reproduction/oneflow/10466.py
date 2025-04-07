import oneflow as flow
import numpy as np

x = flow.ones(2,device='cuda')
y = flow.sin(x)
y = flow.arcsin(y)  # gives [1.0000, 1.0000]
y = flow.arccos(y)
print(y.device,y)  # gives [nan, nan]

x = x.cpu()
y1 = flow.sin(x)
y1 = flow.arcsin(y1) # gives [1.0000, 1.0000]
y1 = flow.arccos(y1)
print(y1.device,y1) # gives [0.0003, 0.0003] should be [0,0]
