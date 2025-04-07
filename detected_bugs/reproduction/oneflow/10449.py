import oneflow as flow
import numpy as np

a = flow.tensor(np.array([[],[]]),dtype=flow.float32)
b = flow.tensor(np.array([[],[]]),dtype=flow.float32)
dims = 0
c = flow.tensordot(a, b, dims=2)
print(c)

#core dumped