import oneflow as flow
import numpy as np

a = flow.tensor([[1,2,3],[1,2,3]],dtype=flow.float32)
print(a)

dim = -3

variance_a = flow.var(a, dim)
print(variance_a)

#core dumped