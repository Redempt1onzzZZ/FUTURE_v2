import oneflow as flow
import numpy as np

input = flow.ones((3,5))*2
index = flow.tensor(np.array([[0,10000,2],[0,1,4]], ), dtype=flow.int32)
src = flow.Tensor(np.array([[0,10,20,30,40],[50,60,70,80,90]]))
out = flow.scatter(input, 1, index, src)
print(out)
