import oneflow as flow
import numpy as np

input = flow.ones((3,5))*2
index = flow.tensor(np.array([[-1], [6], [4]]), dtype=flow.int)
update = flow.tensor(np.array([10.2, 5.1, 12.7]), dtype=flow.float)
out = flow.scatter_nd(index, update, [5])
print(out)
