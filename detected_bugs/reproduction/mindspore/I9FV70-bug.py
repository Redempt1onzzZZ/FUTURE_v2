import mindspore as mind
import numpy as np

input = mind.ops.ones((3,5), mind.int64)*2
index = mind.tensor(np.array([[0,-10000,2],[0,1,4]],))
src = mind.tensor(np.array([[0,10,20],[50,60,70]]))
out = mind.ops.scatter(input, 1, index, src)
print(out)
