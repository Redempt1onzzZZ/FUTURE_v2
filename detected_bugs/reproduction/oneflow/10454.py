import oneflow as flow
import numpy as np

start=0.1
end=2
step=float('inf')

# oneflow
oneflow_range = flow.arange(start, end, step)
print("OneFlow:", oneflow_range)

#numpy
np_range = np.arange(start, end, step)
print('numpy:', np_range)
