import oneflow as flow
import numpy as np

start=0.1
end=float('inf')
step=2

# oneflow
oneflow_range = flow.arange(start, end, step)
print("OneFlow:", oneflow_range)



