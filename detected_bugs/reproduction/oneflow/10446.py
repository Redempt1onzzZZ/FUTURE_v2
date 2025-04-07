import oneflow as flow
import numpy as np

size = (-1, 2)

full_array = flow.full(size, 2)
print(full_array)
