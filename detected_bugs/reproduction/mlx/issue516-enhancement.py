import mlx.core as mx
import numpy as np

a1 = np.array([[1, 2], [3, 4]])
b1 = np.array([[5, 6], [7, 8]])
npresult = np.matmul(a1, b1)
print('numpy', npresult)

a = mx.array([[1, 2], [3, 4]])
b = mx.array([[5, 6], [7, 8]])
mlxresult = mx.matmul(a, b)
print('mlx', mlxresult)

