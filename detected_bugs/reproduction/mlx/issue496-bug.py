import mlx.core as mx
import numpy as np

x = np.array([0,4294967295], dtype=np.uint32)
print("numpy:", np.isposinf(x))

x = mx.array([0,2147483647], dtype=mx.uint32)
result = mx.isposinf(x)
print("mlx1:", result)

x = mx.array([0,2147483648], dtype=mx.uint32)
result = mx.isposinf(x)
print("mlx2:", result)

