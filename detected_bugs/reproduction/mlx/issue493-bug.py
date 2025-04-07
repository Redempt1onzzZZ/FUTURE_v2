import mlx.core as mx
import numpy as np

#int32
x = mx.array([-2147483648,0,2147483647], dtype=mx.int32)
result = mx.isposinf(x)
print("mlx:", result)

x = np.array([-2147483648,0,2147483647], dtype=np.int32)
print("numpy:", np.isposinf(x))

#int16
x = mx.array([-32768,0,32767], dtype=mx.int16)
result = mx.isposinf(x)
print("mlx:", result)

x = np.array([-32768,0,32767], dtype=np.int16)
print("numpy:", np.isposinf(x))

