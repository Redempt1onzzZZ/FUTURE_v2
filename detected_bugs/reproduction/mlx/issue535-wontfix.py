import mlx.core as mx
import numpy as np


print('-'*20,'argmax','-'*20)

print('mlx.core.argmax')
array = mx.array([[1, 2, 3], [float('nan'), 5, 6]],dtype=mx.float32)
mlx_argmax = mx.argmax(array, axis=0)
print(mlx_argmax)

print('numpy')
np_array = np.array([[1, 2, 3], [float('nan'), 5, 6]],dtype=np.float32)
np_argmax = np.argmax(np_array, axis=0)
print(np_argmax)


print('-'*20,'argmin','-'*20)

print('mlx.core.argmin')
mlx_argmin = mx.argmin(array, axis=0)
print(mlx_argmin)

print('numpy')
np_argmin = np.argmin(np_array, axis=0)
print(np_argmin)