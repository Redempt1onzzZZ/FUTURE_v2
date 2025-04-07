import mlx.core as mx
import numpy as np

print('='*10,'mlx.core.argmax','='*10)
array1 = mx.array([[1, 2, 3], [4, 5, 6]],dtype=mx.int32)
mlx_argmax = mx.argmax(array1, axis=0)
print(mlx_argmax)
print(mlx_argmax.dtype)
print(-mlx_argmax)

print('='*10,'mlx.core.argmax','='*10)
array2 = mx.array([[4, 5, 6],[1, 2, 3]],dtype=mx.int32)
mlx_argmin = mx.argmin(array2, axis=0)
print(mlx_argmin)
print(mlx_argmin.dtype)
print(-mlx_argmin)

print('='*10,'mlx.core.argpartition','='*10)
kth = 2
axis = -1
array3 = mx.array([3, 1, 2, 4],dtype=mx.int32)
argpartition = mx.argpartition(array3, kth, axis=axis)
print(argpartition)
print(argpartition.dtype)
print(-argpartition)

print('='*10,'mlx.core.argsort','='*10)
array4 = mx.array([3, 1, 2, 4],dtype=mx.int32)
axis = -1
argsort = mx.argsort(array4, axis=axis)
print(argsort)
print(argsort.dtype)
print(-argsort)

print('='*10,'numpy','='*10)
np_array = np.array([[1, 2, 3], [4, 5, 6]],dtype=np.int32)
np_argmax = np.argmax(np_array, axis=0)
print(np_argmax)
print(np_argmax.dtype)
print(-np_argmax)
