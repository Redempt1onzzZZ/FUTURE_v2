import mlx.core as mx

array1 = mx.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
array2 = mx.array([4, 5, 6])
axis = 1
concatenated = mx.concatenate([array1, array2], axis=axis)
print(concatenated)


array1 = mx.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
array2 = mx.array([4, 5, 6])
array2_reshaped = mx.reshape(array2, (3, 1))
axis = 1
concatenated = mx.concatenate([array1, array2_reshaped], axis=axis)
print(concatenated)

