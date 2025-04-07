import mlx.core as mx

array1 = mx.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
array2 = mx.array([4, 5, 6])
axis = 0
concatenated = mx.concatenate([array1, array2], axis=axis)
print(concatenated)

import numpy as np

np_array1 = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3]])
np_array2 = np.array([4, 5, 6])
axis = 0
np_concatenated = np.concatenate([np_array1, np_array2], axis=axis)
print(np_concatenated)



