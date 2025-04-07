import mlx.core as mx
import numpy as np
array_mlx = mx.array([[1, 2, 3], [float('nan'), 5, 6]], dtype=mx.float32)

mx.set_default_device(mx.gpu)
max_gpu = mx.max(array_mlx, axis=0)
print("Max on GPU:", max_gpu)

mx.set_default_device(mx.cpu)
max_cpu = mx.max(array_mlx, axis=0)
print("Max on CPU:", max_cpu)

a_np = np.array([[1, 2, 3], [float('nan'), 5, 6]], dtype=np.float32)
max_np = np.max(a_np, axis=0)
print("Max using NumPy:", max_np)
# import mlx.core as mx

# # Define arrays for comparison
# a = mx.array([float('nan'), 3, 2])
# b = mx.array([0, 2, 2])

# # Perform greater than comparison on GPU
# mx.set_default_device(mx.gpu)
# result_gpu = mx.greater(a, b)
# print("Greater than comparison on GPU:\n", result_gpu)

# # Perform greater than comparison on CPU
# mx.set_default_device(mx.cpu)
# result_cpu = mx.greater(a, b)
# print("Greater than comparison on CPU:\n", result_cpu)