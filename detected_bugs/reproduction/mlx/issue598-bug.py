import mlx.core as mx
import numpy as np
condition_mx = mx.array([True, False, True, False], dtype=mx.bool_)
x_mx = mx.array([float('inf'), 2, 3, 4], dtype=mx.float32)
y_mx = mx.array([float('inf'), 20, 30, 40], dtype=mx.float32)

condition_np = np.array([True, False, True, False])
x_np = np.array([float('inf'), 2, 3, 4], dtype=np.float32)
y_np = np.array([float('inf'), 20, 30, 40], dtype=np.float32)

result_mlx = mx.where(condition_mx, x_mx, y_mx)
print("Result on Mlx:", result_mlx)

result_np = np.where(condition_np, x_np, y_np)
print("Result using NumPy:", result_np)