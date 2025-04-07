import mlx.core as mx
import numpy as np

start = 1
stop = 10
num = 1

linspace_mlx = mx.linspace(start, stop, num, dtype=mx.float32)
print("Linspace using mlx:", linspace_mlx)

linspace_np = np.linspace(start, stop, num, dtype=np.float32)
print("Linspace using NumPy:", linspace_np)