import mlx.core as mx
import numpy as np


x = np.array([0,20], dtype=np.uint64)
print("numpy:", np.isneginf(x))

x = mx.array([0,20], dtype=mx.uint64)
result = mx.isneginf(x)
print("mlx:", result)
