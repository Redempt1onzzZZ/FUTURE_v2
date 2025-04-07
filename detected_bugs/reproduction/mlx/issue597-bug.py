#bug
import mlx.core as mx
import numpy as np

a_mx = mx.array([1, 2, 3, 4, 5], dtype=mx.float32)
a_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)

axis = None 
keepdims = False
ddof = 6

variance = mx.var(a_mx, axis=axis, keepdims=keepdims, ddof=ddof)
variance_np = np.var(a_np, axis=axis, keepdims=keepdims, ddof=ddof)
print("Variance using Mlx:", variance)
print("Variance using NumPy:", variance_np)

#enhancement
import mlx.core as mx

a_mx = mx.array([1, 2, 3, 4, 5], dtype=mx.float32)

axis = 1 
keepdims = False
ddof = 0

variance = mx.var(a_mx, axis=axis, keepdims=keepdims, ddof=ddof)
print("Variance using Mlx:", variance)