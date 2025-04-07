import mlx.core as mx

a_mx = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.float32)
indices_mx = mx.array([0, 2], dtype=mx.int32)

axis = -2
taken = mx.take(a_mx, indices_mx, axis)
print(taken)
