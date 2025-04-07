import mlx.core as mx

array_mx = mx.array([1, 2, 3], dtype=mx.float32)
repeats = 0
axis = 0 
repeated = mx.repeat(array_mx, repeats, axis)
print(repeated)