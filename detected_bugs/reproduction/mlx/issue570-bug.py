import mlx.core as mx

shape = (-1, 2)
vals = mx.array([-2,2])

full_array = mx.full(shape, vals)
print(full_array)
