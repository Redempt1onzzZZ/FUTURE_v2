import mlx.core as mx

a_mx = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=mx.float32)

indices_or_sections = 3
split = mx.split(a_mx, indices_or_sections, axis=2)
print([arr for arr in split])