import mlx.core as mx

a = mx.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],dtype=mx.float16)
b = mx.arccos(a)
print(b)

c = mx.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],dtype=mx.float32)
d = mx.arccos(c)
print(d)