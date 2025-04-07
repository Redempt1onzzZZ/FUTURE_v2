import mlx.core as mx

a = mx.array([float('nan')])
print(a)
b = mx.clip(a,-1,1)
print(b)