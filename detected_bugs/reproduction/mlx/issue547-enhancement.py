import mlx.core as mx

a = mx.array([])
v = mx.array([1., 0, 0])
mode = 'full' 

convolved = mx.convolve(a, v, mode=mode)
print(convolved)
