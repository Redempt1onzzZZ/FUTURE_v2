import mlx.core as mx
#original array
a = mx.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
print(a)

#gpu
b = mx.round(a,decimals=38)
print(mx.default_device(),b)

#cpu
mx.set_default_device(mx.cpu)
c = mx.round(a,decimals=38)
print(mx.default_device(),c)
