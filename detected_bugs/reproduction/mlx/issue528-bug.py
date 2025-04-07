import mlx.core as mx

x = mx.array([1.0, 2.0, 3.0,float('nan')])
x_cast = x.astype(mx.int32)
print(x_cast)

#结果和tensorflow和numpy不一样