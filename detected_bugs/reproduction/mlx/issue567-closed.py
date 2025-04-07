import mlx.core as mx

#correct
array_a = mx.array([-1.0000001])
result = mx.floor(array_a)
print(result) #array([-2], dtype=float32)

array_b = mx.array([-2.0000002])
result = mx.floor(array_b)
print(result) #array([-3], dtype=float32)

#wrong
array_c = mx.array([-2.0000001])
result = mx.floor(array_c)
print(result) #array([-2], dtype=float32)

array_c = mx.array([-100.0000001])
result = mx.floor(array_c)
print(result) #array([-100], dtype=float32)