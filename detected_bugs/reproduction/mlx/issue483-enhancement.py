import mlx.core as mx

a = mx.array([-2.5, -1.5, -0.31, 0.5, 1.5, 2.5],dtype=mx.float16) 

new_shape = [3, 2]  
reshaped_array = mx.reshape(a, shape=new_shape)
print(reshaped_array)

repeated_array = mx.repeat(reshaped_array, repeats=2, axis=1)
print(repeated_array)

mean_result = mx.mean(repeated_array, axis=None, keepdims=True)
print(mean_result)
