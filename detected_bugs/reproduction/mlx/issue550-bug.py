import mlx.core as mx

quantized_w = mx.array([[1, 2], [3, 4]],dtype=mx.uint32)
scales = mx.array([[0.5, 0.5], [0.5, 0.5]])
biases = mx.array([[1.0, 1.0], [1.0, 1.0]])

group_size = 0
bits = 0

dequantized = mx.dequantize(quantized_w, 
                            scales, 
                            biases, 
                            group_size=group_size, 
                            bits=bits)
print(dequantized)
