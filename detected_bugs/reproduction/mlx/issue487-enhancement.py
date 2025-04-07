import mlx.core as mx

def square(x):
    return x ** 2
    
print(square((mx.array([3], dtype=mx.float32))))
grad_square = mx.grad(square)

gradient_at_3 = grad_square(mx.array([3, 4], dtype=mx.float32))

print(gradient_at_3)
