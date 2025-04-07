import mlx.core as mx
a = mx.random.uniform(0, 65535, shape=(2, 2), dtype=mx.float32)
print(a)
b = mx.random.randint(0, 255, shape=(2, 2), dtype=mx.uint8)
print(b)
