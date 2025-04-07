import mlx.core as mx

a_mx = mx.array([float('nan'), 0.2, 0.3], dtype=mx.float32)
b_mx = mx.array([1, 2, 3], dtype=mx.float32)

logaddexp_gpu = mx.logaddexp(a_mx, b_mx)
print("logaddexp on GPU:", logaddexp_gpu)

mx.set_default_device(mx.cpu)
logaddexp_cpu = mx.logaddexp(a_mx, b_mx)
print("logaddexp on CPU:", logaddexp_cpu)

import mlx.core as mx

a_mx = mx.array([float('nan'), 4, 6], dtype=mx.float32)
b_mx = mx.array([3, 2, 5], dtype=mx.float32)

maximum_gpu = mx.maximum(a_mx, b_mx)
print("Element-wise maximum on GPU:", maximum_gpu)

mx.set_default_device(mx.cpu)
maximum_cpu = mx.maximum(a_mx, b_mx)
print("Element-wise maximum on CPU:", maximum_cpu)
