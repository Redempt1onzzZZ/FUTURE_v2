import mlx.core as mx

a_mx = mx.array([], dtype=mx.float32)
b_mx = mx.array([], dtype=mx.float32)

result_gpu = mx.inner(a_mx, b_mx)
print("Inner product on GPU:", result_gpu)

mx.set_default_device(mx.cpu)
result_cpu = mx.inner(a_mx, b_mx)
print("Inner product on CPU:", result_cpu)
