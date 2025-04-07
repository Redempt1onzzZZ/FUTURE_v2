import mlx.core as mx

a_mx = mx.array([0, -0.5, 1, 1.5, float('inf')], dtype=mx.float32)

log1p_gpu = mx.log1p(a_mx)
print("log1p on GPU:", log1p_gpu)

mx.set_default_device(mx.cpu)
log1p_cpu = mx.log1p(a_mx)
print("log1p on CPU:", log1p_cpu)
