import mlx.core as mx

a_mx = mx.array([], dtype=mx.float16)

res_gpu = mx.hadamard_transform(a_mx)
print("GPU:", res_gpu)

mx.set_default_device(mx.cpu)
res_cpu = mx.hadamard_transform(a_mx)
print("CPU:", res_cpu)
