import mlx.core as mx

c_mx = mx.array([1], dtype=mx.float16)
a_mx = mx.array([], dtype=mx.float32)
b_mx = mx.array([], dtype=mx.float32)

res_gpu = mx.addmm(c_mx,a_mx,b_mx)
print("GPU:", res_gpu)

mx.set_default_device(mx.cpu)
res_cpu = mx.addmm(c_mx,a_mx,b_mx)
print("CPU:", res_cpu)
