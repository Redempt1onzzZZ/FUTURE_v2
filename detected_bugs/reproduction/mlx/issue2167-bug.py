import mlx.core as mx

a_mx = mx.array([1], dtype=mx.float32)
b_mx = mx.array([1], dtype=mx.float32)
c_mx = mx.array([1], dtype=mx.float32)
res_gpu = mx.put_along_axis(a_mx,b_mx,c_mx,axis=None)
print("GPU:", res_gpu)

mx.set_default_device(mx.cpu)
res_cpu = mx.put_along_axis(a_mx,b_mx,c_mx,axis=None)
print("CPU:", res_cpu)
