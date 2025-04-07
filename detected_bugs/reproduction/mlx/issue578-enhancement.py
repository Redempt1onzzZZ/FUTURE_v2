import mlx.core as mx

a_mx = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.float32)

mean_gpu = mx.mean(a_mx, axis=3)
print("Mean on GPU:", mean_gpu)

mx.set_default_device(mx.cpu)
mean_cpu = mx.mean(a_mx, axis=3)
print("Mean on CPU:", mean_cpu)