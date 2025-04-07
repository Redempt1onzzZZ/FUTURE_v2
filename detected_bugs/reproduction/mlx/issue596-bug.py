import mlx.core as mx

a_mx = mx.array([[], []], dtype=mx.float32)
b_mx = mx.array([[], []], dtype=mx.float32)
dims = 2

tensordot_gpu = mx.tensordot(a_mx, b_mx, dims)
print("Tensor dot product on GPU:", tensordot_gpu)

mx.set_default_device(mx.cpu)
tensordot_cpu = mx.tensordot(a_mx, b_mx, dims)
print("Tensor dot product on CPU:", tensordot_cpu)