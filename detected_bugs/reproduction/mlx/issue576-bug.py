import mlx.core as mx

a_mx = mx.array([[float('inf'), 2, 3], [4, 5, 6]], dtype=mx.float32)

logsumexp_gpu = mx.logsumexp(a_mx, axis=1)
print("logsumexp on GPU:", logsumexp_gpu)

mx.set_default_device(mx.cpu)
logsumexp_cpu = mx.logsumexp(a_mx, axis=1)
print("logsumexp on CPU:", logsumexp_cpu)
