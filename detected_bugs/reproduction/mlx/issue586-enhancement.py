import mlx.core as mx

a_mx = mx.array([2.5, 1, 8, 5, 3, 4], dtype=mx.float32)

partition_kth1 = mx.partition(a_mx, kth=1)
print(partition_kth1)

partition_kth2 = mx.partition(a_mx, kth=2)
print(partition_kth2)

partition_kth3 = mx.partition(a_mx, kth=3)
print(partition_kth3)