import mlx.core as mx
x = mx.ones(2)
y = mx.sin(x)
y = mx.arcsin(y)  # inverse operations, gives [1, 1]
y = mx.arccos(y)
print(mx.default_device(),y) # gives [nan, nan]

mx.set_default_device(mx.cpu)
y1 = mx.sin(x)
y1 = mx.arcsin(y1) # gives [1, 1]
y1 = mx.arccos(y1)
print(mx.default_device(),y1) # gives [0, 0]


