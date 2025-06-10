import mlx.core as mx
import numpy as np
arg_1 = np.random.randn()
print(arg_1)
mx_signal = mx.array(arg_1, dtype=mx.float32)
print(mx_signal)
print(mx_signal.shape)
fft_result = mx.fft.irfftn(mx_signal)
print(fft_result)