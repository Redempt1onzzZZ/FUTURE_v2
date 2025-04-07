import mlx.core as mx
import numpy as np

start = 0.1
stop = 2
step = float('nan')

#mlx
range = mx.arange(start, stop, step)
print('mlx:', range)

#numpy
np_range = np.arange(start, stop, step)
print('numpy:', np_range)