from mindspore import set_context
import mindspore.ops as ops
import mindspore as mind
import numpy as np

set_context(device_target="CPU")
x1 = mind.tensor(np.array([[float('inf'), 0, -1, float('nan'), 5]], dtype=np.float32))
y1 = ops.median(x1,axis=1)
print(y1)

set_context(device_target="GPU")
x2 = mind.tensor(np.array([[float('inf'), 0, -1, float('nan'), 5]], dtype=np.float32))
y2 = ops.median(x2,axis=1)
print(y2)

# (Tensor(shape=[1], dtype=Float32, value= [            nan]), Tensor(shape=[1], dtype=Int64, value= [3]))
# (Tensor(shape=[1], dtype=Float32, value= [ 5.00000000e+00]), Tensor(shape=[1], dtype=Int64, value= [4]))