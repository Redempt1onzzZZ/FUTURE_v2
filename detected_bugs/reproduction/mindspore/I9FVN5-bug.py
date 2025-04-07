from mindspore import set_context
import mindspore.ops as ops
import mindspore as mind
import numpy as np

set_context(device_target="CPU")
x1 = mind.tensor(np.array([[float('inf'), 0, -1, float('nan'), 5]], dtype=np.float32))
y1 = ops.softmax(x1)
print(y1)

set_context(device_target="GPU")
x2 = mind.tensor(np.array([[float('inf'), 0, -1, float('nan'), 5]], dtype=np.float32))
y2 = ops.softmax(x2)
print(y2)

#[[nan  0.  0. nan  0.]]
#[[nan nan nan nan nan]]