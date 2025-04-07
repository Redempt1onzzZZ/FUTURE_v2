from mindspore import set_context
import mindspore.ops as ops
import mindspore as mind
import numpy as np

set_context(device_target="CPU")
x1 = mind.tensor(np.array([[float('inf'), 2, 3], [4, 5, 6]], dtype=np.float32))
y1 = ops.logsumexp(x1, axis=1)
print(y1)

set_context(device_target="GPU")
x2 = mind.tensor(np.array([[float('inf'), 2, 3], [4, 5, 6]], dtype=np.float32))
y2 = ops.logsumexp(x1, axis=1)
print(y2)

#logaddexp/logdet