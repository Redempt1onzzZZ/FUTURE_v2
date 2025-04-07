import mindspore as mind
import numpy as np


a = mind.tensor(np.array([[1, 2, 3], [float('nan'), 5, 6]]),dtype=mind.float32)
max1 = mind.ops.permute(a,(0, 0))
print(max1)