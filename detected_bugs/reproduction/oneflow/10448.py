import oneflow as flow
import numpy as np


a = flow.tensor(np.array([[1, 2, 3], [float('nan'), 5, 6]]),dtype=flow.float32)

max_flow = flow.max(a,dim=1)
print(max_flow)

import oneflow as flow
import numpy as np


a = flow.tensor(np.array([[1, 2, 3], [float('nan'), 5, 6]]),dtype=flow.float32)

min_flow = flow.min(a,dim=1)
print(min_flow)
