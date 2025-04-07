import mindspore.ops as ops
import mindspore as mind
import numpy as np

input_tensor = mind.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
min=float('nan')
max=2
output_tensor = ops.clamp(input_tensor,min,max)

# Print the output tensor
print(output_tensor)
