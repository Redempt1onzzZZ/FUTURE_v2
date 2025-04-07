import oneflow as flow
import numpy as np

input_tensor = flow.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))

output_tensor = flow.select(input_tensor, dim=1, select=1)

# Print the output tensor
print("OneFlow Selected Tensor:\n", output_tensor)
