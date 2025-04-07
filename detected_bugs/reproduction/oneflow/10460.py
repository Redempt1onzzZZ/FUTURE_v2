import oneflow as flow
import numpy as np

input_tensor = flow.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))

index_tensor = flow.tensor([-1])

# Select elements from the input tensor along dimension 1
output_tensor = flow.index_select(input_tensor, dim=-2, index=index_tensor)

# Print the output tensor
print("OneFlow Tensor:\n", output_tensor)


