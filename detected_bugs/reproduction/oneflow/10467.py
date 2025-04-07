import oneflow as flow
import numpy as np

# Create an input tensor
input_tensor = flow.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))
min=float('nan')
max=2
output_tensor = flow.clamp(input_tensor,min,max)

# Print the output tensor
print(output_tensor)
