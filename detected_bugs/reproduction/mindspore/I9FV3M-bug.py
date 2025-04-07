import mindspore as mind
import numpy as np

input_tensor = mind.tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32))

index_tensor = mind.tensor([-1])

output_tensor = mind.ops.index_select(input_tensor, axis=-2, index=index_tensor)

print(output_tensor)
