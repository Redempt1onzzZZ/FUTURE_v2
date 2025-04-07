import mindspore as mind
import numpy as np

output_tensor = mind.ops.linspace(start=0, end=float('nan'), steps=5)

# Print the output tensor
print("mindspore:", output_tensor)

