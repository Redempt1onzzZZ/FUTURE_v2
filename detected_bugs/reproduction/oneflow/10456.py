import oneflow as flow

# Create a 1-D tensor with values evenly spaced from start to end
output_tensor = flow.linspace(start=0, end=float('nan'), steps=5)

# Print the output tensor
print("OneFlow Tensor:", output_tensor)
