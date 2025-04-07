import oneflow as flow

# Create a 2-D tensor with ones on the diagonal and zeros elsewhere
output_tensor = flow.eye(n=float('inf'), m=4)

# Print the output tensor
print("OneFlow Tensor:\n", output_tensor)
