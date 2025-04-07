import oneflow as flow

x = flow.tensor([[1, 2, 3],[4, 5, 6]],dtype=flow.float32)
output = flow.permute(x, (0, 0))
print(output)
