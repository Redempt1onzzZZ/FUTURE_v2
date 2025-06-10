import oneflow as flow

x1 = flow.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=flow.float32)
x1 = x1.cuda()
x2 = flow.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=flow.float32)
x2 = x2.cuda()
y1 = flow.quantile(x1,x2)
print(y1)

x1 = flow.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=flow.float32)
x1=x1.cpu()
x2 = flow.tensor([0, -0.5, 1, float('nan'), float('inf')], dtype=flow.float32)
x2 = x2.cpu()
y2 = flow.quantile(x1,x2)
print(y2)