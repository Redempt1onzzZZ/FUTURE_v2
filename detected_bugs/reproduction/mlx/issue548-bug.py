import mlx.core as mx

input_array = mx.array([[[1., 2, 3], [4, 5, 6]]])  
weight_array = mx.array([[[1., 2,3], [4, 5, 6]]]) 

stride = 1
padding = -1
dilation = 1
groups = 1 

conv1d = mx.conv1d(input_array, weight_array, stride=stride, padding=padding, dilation=dilation, groups=groups)
print(conv1d)



input_array = mx.array([[[[1., 2], [3, 4]]]]) 
weight_array = mx.array([[[[1, 0.5], [0.5, 1]]]]) 

stride = 1
padding = -1
dilation = 1
groups = 1  

conv2d = mx.conv2d(input_array, weight_array, stride=stride, padding=padding, dilation=dilation, groups=groups)
print(conv2d)

#conv1d padding为负时abort，应该增加正负检验
#conv2d padding为负时输出空数组，本应该crash
#下面是torch的实现
# import torch
# import torch.nn.functional as F

# input_tensor = torch.tensor([[[1., 2, 3], [4, 5, 6]]]) 
# weight_tensor = torch.tensor([[[1., 2, 3], [4, 5, 6]]])  

# stride = 1
# padding = -1
# dilation = 1
# groups = 1

# conv1d_result = F.conv1d(input_tensor, weight_tensor, stride=stride, padding=padding, dilation=dilation, groups=groups)
# print("Conv1d result:", conv1d_result)

# input_tensor = torch.tensor([[[[1., 2], [3, 4]]]]) 
# weight_tensor = torch.tensor([[[[1, 0.5], [0.5, 1]]]])  

# stride = 1
# padding = -1
# dilation = 1
# groups = 1

# conv2d_result = F.conv2d(input_tensor, weight_tensor, stride=stride, padding=padding, dilation=dilation, groups=groups)
# print("Conv2d result:", conv2d_result)