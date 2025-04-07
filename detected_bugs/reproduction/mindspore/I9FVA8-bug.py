from mindspore import set_context
import mindspore.ops as ops

set_context(device_target="CPU")
x = ops.ones(2)
y = ops.sin(x)
y = ops.arcsin(y) 
y = ops.arccos(y)
print(y)  # gives [0.00034527 0.00034527]

set_context(device_target="GPU")
y1 = ops.sin(x)
y1 = ops.arcsin(y1) 
y1 = ops.arccos(y1)
print(y1) # gives [nan nan] should be [0,0]