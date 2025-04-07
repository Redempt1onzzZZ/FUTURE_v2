import mlx.core as mx

# Create an array for flattening
array_a = mx.array([[1, 2], [3, 4]])

flattened = mx.flatten(array_a, start_axis=2, end_axis=-1)
print(flattened)

#When I tried to use the mlx.core.flatten operation on two-dimensional arrays, there was no crash when I set end_axis to more than 2.