def test_bug():
    import mlx.core as mx
    tensor1 = mx.array([], dtype=mx.float32)
    tensor2 = mx.array([], dtype=mx.float32)
    result = mx.gather_mm(tensor1, tensor2)
    print(result)

if __name__ == '__main__':
    test_bug()