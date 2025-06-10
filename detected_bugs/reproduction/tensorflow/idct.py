import tensorflow as tf

signal = tf.constant([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0,
                      1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],dtype=tf.float32)

type = 4
with tf.device('/cpu:0'):
    dct_result = tf.signal.idct(
        input=signal,
        type=type,
    )

    print("cpu:",dct_result.numpy())
    print(dct_result.dtype)

with tf.device('/gpu:0'):
    dct_result = tf.signal.idct(
        input=signal,
        type=type,
    )

    print("gpu:",dct_result.numpy())
    print(dct_result.dtype)