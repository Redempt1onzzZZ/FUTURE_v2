import tensorflow as tf

signal = tf.constant([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0,
                      1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],dtype=tf.float32)

frame_length = 8 
with tf.device('/cpu:0'):
    mdct_result = tf.signal.mdct(
        signals=signal,
        frame_length=frame_length,
    )

    print("cpu:",mdct_result.numpy())
    print(mdct_result.dtype)

with tf.device('/gpu:0'):
    mdct_result = tf.signal.mdct(
        signals=signal,
        frame_length=frame_length,
    )

    print("gpu:",mdct_result.numpy())
    print(mdct_result.dtype)