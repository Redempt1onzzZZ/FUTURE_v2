import tensorflow as tf

data = tf.constant([float('nan')],dtype=tf.float32)
indices = tf.constant([1],dtype=tf.int32)
segment_ids = tf.constant([0],dtype=tf.int32)
with tf.device('/cpu:0'):
    result1 = tf.sparse.segment_mean(data,indices,segment_ids)
    print(result1)
with tf.device('/gpu:0'):
    result2 = tf.sparse.segment_mean(data,indices,segment_ids)
    print(result2)
