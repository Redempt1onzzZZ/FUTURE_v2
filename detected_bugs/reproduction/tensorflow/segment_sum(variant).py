import tensorflow as tf

data = tf.constant([float('inf')],dtype=tf.float32)
indices = tf.constant([0],dtype=tf.int32)
segment_ids = tf.constant([0],dtype=tf.int32)
num_segments = tf.constant(0,dtype=tf.int32)
with tf.device('/cpu:0'):
    result1 = tf.sparse.segment_sum(data,indices,segment_ids,num_segments)
    print(result1)
with tf.device('/gpu:0'):
    result2 = tf.sparse.segment_sum(data,indices,segment_ids,num_segments)
    print(result2)
