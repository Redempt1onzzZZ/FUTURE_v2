import tensorflow as tf
input1 = tf.constant([[float('inf')],[0]],dtype=tf.float32)
input2 = tf.constant([[0],[0]],dtype=tf.float32)
num_buckets = tf.constant(2,dtype=tf.int32)
with tf.device('/cpu:0'):
    result1 = tf.sparse.cross_hashed([input1,input2],num_buckets)
    print("cpu:",result1)
with tf.device('/gpu:0'):
    result2 = tf.sparse.cross_hashed([input1,input2],num_buckets)
    print("gpu:",result2)
