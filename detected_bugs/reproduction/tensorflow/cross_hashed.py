import tensorflow as tf
input1 = tf.constant([[float('inf')],[0]],dtype=tf.float32)
input2 = tf.constant([[0],[0]],dtype=tf.float32)
with tf.device('/cpu:0'):
    result1 = tf.sparse.cross_hashed([input1,input2])
    print("cpu:",result1)
with tf.device('/gpu:0'):
    result2 = tf.sparse.cross_hashed([input1,input2])
    print("gpu:",result2)
