import tensorflow as tf

sess = tf.Session()
x = tf.ones([2, 3], 'int32')
print(sess.run(x))