
import tensorflow as tf
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_a = tf.random.normal([1000,1000])
gpu_b = tf.random.normal([1000,2000])

c = tf.matmul(gpu_a,gpu_b)

