import tensorflow as tf
import numpy as np

def batch_norm(input_, name='bn'):
    with tf.variable_scope(name) as scope:
        eps = 1e-5
        shape = input_.get_shape().dims[3].value
        gamma = tf.get_variable('gamma', [shape], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [shape], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input_, axes=[0, 1, 2], keep_dims=False)
        return tf.nn.batch_normalization(input_, mean, variance, beta, gamma, variance_epsilon=eps)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name) as scope:
        return tf.maximum(x, leak*x)

def linear(input_, output_size, name='linear_layer'):
    with tf.variable_scope(name) as scope:
        shape = input_.shape[-1]
        stddev = np.sqrt(1./output_size)
        matrix = tf.get_variable('w', [shape, output_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
        return tf.matmul(input_, matrix)

def conv(image, out_dim, name, c=4, k=2, stddev=0.01, bn=True):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('w', [c, c, image.get_shape().dims[-1].value, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        p = int(np.ceil(c/2-1))
        pad_image = tf.pad(image, [[0, 0], [p, p], [p, p], [0, 0]])
        y = tf.nn.conv2d(pad_image, W, strides=[1, k, k, 1], padding='VALID') + b
        if bn: return tf.contrib.layers.batch_norm(y, scale=True, fused=True, scope='bn')
        else: return y

def deconv(image, output_shape, name, c=4, k=2, stddev=0.01, bn=True):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('w', [c, c, output_shape[-1], image.get_shape().dims[-1].value], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        y = tf.nn.conv2d_transpose(image, W, output_shape=output_shape, strides=[1,k,k,1], padding='SAME') + b
        if bn: return batch_norm(y)
        else: return y

def gaussian_noise_layer(x, std=0.05):
    noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=std) 
    return x + noise

def resize_conv(image, output_shape, name, c=4, k=1, bn=True):
    image = tf.image.resize_images(image, [output_shape[1]+1, output_shape[2]+1])
    y = conv(image, output_shape[-1], name=name, c=c, k=k, bn=bn)
    return y

