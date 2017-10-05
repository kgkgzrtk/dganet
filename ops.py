# -*- coding: utf-8 -*-
import tensorflow as tf

def batch_norm(input_, name='bn'):
    with tf.variable_scope(name) as scope:
        shape = input_.get_shape().dims[3].value
        eps = 1e-12
        gamma = tf.get_variable('gamma', [shape], initializer=tf.truncated_normal_initializer(stddev=0.01))
        beta = tf.get_variable('beta', [shape], initializer=tf.truncated_normal_initializer(stddev=0.01))
        mean, variance = tf.nn.moments(input_, [0, 1, 2])
        return gamma * (input_ - mean) / tf.sqrt(variance + eps) + beta

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name) as scope:
        return tf.maximum(x, leak*x)

def linear(input_, output_size, stddev=0.002, name='linear_layer'):
    with tf.variable_scope(name) as scope:
        shape = input_.get_shape().dims

        matrix = tf.get_variable('w', [shape[1].value, output_size], initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, matrix) + bias

def conv(image, out_dim, name, c=4, k=1, stddev=0.002, bn=True, padding='SAME'):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('w', [c, c, image.get_shape().dims[-1].value, out_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        b = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        y = tf.nn.conv2d(image, W, strides=[1, k, k, 1], padding=padding) + b
        if bn: return batch_norm(y)
        else: return y

def pool(x, k=2, name='pooling'):
    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')

def deconv(image, output_shape, name, c=4, k=2, stddev=0.002, bn=True):
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
    image = tf.image.resize_images(image, [output_shape[1], output_shape[2]])
    y = conv(image, output_shape[-1], name=name, c=c, k=k, bn=bn)
    return y

