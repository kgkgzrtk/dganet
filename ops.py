# -*- coding: utf-8 -*-
import tensorflow as tf

def batch_norm(input_, name='bn'):
    with tf.name_scope(name) as scope:
        shape = input_.get_shape().dims[3].value
        eps = 1e-5
        gamma = tf.Variable(tf.truncated_normal([shape], stddev=0.1))
        beta = tf.Variable(tf.truncated_normal([shape], stddev=0.1))
        mean, variance = tf.nn.moments(input_, [0, 1, 2])
        return gamma * (input_ - mean) / tf.sqrt(variance + eps) + beta

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.name_scope(name) as scope:
        return tf.maximum(x, leak*x)

def linear(input_, output_size, stddev=0.02, name='linear_layer'):
    with tf.name_scope(name) as scope:
        shape = input_.get_shape().dims

        matrix = tf.Variable(tf.truncated_normal([shape[1].value, output_size], stddev=stddev))
        bias = tf.Variable(tf.constant(0.0, shape=[output_size]))
        return tf.matmul(input_, matrix) + bias

def conv(image, out_dim, name, c=3, k=1, stddev=0.02, wd=0.00001, bn=True):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, image.get_shape().dims[-1].value, out_dim], stddev=stddev))
        b = tf.Variable(tf.constant(0.0, shape=[out_dim]))
        y = tf.nn.conv2d(image, W, strides=[1, k, k, 1], padding='SAME') + b
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
            tf.add_to_collection('w_loss', weight_decay)
        if bn: return batch_norm(y)
        else: return y

def pool(x, k=2, name='pooling'):
    with tf.name_scope(name) as scope:
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')

def deconv(image, output_shape, name, c=3, k=2, stddev=0.002, wd=0.00001, bn=True):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, output_shape[-1], image.get_shape().dims[-1].value], stddev=stddev))    
        b = tf.Variable(tf.constant(-0.5, shape=[output_shape[-1]]))
        y = tf.nn.deconv2d(image, W, output_shape=output_shape, strides=[1,k,k,1], padding='SAME') + b
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
            tf.add_to_collection('w_loss', weight_decay)
        if bn: return batch_norm(y)
        else: return y

