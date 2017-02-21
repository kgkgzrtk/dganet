# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def l2_loss(y, y_):
    with tf.name_scope('loss') as scope:
        loss = tf.nn.l2_loss(y - y_)
        return loss

def d_loss(h, h_):
    with tf.name_scope('d_loss') as scope:
        d_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(h, tf.ones_like(h)*0.9) )
        d_entropy += tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(h_, tf.zeros_like(h_)) )
        return d_entropy

def g_loss(h, h_):
    with tf.name_scope('g_loss') as scope:
        g_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(h, tf.zeros_like(h)) )
        return g_entropy

def disc_acc(h, h_):
    with tf.name_scope('acc') as scope:
        accuracy = tf.reduce_mean(tf.concat(0,[(1. - h_), h]))
        return accuracy

def gen_image(result):
    BAT_SIZE = 10
    for key, val in result.items():
        encoded_image = result
        size = val.get_shape().dims[1].value
        ch = val.get_shape().dims[3].value
        row = col = np.trunc(np.sqrt(BAT_SIZE)).astype(np.int32)
        images = tf.cast(tf.mul(val, 255.), tf.uint8)
        if ch == 3:
            images = [tf.expand_dims(tf.squeeze(image, [3]), 3) for image in tf.split(3, 3, images)]
            images = tf.concat(3, [images[2], images[1], images[0]])
            images = tf.reshape(images, [BAT_SIZE, size, size, 3])
            images = [tf.squeeze(image, [0]) for image in tf.split(0, BAT_SIZE, images)]
        else:
            images = [tf.squeeze(image, [3]) for image in tf.split(3, ch, images)]
            images = tf.reshape(images[0], [BAT_SIZE, size, size, 1])
            images = [tf.squeeze(image, [0]) for image in tf.split(0, BAT_SIZE, images)]

        rows = []
        for i in range(row):
            rows.append(tf.concat(1,images[col * i + 0:col * i + col]))
        encoded_image[key] = tf.image.encode_png(tf.concat(0, rows))
    return encoded_image


