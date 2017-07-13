# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def mse_loss(y, y_):
    loss = tf.reduce_mean(tf.squared_difference(y, y_))
    return loss

def l2_loss(y, y_):
    loss = tf.nn.l2_loss(y - y_)
    tf.summary.scalar('l2_loss', loss)
    return loss

def d_loss_real(h):
    d_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=tf.ones_like(h)) )
    #d_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(h, tf.zeros_like(h)*0.9) )
    return d_entropy

def d_loss_fake(h):
    d_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=tf.zeros_like(h)) )
    return d_entropy

def g_loss(h):
    g_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=h, labels=tf.ones_like(h)) )
    return g_entropy

def disc_acc(h, h_):
    accuracy = tf.reduce_mean(tf.concat(0,[(1. - h_), h]))
    return accuracy

def gen_image(result):
    BAT_SIZE = 10
    for key, val in result.items():
        encoded_image = result
        size = val.get_shape().dims[1].value
        ch = val.get_shape().dims[3].value
        row = col = np.trunc(np.sqrt(BAT_SIZE)).astype(np.int32)
        images = tf.cast(tf.multiply(val, 255.), tf.uint8)
        if ch == 3:
            images = [tf.expand_dims(tf.squeeze(image, [3]), 3) for image in tf.split(images, 3, 3)]
            images = tf.concat([images[2], images[1], images[0]],3)
            images = tf.reshape(images, [BAT_SIZE, size, size, 3])
            images = [tf.squeeze(image, [0]) for image in tf.split(images, BAT_SIZE, 0)]
        else:
            images = [tf.squeeze(image, [3]) for image in tf.split(images, ch, 3)]
            images = tf.reshape(images[0], [BAT_SIZE, size, size, 1])
            images = [tf.squeeze(image, [0]) for image in tf.split(images, BAT_SIZE, 0)]

        rows = []
        for i in range(row):
            rows.append(tf.concat(images[col * i + 0:col * i + col],1))
        encoded_image[key] = tf.image.encode_png(tf.concat(rows,0))
    return encoded_image

def disp_image(image):
    image = tf.cast(tf.multiply(image, 255.), tf.uint8)
    image = [tf.squeeze(im, [0]) for im in tf.split(image, 10, 0)]
    return tf.image.encode_png(image[0])


