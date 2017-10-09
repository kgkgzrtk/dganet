# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(10.)
    return numerator / denominator

def gray_to_rgb(image): # grayscale image -> color map "JET"
    area = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    shape = image.get_shape()
    image = (image-tf.reduce_min(image))/(tf.reduce_max(image)-tf.reduce_min(image))
    img_filter = lambda x: tf.cast(tf.logical_and(tf.greater_equal(image, x[0]), tf.less(image, x[1])), tf.float32)
    area = [(area[i], area[i+1]) for i in range(len(area)-1)]
    img_ = [img_filter(a) for a in area]
    rad = lambda x: tf.mod(image+0.125, 0.25)/0.25 * img_[x]
    rad_ = lambda x: (1. - tf.mod(image+0.125, 0.25)/0.25) * img_[x]
    col_b = rad(0)          + 1. * img_[1]    + rad_(2)         + 0. * img_[3]  + 0. * img_[4]
    col_g = 0. * img_[0]    + rad(1)          + 1. * img_[2]    + rad_(3)       + 0. * img_[4]
    col_r = 0. * img_[0]    + 0. * img_[1]    + rad(2)          + 1. * img_[3]  + rad_(4)
    return tf.concat([col_r, col_g, col_b], 3)

def gen_image(result):
    BAT_SIZE = 10
    for key, val in result.items():
        encoded_image = result
        size = val.get_shape().dims[1].value
        ch = val.get_shape().dims[3].value
        row = col = np.trunc(np.sqrt(BAT_SIZE)).astype(np.int32)
        images = tf.cast((val-tf.reduce_min(val))/(tf.reduce_max(val)-tf.reduce_min(val))*255., tf.uint8)
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
