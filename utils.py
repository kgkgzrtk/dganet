import numpy as np
import tensorflow as tf

def log10(x):
    eps = 1e-12
    numerator = tf.log(x + eps)
    denominator = tf.log(10.)
    return numerator / denominator

def gray_to_rgb(image): # grayscale image -> color map "JET"
    area = [0.0, 0.125, 0.375, 0.625, 0.875, 1.0]
    batch_size = image.get_shape().dims[0].value
    batch_img = []
    for i in tf.split(image, batch_size):
        batch_img.append( (i-tf.reduce_min(i))/(tf.reduce_max(i)-tf.reduce_min(i)) )
    image = tf.concat(batch_img , 0)
    img_filter = lambda x: tf.cast(tf.logical_and(tf.greater_equal(image, x[0]), tf.less(image, x[1])), tf.float32)
    area = [(area[i], area[i+1]) for i in range(len(area)-1)]
    img_ = [img_filter(a) for a in area]
    rad = lambda x: tf.mod(image+0.125, 0.25)/0.25 * img_[x]
    rad_ = lambda x: (1. - tf.mod(image+0.125, 0.25)/0.25) * img_[x]
    col_b = rad(0)          + 1. * img_[1]    + rad_(2)         + 0. * img_[3]  + 0. * img_[4]
    col_g = 0. * img_[0]    + rad(1)          + 1. * img_[2]    + rad_(3)       + 0. * img_[4]
    col_r = 0. * img_[0]    + 0. * img_[1]    + rad(2)          + 1. * img_[3]  + rad_(4)
    return tf.concat([col_r, col_g, col_b], 3)

