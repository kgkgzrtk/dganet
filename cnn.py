# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
IMAGE_H = 128
IMAGE_W = 128
IMAGE_SIZE = IMAGE_H*IMAGE_W

IMAGE_KEY = ['y_p0', 'y_p1', 'y_p3', 'y_dc0','y_dc1', 'y_dc2']
W_RANGE = [128, 64, 32, 16]
CH_RANGE = [3, 16, 32, 64]
BAT_SIZE = 10

# input image data
train_image = []
train_depth = []

for i in range(848):
    col_dir = "database/voc/color/img_" + str(i).rjust(4,"0") + ".png"
    dep_dir = "database/voc/depth/img_" + str(i).rjust(4,"0") + "_abs_smooth.png"
    col_img = cv2.imread(col_dir)
    dep_img = cv2.imread(dep_dir, cv2.IMREAD_GRAYSCALE)
    col_img = cv2.resize(col_img, (IMAGE_H, IMAGE_W))
    dep_img = cv2.resize(dep_img, (IMAGE_H, IMAGE_W))
    train_image.append(col_img.flatten().astype(np.float32)/255.0)
    train_depth.append(dep_img.flatten().astype(np.float32)/255.0)

train_image = np.asarray(train_image)
train_depth = np.asarray(train_depth)

def b_n(input):
    shape = input.get_shape().dims[-1].value
    eps = 1e-5
    gamma = tf.Variable(tf.truncated_normal([shape], stddev=0.1))
    beta = tf.Variable(tf.truncated_normal([shape], stddev=0.1))
    mean, variance = tf.nn.moments(input, [0])
    return gamma * (input - mean) / tf.sqrt(variance + eps) + beta

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, stddev=0.02):
    shape = input_.get_shape().dims

    matrix = tf.Variable(tf.truncated_normal([shape[1].value, output_size], stddev=stddev))
    bias = tf.Variable(tf.constant(0.0, shape=[output_size]))
    return tf.matmul(input_, matrix) + bias

def conv(image, out_dim, name, c=5, k=1, stddev=0.02):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, image.get_shape().dims[-1].value, out_dim], stddev=stddev))
        b = tf.Variable(tf.constant(0.0, shape=[out_dim]))
        y = tf.nn.conv2d(image, W, strides=[1, k, k, 1], padding='SAME') + b
        return b_n(y)

def pool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')

def deconv(image, output_shape, name, c=5, k=2, stddev=0.02):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, output_shape[-1], image.get_shape().dims[-1].value], stddev=0.02))    
        b = tf.Variable(tf.constant(0.0, shape=[output_shape[-1]]))
        y = tf.nn.deconv2d(image, W, output_shape=output_shape, strides=[1,k,k,1], padding='SAME') + b
        return b_n(y)

def discriminator(image):
    dim = 16
    with tf.name_scope('discriminator') as scope:
        h0 = lrelu(conv(image, dim, k=2, name='h0_conv'))
        h1 = lrelu(conv(h0, dim*2, k=2, name='h1_conv'))
        h2 = lrelu(conv(h0, dim*4, k=2, name='h2_conv'))
        h3 = lrelu(conv(h0, dim*8, k=2, name='h3_conv'))
        h4 = linear(tf.reshape(h3,[BAT_SIZE, -1]), BAT_SIZE)
        return tf.nn.sigmoid(h4)

def inference(input_ph):
    with tf.name_scope('inference') as scope:

        input_mat = tf.reshape(input_ph, [-1, IMAGE_H, IMAGE_W, 3])

        #convolutional layers
        y_c0 = tf.nn.relu(conv(input_mat, CH_RANGE[1], name='c0'))
        y_p0 = pool(y_c0)
        y_c1 = tf.nn.relu(conv(y_p0, CH_RANGE[2], name='c1'))
        y_p1 = pool(y_c1)
        y_c2 = tf.nn.relu(conv(y_p1, CH_RANGE[2], name='c2'))
        y_c3 = tf.nn.relu(conv(y_c2, CH_RANGE[3], c=3, name='c3'))
        y_p3 = pool(y_c3)

        #generator
        y_dc0 = tf.nn.relu(deconv(y_p3, [BAT_SIZE, W_RANGE[2], W_RANGE[2], CH_RANGE[2]], c=3, name='dc0'))
        y_dc1 = tf.nn.relu(deconv(y_dc0, [BAT_SIZE, W_RANGE[1], W_RANGE[1], CH_RANGE[1]], c=5, name='dc1'))
        y_in = y_dc1 + y_p0
        y_dc2 = tf.nn.relu(deconv(y_in, [BAT_SIZE, W_RANGE[0], W_RANGE[0], 1], c=5, name='dc2'))
        
        return {'y_p0':y_p0, 'y_p1':y_p1, 'y_p3':y_p3, 'y_dc0':y_dc0, 'y_dc1':y_dc1,'y_dc2':y_dc2}


def loss(y, y_):
    with tf.name_scope('loss') as scope:
        y_shaped = tf.reshape(y_, [-1, IMAGE_H, IMAGE_W, 1])
        pow2_loss = tf.nn.l2_loss(y_shaped - y )
        tf.scalar_summary("loss", pow2_loss)
    return pow2_loss

def desc_loss(h, h_):
    with tf.name_scope('d_loss') as scope:
        cross_entropy = tf.reduce_mean(- (h_ * tf.log(h + 1e-12) + (1. - h_) * tf.log(1. - h + 1e-12)))
        tf.scalar_summary("d_entropy", cross_entropy)
        correct_prediction = tf.reduce_mean(h_ * h + (1. - h_)*(1. - h))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return cross_entropy, accuracy


def training(loss):
    with tf.name_scope('training') as scope:
        i_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='inference')
        train_step = tf.train.AdamOptimizer(0.0002).minimize(loss, var_list=i_vars)
    return train_step

def d_train(d_loss):
    with tf.name_scope('d_train') as scope:
        d_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='discriminator')
        train_step = tf.train.AdamOptimizer(0.0002).minimize(d_loss, var_list=d_vars)
    return train_step

def gen_image(result):
    encoded_image = []
    for y in result.values():
        size = y.get_shape().dims[1].value
        ch = y.get_shape().dims[3].value

        row = col = 1
        images = tf.cast(tf.mul(y,255.), tf.uint8)
        images = [tf.squeeze(image, [3]) for image in tf.split(3, ch, images)]
        images = tf.reshape(images[0], [BAT_SIZE, size, size, 1])
        images = [tf.squeeze(image, [0]) for image in tf.split(0, BAT_SIZE, images)]
        rows = []
        for i in range(row):
            rows.append(tf.concat(1,images[col * i + 0:col * i + col]))
        encoded_image.append(tf.image.encode_png(tf.concat(0, rows)))
        
    return encoded_image 


with tf.Graph().as_default():

    x = tf.placeholder("float", [None, IMAGE_SIZE * 3], name="x")
    y_ = tf.placeholder("float", [None, IMAGE_SIZE], name="y_")

    y_shaped = tf.reshape(y_, [BAT_SIZE, IMAGE_H, IMAGE_W, 1])

    result = inference(x)
    y = result['y_dc2']
    
    rf = []
    label = []
    real = [tf.squeeze(ys, [0]) for ys in tf.split(0, BAT_SIZE, y_shaped)]
    fake = [tf.squeeze(yf, [0]) for yf in tf.split(0, BAT_SIZE, y)]
    n = np.random.uniform(-1,1,BAT_SIZE)
    for i in range(BAT_SIZE):
        if n[i] > 0.:
            rf.append(real[i])
            label.append(0.)
        else:
            rf.append(fake[i])
            label.append(1.)
    d_in = tf.pack(rf)
    h_ = tf.pack(label)

    h = discriminator(d_in)

    
    d_loss, acc = desc_loss(h, h_)
    
    d_train_op = d_train(d_loss)

    loss = loss(y, y_)
    alpha = 0.2
    train_op = training( loss /(alpha + d_loss) )

    saver = tf.train.Saver()
    summary_op = tf.merge_all_summaries()
    
    images = gen_image(result)
    res_image = gen_image(dict(zip(range(0,9), [y_shaped])))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter("tmp/testlog", sess.graph_def)
        sess.run(init)
 
        # train
        batch_size = 10
        for step in range(15000):
            for i in range(len(train_image)/batch_size):
                batch = batch_size*i
                feed = {x: train_image[batch:batch+batch_size],
                        y_: train_depth[batch:batch+batch_size]}
                sess.run([train_op, d_train_op], feed_dict = feed)
            
            if step == 0:
                res = sess.run(res_image, {y_: train_depth[:batch_size]})
                for j in range(len(res)):
                    with open("database/image/y_-%s.png" % (IMAGE_KEY[j]), 'wb') as f:
                        f.write(res[j])

            if step % 10 == 0:
                feed = {x: train_image[:batch_size], y_: train_depth[:batch_size]}
                # output results
                result = sess.run([summary_op, loss, d_loss, acc], feed_dict=feed)
                print("loss at step %s: %.10f" % (step, result[1]))
                print("d_loss : %.10f" % result[2])
                print("accuracy : %f" % result[3])
                print("")
                summary_str = sess.run(summary_op,{x: train_image[:10], y_:train_depth[:10]})
                summary_writer.add_summary(summary_str,step)
            
            if step % 100 == 0:
                num = step/100
                res = sess.run(images, {x: train_image[:10], y_: train_depth[:batch_size]})
                for j in range(len(res)):
                    with open("database/image/result%s-%03d.png" % (IMAGE_KEY[j], num), 'wb') as f:
                        f.write(res[j])
        
        save_path = saver.save(sess, "CDC_I-O_128.model")
        sess.close() 
