# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
IMAGE_H = 128
IMAGE_W = 128
IMAGE_SIZE = IMAGE_H*IMAGE_W

IMAGE_KEYS = ['y_p0', 'y_p1', 'y_p3', 'y_in', 'y_dc2', 'y_dc3']

W_RANGE = [128, 64, 32, 16, 8, 4]
CH_RANGE = [3, 16, 32, 64, 128, 256]
BAT_SIZE = 10
ALPHA = 0

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
    
    _dep_img = cv2.applyColorMap(dep_img, cv2.COLORMAP_JET)

    train_image.append(col_img.flatten().astype(np.float32)/255.)
    train_depth.append(dep_img.flatten().astype(np.float32)/255.)

train_image = np.asarray(train_image)
train_depth = np.asarray(train_depth)


def b_n(input_):
    shape = input_.get_shape().dims[3].value
    eps = 1e-5
    gamma = tf.Variable(tf.truncated_normal([shape], stddev=0.1))
    beta = tf.Variable(tf.truncated_normal([shape], stddev=0.1))
    mean, variance = tf.nn.moments(input_, [0, 1, 2])
    return gamma * (input_ - mean) / tf.sqrt(variance + eps) + beta

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear(input_, output_size, stddev=0.02):
    shape = input_.get_shape().dims

    matrix = tf.Variable(tf.truncated_normal([shape[1].value, output_size], stddev=stddev))
    bias = tf.Variable(tf.constant(0.0, shape=[output_size]))
    return tf.matmul(input_, matrix) + bias

def conv(image, out_dim, name, c=3, k=1, stddev=0.1, wd=1e-5):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, image.get_shape().dims[-1].value, out_dim], stddev=stddev))
        b = tf.Variable(tf.constant(0.1, shape=[out_dim]))
        y = tf.nn.conv2d(image, W, strides=[1, k, k, 1], padding='SAME') + b
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
            tf.add_to_collection('w_loss', weight_decay)
        return y

def pool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')


def deconv(image, output_shape, name, c=5, k=1, stddev=0.1, wd=1e-5):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, output_shape[-1], image.get_shape().dims[-1].value], stddev=stddev))    
        b = tf.Variable(tf.constant(0.1, shape=[output_shape[-1]]))
        y = tf.nn.deconv2d(image, W, output_shape=output_shape, strides=[1,k,k,1], padding='SAME') + b
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
            tf.add_to_collection('w_loss', weight_decay)
        return y

def discriminator(image, depth):
    dim = 12
    with tf.name_scope('disc') as scope:
        image = tf.reshape(image, [-1, IMAGE_H, IMAGE_W, 3])
        depth = tf.reshape(depth, [-1, IMAGE_H, IMAGE_W, 1])

        h0 = lrelu(conv(image, dim, k=2, name='h0_conv'))
        h1 = lrelu(conv(h0, dim*2, k=2, name='h1_conv'))

        l0 = lrelu(conv(depth, dim, k=2, name='l0_conv'))
        l1 = lrelu(conv(l0, dim*2, k=2, name='l1_conv'))
        
        hl0 = tf.concat(3,[h1,l1])
        hl1 = lrelu(conv(hl0, dim*4, k=2, name='hl1_conv'))
        hl2 = linear(tf.reshape(hl1,[BAT_SIZE, -1]), BAT_SIZE)
        return tf.nn.sigmoid(hl2)


def inference(input_):
    with tf.name_scope('conv') as scope:
        dim = 16
        input_ = tf.reshape(input_, [BAT_SIZE, IMAGE_H, IMAGE_W, 3])
        #convolutional layers
        
        y_c0 = tf.nn.relu(conv(input_, CH_RANGE[1], c=5, name='c0'))
        y_p0 = pool(y_c0)
        y_c1 = tf.nn.relu(conv(y_p0, CH_RANGE[2], c=5, name='c1'))
        y_p1 = pool(y_c1)
        y_c2 = tf.nn.relu(conv(y_p1, CH_RANGE[2], c=3, name='c2'))
        y_c3 = tf.nn.relu(conv(y_c2, CH_RANGE[3], c=3, name='c3'))
        y_p3 = pool(y_c3)

    with tf.name_scope('gen') as scope:
        #generator
        y_dc0 = tf.nn.relu(deconv(y_p3, [BAT_SIZE, W_RANGE[2], W_RANGE[2], CH_RANGE[2]], k=2, c=3, name='dc0'))
        y_dc1 = tf.nn.relu(deconv(y_dc0, [BAT_SIZE, W_RANGE[2], W_RANGE[2], CH_RANGE[2]], c=5, name='dc1'))
        y_dc2 = tf.nn.relu(deconv(y_dc1, [BAT_SIZE, W_RANGE[1], W_RANGE[1], CH_RANGE[1]], k=2, c=5, name='dc2'))
        y_in = y_p0 + y_dc2
        y_dc3 = tf.nn.sigmoid(deconv(y_in, [BAT_SIZE, W_RANGE[0], W_RANGE[0], 1], k=2, c=5, name='dc3'))
        
    y = [y_p0 ,y_p1, y_p3, y_in, y_dc2, y_dc3]
    return dict(zip(IMAGE_KEYS, y))


def loss(y, y_):
    with tf.name_scope('loss') as scope:
        loss = tf.nn.l2_loss(y_ - y)
        tf.scalar_summary("loss", loss)
    return loss

def gen_loss(h, h_):
    with tf.name_scope('g_loss') as scope:
        zero_h = tf.zeros_like(h_)
        g_entropy = tf.reduce_mean(- (zero_h * tf.log(h + 1e-7) + (1. - zero_h) * tf.log(1. - h + 1e-7)))
        tf.scalar_summary("g_entropy", g_entropy)
    return g_entropy

def disc_loss(h, h_):
    with tf.name_scope('d_loss') as scope:
        d_entropy = tf.reduce_mean(- (h_ * tf.log(h + 1e-7) + (1. - h_) * tf.log(1. - h + 1e-7)))
        tf.scalar_summary("d_entropy", d_entropy)
        correct_prediction = tf.reduce_mean(h_ * h + (1. - h_)*(1. - h))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return d_entropy, accuracy

def train(loss):
    with tf.name_scope('train') as scope:
        c_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv')
        g_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
        #train_step = tf.train.AdamOptimizer(2e-6).minimize(loss, var_list=list(c_vars + g_vars))
        train_step = tf.train.AdamOptimizer(2e-6).minimize(loss)
    return train_step

def d_train(d_loss):
    with tf.name_scope('d_train') as scope:
        d_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='disc')
        train_step = tf.train.AdamOptimizer(3e-4).minimize(d_loss, var_list=d_vars)
    return train_step

def gen_image(result):
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

with tf.Graph().as_default():

    x = tf.placeholder("float", [None, IMAGE_SIZE * 3], name="x")
    y_ = tf.placeholder("float", [None, IMAGE_SIZE], name="y_")

    x_ph = tf.reshape(x, [BAT_SIZE, IMAGE_H, IMAGE_W, 3])
    y_ph = tf.reshape(y_, [BAT_SIZE, IMAGE_H, IMAGE_W, 1])

    result = inference(x)
    y_out = result[IMAGE_KEYS[-1]]
    
    rf = []
    label = []
    real = [tf.squeeze(ys, [0]) for ys in tf.split(0, BAT_SIZE, y_ph)]
    fake = [tf.squeeze(yf, [0]) for yf in tf.split(0, BAT_SIZE, y_out)]
    n = np.random.uniform(-1,1,BAT_SIZE)
    for i in range(BAT_SIZE):
        if n[i] > 0.:
            rf.append(real[i])
            label.append(0.)
        else:
            rf.append(fake[i])
            label.append(1.)

    sample = tf.pack(rf)

    h_ = tf.pack(label)
    h = discriminator(x, sample)

    loss = loss(y_out, y_ph)
    d_loss, acc = disc_loss(h, h_)
    g_loss = gen_loss(h, h_)

    train_op = train(loss + ALPHA * tf.add_n(tf.get_collection('w_loss')) + ALPHA * g_loss)

    d_train_op = d_train(d_loss)

    saver = tf.train.Saver()
    summary_op = tf.merge_all_summaries()
    
    images = gen_image(result)
    in_image = gen_image(dict(zip(range(0,9), [x_ph])))
    res_image = gen_image(dict(zip(range(0,9), [y_ph])))

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter("tmp/testlog", sess.graph_def)
        sess.run(init)
 
        # train
        batch_size = 10
        train_flag = 0
        for step in range(20000):
            batch = batch_size*i
            train_feed = {x: train_image[batch:batch+batch_size],
                          y_: train_depth[batch:batch+batch_size]}
            test_feed = {x: train_image[:batch_size], y_: train_depth[:batch_size]}
            
            for i in range(len(train_image)/batch_size):
                train_res = sess.run(train_op, feed_dict = train_feed)

            if step == 0:
                # save input & target image
                in_img = sess.run(in_image[0], {x: train_image[:10]})
                with open("database/image/input.png", 'wb') as f:
                    f.write(in_img)
                tar_img = sess.run(res_image[0], {y_: train_depth[:10]})
                with open("database/image/target.png", 'wb') as f:
                    f.write(tar_img)

            if step % 50 + 1 == 0:
                sess.run(d_train_op, feed_dict = train_feed)

            if step % 10 == 0:
                # output results
                result = sess.run([summary_op, loss, g_loss, d_loss, acc], feed_dict = test_feed)
                print("loss at step %s: %.10f" % (step, result[1]))
                print("g_loss : %.10f" % result[2])
                print("d_loss : %.10f" % result[3])
                print("accuracy : %f" % result[4])
                print("")
                summary_str = sess.run(summary_op,{x: train_image[:10], y_:train_depth[:10]})
                summary_writer.add_summary(summary_str,step)
                
                # save image
                num = step/10
                for j in range(len(IMAGE_KEYS)):
                    img = sess.run(images[IMAGE_KEYS[j]], {x: train_image[:10], y_: train_depth[:10]})
                    with open("database/image/result_%s_%03d.png" % (IMAGE_KEYS[j], num), 'wb') as f:
                        f.write(img)
        
        save_path = saver.save(sess, "dganet_I-O_128.model")
        sess.close() 
