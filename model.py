# -*- coding: utf-8 -*-
<<<<<<< HEAD
import os
import cv2
import numpy as np
import tensorflow as tf

from ops import *
from utils import *

class dganet(object):
    def __init__(self, sess, image_h=128, image_w=128,
            batch_size=10,
            g_dim=16, d_dim=12, input_ch=3, output_ch=1,
            dataset_dir=None, checkpoint_dir=None, outdata_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_size = image_h*image_w

        self.g_dim = g_dim
        self.d_dim = d_dim

        self.input_ch = input_ch
        self.output_ch = output_ch

        self.checkpoint_dir = checkpoint_dir
        self.dataset_dir = dataset_dir
        self.outdata_dir = outdata_dir
        self.build()

    def build(self):


        self.image_keys = ['y_p1', 'y_p4', 'y_p5', 'y_dc2', 'y_dc5', 'y_dc6']
        self.w_range = [128, 64, 32, 16, 8, 4]

        self.x = tf.placeholder("float", [None, self.image_size * 3], name="x")
        self.y_ = tf.placeholder("float", [None, self.image_size], name="y_")

        self.x_in = tf.reshape(self.x, [self.batch_size, self.image_h, self.image_w, 3])
        self.y_tar = tf.reshape(self.y_, [self.batch_size, self.image_h, self.image_w, 1])

        self.y = self.inference(self.x_in)
        self.y_out = self.y['y_dc6']

        self.yd_fake = self.discriminator(self.x_in, self.y_out)
        self.yd_real = self.discriminator(self.x_in, self.y_tar)

        self.loss = tf.nn.l2_loss(self.y_tar - self.y_out)
        self.d_loss = d_loss(self.yd_fake, self.yd_real)
        self.g_loss = g_loss(self.yd_fake, self.yd_real)
        self.acc = disc_acc(self.yd_fake, self.yd_real)

        self.c_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv')
        self.g_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
        self.d_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='disc')
               
        self.loss_sum = tf.scalar_summary("loss", self.loss)
        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        self.images = gen_image(self.y)
        self.in_image = gen_image(dict(zip(range(0,9), [self.x_in])))
        self.tar_image = gen_image(dict(zip(range(0,9), [self.y_tar])))
        
        self.merged = tf.merge_all_summaries()
        self.saver = tf.train.Saver()


    def discriminator(self, image, depth):
        dim = self.d_dim
        with tf.name_scope('disc') as scope:
            image = tf.reshape(image, [-1, self.image_h, self.image_w, 3])
            depth = tf.reshape(depth, [-1, self.image_h, self.image_w, 1])
            input_ = tf.concat(3,[image, depth])

            h0 = lrelu(conv(input_, dim, k=2, name='h0_conv', bn=False))
            h1 = lrelu(conv(h0, dim*2, k=2, name='h1_conv', bn=False))
            h2 = lrelu(conv(h1, dim*4, k=2, name='h2_conv', bn=False))
            h3 = lrelu(conv(h2, dim*8, k=2, name='h3_conv', bn=False))
            h4 = lrelu(conv(h3, dim*8, k=1, name='h4_conv', bn=False))
            h5 = lrelu(conv(h4, dim*16, k=2, name='h5_conv', bn=False))
            l0 = linear(tf.reshape(h5,[self.batch_size, -1]), self.batch_size)
            return tf.nn.sigmoid(l0)


    def inference(self, input_):
        with tf.name_scope('conv') as scope:
            dim = self.g_dim
            input_ = tf.reshape(input_, [self.batch_size, self.image_h, self.image_w, 3])
            #convolutional layers

            y_c0 = tf.nn.relu(conv(input_, dim, c=11, name='c0'))
            y_c1 = tf.nn.relu(conv(y_c0, dim * 2, c=11, name='c1'))
            y_p1 = pool(y_c1, name='pool1')
            y_c2 = tf.nn.relu(conv(y_p1, dim * 2, c=7, name='c2'))
            y_c3 = tf.nn.relu(conv(y_c2, dim * 4, c=7, k=2, name='c3'))
            y_c4 = tf.nn.relu(conv(y_c3, dim * 8, c=5, name='c4'))
            y_p4 = pool(y_c4, name='pool4')
            y_c5 = tf.nn.relu(conv(y_p4, dim * 16, c=3, name='c5'))
            y_p5 = pool(y_c5, name='pool5')

        with tf.name_scope('gen') as scope:
            #generator
            y_dc0 = tf.nn.elu(deconv(y_p5, [self.batch_size, self.w_range[3], self.w_range[3], dim * 8], c=3, name='dc0'))
            y_dc0 = tf.nn.dropout(y_dc0, 0.5)
            y_dc1 = tf.nn.elu(deconv(y_dc0, [self.batch_size, self.w_range[3], self.w_range[3], dim * 4], c=3, k=1, name='dc1'))
            y_dc2 = tf.nn.elu(deconv(y_dc1, [self.batch_size, self.w_range[2], self.w_range[2], dim * 2], c=3, name='dc2'))
            y_dc3 = tf.nn.elu(deconv(y_dc2, [self.batch_size, self.w_range[1], self.w_range[1], dim * 2], c=5, name='dc3'))
            y_dc3 = tf.nn.dropout(y_dc3, 0.5)
            y_dc4 = tf.nn.elu(deconv(y_dc3, [self.batch_size, self.w_range[0], self.w_range[0], dim], c=7, name='dc4'))
            y_in = tf.add(y_dc4, y_c0)
            y_dc5 = tf.nn.elu(deconv(y_in, [self.batch_size, self.w_range[0], self.w_range[0], dim], c=11, k=1, name='dc5'))
            y_dc5 = tf.nn.dropout(y_dc5, 0.5)
            y_dc6 = tf.nn.sigmoid(deconv(y_dc5, [self.batch_size, self.w_range[0], self.w_range[0], 1], k=1, c=11, name='dc6', bn=False))

            y = [y_p1 ,y_p4, y_p5, y_dc2, y_dc5, y_dc6]
            return dict(zip(self.image_keys, y))


    def train(self, i_lr=0.00002, d_lr=0.00002, train_epoch=5000):

        self.load_image()
        #train_op = i_train(loss + tf.add_n(tf.get_collection('w_loss')) + ALPHA * g_loss)

        i_train_op = tf.train.AdamOptimizer(2e-5).minimize(self.g_loss, var_list=list(self.c_vars + self.g_vars))
        d_train_op = tf.train.AdamOptimizer(2e-5).minimize(self.d_loss, var_list=self.d_vars)
        
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.loss_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.d_loss_sum])
        self.summary_writer = tf.train.SummaryWriter("tmp/log", self.sess.graph_def)

        gl = 10.
        dl = 10.
        batch_size = self.batch_size
        
        # save input & target image
        in_img = self.sess.run(self.in_image[0], {self.x: self.train_image[:10]})
        with open(self.outdata_dir+"/init"+"/input.png", 'wb') as f:
            f.write(in_img)
        tar_img = self.sess.run(self.tar_image[0], {self.y_: self.train_depth[:10]})
        with open(self.outdata_dir+"/init"+"/target.png", 'wb') as f:
            f.write(tar_img)
        
        for epoch in range(train_epoch):
            test_feed = {self.x: self.train_image[:batch_size], self.y_: self.train_depth[:batch_size]}

            for i in range(len(self.train_image)/batch_size):
                batch = batch_size*i
                train_feed = {self.x: self.train_image[batch:batch+batch_size],
                        self.y_: self.train_depth[batch:batch+batch_size]}

                _, gl = self.sess.run([i_train_op, self.g_loss], feed_dict = train_feed)

                if gl <= dl:
                    _, dl = self.sess.run([d_train_op, self.d_loss], feed_dict = train_feed)
                #if step % 50 == 0:
                #    sess.run(d_train_op, feed_dict = train_feed)

            if epoch % 10 == 0:
                # output results
                #result = sess.run([summary_op, loss, g_loss, d_loss, acc], feed_dict = test_feed)
                summary_str, l, gl, dl, ac = self.sess.run([self.merged, self.loss, self.g_loss, self.d_loss, self.acc], feed_dict = test_feed)
                self.summary_writer.add_summary(summary_str,epoch)
                print("loss at epoch %s: %.10f" % (epoch, l))
                print("g_loss : %.10f" % gl)
                print("d_loss : %.10f" % dl)
                print("accuracy : %f" % ac)
                print("")
                
                # save image
                for j in range(len(self.image_keys)):
                    img = self.sess.run(self.images[self.image_keys[j]], {self.x: self.train_image[:10], self.y_: self.train_depth[:10]})
                    out_dir = os.path.join(self.outdata_dir , self.image_keys[j])
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    with open( os.path.join(out_dir, "result_e%05d.png" % int(epoch)),'wb') as f:
                        f.write(img)
                
            if epoch % 100 == 0:
                self.save_model(epoch)
    
    def load_image(self):
        # input image data
        train_image = []
        train_depth = []

        for i in range(848):
            col_dir = self.dataset_dir + "/color/img_" + str(i).rjust(4,"0") + ".png"
            dep_dir = self.dataset_dir + "/depth/img_" + str(i).rjust(4,"0") + "_abs_smooth.png"
            col_img = cv2.imread(col_dir)
            dep_img = cv2.imread(dep_dir, cv2.IMREAD_GRAYSCALE)
            col_img = cv2.resize(col_img, (self.image_h, self.image_w))
            dep_img = cv2.resize(dep_img, (self.image_h, self.image_w))

            train_image.append(col_img.flatten().astype(np.float32)/255.)
            train_depth.append(dep_img.flatten().astype(np.float32)/255.)

        self.train_image = np.asarray(train_image)
        self.train_depth = np.asarray(train_depth)

    def save_model(self, epoch):
        model_name = "dganet_e%05d.model" % epoch
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name))

    def load_model(self, model_name):
        self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, model_name))

        res = self.sess.run(self.images[0], {self.x: self.train_image})
        with open("data/display/depth.png", 'wb') as f:
            f.write(res)
=======

import cv2
import numpy as np
import tensorflow as tf
IMAGE_H = 128
IMAGE_W = 128
IMAGE_SIZE = IMAGE_H*IMAGE_W

IMAGE_KEYS = ['y_p1','y_p4' ,'y_p5' ,'y_dc2' ,'y_dc5', 'y_dc6']

W_RANGE = [128, 64, 32, 16, 8, 4]
CH_RANGE = [3, 16, 32, 64, 128, 256]
BAT_SIZE = 10
ALPHA = 1

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

def conv(image, out_dim, name, c=3, k=1, stddev=0.02, wd=0.0001):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, image.get_shape().dims[-1].value, out_dim], stddev=stddev))
        b = tf.Variable(tf.constant(0.0, shape=[out_dim]))
        y = tf.nn.conv2d(image, W, strides=[1, k, k, 1], padding='SAME') + b
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
            tf.add_to_collection('w_loss', weight_decay)
        return b_n(y)

def pool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1], padding='SAME')

def deconv(image, output_shape, name, c=3, k=2, stddev=0.02, wd=0.0001):
    with tf.name_scope(name) as scope:
        W = tf.Variable(tf.truncated_normal([c, c, output_shape[-1], image.get_shape().dims[-1].value], stddev=stddev))    
        b = tf.Variable(tf.constant(0.0, shape=[output_shape[-1]]))
        y = tf.nn.deconv2d(image, W, output_shape=output_shape, strides=[1,k,k,1], padding='SAME') + b
        if wd:
            weight_decay = tf.mul(tf.nn.l2_loss(W), wd, name='weight_loss')
            tf.add_to_collection('w_loss', weight_decay)
        return y

def discriminator(image, depth):
    dim = 16
    with tf.name_scope('disc') as scope:
        image = tf.reshape(image, [-1, IMAGE_H, IMAGE_W, 3])
        depth = tf.reshape(depth, [-1, IMAGE_H, IMAGE_W, 1])

        h0 = lrelu(conv(image, dim, k=2, name='h0_conv'))
        h1 = lrelu(conv(h0, dim*2, k=2, name='h1_conv'))
        h2 = lrelu(conv(h1, dim*4, k=2, name='h2_conv'))

        l0 = lrelu(conv(depth, dim, k=2, name='l0_conv'))
        l1 = lrelu(conv(l0, dim*2, k=2, name='l1_conv'))
        l2 = lrelu(conv(l1, dim*4, k=2, name='l2_conv'))
        
        hl0 = tf.concat(3,[h2,l2])
        hl1 = linear(tf.reshape(hl0,[BAT_SIZE, -1]), BAT_SIZE)
        return tf.nn.sigmoid(hl1)


def inference(input_):
    with tf.name_scope('conv') as scope:
        dim = 16
        input_ = tf.reshape(input_, [BAT_SIZE, IMAGE_H, IMAGE_W, 3])
        #convolutional layers
        
        y_c0 = tf.nn.relu(conv(input_, dim, c=11, name='c0'))
        y_c1 = tf.nn.relu(conv(y_c0, dim * 2, c=11, name='c1'))
        y_p1 = pool(y_c1)
        y_c2 = tf.nn.relu(conv(y_p1, dim * 2, c=7, name='c2'))
        y_c3 = tf.nn.relu(conv(y_c2, dim * 4, c=7, k=2, name='c3'))
        y_c4 = tf.nn.relu(conv(y_c3, dim * 8, c=5, name='c4'))
        y_p4 = pool(y_c4)
        y_c5 = tf.nn.relu(conv(y_p4, dim * 16, c=3, name='c5'))
        y_p5 = pool(y_c5)

    with tf.name_scope('gen') as scope:
        #generator
        y_dc0 = tf.nn.relu(b_n(deconv(y_p5, [BAT_SIZE, W_RANGE[3], W_RANGE[3], dim * 8], c=3, name='dc0')))
        y_dc1 = tf.nn.relu(b_n(deconv(y_dc0, [BAT_SIZE, W_RANGE[3], W_RANGE[3], dim * 4], c=3, k=1, name='dc1')))
        y_dc2 = tf.nn.relu(b_n(deconv(y_dc1, [BAT_SIZE, W_RANGE[2], W_RANGE[2], dim * 2], c=3, name='dc2')))
        y_dc3 = tf.nn.relu(b_n(deconv(y_dc2, [BAT_SIZE, W_RANGE[1], W_RANGE[1], dim * 2], c=5, name='dc3')))
        y_dc4 = tf.nn.relu(b_n(deconv(y_dc3, [BAT_SIZE, W_RANGE[0], W_RANGE[0], dim], c=7, name='dc4')))
        y_dc5 = tf.nn.relu(b_n(deconv(y_dc4 + y_c0, [BAT_SIZE, W_RANGE[0], W_RANGE[0], dim], c=11, k=1, name='dc5')))
        y_dc6 = tf.nn.sigmoid(deconv(y_dc5, [BAT_SIZE, W_RANGE[0], W_RANGE[0], 1], k=1, c=11, name='dc6'))
        
    y = [y_p1 ,y_p4, y_p5, y_dc2, y_dc5, y_dc6]
    return dict(zip(IMAGE_KEYS, y))


def loss(y, y_):
    with tf.name_scope('loss') as scope:
        loss = tf.nn.l2_loss(y - y_)
        tf.scalar_summary("loss", loss)
    return loss

def d_loss(h, h_):
    with tf.name_scope('d_loss') as scope:
        d_loss_fake = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(h,tf.ones_like(h)) )
        d_loss_real = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(h_,tf.zeros_like(h_)) )
        d_entropy = d_loss_fake + d_loss_real
        tf.scalar_summary("d_entropy", d_entropy)
    return d_entropy

def g_loss(h, h_):
    with tf.name_scope('g_loss') as scope:
        g_entropy = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(h,tf.zeros_like(h)) )
        tf.scalar_summary("g_entropy", g_entropy)
    return g_entropy

def disc_acc(h, h_):
    with tf.name_scope('acc') as scope:
        correct_prediction = tf.reduce_mean(h_ * h + (1. - h_)*(1. - h))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def train(loss):
    with tf.name_scope('train') as scope:
        c_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv')
        g_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss, var_list=list(c_vars + g_vars))
    return train_step

def d_train(d_loss):
    with tf.name_scope('d_train') as scope:
        d_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='disc')
        train_step = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list=d_vars)
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
    d_loss = d_loss(h, h_)
    g_loss = g_loss(h, h_)
    acc = disc_acc(h, h_)

    train_op = train(loss + tf.add_n(tf.get_collection('w_loss')) + ALPHA * g_loss)
    d_train_op = d_train(d_loss)
    
    images = gen_image(result)
    in_image = gen_image(dict(zip(range(0,9), [x_ph])))
    res_image = gen_image(dict(zip(range(0,9), [y_ph])))

    saver = tf.train.Saver()
    summary_op = tf.merge_all_summaries()
    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        summary_writer = tf.train.SummaryWriter("tmp/testlog", sess.graph_def)
        sess.run(init)
 
        # train
        batch_size = 10
        train_flag = 0
        for step in range(5000):
            test_feed = {x: train_image[:batch_size], y_: train_depth[:batch_size]}
            
            for i in range(len(train_image)/batch_size):
                batch = batch_size*i
                train_feed = {x: train_image[batch:batch+batch_size],
                            y_: train_depth[batch:batch+batch_size]}
                train_res = sess.run(train_op, feed_dict = train_feed)
                
                if step % 50 == 0:
                    sess.run(d_train_op, feed_dict = train_feed)

            if step == 0:
                # save input & target image
                in_img = sess.run(in_image[0], {x: train_image[:10]})
                with open("database/image/input.png", 'wb') as f:
                    f.write(in_img)
                tar_img = sess.run(res_image[0], {y_: train_depth[:10]})
                with open("database/image/target.png", 'wb') as f:
                    f.write(tar_img)


            if step % 10 == 0:
                # output results
                result = sess.run([summary_op, loss, g_loss, d_loss, acc], feed_dict = test_feed)
                summary_str = result[0];
                summary_writer.add_summary(summary_str,step)

                print("loss at step %s: %.10f" % (step, result[1]))
                print("g_loss : %.10f" % result[2])
                print("d_loss : %.10f" % result[3])
                print("accuracy : %f" % result[4])
                print("")
                
                # save image
                num = step/10
                for j in range(len(IMAGE_KEYS)):
                    img = sess.run(images[IMAGE_KEYS[j]], {x: train_image[:10], y_: train_depth[:10]})
                    with open("database/image/result_%s_%03d.png" % (IMAGE_KEYS[j], num), 'wb') as f:
                        f.write(img)

        
        save_path = saver.save(sess, "dganet_I-O_128.model")
        sess.cllose() 

>>>>>>> 8ded99a4c185b25e0d1e20d12e6d96164b91f1bd
