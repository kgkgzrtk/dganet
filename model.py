# -*- coding: utf-8 -*-
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
