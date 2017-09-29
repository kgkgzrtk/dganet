# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import tensorflow as tf

from ops import *
from utils import *

class dganet(object):
    def __init__(self, sess, image_h=128, image_w=128,
            dataset_num=849, test_data_num=129, batch_size=10, k_num=1,
            g_dim=16, d_dim=16, input_ch=3, output_ch=1,
            dataset_dir=None, checkpoint_dir=None, outdata_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_size = image_h*image_w
        
        self.dataset_num = dataset_num
        self.test_data_num = test_data_num
        self.k_num = k_num

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

        #self.yd_fake = self.discriminator(self.x_in, self.y_out)
        #self.yd_real = self.discriminator(self.x_in, self.y_tar)

        self.loss = mse_loss(self.y_tar , self.y_out)
        #self.d_loss = d_loss(self.yd_fake, self.yd_real)
        #self.g_loss = g_loss(self.yd_fake, self.yd_real)
        #self.acc = disc_acc(self.yd_fake, self.yd_real)

        self.c_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv')
        self.g_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='gen')
        #self.d_vars = tf.get_collection(tf.GraphKeys.VARIABLES, scope='disc')
               
        #self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        #self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)


        tl_sum = tf.summary.scalar("training_loss", self.loss/self.batch_size)
        vl_sum = tf.summary.scalar("validation_loss", self.loss/self.batch_size)
        self.t_sum_op = tf.summary.merge([tl_sum])
        self.v_sum_op = tf.summary.merge([vl_sum])


        self.images = gen_image(self.y)
        self.in_image = gen_image(dict(zip(range(0,9), [self.x_in])))
        self.tar_image = gen_image(dict(zip(range(0,9), [self.y_tar])))
        
        self.dd_ =  disp_image(self.y_out)
        self.init = tf.initialize_all_variables()
        self.merged = tf.summary.merge_all()
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
            l0 = linear(tf.reshape(h2,[self.batch_size, -1]), self.batch_size)
            return tf.nn.sigmoid(l0)

    def inference(self, input_):
        with tf.name_scope('conv') as scope:
            dim = self.g_dim
            input_ = tf.reshape(input_, [self.batch_size, self.image_h, self.image_w, 3])
            #convolutional layers

            y_c0 = conv(input_, dim, c=3, name='c0', bn=False)
            y_c1 = tf.nn.relu(conv(tf.nn.relu(batch_norm(y_c0)), dim * 2, c=3, name='c1'))
            y_p1 = pool(y_c1, name='pool1')
            y_c2 = tf.nn.relu(conv(y_p1, dim * 2, c=3, name='c2'))
            y_c3 = tf.nn.relu(conv(y_c2, dim * 4, c=3, k=2, name='c3'))
            y_c4 = tf.nn.relu(conv(y_c3, dim * 8, c=3, name='c4'))
            y_p4 = pool(y_c4, name='pool4')
            y_c5 = tf.nn.relu(conv(y_p4, dim * 16, c=3, name='c5'))
            y_p5 = pool(y_c5, name='pool5')

        with tf.name_scope('gen') as scope:
            #generator
            y_dc0 = tf.nn.relu(resize_conv(y_p5, [self.batch_size, self.w_range[3], self.w_range[3], dim * 8], c=3, name='dc0'))
            y_dc1 = tf.nn.relu(resize_conv(y_dc0, [self.batch_size, self.w_range[3], self.w_range[3], dim * 4], c=3, name='dc1'))
            y_dc2 = tf.nn.relu(resize_conv(y_dc1, [self.batch_size, self.w_range[2], self.w_range[2], dim * 2], c=3, name='dc2'))
            y_dc3 = tf.nn.relu(resize_conv(y_dc2, [self.batch_size, self.w_range[1], self.w_range[1], dim * 2], c=3, name='dc3'))
            y_dc4 = tf.nn.relu(resize_conv(y_dc3, [self.batch_size, self.w_range[0], self.w_range[0], dim], c=1, name='dc4'))
            y_in = tf.add(y_dc4, y_c0)
            y_dc5 = tf.nn.relu(resize_conv(y_in, [self.batch_size, self.w_range[0], self.w_range[0], dim], c=3, name='dc5'))
            y_dc6 = tf.nn.sigmoid(resize_conv(y_dc5, [self.batch_size, self.w_range[0], self.w_range[0], 1], c=1, name='dc6', bn=False))

            y = [y_p1 ,y_p4, y_p5, y_dc2, y_dc5, y_dc6]
            return dict(zip(self.image_keys, y))


    def train(self, i_lr=0.00002, d_lr=0.00002, train_epoch=1000):
        self.load_image()

        #weights = tf.trainable_variables()
        w_i = self.c_vars + self.g_vars
        l1_penalty = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in w_i]) * 0.005
        op_loss = self.loss + l1_penalty
        i_train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss, var_list=list(self.c_vars + self.g_vars))
        #d_train_op = tf.train.AdamOptimizer(1e-4).minimize(self.d_loss, var_list=self.d_vars)
        
        tf.initialize_all_variables().run()
        #self.load_model("dganet_e00502.model")

        #self.g_sum = tf.merge_summary([self.loss_sum, self.g_loss_sum])
        #self.d_sum = tf.merge_summary([self.d_loss_sum])
        self.summary_writer = tf.summary.FileWriter("tmp/log", self.sess.graph_def)

        #gl = 10.
        #dl = 10.
        batch_size = self.batch_size
        step = 0
        train_loss = []

        # save input & target image
        in_img = self.sess.run(self.in_image[0], {self.x: self.test_image[:10]})
        with open(self.outdata_dir+"/init"+"/input.png", 'wb') as f:
            f.write(in_img)
        tar_img = self.sess.run(self.tar_image[0], {self.y_: self.test_depth[:10]})
        with open(self.outdata_dir+"/init"+"/target.png", 'wb') as f:
            f.write(tar_img)
        
        for epoch in range(train_epoch):
            
            eperm = np.random.permutation(len(self.test_image))
            test_feed = {self.x: self.test_image[eperm[:batch_size]], self.y_: self.test_depth[eperm[:batch_size]]}

            for i in range(len(self.train_image)//batch_size):
                #if epoch < train_epoch/2:
                if epoch < 300:
                    perm = [x for x in range(len(self.train_image))]
                else:
                    perm = np.random.permutation(len(self.train_image))
                batch = batch_size*i

                train_feed = {self.x: self.train_image[perm[batch:batch+batch_size]], self.y_: self.train_depth[perm[batch:batch+batch_size]]}
                _, tl = self.sess.run([i_train_op, self.loss], feed_dict = train_feed)
                train_loss.append(tl/batch_size)

            if epoch % 10 == 0:
                # output results
                #summary_str, l, gl, dl, ac = self.sess.run([self.merged, self.loss, self.g_loss, self.d_loss, self.acc], feed_dict = test_feed)
                vl, vl_op = self.sess.run([self.loss, self.v_sum_op], feed_dict = test_feed)
                tl, tl_op = self.sess.run([self.loss, self.t_sum_op], feed_dict = train_feed)
                self.summary_writer.add_summary(tl_op, epoch)
                self.summary_writer.add_summary(vl_op, epoch)
                self.summary_writer.flush()
                print("loss at epoch %s: %.10f" % (epoch, vl/self.batch_size))
                print("avg_loss : %.10f" % np.array(train_loss).mean())
                train_loss = []
                #print("g_loss : %.10f" % gl)
                #print("d_loss : %.10f" % dl)
                #print("accuracy : %f" % ac)
                print("")
                
                # save image
                for j in range(len(self.image_keys)):
                    img = self.sess.run(self.images[self.image_keys[j]], {self.x: self.test_image[:10], self.y_: self.test_depth[:10]})
                    out_dir = os.path.join(self.outdata_dir , self.image_keys[j])
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    with open( os.path.join(out_dir, "result_e%05d.png" % int(epoch)),'wb') as f:
                        f.write(img)
                
            if epoch % 100 == 2:
                self.save_model(epoch)

    def eq_hist(self, image):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img_out = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_out

    
    def load_image(self):
        # input image data
        train_image = []
        train_depth = []
        test_image = []
        test_depth = []

        for i in range(self.dataset_num):
            col_dir = self.dataset_dir + "/color/img_" + str(i).rjust(4,"0") + ".png"
            dep_dir = self.dataset_dir + "/depth/img_" + str(i).rjust(4,"0") + "_abs_smooth.png"
            col_img = cv2.imread(col_dir)
            dep_img = cv2.imread(dep_dir, cv2.IMREAD_GRAYSCALE)
            col_img_hist = self.eq_hist(col_img)
            col_img = cv2.resize(col_img, (self.image_h, self.image_w), interpolation = cv2.INTER_AREA)
            dep_img = cv2.resize(dep_img, (self.image_h, self.image_w), interpolation = cv2.INTER_AREA)
            col_img_hist = cv2.resize(col_img_hist, (self.image_h, self.image_w), interpolation = cv2.INTER_AREA)

            train_image.append(col_img.flatten().astype(np.float32)/255.)
            train_depth.append(dep_img.flatten().astype(np.float32)/255.)
            train_image.append(col_img_hist.flatten().astype(np.float32)/255.)
            train_depth.append(dep_img.flatten().astype(np.float32)/255.)

        data_list = list(zip(train_image, train_depth))
        test_list = [v for i, v in enumerate(data_list) if i%7==0]
        train_list = [v for i, v in enumerate(data_list) if i%7!=0]
        (test_image, test_depth) = list(zip(*test_list))
        (train_image, train_depth) = list(zip(*train_list))


        self.train_image = np.asarray(train_image)
        self.train_depth = np.asarray(train_depth)
        self.test_image = np.asarray(test_image)
        self.test_depth = np.asarray(test_depth)

        print("Finish load training images:")
        print(" train : %d images"%len(self.train_image))
        print(" test : %d images"%len(self.test_image))
        print("")

    def save_model(self, epoch):
        model_name = "dganet_e%05d.model" % epoch
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name))

    def load_model(self, model_name):
        self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, model_name))
        print("Model restored.")
    
    def draw_depth(self, image_dir):
        image = cv2.imread(image_dir)
        image = cv2.resize(image, (self.image_h, self.image_w), interpolation = cv2.INTER_AREA)
        image = image.flatten().astype(np.float32)/255.
        in_, out_ = self.sess.run([self.in_image[0], self.dd_], {self.x: [image for x in range(self.batch_size)]})
        with open("data/display/depth.png", 'wb') as f:
            f.write(out_)
        with open("data/display/input.png", 'wb') as f:
            f.write(in_)
