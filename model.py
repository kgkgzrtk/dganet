import os
import cv2
import numpy as np
import tensorflow as tf
import glob
import re
import h5py
from tqdm import tqdm, trange
from PIL import Image

from ops import *
from utils import *

class dganet(object):
    def __init__(self, sess, image_h=256, image_w=256, batch_size=4,
            input_ch=3, output_ch=1, g_lr=1e-4, d_lr=1e-4, beta1=0.9, beta2=0.999,
            reg_scale=1e-6, alpha=0.5, gp_scale=10., L_scale=1e+3, GH_scale=1e+8, DH_scale=0.,
            g_dim=64, d_dim=64, critic_k=1, keep_prob=0.5, noise_std=0.,
            rotation=10., crop_scale=[1.0, 1.5], col_scale=[0.8, 1.2], bright_scale=[0.7, 1.3],
            dataset_path=None, checkpoint_dir=None, outdata_dir=None, summary_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.read_image_h, self.read_image_w = (360, 360)
        self.image_size = self.image_h * self.image_w
        self.input_ch = input_ch
        self.output_ch = output_ch
        
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.L_scale, self.GH_scale, self.DH_scale = (L_scale, GH_scale, DH_scale)
        self.rotation = rotation
        self.crop_scale = crop_scale
        self.col_scale = col_scale
        self.bright_scale = bright_scale
        self.critic_k = critic_k
        self.reg_scale = reg_scale
        self.alpha = alpha
        self.gp_scale = gp_scale
        self.keep_prob = keep_prob
        self.noise_std = noise_std
        self.seed = 123
        
        #self.r_s = lambda x: (10.**((x+1.)/2.)-1.)/0.9  #[-1, 1] => [0, 10]
        self.r_s = lambda x : (x+1.)*5.
        self.test_writer = tf.summary.FileWriter(summary_dir+'/test', sess.graph)
        self.train_writer = tf.summary.FileWriter(summary_dir+'/train', sess.graph)
        self.checkpoint_dir = checkpoint_dir
        self.dataset_path = dataset_path
        self.outdata_dir = outdata_dir
        self.summary_dir = summary_dir
        self.build()
    
    def build(self):

        self.image_keys = ['y_x', 'y_dc1', 'y_dc2', 'y_dc3', 'y_dc4', 'y_dc5', 'y_dc6']
        self.h_range = [256, 128, 64, 32, 16, 8, 4]
        self.w_range = [256, 128, 64, 32, 16, 8, 4]

        self.x = tf.placeholder(tf.float32, [None, self.read_image_h * self.read_image_w * 3], name="x")
        self.y_ = tf.placeholder(tf.float32, [None, self.read_image_h * self.read_image_w], name="y_")
        self.training = tf.placeholder(tf.bool)

        self.x_in = tf.reshape(self.x, [self.batch_size, self.read_image_h, self.read_image_w, 3])
        self.y_tar = tf.reshape(self.y_, [self.batch_size, self.read_image_h, self.read_image_w, 1])
        
        self.dist_x, self.dist_y_tar = self.generate_image_batch(self.x_in, self.y_tar)

        self.y = self.inference(self.dist_x)

        d_real = self.discriminator(self.dist_x, self.dist_y_tar)
        d_fake = self.discriminator(self.dist_x, self.y, reuse=True)

        self.d_y_real = d_real[-1]
        self.d_y_fake = d_fake[-1]
        self.d_h_real = d_real[:-1]
        self.d_h_fake = d_fake[:-1]

        t_vars = tf.trainable_variables()
        self.d_vars = [v for v in t_vars if 'disc' in v.name]
        self.g_vars = [v for v in t_vars if 'gen' in v.name]
        
        #L1_Regularization
        self.L1_weight_penalty = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.g_vars if 'w' in w.name])
        
        #L2_Regularization 
        self.L2_weight_penalty = tf.add_n([tf.nn.l2_loss(w) for w in self.g_vars if 'w' in w.name])
        
        merged_weight_penalty = self.alpha * self.L1_weight_penalty + (1. - self.alpha) * self.L2_weight_penalty
        self.weight_penalty = self.reg_scale * merged_weight_penalty
        
        #L1 loss
        loss_ = tf.reduce_sum(tf.abs(self.dist_y_tar - self.y), [1, 2, 3])
        self.loss = tf.reduce_mean(loss_) / self.image_size

        #Discriminator hidden layer loss
        d_h_loss_list = [tf.reduce_mean(tf.abs(d_r - d_f)) for d_r, d_f in zip(self.d_h_real, self.d_h_fake)]
        d_hidden_loss = tf.reduce_mean(d_h_loss_list)
        
        #Generator Loss
        self.gan_loss = tf.reduce_mean(-self.d_y_fake)
        self.g_loss = self.gan_loss + self.loss * self.L_scale + d_hidden_loss * self.GH_scale + self.weight_penalty
        #self.g_loss = self.loss * self.L_scale + d_hidden_loss * self.GH_scale + self.weight_penalty

        #Discriminator loss + gp
        self.gp = self.gradient_penalty() * self.gp_scale
        self.w_distance = tf.reduce_mean(self.d_y_real - self.d_y_fake)
        self.d_loss = -self.w_distance + self.gp - d_hidden_loss * self.DH_scale
       
        d_gt, d_out = (self.r_s(self.dist_y_tar), self.r_s(self.y))
        self.RMS_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(d_gt - d_out), [1, 2, 3])/self.image_size))
        self.REL_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(d_gt - d_out)/d_gt, [1, 2, 3]))/self.image_size
        self.Log10_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(log10(d_gt)-log10(d_out)), [1, 2, 3]))/self.image_size
        
        scalar_summary_list = {
            'loss/RMS':             self.RMS_loss,
            'loss/REL':             self.REL_loss,
            'loss/Log10':           self.Log10_loss,
            'gen/L1_loss':          self.loss*self.L_scale,
            'gen/g_loss':           self.g_loss,
            'gen/GAN_loss':         self.gan_loss,
            'gen/GH_loss':          d_hidden_loss*self.GH_scale,
            'gen/weight_penalty':   self.weight_penalty,
            'd_hidden_loss/mean':   d_hidden_loss,
            'd_hidden_loss/h1':     d_h_loss_list[0],
            'd_hidden_loss/h2':     d_h_loss_list[1],
            'd_hidden_loss/h3':     d_h_loss_list[2],
            'd_hidden_loss/h4':     d_h_loss_list[3],
            'disc/d_loss':          self.d_loss,
            'disc/DH_loss':         d_hidden_loss*self.DH_scale,
            'disc/gp':              self.gp,
            'disc/WD_loss':         -self.w_distance,
            'wasser_distance':      self.w_distance
        }
        
        self.update_met_list = [] 
        for k, v in scalar_summary_list.items():
            mean_val, update_op = tf.contrib.metrics.streaming_mean(v, name=k)
            tf.summary.scalar(k, mean_val, collections=['train', 'test'])
            self.update_met_list.append(update_op)
        
        rgb_img = self.dist_x
        hsv_img = tf.image.grayscale_to_rgb(tf.concat(tf.split(self.dist_x,3,3),2))
        self.result_img = tf.concat([rgb_img, gray_to_rgb(tf.concat([self.y, self.dist_y_tar], 2))], 2)
        self.merged_img = tf.concat([hsv_img, self.result_img], 1)
        tf.summary.image('result-images', tf.cast(self.merged_img*255., tf.uint8), self.batch_size, collections=['train', 'test'])
        
        [tf.summary.histogram(var.name, var, collections=['train']) for var in t_vars if (('w' in var.name) or ('bn' in var.name))]

        self.output_img = tf.image.grayscale_to_rgb(d_out)

        self.train_merged = tf.summary.merge_all(key='train')
        self.test_merged = tf.summary.merge_all(key='test')
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()

    
    def discriminator(self, image, depth, reuse=False):
        stddev = 0.002
        with tf.variable_scope('disc') as scope:
            if reuse:
                scope.reuse_variables()
            dim = self.d_dim
            image = tf.reshape(image, [-1, self.image_h, self.image_w, 3])
            depth = tf.reshape(depth, [-1, self.image_h, self.image_w, 1])
            #depth = gaussian_noise_layer(depth, std=self.noise_std)
            input_ = tf.concat([image, depth],3)

            h0_0 = conv(input_, dim, k=1, stddev=stddev, bn=False, name='h0_0_conv')
            h0_1 = lrelu(conv(h0_0, dim, k=2, stddev=stddev, bn=False, name='h0_1_conv'))
            h1_0 = conv(h0_1, dim, k=1, stddev=stddev, bn=False, name='h1_0_conv')
            h1_1 = lrelu(conv(h1_0, dim*2, k=2, stddev=stddev, bn=False, name='h1_1_conv'))
            h2_0 = conv(h1_1, dim*2, k=1, stddev=stddev, bn=False, name='h2_0_conv')
            h2_1 = lrelu(conv(h2_0, dim*4, k=2, stddev=stddev, bn=False, name='h2_1_conv'))
            h3_0 = conv(h2_1, dim*4, k=1, stddev=stddev, bn=False, name='h3_0_conv')
            h3_1 = lrelu(conv(h3_0, dim*8, k=2, stddev=stddev, bn=False, name='h3_1_conv'))
            h4_0 = conv(h3_1, dim*4, k=1, stddev=stddev, bn=False, name='h4_0_conv')
            h4_1 = lrelu(conv(h4_0, dim*8, k=2, stddev=stddev, bn=False, name='h4_1_conv'))
            l_in = tf.reshape(h4_1, [self.batch_size, -1])
            l0 = linear(l_in, 1)

            d_y = [h0_1, h1_1, h2_1, h3_1, l0]
            return d_y

    def inference(self, input_):
        with tf.variable_scope('gen') as scope:
            keep_prob = self.keep_prob
            dim = self.g_dim
            input_ = tf.reshape(input_, [-1, self.image_h, self.image_w, 3])
            
            #convolutional layers
            
            y_c0 = conv(input_, dim, k=2, name='c0')
            y_c1 = conv(lrelu(y_c0), dim * 2, k=2, name='c1')
            y_c2 = conv(lrelu(y_c1), dim * 4, k=2, name='c2')
            y_c3 = conv(lrelu(y_c2), dim * 8, k=2, name='c3')
            y_c4 = conv(lrelu(y_c3), dim * 8, k=2, name='c4')
            y_c5 = conv(lrelu(y_c4), dim * 8, k=2, name='c5')

            y_dc4 = resize_conv(lrelu(y_c5), [self.batch_size, self.h_range[5], self.w_range[5], dim * 8], name='dc1')
            y_dc4 = tf.nn.dropout(y_dc4, keep_prob)
            y_cc4 = tf.concat([y_dc4, y_c4],3)

            y_dc3 = resize_conv(lrelu(y_cc4), [self.batch_size, self.h_range[4], self.w_range[4], dim * 8], name='dc2')
            y_dc3 = tf.nn.dropout(y_dc3, keep_prob)
            y_cc3 = tf.concat([y_dc3, y_c3],3)

            y_dc2 = resize_conv(lrelu(y_cc3), [self.batch_size, self.h_range[3], self.w_range[3], dim * 4], name='dc3')
            y_cc2 = tf.concat([y_dc2, y_c2],3)

            y_dc1 = resize_conv(lrelu(y_cc2), [self.batch_size, self.h_range[2], self.w_range[2], dim * 2], name='dc4')
            y_cc1 = tf.concat([y_dc1, y_c1],3)

            y_dc0 = resize_conv(lrelu(y_cc1), [self.batch_size, self.h_range[1], self.w_range[1], dim], name='dc5')
            y_cc0 = tf.concat([y_dc0, y_c0],3)

            y_out = resize_conv(lrelu(y_cc0), [self.batch_size, self.h_range[0], self.w_range[0], self.output_ch], bn=False, name='dc7')
            return tf.tanh(y_out)
    
    
    def train(self, train_epoch):

        self.get_nyu_dataset()
        g_train_op = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=self.g_vars)
        d_train_op = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=self.d_vars)

        tf.global_variables_initializer().run()
        #self.load_model("dganet_e00502.model")
        batch_size = self.batch_size
        
        
        for epoch in trange(train_epoch, desc='epoch'):

            tf.local_variables_initializer().run()
            if epoch < 0:
                perm = [x for x in range(len(self.train_image))]
            else:
                perm = np.random.permutation(len(self.train_image))

            for i in trange(len(self.train_image)//batch_size, desc='iter'):
                
                batch = batch_size*i
                
                train_image = self.train_image[perm[batch:batch+batch_size]]
                train_depth = self.train_depth[perm[batch:batch+batch_size]]
                train_feed = {self.x: train_image, self.y_: train_depth, self.training: True}

                for i in range(self.critic_k):
                    self.sess.run(d_train_op, feed_dict = train_feed)
                self.sess.run([g_train_op] + self.update_met_list, feed_dict = train_feed)

            if epoch > -1:
                # output results
                train_feed ={
                                self.x: self.train_image[:batch_size],
                                self.y_: self.train_depth[:batch_size],
                                self.training: False
                            }
                train_summary = self.sess.run(self.train_merged, feed_dict = train_feed)
                self.train_writer.add_summary(train_summary, epoch)
                
                tf.local_variables_initializer().run()
                for j in range(len(self.test_image)//batch_size):
                    batch = batch_size*j
                    test_feed = {
                                    self.x: self.test_image[batch:batch+self.batch_size],
                                    self.y_: self.test_depth[batch:batch+self.batch_size],
                                    self.training: False
                                }
                    self.sess.run(self.update_met_list, feed_dict = test_feed)
                
                test_feed = {
                                self.x: self.test_image[:batch_size],
                                self.y_: self.test_depth[:batch_size],
                                self.training: False
                            }
                test_summary = self.sess.run(self.test_merged, feed_dict = test_feed)
                self.test_writer.add_summary(test_summary, epoch)

            if epoch % 100 == 2:
                self.save_model(epoch)
    
    #gradient penalty https://arxiv.org/pdf/1704.00028
    def gradient_penalty(self):
        eps = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        y_hat = eps * self.y + (1. - eps) * self.dist_y_tar
        d_y_hat = self.discriminator(self.dist_x, y_hat, reuse=True)[-1]
        ddy = tf.gradients(d_y_hat, [y_hat])[0]
        ddy = tf.sqrt(tf.reduce_sum(tf.square(ddy), [1, 2, 3]))
        return tf.reduce_mean(tf.square(ddy - 1.))
    
    def get_nyu_dataset(self):
        f = h5py.File(self.dataset_path)
        data_list = list(zip(f['images'], f['depths']))
        img_list = []
        dep_list = []
        defpad = 7
        for (image, depth) in tqdm(data_list):
            img = image.transpose(2, 1, 0)
            dep = depth.transpose(1, 0)
            img = img[defpad:-defpad, defpad:-defpad]
            dep = dep[defpad:-defpad, defpad:-defpad]
            img = cv2.resize(img, (self.read_image_w, self.read_image_h), interpolation = cv2.INTER_AREA)
            dep = cv2.resize(dep, (self.read_image_w, self.read_image_h), interpolation = cv2.INTER_AREA)
            img = img.flatten().astype(np.float32)
            dep = dep.flatten().astype(np.float32)
            img_list.append(img)
            dep_list.append(dep)
        data_list = list(zip(img_list, dep_list))
        test_list = [v for i, v in enumerate(data_list) if i%7==0]
        train_list = [v for i, v in enumerate(data_list) if i%7!=0]
        (self.test_image, self.test_depth) = list(zip(*test_list))
        (self.train_image, self.train_depth) = list(zip(*train_list))
        self.test_image = np.asarray(self.test_image)
        self.test_depth = np.asarray(self.test_depth)
        self.train_image = np.asarray(self.train_image)
        self.train_depth = np.asarray(self.train_depth)
    
    #data augmentation
    def data_augment(self, img, tar):
        img_tar_set = tf.concat([img, tar], 2)
        #Random horizontal flip
        img_tar_set = tf.image.random_flip_left_right(img_tar_set)

        #Random rotation
        rnd_theta = np.random.uniform(-self.rotation*(np.pi/180.), self.rotation*(np.pi/180.))
        img_tar_set = tf.contrib.image.rotate(img_tar_set, rnd_theta)
        abs_theta = np.absolute(rnd_theta)
        crop_size_h = int(self.read_image_h * ((np.cos(abs_theta)/(np.sin(abs_theta)+np.cos(abs_theta)))**2))
        crop_size_w = int(crop_size_h*(self.read_image_w/self.read_image_h))
        img_tar_set = tf.image.resize_image_with_crop_or_pad(img_tar_set, crop_size_h, crop_size_w)

        #Random crop
        rnd_scale = np.random.uniform(self.crop_scale[0], self.crop_scale[1])
        img_tar_set = tf.image.resize_images(img_tar_set, [int(self.image_h*rnd_scale), int(self.image_w*rnd_scale)])
        img_tar_set = tf.random_crop(img_tar_set, [self.image_h, self.image_w, 4])

        img, tar = tf.split(img_tar_set, [3, 1], 2)

        #Random color & brightness
        img = tf.image.rgb_to_hsv(img/255.)
        h, s, v = tf.split(img, [1, 1, 1], 2)
        rnd_col = np.random.uniform(self.col_scale[0], self.col_scale[1])
        rnd_bright = np.random.uniform(self.bright_scale[0], self.bright_scale[1])
        img = tf.clip_by_value(tf.concat([h*rnd_col, s, v*rnd_bright], 2), 0., 1.)
        img = tf.image.hsv_to_rgb(img)*255.


        return img, tar

    def generate_image_batch(self, images, targets):
        img_list = [tf.squeeze(img, [0]) for img in tf.split(images, self.batch_size, 0)]
        tar_list = [tf.squeeze(tar, [0]) for tar in tf.split(targets, self.batch_size, 0)]
        d_images = []
        images_ = []
        targets_ = []
        for img, tar in zip(img_list, tar_list):
            img, tar = tf.cond(self.training,
                    lambda: self.data_augment(img, tar),
                    lambda: (tf.image.resize_images(img, [self.image_h, self.image_w]),
                             tf.image.resize_images(tar, [self.image_h, self.image_w]))
                    )

            img /= 255.                     #img: [0, 255] -> [0, 1]  
            #tar = log10(tar*0.9+1.)*2.-1.   #tar: [0, 10] --Log10--> [-1, 1] 
            tar = tar*0.2-1.

            img = tf.expand_dims(img, 0)
            tar = tf.expand_dims(tar, 0)
            images_.append(img)
            targets_.append(tar)
        images_, targets_ = tf.concat(images_, 0), tf.concat(targets_, 0)
        #images_ = tf.image.rgb_to_hsv(images_)
        return (images_, targets_)

    def save_model(self, epoch):
        model_name = "trams_e%05d.model" % epoch
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name))

    def load_model(self, model_name):
        self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, model_name))
        print("Model restored.")
    
    def gen_depth(self, save_dir):
        self.get_nyu_dataset()
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        for i, (image, depth) in enumerate(zip(self.test_image, self.test_depth)):
            image_batch = [image]*self.batch_size
            depth_batch = [depth]*self.batch_size
            feed = {self.x: image_batch, self.y_: depth_batch, self.training: False}
            #img = self.sess.run(self.result_img, feed_dict=feed)
            img = self.sess.run(self.output_img, feed_dict=feed)
            img = np.mean(img*255., axis=0).astype(np.uint8)
            print(np.shape(img))
            im = Image.fromarray(img)
            file_name = "%03d.png" % i
            im.save(os.path.join(save_dir, file_name))

