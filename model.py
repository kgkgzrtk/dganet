import os
import cv2
import numpy as np
import tensorflow as tf
import glob
import re
import h5py
from tqdm import tqdm

from ops import *
from utils import *

class dganet(object):
    def __init__(self, sess, image_h=128, image_w=128, batch_size=10,
            input_ch=3, output_ch=1, g_lr=1e-4, d_lr=1e-4, beta1=0.5, beta2=0.999, reg_scale=1e-4, alpha=0.5, gp_scale=10., h_scale=1e+2,
            g_dim=64, d_dim=64, K=10., critic_k=1, keep_prob=0.5, noise_std=0.,
            dataset_path=None, checkpoint_dir=None, outdata_dir=None, summary_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.image_h = image_h
        self.image_w = image_w
        self.image_size = self.image_h * self.image_w
        self.input_ch = input_ch
        self.output_ch = output_ch
        
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.K = K
        self.critic_k = critic_k
        self.reg_scale = reg_scale
        self.h_scale = h_scale
        self.alpha = alpha
        self.gp_scale = gp_scale
        self.keep_prob = keep_prob
        self.noise_std = noise_std
        self.seed = 123
        
        self.r_s = lambda x: (x+1.)/2.*10.
        self.test_writer = tf.summary.FileWriter(summary_dir+'/test', sess.graph)
        self.train_writer = tf.summary.FileWriter(summary_dir+'/train', sess.graph)
        self.checkpoint_dir = checkpoint_dir
        self.dataset_path = dataset_path
        self.outdata_dir = outdata_dir
        self.summary_dir = summary_dir
        self.build()
    
    def build(self):

        self.image_keys = ['y_x', 'y_dc1', 'y_dc2', 'y_dc3', 'y_dc4', 'y_dc5', 'y_dc6']
        self.w_range = [128, 64, 32, 16, 8, 4, 2, 1]

        self.x = tf.placeholder(tf.float32, [None, self.image_size * 3], name="x")
        self.y_ = tf.placeholder(tf.float32, [None, self.image_size], name="y_")
        self.training = tf.placeholder(tf.bool)

        self.x_in = tf.reshape(self.x, [self.batch_size, self.image_h, self.image_w, 3])
        self.y_tar = tf.reshape(self.y_, [self.batch_size, self.image_h, self.image_w, 1])
        
        self.dist_x, self.dist_y_tar = self.generate_image_batch(self.x_in, self.y_tar)

        self.y = self.inference(self.dist_x)
        self.y_out = self.y['y_dc6']

        d_real = self.discriminator(self.dist_x, self.dist_y_tar)
        d_fake = self.discriminator(self.dist_x, self.y_out, reuse=True)

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
        
        self.weight_penalty = self.alpha * self.L1_weight_penalty + (1. - self.alpha) * self.L2_weight_penalty
        
        #L1 loss
        loss_ = tf.reduce_sum(tf.abs(self.dist_y_tar - self.y_out), [1, 2, 3])
        self.loss = tf.reduce_mean(loss_)

        #Discriminator hidden layer loss
        d_h_loss_list = [tf.reduce_mean(tf.abs(self.d_h_real[i] - self.d_h_fake[i])) for i in range(len(self.d_h_real))]
        d_hidden_loss = tf.reduce_mean(d_h_loss_list) * self.h_scale
        
        #Generator Loss
        self.gan_loss = tf.reduce_mean(-self.d_y_fake) + d_hidden_loss
        self.g_loss = self.loss + self.K * self.gan_loss + self.reg_scale * self.weight_penalty

        #Discriminator loss + gp
        self.gp = self.gradient_penalty() * self.gp_scale
        self.w_distance = tf.reduce_mean(self.d_y_real) - tf.reduce_mean(self.d_y_fake)
        self.d_loss = -self.w_distance + self.gp - d_hidden_loss
       
        d_gt, d_out = (self.r_s(self.dist_y_tar), self.r_s(self.y_out))
        self.RMS_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(d_gt - d_out), [1, 2, 3])/self.image_size))
        self.REL_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(d_gt - d_out)/d_gt, [1, 2, 3]))/self.image_size
        self.Log10_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(log10(d_gt)-log10(d_out)), [1, 2, 3]))/self.image_size
        
        self.scalar_summary_list = {
            'L1_loss':              self.loss,
            'RMS_loss':             self.RMS_loss,
            'REL_loss':             self.REL_loss,
            'Log10_loss':           self.Log10_loss,
            'generator_loss':       self.g_loss,
            'discriminator_loss':   self.d_loss,
            'GAN_loss':             self.gan_loss,
            'd_hidden_loss':        d_hidden_loss,
            'generator_reg':        self.weight_penalty,
            'wasser_distance':      self.w_distance,
            'gradient_penalty':     self.gp    
        }
        
        for k, v in self.scalar_summary_list.items():
            tf.summary.scalar(k, v, collections=['train', 'test'])
        
        rgb_img = tf.image.hsv_to_rgb(self.dist_x)
        merged_img = tf.concat([rgb_img, self.dist_x, gray_to_rgb(tf.concat([self.y_out, self.dist_y_tar], 2))], 2)
        tf.summary.image('result-images', merged_img, self.batch_size, collections=['train', 'test'])
        
        [tf.summary.histogram(var.name, var, collections=['train']) for var in t_vars if (('w' in var.name) or ('bn' in var.name))]

        self.images = gen_image(self.y)
        self.in_image = gen_image(dict(zip(range(0,9), [self.x_in])))
        self.tar_image = gen_image(dict(zip(range(0,9), [gray_to_rgb(self.y_tar)])))

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

            h0 = lrelu(conv(input_, dim, k=2, stddev=stddev, bn=False, name='h0_conv'))
            h1 = lrelu(conv(h0, dim*2, k=2, stddev=stddev, bn=False, name='h1_conv'))
            h2 = lrelu(conv(h1, dim*4, k=2, stddev=stddev, bn=False, name='h2_conv'))
            h3 = lrelu(conv(h2, dim*8, k=2, stddev=stddev, bn=False, name='h3_conv'))
            h4 = lrelu(conv(h3, dim*8, k=1, stddev=stddev, bn=False, name='h4_conv'))
            l_in = tf.reshape(h4, [self.batch_size, -1])
            l0 = linear(l_in, 1)

            d_y = [h0, h1, h2, h3, l0]
            return d_y

    def inference(self, input_):
        with tf.variable_scope('gen') as scope:
            keep_prob = self.keep_prob
            dim = self.g_dim
            input_ = tf.reshape(input_, [self.batch_size, self.image_h, self.image_w, 3])
            #convolutional layers

            y_c0 = conv(input_, dim, c=4, k=2, name='c0', bn=False)
            y_c1 = conv(lrelu(y_c0), dim * 2, k=2, name='c1')
            y_c2 = conv(lrelu(y_c1), dim * 4, k=2, name='c2')
            y_c3 = conv(lrelu(y_c2), dim * 8, k=2, name='c3')
            y_c4 = conv(lrelu(y_c3), dim * 8, k=2, name='c4')
            y_c5 = conv(lrelu(y_c4), dim * 8, k=2, name='c5')
            y_c6 = conv(lrelu(y_c5), dim * 8, k=2, name='c6')


            y_dc0 = resize_conv(lrelu(y_c6), [self.batch_size, self.w_range[6], self.w_range[6], dim * 8 * 2], name='dc0')
            y_dc0 = tf.nn.dropout(y_dc0, keep_prob)

            y_in1 = tf.concat([y_dc0, y_c5],3)
            y_dc1 = resize_conv(lrelu(y_in1), [self.batch_size, self.w_range[5], self.w_range[5], dim * 8 * 2], name='dc1')
            y_dc1 = tf.nn.dropout(y_dc1, keep_prob)

            y_in2 = tf.concat([y_dc1, y_c4],3)
            y_dc2 = resize_conv(lrelu(y_in2), [self.batch_size, self.w_range[4], self.w_range[4], dim * 8 * 2], name='dc2')
            y_dc2 = tf.nn.dropout(y_dc2, keep_prob)

            y_in3 = tf.concat([y_dc2, y_c3],3)
            y_dc3 = resize_conv(lrelu(y_in3), [self.batch_size, self.w_range[3], self.w_range[3], dim * 4 * 2], name='dc3')

            y_in4 = tf.concat([y_dc3, y_c2],3)
            y_dc4 = resize_conv(lrelu(y_in4), [self.batch_size, self.w_range[2], self.w_range[2], dim * 2 * 2], name='dc4')

            y_in5 = tf.concat([y_dc4, y_c1],3)
            y_dc5 = resize_conv(lrelu(y_in5), [self.batch_size, self.w_range[1], self.w_range[1], dim * 1 * 2], name='dc5')

            y_in6 = tf.concat([y_dc5, y_c0],3)
            y_dc6 = tf.tanh(resize_conv(lrelu(y_in6), [self.batch_size, self.w_range[0], self.w_range[0], self.output_ch], name='dc6'))

            y = [input_, y_dc1, y_dc2, y_dc3, y_dc4, y_dc5, y_dc6]
            
            return dict(zip(self.image_keys, y))
    
    
    def train(self, train_epoch):

        self.get_nyu_dataset()
        g_train_op = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.g_loss, var_list=self.g_vars)
        d_train_op = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1, beta2=self.beta2).minimize(self.d_loss, var_list=self.d_vars)

        tf.global_variables_initializer().run()
        #self.load_model("dganet_e00502.model")
        batch_size = self.batch_size
        
        # save input & target image
        in_img = self.sess.run(self.in_image[0], {self.x: self.test_image[:batch_size]})
        with open(self.outdata_dir+"/init"+"/input.png", 'wb') as f:
            f.write(in_img)
        tar_img = self.sess.run(self.tar_image[0], {self.y_: self.test_depth[:batch_size]})
        with open(self.outdata_dir+"/init"+"/target.png", 'wb') as f:
            f.write(tar_img)
        
        for epoch in range(train_epoch):

            if epoch < 0:
                perm = [x for x in range(len(self.train_image))]
            else:
                perm = np.random.permutation(len(self.train_image))

            for i in tqdm(range(len(self.train_image)//batch_size)):
                
                batch = batch_size*i
                
                train_image = self.train_image[perm[batch:batch+batch_size]]
                train_depth = self.train_depth[perm[batch:batch+batch_size]]
                
                train_feed = {self.x: train_image, self.y_: train_depth, self.training: True}
                for i in range(self.critic_k):
                    self.sess.run(d_train_op, feed_dict = train_feed)
                self.sess.run(g_train_op, feed_dict = train_feed)

            if epoch % 1 == 0:
                # output results
                losses_list = []
                t_perm = np.random.permutation(len(self.test_image))
                for j in range(len(self.test_image)//self.batch_size):
                    batch = self.batch_size*j
                    test_feed = {self.x: self.test_image[batch:batch+self.batch_size], self.y_: self.test_depth[batch:batch+self.batch_size], self.training: False}
                    losses = self.sess.run([self.loss, self.RMS_loss, self.REL_loss, self.Log10_loss], feed_dict = test_feed)
                    losses_list.append(losses)
                vl, rms, rel, log10 = (np.asarray(loss).mean() for loss in list(zip(*losses_list)))
                test_feed = {self.x: self.test_image[t_perm[:10]], self.y_: self.test_depth[t_perm[:10]], self.training: False}
                test_summary = self.sess.run(self.test_merged, feed_dict = test_feed)
                self.test_writer.add_summary(test_summary, epoch)
                
                train_summary, l = self.sess.run([self.train_merged, self.loss], feed_dict = train_feed)
                self.train_writer.add_summary(train_summary, epoch)

                print("[ Epoch : %s ]"% epoch)
                print("val_loss : %.10f"% vl)
                print("RMS_loss : %.10f"% rms)
                print("REL_loss : %.10f"% rel)
                print("Log10_loss : %.10f"% log10)
                print("train_loss : %.10f"% l)
                print("")

                """ 
                # save image
                for j in range(len(self.image_keys)):
                    img = self.sess.run(self.images[self.image_keys[j]], {self.x: self.test_image[:10], self.y_: self.test_depth[:10], self.test: False})
                    out_dir = os.path.join(self.outdata_dir , self.image_keys[j])
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    with open( os.path.join(out_dir, "result_e%05d.png" % int(epoch)),'wb') as f:
                        f.write(img)
                """ 
            if epoch % 100 == 2:
                self.save_model(epoch)
    
    #gradient penalty https://arxiv.org/pdf/1704.00028
    def gradient_penalty(self):
        eps = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        y_hat = eps * self.y_out + (1. - eps) * self.dist_y_tar
        d_y_hat = self.discriminator(self.dist_x, y_hat, reuse=True)[-1]
        ddy = tf.gradients(d_y_hat, [y_hat])[0]
        ddy = tf.sqrt(tf.reduce_sum(tf.square(ddy), [1, 2, 3]))
        return tf.reduce_mean(tf.square(ddy - 1.))
    
    def get_nyu_dataset(self):
        f = h5py.File(self.dataset_path)
        data_list = list(zip(f['images'], f['depths']))
        img_list = []
        dep_list = []
        for (image, depth) in tqdm(data_list):
            img = image.transpose(2, 1, 0)
            dep = depth.transpose(1, 0)
            img = cv2.resize(img, (self.image_h, self.image_w), interpolation = cv2.INTER_AREA)
            dep = cv2.resize(dep, (self.image_h, self.image_w), interpolation = cv2.INTER_AREA)
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
        img_tar_set = tf.image.random_flip_left_right(img_tar_set)
        img, tar = tf.split(img_tar_set, [3, 1], 2)
        img = tf.image.random_brightness(img, max_delta=30)
        img = tf.image.random_contrast(img, lower=0.7, upper=1.5)
        return img, tar

    def generate_image_batch(self, images, targets):
        img_list = [tf.squeeze(img, [0]) for img in tf.split(images, self.batch_size, 0)]
        tar_list = [tf.squeeze(tar, [0]) for tar in tf.split(targets, self.batch_size, 0)]
        d_images = []
        images_ = []
        targets_ = []
        for img, tar in zip(img_list, tar_list):
            img, tar = tf.cond(self.training, lambda: self.data_augment(img, tar), lambda: (img, tar))
            img /= 255.; tar = tar/10.*2.-1.
            img = tf.expand_dims(img, 0)
            tar = tf.expand_dims(tar, 0)
            images_.append(img)
            targets_.append(tar)
        images_, targets_ = tf.concat(images_, 0), tf.concat(targets_, 0)
        images_ = tf.image.rgb_to_hsv(images_)
        return (images_, targets_)

    def save_model(self, epoch):
        model_name = "uw-gan_e%05d.model" % epoch
        if not os.path.exists(self.checkpoint_dir): os.makedirs(self.checkpoint_dir)
        self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name))

    def load_model(self, model_name):
        self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, model_name))
        print("Model restored.")



