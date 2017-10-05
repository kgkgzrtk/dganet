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
            input_ch=3, output_ch=1, g_lr=1e-4, d_lr=1e-4, l1_reg_scale=3e-3, l2_reg_scale=1e-4, gp_scale=10.,
            g_dim=64, d_dim=64, K=0.5, critic_k=1, c=0.01, keep_prob=0.5, noise_std=0,
            dataset_path=None, checkpoint_dir=None, outdata_dir=None, summary_dir=None):

        self.sess = sess
        self.batch_size = batch_size
        self.read_image_h = int(image_h * 1.1)
        self.read_image_w = int(image_w * 1.1)
        self.image_h = image_h
        self.image_w = image_w
        self.image_size = self.read_image_h * self.read_image_w
        self.input_ch = input_ch
        self.output_ch = output_ch
        
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.g_dim = g_dim
        self.d_dim = d_dim
        self.K = K
        self.critic_k = critic_k
        self.c = c
        self.l1_reg_scale = l1_reg_scale
        self.l2_reg_scale = l2_reg_scale
        self.gp_scale = gp_scale
        self.keep_prob = keep_prob
        self.noise_std = noise_std
        self.seed = 123
        
        self.test_writer = tf.summary.FileWriter(summary_dir+'/test', sess.graph)
        self.train_writer = tf.summary.FileWriter(summary_dir+'/train', sess.graph)
        self.checkpoint_dir = checkpoint_dir
        self.dataset_path = dataset_path
        self.outdata_dir = outdata_dir
        self.summary_dir = summary_dir
        self.build()
    
    def build(self):

        self.image_keys = ['y_x', 'y_c1', 'y_c2', 'y_c4', 'y_in2', 'y_in4', 'y_in5', 'y_dc6']
        self.w_range = [128, 64, 32, 16, 8, 4, 2, 1]

        self.x = tf.placeholder(tf.float32, [None, self.image_size * 3], name="x")
        self.y_ = tf.placeholder(tf.float32, [None, self.image_size], name="y_")

        self.x_in = tf.reshape(self.x, [self.batch_size, self.read_image_h, self.read_image_w, 3])
        self.y_tar = tf.reshape(self.y_, [self.batch_size, self.read_image_h, self.read_image_w, 1])
        
        self.test = tf.placeholder(tf.int32)
        self.dist_x, self.dist_y_tar = self.generate_image_batch(self.x_in, self.y_tar, self.test)

        self.y = self.inference(self.dist_x)
        self.y_out = self.y['y_dc6']

        self.d_y_real = self.discriminator(self.dist_x, self.dist_y_tar)
        self.d_y_fake = self.discriminator(self.dist_x, self.y_out, reuse=True)



        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'disc' in var.name]
        self.g_vars = [var for var in t_vars if 'gen' in var.name]
        
        #L1_Regularization
        self.L1_weight_penalty = tf.add_n([tf.reduce_sum(tf.abs(w)) for w in self.g_vars if 'w' in w.name]) * self.l1_reg_scale
        
        #L2_Regularization 
        self.L2_weight_penalty = tf.add_n([tf.nn.l2_loss(w) for w in self.g_vars if 'w' in w.name]) * self.l2_reg_scale

        self.weight_penalty = self.L1_weight_penalty #+ self.L2_weight_penalty
        
        gp = self.gradient_penalty() * self.gp_scale
        self.clip_updates = [w.assign(tf.clip_by_value(w, -self.c, self.c)) for w in self.d_vars]
        
        #L2_norm
        loss_ = tf.reduce_sum(tf.square(self.dist_y_tar - self.y_out)/2., [1, 2, 3])
        self.loss = tf.reduce_mean(loss_)
        
        #L2_loss + (L2_loss/GAN_loss)*GAN_LOSS + L1_reg
        gan_loss_ = tf.reduce_sum(-self.d_y_fake, [1, 2, 3])
        gan_loss = -tf.reduce_mean(gan_loss_)
        self.g_loss = tf.reduce_mean(loss_ + self.K * tf.abs(self.loss/gan_loss) * gan_loss_) + self.weight_penalty
        #Critic loss + gp
        self.d_loss = tf.reduce_mean(tf.reduce_sum(self.d_y_fake - self.d_y_real, [1, 2, 3])) + gp
        
        tf.summary.scalar('l2_loss', self.loss)
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('gan_loss', gan_loss)
        tf.summary.scalar('wassaer_distance', tf.reduce_mean(tf.reduce_sum(self.d_y_real - self.d_y_fake, [1, 2, 3])))
        tf.summary.scalar('g_reg', self.weight_penalty)
        tf.summary.scalar('gradient_penalty', gp)
        
        tf.summary.image('y_out', tf.reshape(self.y_out, [-1, self.y_out.get_shape().dims[1].value, self.y_out.get_shape().dims[2].value, 1]), 10)

        self.images = gen_image(self.y)
        self.in_image = gen_image(dict(zip(range(0,9), [self.x_in])))
        self.tar_image = gen_image(dict(zip(range(0,9), [self.y_tar])))
        
        self.dd_ =  disp_image(self.y_out)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()


    def discriminator(self, image, depth, reuse=False):
        with tf.variable_scope('disc') as scope:
            if reuse:
                scope.reuse_variables()
            dim = self.d_dim
            image = tf.reshape(image, [-1, self.image_h, self.image_w, 3])
            depth = tf.reshape(depth, [-1, self.image_h, self.image_w, 1])
            depth = gaussian_noise_layer(depth, std=self.noise_std)
            input_ = tf.concat([image, depth],3)

            h0 = lrelu(conv(input_, dim, k=2, bn=False, name='h0_conv'))
            h1 = lrelu(conv(h0, dim*2, k=2, bn=False, name='h1_conv'))
            h2 = lrelu(conv(h1, dim*4, k=2, bn=False, name='h2_conv'))
            h3 = lrelu(conv(h2, dim*8, k=2, bn=False, name='h3_conv'))
            h4 = tf.sigmoid(conv(h3, 1, k=1, bn=False, name='h4_conv'))
            return h4

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
            y_dc5 = deconv(lrelu(y_in5), [self.batch_size, self.w_range[1], self.w_range[1], dim * 1 * 2], name='dc5')

            y_in6 = tf.concat([y_dc5, y_c0],3)
            y_dc6 = tf.sigmoid(resize_conv(lrelu(y_in6), [self.batch_size, self.w_range[0], self.w_range[0], self.output_ch], name='dc6'))

            y = [input_, y_c1 ,y_c2, y_c4, y_in2, y_in4, y_in5, y_dc6]
            
            return dict(zip(self.image_keys, y))


    def train(self, train_epoch):

        self.get_nyu_dataset()

        g_train_op = tf.train.AdamOptimizer(self.g_lr, beta1=0., beta2=0.9).minimize(self.g_loss, var_list=self.g_vars)
        d_train_op = tf.train.AdamOptimizer(self.d_lr, beta1=0., beta2=0.9).minimize(self.d_loss, var_list=self.d_vars)

        tf.global_variables_initializer().run()
        #self.load_model("dganet_e00502.model")

        batch_size = self.batch_size
        step = 0
        train_loss = []
        g_train_loss = []
        d_train_loss = []
        
        # save input & target image
        in_img = self.sess.run(self.in_image[0], {self.x: self.test_image[:batch_size]})
        with open(self.outdata_dir+"/init"+"/input.png", 'wb') as f:
            f.write(in_img)
        tar_img = self.sess.run(self.tar_image[0], {self.y_: self.test_depth[:batch_size]})
        with open(self.outdata_dir+"/init"+"/target.png", 'wb') as f:
            f.write(tar_img)


        
        for epoch in range(train_epoch):
            
            eperm = np.random.permutation(len(self.test_image))
            test_feed = {self.x: self.test_image[eperm[:10]], self.y_: self.test_depth[eperm[:10]], self.test: 1}

            for i in tqdm(range(len(self.train_image)//batch_size)):
                if epoch < 100:
                    perm = [x for x in range(len(self.train_image))]
                else:
                    perm = np.random.permutation(len(self.train_image))
                
                batch = batch_size*i
                
                train_image = self.train_image[perm[batch:batch+batch_size]]
                train_depth = self.train_depth[perm[batch:batch+batch_size]]
                train_feed = {self.x: train_image, self.y_: train_depth, self.test: 0}
               
                for i in range(self.critic_k):
                    _, dl = self.sess.run([d_train_op, self.d_loss], feed_dict = train_feed)
                    self.sess.run(self.clip_updates)
                _, l, gl  = self.sess.run([g_train_op, self.loss, self.g_loss], feed_dict = train_feed)

                train_loss.append(l)
                g_train_loss.append(gl)
                d_train_loss.append(dl)

            if epoch % 1 == 0:
                # output results

                test_summary, vl = self.sess.run([self.merged, self.loss], feed_dict = test_feed)
                train_summary, cl = self.sess.run([self.merged, self.loss], feed_dict = train_feed)
                g_train_summary, gl = self.sess.run([self.merged, self.g_loss], feed_dict = train_feed)
                d_train_summary, dl = self.sess.run([self.merged, self.d_loss], feed_dict = train_feed)

                self.train_writer.add_summary(train_summary, epoch)
                self.test_writer.add_summary(test_summary, epoch)

                print("loss at epoch %s: %.10f" % (epoch, vl))
                print("avg_loss : %.10f" % np.array(train_loss).mean())
                print("g_loss : %.10f" % np.array(g_train_loss).mean())
                print("d_loss : %.10f" % np.array(d_train_loss).mean())
                train_loss = []
                g_train_loss = []
                d_train_loss = []

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
    
    #gradient penalty https://arxiv.org/pdf/1704.00028
    def gradient_penalty(self):
        eps = tf.random_uniform([self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        y_hat = eps * self.y_out + (1. - eps) * self.dist_y_tar
        d_y_hat = self.discriminator(self.dist_x, y_hat, reuse=True)
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
            img = cv2.resize(img, (self.read_image_h, self.read_image_w), interpolation = cv2.INTER_AREA)
            dep = cv2.resize(dep, (self.read_image_h, self.read_image_w), interpolation = cv2.INTER_AREA)
            img = img.flatten().astype(np.float32)
            dep = dep.flatten().astype(np.float32)
            img_list.append(img/np.amax(img))
            dep_list.append(dep/np.amax(dep))

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
    def generate_image_batch(self, images, targets, test_flag):
        img_list = [tf.squeeze(img, [0]) for img in tf.split(images, self.batch_size, 0)]
        tar_list = [tf.squeeze(tar, [0]) for tar in tf.split(targets, self.batch_size, 0)]
        images_ = []
        targets_ = []
        if test_flag == 0:
            for img, tar in zip(img_list, tar_list):
                img_tar_set = tf.concat([img, tar], 2)
                img_tar_set = tf.random_crop(img_tar_set, [self.image_h, self.image_w, 4])
                img_tar_set = tf.image.random_flip_left_right(img_tar_set)
                img, tar = tf.split(img_tar_set, [3, 1], 2)
                img = tf.image.random_brightness(img, max_delta=63/255.)
                img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
                #img = tf.image.per_image_standardization(img)

                img = tf.expand_dims(img, 0)
                tar = tf.expand_dims(tar, 0)
                images_.append(img)
                targets_.append(tar)
            return tf.concat(images_, 0), tf.concat(targets_, 0)
        else:
            return (tf.image.resize_images(images, [self.image_h, self.image_w]),
                    tf.image.resize_images(targets, [self.image_h, self.image_w]))


        
    def save_model(self, epoch):
        model_name = "uw-gan_e%05d.model" % epoch
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



