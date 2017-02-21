# -*- coding: utf-8 -*-
import tensorflow as tf
from model import dganet

DATASET_DIR = "./dataset/voc"
CHECKPOINT_DIR = "./checkpoint"
OUTDATA_DIR = "./data"
SUMMARY_DIR = "./tmp/log"

def main(_):
    with tf.Session() as sess:
        model = dganet(sess, dataset_dir=DATASET_DIR, checkpoint_dir=CHECKPOINT_DIR, outdata_dir=OUTDATA_DIR)
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph_def)

        model.train(sess, train_epoch=5000)
        save_path = saver.save(sess, "dganet_I-O_128.model")
        sess.close() 

if __name__ == '__main__':
    tf.app.run()
