import tensorflow as tf
from model import dganet

DATASET_PATH = "./dataset/nyu_depth_v2_labeled.mat"
CHECKPOINT_DIR = "./checkpoint"
OUTDATA_DIR = "./data"
SUMMARY_DIR = "./tmp/log"

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = dganet(sess, dataset_path=DATASET_PATH, checkpoint_dir=CHECKPOINT_DIR, outdata_dir=OUTDATA_DIR, summary_dir=SUMMARY_DIR)
        model.train(train_epoch=1800)
        #model.draw_depth('./data/display/oyaken_chair.png')
        #save_path = saver.save(sess, "dganet_I-O_128.model")
        sess.close() 

if __name__ == '__main__':
    tf.app.run()
