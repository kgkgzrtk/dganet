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
        model.train(train_epoch=5000)
        #model.load_model('uw-gan-gp_e01302.model')
        #model.gen_depth('./data/display/01depth')
        save_path = saver.save(sess, "view.model")
        sess.close() 

if __name__ == '__main__':
    tf.app.run()
