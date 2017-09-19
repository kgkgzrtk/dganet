import tensorflow as tf
from model import dganet

DATASET_DIR = "./dataset/voc"
CHECKPOINT_DIR = "./checkpoint"
OUTDATA_DIR = "./data"
SUMMARY_DIR = "./tmp/log"

def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        model = dganet(sess, dataset_dir=DATASET_DIR, checkpoint_dir=CHECKPOINT_DIR, outdata_dir=OUTDATA_DIR, summary_dir=SUMMARY_DIR)
        model.train(sess, train_epoch=3600)
        #model.draw_depth('./data/display/oyaken_chair.png')
        #save_path = saver.save(sess, "dganet_I-O_128.model")
        sess.close() 

if __name__ == '__main__':
    tf.app.run()
