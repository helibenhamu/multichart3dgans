import tensorflow as tf
import os


class BaseModel:
    def __init__(self, data, config):
        self.config = config
        self.dataLoader = data
        # init the global step
        self.init_global_step()
        # init the epoch counter
        self.init_cur_epoch()

    # saves the checkpoint in the path defined in the config file
    def save(self, sess):
        print("Saving model...")
        cur_epoch = sess.run(self.cur_epoch_tensor)
        self.saver.save(sess, os.path.join(self.config.checkpoint_dir, "epoch_" + str(cur_epoch)), self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        if self.config.init_path is not None:
            checkpoint_dir = self.config.init_path
        else:
            checkpoint_dir = self.config.checkpoint_dir
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
            self.increment_cur_epoch_tensor

    # initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

