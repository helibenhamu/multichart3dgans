from trainers.base_trainer import BaseTrainer
from tqdm import tqdm
import numpy as np
import os
import scipy.io as sio
import random
import tensorflow as tf


class GANTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger):
        super(GANTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))

        # filenames of tfRecords
        src_dir = os.path.join(self.config.database_dir, "train")
        training_filenames = [os.path.join(src_dir, tfr_file) for tfr_file in os.listdir(src_dir)]
        random.shuffle(training_filenames)
        self.sess.run(self.data.iterator.initializer, feed_dict={self.data.filenames: training_filenames})

        for _ in loop:
            fake_charts_np, disc_cost_np, gen_cost_np = self.train_step()

        print('epoch ' + str(self.sess.run(self.model.cur_epoch_tensor)) + " - disc cost:" + str(disc_cost_np) + ", gen cost:" + str(gen_cost_np))

        if self.sess.run(self.model.cur_epoch_tensor) % self.config.save_period == 0:
            noise = self.sess.run(self.model.noise)
            c = self.sess.run(self.model.generator(tf.convert_to_tensor(noise)))
            save_dir = os.path.join(self.config.results_dir,str(self.sess.run(self.model.cur_epoch_tensor))+'_epoch')
            sio.savemat(save_dir , {'c': c})

        if self.sess.run(self.model.cur_epoch_tensor) % self.config.checkpoint_period == 0:
            self.model.save(self.sess)

    def train_step(self):

        for i in range(self.config.critic_iters):
            _, _disc_cost, _real_charts = self.sess.run([self.model.disc_train_op, self.model.disc_cost, self.model.real_charts])

        _, _gen_cost, _fake_charts = self.sess.run([self.model.gen_train_op, self.model.gen_cost, self.model.fake_charts])

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'disc_cost': _disc_cost,
            'gen_cost': _gen_cost
              }

        if cur_it % 10 == 0:
            self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        return _fake_charts, _disc_cost, _gen_cost

