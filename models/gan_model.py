import os
import numpy as np
import scipy.io as sio
import tensorflow as tf
from models.base_model import BaseModel
import layers.conv_block
from layers.periodic_conv2d import Conv2D
from layers.projection import toric_symmetry
from layers.projection import project_ST
from layers.projection import align_ST
import layers.linear


class GANModel(BaseModel):
    def __init__(self, data, config):
        super(GANModel, self).__init__(data, config)
        self.get_alignment_params()     # load dataset parameters
        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.device('/cpu:0'):

            # define optimizers for generator and discriminator
            self.gen_opt = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2, name="gen_opt")
            self.disc_opt = tf.train.AdamOptimizer(learning_rate=self.config.lr, beta1=self.config.beta1, beta2=self.config.beta2, name="disc_opt")

            # data and noise
            real_charts = tf.reshape(self.dataLoader.get_next, [self.config.batch_size, self.config.depth_dim, self.config.spatial_dim, self.config.spatial_dim])
            # post process raw batch
            self.real_charts = self.process_batch(real_charts)

            # split to towers
            batches = tf.split(self.real_charts, self.config.n_gpus)
            self.noise = tf.random_normal([tf.shape(self.real_charts)[0], self.config.latent_vec_dim])
            noises = tf.split(self.noise, self.config.n_gpus)

            # define generator and discriminator on CPU
            self.generator(self.noise)
            self.discriminator(self.real_charts)

            # build towers
            self.tower_gen_grads = []
            self.tower_disc_grads =[]
            self.disc_cost = []
            self.gen_cost = []
            self.fake_charts = []
            for device_index, (device, batch, noise) in enumerate(zip(self.config.devices, batches, noises)):
                with tf.device(device):
                    with tf.name_scope('tower.{0}'.format(device_index)) as scope:

                        disc_cost, gen_cost, fake_charts = self.build_tower(noise, batch)
                        self.disc_cost.append(disc_cost)
                        self.gen_cost.append(gen_cost)
                        self.fake_charts.append(fake_charts)
                        if device_index == 0:
                            t_vars = tf.trainable_variables()
                            d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
                            g_vars = [var for var in t_vars if var.name.startswith('generator')]

                        # Calculate the gradients for the batch of data on this tower.
                        grads_gen  = self.gen_opt.compute_gradients(gen_cost, g_vars)
                        grads_disc = self.disc_opt.compute_gradients(disc_cost, d_vars)

                        # Keep track of the gradients across all towers.
                        self.tower_gen_grads.append(grads_gen)
                        self.tower_disc_grads.append(grads_disc)

            self.disc_cost = tf.add_n(self.disc_cost)/self.config.n_gpus
            self.gen_cost = tf.add_n(self.gen_cost)/self.config.n_gpus

            self.grads_gen = self.average_gradients(self.tower_gen_grads)
            self.grads_disc = self.average_gradients(self.tower_disc_grads)

            self.gen_train_op = self.gen_opt.apply_gradients(self.grads_gen)
            self.disc_train_op = self.disc_opt.apply_gradients(self.grads_disc, global_step=self.global_step_tensor)

    def process_batch(self, real_charts):
        '''
        Post-process of charts given a chosen experiment setting
        :param real_charts: real charts from database
        :return: processed_real_charts: processed charts based on the choice of experiment
        '''

        if self.config.model == 'single_chart':
            # train generator to generate a single chart for each noise vector
            real_charts = tf.reshape(real_charts, [self.config.batch_size, self.config.depth_dim//3, 3, self.config.spatial_dim, self.config.spatial_dim])
            N_indices = tf.constant(np.arange(self.config.batch_size), dtype=tf.int32)
            C_indices = tf.random_uniform([self.config.batch_size], 0, self.config.depth_dim//3, dtype=tf.int32)
            indices = tf.stack([N_indices, C_indices], axis=1)
            real_charts = tf.gather_nd(real_charts, indices)
            self.config.depth_dim = 3
        elif self.config.model == 'aligned':
            # train on aligned charts (i.e. not normalized)
            real_charts = align_ST(self, real_charts)

        if self.config.normalize_charts:
            # normalize charts
            real_charts = tf.subtract(real_charts, tf.reduce_mean(real_charts, axis=[2, 3], keepdims=True))
            real_charts = tf.reshape(real_charts, [self.config.batch_size, self.config.depth_dim // 3, -1])
            _, data_variance = tf.nn.moments(real_charts, axes=[2], keep_dims=True)
            processed_real_charts = tf.reshape(tf.divide(real_charts, tf.sqrt(data_variance)),
                                          [self.config.batch_size, self.config.depth_dim, self.config.spatial_dim,
                                           self.config.spatial_dim])
        else:
            processed_real_charts = tf.subtract(real_charts, tf.reduce_mean(real_charts, axis=[1, 2, 3], keepdims=True))

        return processed_real_charts

    def build_tower(self, noise, real_charts):

        '''
        a tower works on a split of the data according to the number of GPUS. Each GPU defines a tower.
        :param noise: input to generator
        :param real_charts: input to discriminator
        :return: disc_cost, gen_cost, fake_charts:
        '''

        fake_charts = self.generator(noise)
        disc_fake   = self.discriminator(fake_charts)
        disc_real   = self.discriminator(real_charts)

        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Gradient penalty
        alpha = tf.random_uniform(
            shape=[tf.shape(real_charts)[0], 1, 1, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_charts - real_charts
        interpolates = real_charts + (alpha * differences)
        disc_interpolates = self.discriminator(interpolates)
        gradients = tf.reshape(tf.gradients(disc_interpolates, [interpolates])[0], [tf.shape(interpolates)[0], -1])
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        # TODO: reconsider LAMBDA name in config
        disc_cost += self.config.LAMBDA * gradient_penalty

        return disc_cost, gen_cost, fake_charts

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def generator(self, noise):

        if self.config.model == 'single_chart':
            return self.generator_single_chart(noise)
        else:
            with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
                depths = self.config.depths
                spatial = self.config.spatial
                output = layers.linear.Linear('input_FC', self.config.latent_vec_dim, (spatial[0]**2)*depths[0], noise)
                output = tf.nn.relu(output)
                output = tf.reshape(output, [-1, depths[0], spatial[0], spatial[0]])

                # convolution layers
                output = Conv2D('conv_in_{0}-{1}_{2}'.format(depths[0], depths[0], spatial[0]),
                                depths[0], depths[0], 3, output, spatial[0]) # N x dim x 4 x 4
                output = tf.nn.relu(output)
                for ind, depth in enumerate(depths[:-1]):
                    output, _, _ = layers.conv_block.GenConvBlock('conv_{3}_{0}-{1}_{2}'.format(depth, depths[ind+1], spatial[ind], ind),
                                                                  depth, depths[ind+1], 3, output, spatial[ind])
                output = Conv2D('conv_out', depths[ind+1], self.config.depth_dim, 1, output, self.config.spatial_dim)

                # enforce toric symmetry
                output = toric_symmetry(output, self.config.spatial_dim)

                # landmark consistency layer
                def project_ST_op():
                    if self.config.gamma_decay is not None:
                        gamma = tf.multiply(tf.pow(self.config.gamma_decay,tf.cast(tf.subtract(self.cur_epoch_tensor,self.config.kick_projection),dtype='float32')),self.config.gamma)
                    else:
                        gamma = self.config.gamma
                    op1 = project_ST(self, output, gamma)
                    with tf.control_dependencies([op1]):
                        return tf.identity(op1)

                # the landmark consistency layer kicks in from a certain given epoch
                output = tf.cond(tf.greater(self.cur_epoch_tensor, self.config.kick_projection),
                                 project_ST_op,
                                 lambda: output)

                # reduce mean
                if self.config.normalize_charts:
                    data_mean, _ = tf.nn.moments(output, axes=[2, 3], keep_dims=True)
                    output = tf.subtract(output, data_mean)

            return output

    def generator_single_chart(self, noise):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
            depths = self.config.depths
            spatial = self.config.spatial
            output = layers.linear.Linear('input_FC', self.config.latent_vec_dim, (spatial[0]**2)*depths[0], noise)
            output = tf.nn.relu(output)
            output = tf.reshape(output, [-1, depths[0], spatial[0], spatial[0]])

            # convolution layers
            output = Conv2D('conv_in_{0}-{1}_{2}'.format(depths[0], depths[0], spatial[0]),
                            depths[0], depths[0], 3, output, spatial[0]) # N x dim x 4 x 4
            output = tf.nn.relu(output)
            for ind, depth in enumerate(depths[:-1]):
                output, _, _ = layers.conv_block.GenConvBlock('conv__{3}_{0}-{1}_{2}'.format(depth, depths[ind+1], spatial[ind], ind),
                                                              depth, depths[ind+1], 3, output, spatial[ind])
            output = Conv2D('conv_out', depths[ind+1], self.config.depth_dim, 1, output, self.config.spatial_dim)

            # enforce toric symmetry
            output = toric_symmetry(output, self.config.spatial_dim)

            # reduce mean
            if self.config.normalize_charts:
                data_mean, _ = tf.nn.moments(output, axes=[2, 3], keep_dims=True)
                output = tf.subtract(output, data_mean)

        return output

    def discriminator(self, inputs):

        if self.config.model == "single_chart":
            return self.discriminator_single_chart(inputs)
        else:
            with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
                depths = self.config.depths[::-1]
                spatial = self.config.spatial[::-1]
                # convolutional network discriminator
                output = Conv2D('conv_in', self.config.depth_dim, depths[0], 1, inputs, self.config.spatial_dim)
                output = tf.nn.leaky_relu(output)
                for ind, depth in enumerate(depths[:-1]):
                    output, _, _ = layers.conv_block.DiscConvBlock(
                        'conv_{3}_{0}-{1}_{2}'.format(depth, depths[ind + 1], spatial[ind], ind),
                        depth, depths[ind + 1], 3, output, spatial[ind])

                output = Conv2D('conv_{0}-{1}_{2}_1'.format(depths[ind + 1], depths[ind + 1], spatial[ind+1]),
                                depths[ind + 1], depths[ind + 1], 3, output, spatial[ind+1])
                output = tf.nn.leaky_relu(output)
                output = tf.reshape(Conv2D('conv_{0}-{1}_{2}_2'.format(depths[ind + 1], depths[ind + 1], spatial[ind+1]),
                                           depths[ind + 1], depths[ind + 1], 4, output, spatial[ind+1], periodic=False), [-1, depths[ind + 1]])
                output = tf.nn.leaky_relu(output)
                total_loss = tf.reshape(layers.linear.Linear('output_FC', depths[ind + 1], 1, output), [-1])

            return total_loss

    def discriminator_single_chart(self, inputs):

        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
            depths = self.config.depths[::-1]
            spatial = self.config.spatial[::-1]
            # convolutional network discriminator
            output = Conv2D('conv_in', self.config.depth_dim, depths[0], 1, inputs, self.config.spatial_dim)
            output = tf.nn.leaky_relu(output)
            for ind, depth in enumerate(depths[:-1]):
                output, _, _ = layers.conv_block.DiscConvBlock(
                    'conv_{3}_{0}-{1}_{2}'.format(depth, depths[ind + 1], spatial[ind], ind),
                    depth, depths[ind + 1], 3, output, spatial[ind])

            output = Conv2D('conv_{0}-{1}_{2}_1'.format(depths[ind + 1], depths[ind + 1], spatial[ind + 1]),
                            depths[ind + 1], depths[ind + 1], 3, output, spatial[ind + 1])
            output = tf.nn.leaky_relu(output)
            output = tf.reshape(
                Conv2D('conv_{0}-{1}_{2}_2'.format(depths[ind + 1], depths[ind + 1], spatial[ind + 1]),
                       depths[ind + 1], depths[ind + 1], 4, output, spatial[ind + 1], periodic=False),
                [-1, depths[ind + 1]])
            output = tf.nn.leaky_relu(output)
            total_loss = tf.reshape(layers.linear.Linear('output_FC', depths[ind + 1], 1, output), [-1])

        return total_loss

    def get_alignment_params(self):

        # load matlab files
        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'triplets_mat.mat'))
        self.triplets_mat = tf.cast(mat_content['triplets_mat'].toarray(), tf.float32)

        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'triplets_mat_reg.mat'))
        self.triplets_mat_reg = tf.cast(mat_content['triplets_mat'].toarray(), tf.float32)

        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'data_std.mat'))
        self.STD = mat_content['STD']

        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'gather_indices_p.mat'))
        self.gather_indices_p = mat_content['gather_indices_p']

        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'gather_indices_m.mat'))
        self.gather_indices_m = mat_content['gather_indices_m']

        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'scatter_indices_p.mat'))
        self.scatter_indices_p = mat_content['scatter_indices_p']

        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'scatter_indices_m.mat'))
        self.scatter_indices_m = mat_content['scatter_indices_m']

        mat_content = sio.loadmat(os.path.join(self.config.database_dir, 'triplets_masks.mat'))
        self.triplets_masks = mat_content['triplets_masks']

    def init_saver(self):
        # initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
