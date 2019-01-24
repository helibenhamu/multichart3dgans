import numpy as np
import tensorflow as tf
from layers.periodic_padding import periodic_padding
from layers.periodic_conv2d import Conv2D

def GenConvBlock(name,
           input_dim,
           output_dim,
           filter_size,
           inputs,
           spatial_dim,
           upsample=True,
           data_format='NCHW',
           method='slice',
           initialization='he',
           stride=1,
           weightnorm=None,
           biases=True,
           gain=1.,
           padding='VALID'):
    output = inputs
    if upsample:
        if data_format == 'NCHW':
            output = tf.transpose(output, [0, 2, 3, 1])
        output = periodic_padding(output, 0, 1, 0, 1, spatial_dim, data_format='NHWC', method=method)

        output = tf.image.resize_images(output, [spatial_dim * 2 + 1, spatial_dim * 2 + 1], align_corners=True)
        output = output[:, :spatial_dim*2, :spatial_dim*2, :]

        if data_format == 'NCHW':
            output = tf.transpose(output, [0, 3, 1, 2])
        spatial_dim = spatial_dim*2

    output = Conv2D('{0}_1'.format(name), input_dim, output_dim, filter_size, output, spatial_dim)
    output = tf.nn.relu(output)
    output = Conv2D('{0}_2'.format(name), output_dim, output_dim, filter_size, output, spatial_dim)
    output = tf.nn.relu(output)

    return output, output_dim, spatial_dim


def DiscConvBlock(name,
           input_dim,
           output_dim,
           filter_size,
           inputs,
           spatial_dim,
           downsample=True,
           data_format='NCHW',
           method='slice',
           initialization='he',
           stride=1,
           biases=True,
           gain=1.,
           padding='VALID'):

    output = Conv2D('{0}_1'.format(name), input_dim, input_dim, filter_size, inputs, spatial_dim)
    output = tf.nn.leaky_relu(output)
    output = Conv2D('{0}_2'.format(name), input_dim, output_dim, filter_size, output, spatial_dim)
    output = tf.nn.leaky_relu(output)

    if downsample:
        if data_format == 'NCHW':
            output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.nn.avg_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',
                                data_format='NHWC')
        if data_format == 'NCHW':
            output = tf.transpose(output, [0, 3, 1, 2])

    return output, output_dim, spatial_dim//2


