import numpy as np
import tensorflow as tf
from layers.periodic_padding import periodic_padding
from layers.periodic_padding import calculate_required_padding


def Conv2D(name,
           input_dim,
           output_dim,
           filter_size,
           inputs,
           spatial_dim,
           data_format='NCHW',
           periodic=True,
           initialization='he',
           stride=1,
           biases=True,
           gain=1.,
           padding='VALID'):

    with tf.variable_scope(name) as scope:

        fan_in = input_dim * filter_size**2
        fan_out = output_dim * filter_size**2 / (stride**2)

        if initialization == 'he':
            filters_stdev = np.sqrt(4./(fan_in+fan_out))
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim))
        elif initialization == 'ones':
            filter_values = np.ones((filter_size, filter_size, input_dim, output_dim))
        else: # Normalized init (Glorot & Bengio)
            filters_stdev = np.sqrt(2./(fan_in+fan_out))
            filter_values = uniform(
                filters_stdev,
                (filter_size, filter_size, input_dim, output_dim))

        filter_values *= gain

        filters = tf.get_variable('Filters', initializer=filter_values.astype('float32'))

        # Define t he 4 element stride
        if data_format=='NCHW':
            strides = [1, 1, stride, stride]
        elif data_format=='NHWC':
            strides = [1, stride, stride, 1]
        else:
            raise KeyError(
                'data_format {0} is invalid. expected NCHW or NHWC'.format(data_format))

        if periodic:
            pad_left, pad_right, pad_top, pad_bottom = calculate_required_padding(filter_size, stride, spatial_dim)
            inputs = periodic_padding(inputs, pad_left, pad_right, pad_top, pad_bottom, spatial_dim, data_format=data_format)

        result = tf.nn.conv2d(
            input=inputs,
            filter=filters,
            strides=strides,
            padding=padding,
            data_format=data_format
        )

        if biases:
            _biases = tf.get_variable('b', initializer=np.zeros(output_dim, dtype='float32'))
            result = tf.nn.bias_add(result, _biases, data_format=data_format)

        return result


def uniform(stdev, size):
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype('float32')
