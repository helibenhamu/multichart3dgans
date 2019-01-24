import numpy as np
import tensorflow as tf


def Linear(
        name,
        input_dim,
        output_dim,
        inputs,
        biases=True,
        gain=1.):

    with tf.variable_scope(name) as scope:

        weight_values = uniform(np.sqrt(2. / (input_dim + output_dim)),(input_dim, output_dim))
        weight_values *= gain
        weight = tf.get_variable('W', initializer=weight_values.astype('float32'))

        if inputs.get_shape().ndims == 2:
            result = tf.matmul(inputs, weight)
        else:
            reshaped_inputs = tf.reshape(inputs, [-1, input_dim])
            result = tf.matmul(reshaped_inputs, weight)
            result = tf.reshape(result, tf.stack(tf.unstack(tf.shape(inputs))[:-1] + [output_dim]))

        if biases:
            result = tf.nn.bias_add(
                result,
                tf.get_variable('b', initializer=np.zeros(output_dim, dtype='float32'))
            )

        return result


def uniform(stdev, size):
    return np.random.uniform(
        low=-stdev * np.sqrt(3),
        high=stdev * np.sqrt(3),
        size=size
    ).astype('float32')
