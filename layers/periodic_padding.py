import tensorflow as tf
import numpy as np


def periodic_padding(inputs, pad_left, pad_right, pad_top, pad_bottom, spatial_dim, data_format='NCHW', method='slice'):

    if method == 'ein':
        padded_input = periodic_padding_ein(inputs, pad_left, pad_right, pad_top, pad_bottom, spatial_dim, data_format=data_format)
    elif method == 'slice':
        padded_input = periodic_padding_slice(inputs, pad_left, pad_right, pad_top, pad_bottom, spatial_dim, data_format=data_format)
    return padded_input


def periodic_padding_ein(inputs, pad_left, pad_right, pad_top, pad_bottom, spatial_dim, data_format='NCHW'):

    # Now that we have calculated the required padding, lets create the padded input.
    I = np.identity(spatial_dim)

    if data_format == 'NCHW':
        tb_pad = tf.constant(np.expand_dims(np.expand_dims(np.pad(I,((pad_top,pad_bottom),(0, 0)),'wrap'), 0), 0),
                             dtype='float32')
        rl_pad = tf.constant(
            np.expand_dims(np.expand_dims(np.pad(I,((0,0),(pad_left, pad_right)),'wrap'), 0), 0),
            dtype='float32')
        tb_padded_input = tf.einsum('cdij,abjk->abik', tb_pad, inputs)
        padded_input = tf.einsum('cdij,abjk->cdik', tb_padded_input, rl_pad)
    elif data_format == 'NHWC':
        tb_pad = tf.constant(np.expand_dims(np.expand_dims(np.pad(I, ((pad_top, pad_bottom), (0, 0)), 'wrap'), 0), 3),
                             dtype='float32')
        rl_pad = tf.constant(
            np.expand_dims(np.expand_dims(np.pad(I, ((0, 0), (pad_left, pad_right)), 'wrap'), 0), 3),
            dtype='float32')
        tb_padded_input = tf.einsum('cijd,ajkb->aikb', tb_pad, inputs)
        padded_input = tf.einsum('cijd,ajkb->cikd', tb_padded_input, rl_pad)

    # The result has the required periodic padding
    return padded_input


def periodic_padding_slice(inputs, pad_left, pad_right, pad_top, pad_bottom, spatial_dim, data_format='NCHW'):

    # We will organize all the data in NHWC and than revert it to it's original format
    if data_format == 'NHWC':
        # Convert input
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

    # Extract the input height and width
    in_shape = tf.shape(inputs)
    in_N = tf.to_int32(in_shape[0])
    in_depth = tf.to_int32(in_shape[1])
    in_height = tf.to_int32(in_shape[2])
    in_width = tf.to_int32(in_shape[3])

    # Now that we have calculated the required padding, lets create the padded input.
    # First, slice the top and bottom paddings from the input
    bottom_padding = tf.slice(inputs, [0, 0, 0, 0],[in_N, in_depth, pad_bottom, in_width])
    top_padding = tf.slice(inputs, [0, 0, in_height-pad_top, 0], [in_N, in_depth, pad_top, in_width])
    # pad top and bottom
    top_bottom_padded_input = tf.concat([top_padding, inputs,bottom_padding], 2)
    # find new height of image
    in_height = in_height + pad_top + pad_bottom

    # slice right and left paddings
    right_padding = tf.slice(top_bottom_padded_input, [0, 0, 0, 0], [in_N, in_depth, in_height, pad_right])
    left_padding = tf.slice(top_bottom_padded_input, [0, 0, 0, in_width-pad_left], [in_N, in_depth, in_height, pad_left])
    # final padded input
    inputs_pad = tf.concat([left_padding, top_bottom_padded_input, right_padding], 3)

    # transform the padded input into the original data format
    if data_format == 'NHWC':
        inputs_pad = tf.transpose(inputs_pad, [0, 2, 3, 1])

    # The result has the required periodic padding
    return inputs_pad


def calculate_required_padding(kernel_size, stride, spatial_dim):

    if kernel_size % 2 != 1:
        raise KeyError('Only kernels with odd dimension lengths are supported. '
                       'received kernel of dimensions {0}'.format(kernel_size))


    # The 2-d convolution starts from the first top-left pixel that is in the
    # valid zone.
    # So the padding needed from the left and above is k_width/2 and k_height/2 accordingly.
    pad_left = kernel_size // 2
    pad_top = kernel_size // 2

    # From the top-left pixel, the 2-d convolution will continue according to the strides.
    # the distance between the rightmost pixel we will reach with the strides
    # from the rightmost pixel will be d_right = (in_width-1) mod stride_width.
    # So the padding needed from the right is (k_width / 2) - d_right

    # Accordingly, the distance between the bottommost pixel we will reach with the
    # strides from the bottomost pixel will be d_bottom =  (in_height-1) mod stride_height.
    # So the padding needed from below is (k_height / 2) - d_bottom
    d_right = np.mod(spatial_dim - 1, stride)
    d_bottom = np.mod(spatial_dim - 1, stride)

    pad_right = kernel_size // 2 - d_right
    pad_bottom = kernel_size // 2 - d_bottom

    return pad_left, pad_right, pad_top, pad_bottom


def old_calculate_required_padding(kernel, strides, in_height, in_width):
    # Extract the kernel height and width
    k_height = kernel.shape[0]
    k_width = kernel.shape[1]
    if k_height % 2 != 1 or k_width % 2 != 1:
        raise KeyError('Only kernels with odd dimension lengths are supported. '
                       'received kernel of dimensions {0}'.format(kernel.shape))

    # Extract the strides in the height and width dimensions
    stride_height = strides[1]
    stride_width = strides[2]

    # The 2-d convolution starts from the first top-left pixel that is in the
    # valid zone.
    # So the padding needed from the left and above is k_width/2 and k_height/2 accordingly.
    pad_left = tf.div(k_width, 2)
    pad_top = tf.div(k_height, 2)

    # From the top-left pixel, the 2-d convolution will continue according to the strides.
    # the distance between the rightmost pixel we will reach with the strides
    # from the rightmost pixel will be d_right = (in_width-1) mod stride_width.
    # So the padding needed from the right is (k_width / 2) - d_right

    # Accordingly, the distance between the bottommost pixel we will reach with the
    # strides from the bottomost pixel will be d_bottom =  (in_height-1) mod stride_height.
    # So the padding needed from below is (k_height / 2) - d_bottom
    d_right = tf.mod(in_width - 1, stride_width)
    d_bottom = tf.mod(in_height - 1, stride_height)

    pad_right = tf.subtract(k_width // 2, d_right)
    pad_bottom = tf.subtract(k_height // 2, d_bottom)

    return pad_left, pad_right, pad_top, pad_bottom
