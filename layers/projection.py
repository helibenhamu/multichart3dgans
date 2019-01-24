import tensorflow as tf
import numpy as np
from layers.periodic_padding import periodic_padding


def toric_symmetry(inputs, spatial_dim, data_format='NCHW', pooling='max', method='slice'):

    # We will work with NHWC data format (image rotation needs it)
    if data_format == 'NCHW':
        inputs = tf.transpose(inputs, [0, 2, 3, 1])

    # Extract the input height and width
    in_shape = tf.shape(inputs)
    in_N = tf.to_int32(in_shape[0])
    in_height = tf.to_int32(in_shape[1])
    in_width = tf.to_int32(in_shape[2])
    in_depth = tf.to_int32(in_shape[3])

    inputs_pad = periodic_padding(inputs, 0, 1, 0, 1, spatial_dim, data_format='NHWC', method=method)

    # rotate the inputs
    inputs_90 = rot90(inputs_pad)
    inputs_180 = rot90(inputs_90)
    inputs_270 = rot90(inputs_180)

    # Stack the rotations and max over them
    rotations_stacked = tf.stack([inputs_pad, inputs_90, inputs_180, inputs_270])
    if pooling == 'max':
        result = tf.reduce_max(rotations_stacked, axis=0)
    elif pooling == 'avg':
        result = tf.reduce_mean(rotations_stacked, axis=0)
    result = tf.slice(result, [0, 0, 0, 0], [in_N, in_height, in_width, in_depth])

    mask = np.zeros([1, spatial_dim, spatial_dim, 1], dtype='float32')
    mask[:, 0, :, :] = 1
    mask[:, :, 0, :] = 1
    n_mask = abs(mask - 1)

    tf_mask = tf.constant(mask)
    tf_n_mask = tf.constant(n_mask)
    # apply symmetry to boundary
    boundary = tf.multiply(result, tf_mask)
    inner_part = tf.multiply(result, tf_n_mask)
    if pooling == 'max':
        projected_boundary = tf.reduce_max(tf.stack([boundary, tf.transpose(boundary,[0, 2, 1, 3])]), axis=0)
    elif pooling == 'avg':
        projected_boundary = tf.reduce_mean(tf.stack([boundary, tf.transpose(boundary,[0, 2, 1, 3])]), axis=0)

    result = projected_boundary + inner_part

    # if needed, convert back to NCHW format
    if data_format == 'NCHW':
        result = tf.transpose(result, [0, 3, 1, 2])

    return result


def rot90(inputs):
    inputs_rot90 = tf.map_fn(lambda Input: tf.image.rot90(Input), inputs)
    return inputs_rot90


def solve_ST(model, inputs, gamma=None, normalize_cones=False):
    '''

    :param model:
    :param inputs: in HWCN data format
    :return:
    '''

    # Extract the input height and width
    in_shape = tf.shape(inputs)
    in_height = tf.to_int32(in_shape[0])
    in_width = tf.to_int32(in_shape[1])
    in_depth = tf.to_int32(in_shape[2])
    in_N = tf.to_int32(in_shape[3])
    center = in_height//2

    # gather and scatter operations work on first dimensions, thus the transfer of the batch dimension to the end
    orig_cones = tf.gather_nd(inputs, [[center, center], [0, 0], [0, center]])  # 3 x in_depth x in_N
    if normalize_cones:
        epsilon = 1e-5
        T_cones = tf.reshape(orig_cones, [3, in_depth//3, 3, in_N])              # 3 x triplets x 3(xyz) x in_N
        cones_mean, _ = tf.nn.moments(T_cones, axes=0, keep_dims=True)           # 1 x triplets x 3(xyz) x in_N
        _, cones_variance = tf.nn.moments(T_cones, axes=[0,2], keep_dims=True)   # 1 x triplets x 1      x in_N
        cones = tf.reshape(tf.divide(tf.subtract(T_cones, cones_mean), epsilon+tf.sqrt(cones_variance)), [3, in_depth, in_N])
    else:
        cones = orig_cones

    updates_p = tf.gather_nd(cones, model.gather_indices_p)                # num_of_equations x in_N
    updates_m = tf.gather_nd(cones, model.gather_indices_m)                # num_of_equations x in_N

    if gamma is not None:
        fixed_chart = tf.random_uniform([1], minval=0, maxval=model.config.depth_dim//3, dtype=tf.int32)[0]
        num_of_equations = tf.to_int32(np.shape(model.triplets_mat_reg)[0])
        num_of_params = tf.to_int32(np.shape(model.triplets_mat_reg)[1])
        triplets_mat = tf.multiply(model.triplets_mat_reg, tf.concat(
            [tf.ones([num_of_equations - num_of_params // 4, 1]), tf.sqrt(model.config.gamma) * tf.ones([num_of_params // 4, 1])],
            axis=0))
        fix_chart_equations = tf.scatter_nd(
            [[0, fixed_chart * 4], [1, fixed_chart * 4 + 1], [2, fixed_chart * 4 + 2], [3, fixed_chart * 4 + 3]],
            tf.ones(4, 1), [4, num_of_params])
        triplets_mat = tf.concat([fix_chart_equations, triplets_mat[4:, :]], axis=0)
        STD = tf.reshape(model.STD, [1, num_of_params // 4, 1])
        STD = tf.divide(STD, tf.gather_nd(STD, [[[0, fixed_chart, 0]]]))
        b = tf.tile(tf.concat(
            [tf.ones([1, 1, 1]), tf.zeros([1, num_of_equations - 1 - num_of_params // 4, 1]), tf.sqrt(model.config.gamma) * STD], 1),
                    [in_N, 1, 1])
    else:
        fixed_chart = 0
        num_of_equations = tf.to_int32(np.shape(model.triplets_mat)[0])
        num_of_params = tf.to_int32(np.shape(model.triplets_mat)[1])
        triplets_mat = model.triplets_mat
        b = tf.tile(tf.concat([tf.ones([1, 1, 1]), tf.zeros([1, num_of_equations - 1, 1])], 1), [in_N, 1, 1])

    triplets_mat = tf.expand_dims(triplets_mat, 2)
    batched_triplets_mat = tf.add(triplets_mat, tf.scatter_nd(model.scatter_indices_p, updates_p, [num_of_equations, num_of_params, in_N]))
    batched_triplets_mat = tf.subtract(batched_triplets_mat, tf.scatter_nd(model.scatter_indices_m, updates_m, tf.shape(batched_triplets_mat)))
    batched_triplets_mat = tf.transpose(batched_triplets_mat, [2, 0, 1])  # in_N x num_of_equations x num_of_params

    # solve LS
    x = tf.matrix_solve_ls(batched_triplets_mat, b, l2_regularizer=model.config.ls_reg)  # in_N x num_of_params x 1
    x = tf.reshape(tf.transpose(x, [2, 1, 0]), [1, num_of_params//4, 4, in_N])  # 1 x triplets x 4 x in_N

    # extract scale and translation
    if normalize_cones:
        S_temp = tf.tile(
            tf.divide(tf.slice(x, [0, 0, 0, 0], [1, num_of_params // 4, 1, in_N]), epsilon + tf.sqrt(cones_variance)),
            [1, 1, 3, 1])
        S_fixed = tf.gather_nd(S_temp, [[[0, fixed_chart, 0]]])  # 1 x 1 x in_N
        S = tf.reshape(S_temp, [1, in_depth, in_N])  # 1 x in_depth x in_N
        S = tf.divide(S, S_fixed)
        T = tf.reshape(tf.slice(x, [0, 0, 1, 0], [1, num_of_params // 4, 3, in_N]) - tf.multiply(S_temp, cones_mean),
                       [1, 3 * num_of_params // 4, in_N])  # 1 x in_depth x in_N
        T = tf.divide(T, S_fixed)
    else:
        S = tf.reshape(tf.tile(tf.slice(x,[0, 0, 0, 0], [1, num_of_params//4, 1, in_N]), [1, 1, 3, 1]), [1, in_depth, in_N]) # 1 x in_depth x in_N
        T = tf.reshape(tf.slice(x,[0, 0, 1, 0], [1, num_of_params//4, 3, in_N]), [1, 3*num_of_params//4, in_N]) # 1 x in_depth x in_N

    return S, T, orig_cones


def align_ST(model, inputs, data_format='NCHW'):

    if data_format == 'NCHW':
        T_inputs = tf.transpose(inputs, [2, 3, 1, 0])  # HWCN
    elif data_format == 'NHWC':
        T_inputs = tf.transpose(inputs, [1, 2, 3, 0])  # HWCN

    S, T, _ = solve_ST(model, T_inputs)

    S = tf.expand_dims(S, 0)
    T = tf.expand_dims(T, 0)

    results = tf.add(tf.multiply(T_inputs, S), T)  # H x W x in_depth x in_N

    if data_format == 'NCHW':
        results = tf.transpose(results, [3, 2, 0, 1])
    elif data_format == 'NHWC':
        results = tf.transpose(results, [3, 0, 1, 2])

    return results


def project_ST(model, inputs, gamma=None, data_format='NCHW'):

    if data_format == 'NCHW':
        T_inputs = tf.transpose(inputs, [2, 3, 1, 0])  # HWCN
    elif data_format == 'NHWC':
        T_inputs = tf.transpose(inputs, [1, 2, 3, 0])  # HWCN

    # Extract the input height and width
    in_shape = tf.shape(T_inputs)
    in_height = tf.to_int32(in_shape[0])
    in_width = tf.to_int32(in_shape[1])
    in_depth = tf.to_int32(in_shape[2])
    in_N = tf.to_int32(in_shape[3])
    center = in_height//2
    num_of_charts = tf.constant(np.shape(model.triplets_mat)[1]//4, dtype='int32')

    S, T, cones = solve_ST(model, T_inputs, gamma)
    ST_cones = tf.add(tf.multiply(cones, S), T)  # 3 x in_depth x in_N
    # reshape to: 1 x 3(cones) x num_of_charts x 3(xyz) x in_N
    ST_cones_reshaped = tf.reshape(ST_cones, [1, 3, num_of_charts, 3, -1])
    # triples_masks: num_ucones x 3(cones) x num_of_charts x 1 x 1
    triplets_masks = tf.constant(np.expand_dims(np.expand_dims(model.triplets_masks,3),4), dtype='float32')

    friends_mat = tf.multiply(ST_cones_reshaped, triplets_masks)
    mean_masks = tf.divide(tf.reduce_sum(friends_mat, [1,2], keepdims=True),  # num_ucones x 1 x 1 x 3(xyz) x in_N
                           tf.reduce_sum(triplets_masks, [1,2], keepdims=True)  # num_ucones x 1 x 1 x 1 x 1
                           )  # num_ucones x 1 x 1 x 3(xyz) x in_N
    projected_cones = tf.reshape(tf.reduce_sum(tf.multiply(triplets_masks, mean_masks), 0), tf.shape(ST_cones))  # 3 x in_depth x in_N

    # reverse alignment
    reduced_cones = tf.subtract(tf.divide(tf.subtract(projected_cones, T), S), cones)  # 3 x in_depth x in_N
    reduced_cones = tf.concat([reduced_cones, tf.slice(reduced_cones,[2, 0, 0],[1, in_depth, in_N ])],0)
    new_cones = tf.scatter_nd([[center, center], [0, 0], [0, center], [center, 0]],reduced_cones, tf.shape(T_inputs))

    results = tf.add(T_inputs, new_cones)

    if data_format == 'NCHW':
        results = tf.transpose(results, [3, 2, 0, 1])
    elif data_format == 'NHWC':
        results = tf.transpose(results, [3, 0, 1, 2])

    return results





