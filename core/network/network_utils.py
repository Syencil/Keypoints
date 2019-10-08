#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-10
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim


def residual_block_v2(inputs, output_dim, stride,
                      is_training=True, scope='residual_block'):
    """
    Pre-act mode
    modified residual block
    bottle neck depth = output_dim / 2
    output = conv + short-cut
    :param inputs: (Tensor) input tensor BxHxWxC
    :param output_dim: (int) multiple of 2
    :param stride: (int) if down-sample
    :param scope: (str) scope name
    :param is_training: (bool)bn is in training phase
    :return: (Tensor) Bx(H/stride)x(W/stride)xC
    """
    dim = output_dim / 2
    if output_dim % 2 != 0:
        raise ValueError('residual block output dim must be a multiple of 2')
    with tf.variable_scope(scope):
        depth_in = inputs.get_shape().as_list()[-1]
        pre_act = slim.batch_norm(
            inputs=inputs,
            activation_fn=tf.nn.relu,
            is_training=is_training,
            scope='pre_act',
            scale=True
        )
        if output_dim == depth_in:
            short_cut = slim.max_pool2d(
                inputs=inputs,
                kernel_size=[1, 1],
                stride=stride,
                scope='short_cut'
            )
        else:
            short_cut = slim.conv2d(
                inputs=pre_act,
                num_outputs=output_dim,
                kernel_size=[1, 1],
                stride=stride,
                activation_fn=None,
                normalizer_fn=None,
                scope='short_cut'
            )
        tf.summary.histogram(short_cut.name + '/activations', short_cut)

        residual = slim.conv2d(
            inputs=pre_act,
            num_outputs=dim,
            kernel_size=[1, 1],
            stride=1,
            activation_fn=None,
            normalizer_fn=None,
            scope='conv1'
        )
        residual = slim.batch_norm(
            residual,
            activation_fn=tf.nn.relu,
            is_training=is_training,
            scope='conv1/bn',
            scale=True)
        tf.summary.histogram(residual.name + '/activations', residual)

        residual = slim.conv2d(
            inputs=residual,
            num_outputs=dim,
            kernel_size=[3, 3],
            stride=stride,
            activation_fn=None,
            normalizer_fn=None,
            scope='conv2'
        )
        residual = slim.batch_norm(
            residual,
            activation_fn=tf.nn.relu,
            is_training=is_training,
            scope='conv2/bn',
            scale=True)
        tf.summary.histogram(residual.name + '/activations', residual)

        residual = slim.conv2d(
            inputs=residual,
            num_outputs=output_dim,
            kernel_size=[1, 1],
            stride=1,
            activation_fn=None,
            normalizer_fn=None,
            scope='conv3'
        )
        tf.summary.histogram(residual.name + '/activations', residual)

        output = short_cut + residual
    return output


def hourglass_block(inputs, num_depth, residual_dim,
                    is_training=True, is_maxpool=False,
                    is_nearest=True, scope='hourglass_block'):
    """
    modified hourglass block fellow by "CornerNet"
    There 2 residual blocks in short-cut istead of 1
    There 2 residual blocks after upsampling
    There 4 residual blocks with depth dim (512 in paper) in the middle of hourglass
    Attention! residual blocks are in pre-act mode
    inputs must be not processed by actived or normlized
    :param inputs: (Tensor) BxHxWxC
    :param num_depth: (int) depth of downsample
    :param residual_dim: (list) dim of residual block. len(residual_dim)=num_depth+1
    :param is_training: (bool) bn is in training phase
    :param is_maxpool: (bool) if it's True, downsample mode will be maxpool. Otherwise, downsample mode will be stride=2
    :param is_nearest: (bool) if it's True, upsample mode will be neareast upsample. Otherwise, upsample mode will be deconv.
    :param scope: (str) scope name
    :return: (Tensor) BxHxWxC
    """
    cur_res_dim = inputs.get_shape().as_list()[-1]
    next_res_dim = residual_dim[0]

    with tf.variable_scope(scope):
        up_1 = residual_block_v2(
            inputs=inputs,
            output_dim=cur_res_dim,
            stride=1,
            is_training=is_training,
            scope='up_1'
        )
        if is_maxpool:
            low_1 = slim.max_pool2d(
                inputs=inputs,
                kernel_size=2,
                stride=2,
                padding='VALID'
            )
            low_1 = residual_block_v2(
                inputs=low_1,
                output_dim=next_res_dim,
                stride=1,
                is_training=is_training,
                scope='low_1'
            )
        else:
            low_1 = residual_block_v2(
                inputs=inputs,
                output_dim=next_res_dim,
                stride=2,
                is_training=is_training,
                scope='low_1'
            )

        if num_depth > 1:
            low_2 = hourglass_block(
                inputs=low_1,
                num_depth=num_depth - 1,
                residual_dim=residual_dim[1:],
                is_training=is_training,
                is_maxpool=is_maxpool,
                is_nearest=is_nearest,
                scope='hourglass_block_%d' % (num_depth - 1)
            )
        else:
            low_2 = residual_block_v2(
                inputs=low_1,
                output_dim=next_res_dim,
                stride=1,
                is_training=is_training,
                scope='low_2'
            )
        low_3 = residual_block_v2(
            inputs=low_2,
            output_dim=cur_res_dim,
            stride=1,
            is_training=is_training,
            scope='low_3'
        )
        if is_nearest:
            up_2 = tf.image.resize_nearest_neighbor(
                images=low_3,
                size=tf.shape(low_3)[1:3] * 2,
                name='up_2'
            )
        else:
            up_2 = slim.conv2d_transpose(
                inputs=low_3,
                num_outputs=cur_res_dim,
                kernel_size=[3, 3],
                stride=2,
                scope='up_2'
            )
        merge = up_1 + up_2
    return merge

