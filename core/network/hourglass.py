#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-10
"""
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
from core.network.network_utils import residual_block_v2, hourglass_block


class Hourglass():
    def __init__(self, inputs, num_class,
                 num_block=2,
                 num_depth=5,
                 residual_dim=(256, 256, 384, 384, 384, 512),
                 is_training=True,
                 is_maxpool=False,
                 is_nearest=True,
                 reuse=False
                 ):
        """
        Modified hourglass. See more in network_utils.py
        :param inputs: (Tensor) BxHxWxC images
        :param num_class: (int) num of classes
        :param num_block: (int) num of hourglass block
        :param num_depth: (int) num of down-sampling steps
        :param residual_dim: (list(int)) output dim for each residual block. Length should be num_depth+1
        :param is_training: (bool) is in training parse
        :param is_maxpool: (bool) if true, using max-pool down-sampling. Otherwise, residual block stride will be 2
        :param is_nearest: (bool) if true, using nearest up-sampling. Otherwise, using deconvolution
        :param reuse:(bool) reuse the variable
        """
        self.inputs = inputs
        self.num_class = num_class
        self.num_block = num_block
        self.num_depth = num_depth
        self.residual_dim = residual_dim
        self.is_training = is_training
        self.is_maxpool = is_maxpool
        self.is_nearest = is_nearest
        self.reuse = reuse

        self.features = self.graph_hourglass(self.inputs)

    def pre_process(self, inputs, scope='pre_process'):
        """
        pre-process conv7x7/s=2 -> residual/s=2
        :param inputs: (Tensor) BxHxWxC
        :param scope: (str) scope
        :return: (Tensor) BxH/4xW/4xC
        """
        with tf.variable_scope(scope):
            net = slim.conv2d(
                inputs=inputs,
                num_outputs=128,
                kernel_size=[7, 7],
                stride=2,
                activation_fn=None,
                normalizer_fn=None,
                reuse=self.reuse,
                scope='conv1'
            )
            tf.summary.histogram(net.name + '/activations', net)

            net = residual_block_v2(
                inputs=net,
                output_dim=256,
                stride=2,
                is_training=self.is_training,
                reuse=self.reuse,
                scope='residual_1'
            )
        return net

    def inter_process(self, inputs_1, inputs_2, scope='inter_process'):
        with tf.variable_scope(scope):
            branch_1 = slim.batch_norm(
                inputs=inputs_1,
                activation_fn=tf.nn.relu,
                is_training=self.is_training,
                scope='branch_1/bn',
                reuse=self.reuse,
                scale=True
            )
            tf.summary.histogram(branch_1.name + '/activations', branch_1)

            branch_1 = slim.conv2d(
                inputs=branch_1,
                num_outputs=inputs_1.get_shape().as_list()[-1],
                kernel_size=[1, 1],
                stride=1,
                activation_fn=None,
                normalizer_fn=None,
                reuse=self.reuse,
                scope='branch_1/conv'
            )
            tf.summary.histogram(branch_1.name + '/activations', branch_1)

            branch_2 = slim.batch_norm(
                inputs=inputs_2,
                activation_fn=tf.nn.relu,
                is_training=self.is_training,
                scope='branch_2/bn',
                reuse=self.reuse,
                scale=True)
            tf.summary.histogram(branch_2.name + '/activations', branch_2)

            branch_2 = slim.conv2d(
                inputs=branch_2,
                num_outputs=inputs_2.get_shape().as_list()[-1],
                kernel_size=[1, 1],
                stride=1,
                activation_fn=None,
                normalizer_fn=None,
                reuse=self.reuse,
                scope='branch_2/conv'
            )
            tf.summary.histogram(branch_2.name + '/activations', branch_2)

            output = tf.add(branch_1, branch_2)
        return output

    def hinge(self, inputs, output_dim, scope='hinge'):
        with tf.variable_scope(scope):
            pre = slim.batch_norm(
                inputs=inputs,
                activation_fn=tf.nn.relu,
                is_training=self.is_training,
                scope='bn',
                reuse=self.reuse,
                scale=True
            )
            tf.summary.histogram(pre.name + '/activations', pre)

            outputs = slim.conv2d(
                inputs=pre,
                num_outputs=output_dim,
                kernel_size=[1, 1],
                stride=1,
                activation_fn=None,
                normalizer_fn=None,
                reuse=self.reuse,
                scope='conv'
            )
            tf.summary.histogram(outputs.name + '/activations', outputs)
        return outputs

    def keypoint(self, features, scope='keypoint'):
        """
        key-point branch. return final feature map
        :param features: (Tensor) final backbone features without bn and activated
        :param scope: (str) scope
        :return: [Tensor,...]
        """
        keypoint_feature = []
        if type(features) is not list:
            features = [features]
        for i in range(len(features)):
            with tf.variable_scope(scope+'_%d' % i):
                feature = slim.batch_norm(inputs=features[i],
                                          activation_fn=tf.nn.relu,
                                          is_training=self.is_training,
                                          scope='pre_bn',
                                          reuse=self.reuse,
                                          scale=True)
                tf.summary.histogram(feature.name + '/activations', feature)
                feature = slim.conv2d(
                    inputs=feature,
                    num_outputs=self.num_class,
                    kernel_size=[3, 3],
                    stride=1,
                    activation_fn=tf.nn.sigmoid,
                    normalizer_fn=None,
                    reuse=self.reuse,
                    scope='conv'
                )
                tf.summary.histogram(feature.name + '/activations', feature)
                keypoint_feature.append(feature)

        return keypoint_feature

    def graph_backbone(self, inputs):
        """
        Extract features
        :param inputs: (Tensor) BxHxWxC images
        :return: [Tensor] BxH/4xW/4xC. Pre is for inter-mediate supervision, last if for prediction.
        """
        t0 = time.time()
        print('-Begin to creat model')
        with tf.variable_scope('backbone'):
            start_time = time.time()
            pre = self.pre_process(inputs)
            print('--%s has been created in %.3fs' %
                  ('pre_process', time.time() - start_time))
            net = pre
            features = []
            for i in range(self.num_block):
                start_time = time.time()
                hourglass = hourglass_block(
                    inputs=net,
                    num_depth=self.num_depth,
                    residual_dim=self.residual_dim,
                    is_training=self.is_training,
                    is_maxpool=self.is_maxpool,
                    is_nearest=self.is_nearest,
                    reuse=self.reuse,
                    scope='hourglass_%d' % i
                )
                hinge = self.hinge(hourglass, self.residual_dim[0], 'hinge_%d' % i)
                features.append(hinge)
                print('--%s has been created in %.3fs' % ('hourglass_%d' % i, time.time() - start_time))
                start_time = time.time()
                if i < self.num_block - 1: net = self.inter_process(net, hinge, 'inter_process_%d' % i)
                print('--%s has been created in %.3fs' % ('inter_process_%d' % i, time.time() - start_time))

        print('-Model has been created in %.3fs' % (time.time() - t0))
        return features

    def graph_hourglass(self, inputs, scope='HourglassNet'):
        """
        graph hourglass net.
        :param inputs: (Tensor) images
        :param scope: (str) scope
        :return: [[Tensor B x H/4 x W/4 x num_class,...]]
        """
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.initializers.variance_scaling(scale=1, mode='fan_avg')):
                features = self.graph_backbone(inputs)
                all_features = [self.keypoint(features)]
        print('--PB file input node is %s' % inputs.name)
        print('--PB file output node is %s' % all_features[0][-1].name)
        return all_features

