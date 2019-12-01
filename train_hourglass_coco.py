#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-12
"""
from core.train.trainer import Trainer
from core.network.keypoints import Keypoints
from core.dataset.data_generator import Dataset
import config.config_hourglass_coco as cfg
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.append('.')


class TrainHourglass(Trainer):
    def __init__(self, model, dataset, cfg):
        super(TrainHourglass, self).__init__(model, dataset, cfg)

    def init_model(self):
        # BN decay 0.9
        with slim.arg_scope([slim.batch_norm], decay=0.96):
            Trainer.init_model(self)

    def init_train_op(self):
        start_time = time.time()
        # TRAIN_OP
        with tf.name_scope("Train_op"):
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate)
            # optimizer = tf.train.MomentumOptimizer(
            #     self.learning_rate, 0.9)
            gvs = optimizer.compute_gradients(self.loss)
            clip_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
                        for grad, var in gvs]
            if self.is_debug:
                self.mean_gradient = tf.reduce_mean(
                    [tf.reduce_mean(g) for g, v in gvs])
                tf.summary.scalar("mean_gradient", self.mean_gradient)
                print('Debug mode is on !!!')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # It's important!
            # Update moving-average in BN
            self.train_op = optimizer.apply_gradients(
                clip_gvs, global_step=self.global_step)
        print('-Creat train op in %.3f' % (time.time() - start_time))

    def init_loader_saver(self):
        start_time = time.time()
        with tf.name_scope('loader_and_saver'):
            if self.pre_trained_ckpt is not None:
                from tensorflow.python import pywrap_tensorflow
                reader = pywrap_tensorflow.NewCheckpointReader(self.pre_trained_ckpt)
                var_to_shape_map = reader.get_variable_to_shape_map()
                var_to_restore = [k for k in var_to_shape_map]
                # var_ = [var for var in tf.global_variables() if var.name.strip(':0') in var_to_restore and var.name.strip(':0')!="Learning_rate/global_step" and "Momentum" not in var.name.strip(':0')]
                var_ = [var for var in tf.global_variables() if var.name.strip(':0') in var_to_restore and var.name.strip(':0')!="Learning_rate/global_step"]
                print('restore var total is %d' % len(var_))
                self.loader = tf.train.Saver(var_list=var_)
            self.saver = tf.train.Saver(
                var_list=tf.global_variables(),
                max_to_keep=self.max_keep)
        print(
            '-Creat loader saver in %.3f' %
            (time.time() - start_time))


if __name__ == '__main__':
    trainer = TrainHourglass(Keypoints, Dataset, cfg)
    trainer.train_launch()
