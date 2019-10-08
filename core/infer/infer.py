#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-21
"""
import os
import time
import numpy as np
import tensorflow as tf
from core.loss.loss import focal_loss


class Infer():
    def __init__(self, model_class, dataset_class, cfg):
        start_time = time.time()
        # HARDWARE
        self.CUDA_VISIBLE_DEVICES_INFER = cfg.CUDA_VISIBLE_DEVICES_INFER
        if self.CUDA_VISIBLE_DEVICES_INFER is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA_VISIBLE_DEVICES_INFER
        self.MULTI_THREAD_NUM = cfg.MULTI_THREAD_NUM
        # self.MULTI_GPU = cfg.MULTI_GPU
        # self.NUM_GPU = len(self.MULTI_GPU)

        # NETWORK
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.heatmap_size
        self.stride = cfg.stride
        self.num_block = cfg.num_block
        self.num_depth = cfg.num_depth
        self.residual_dim = cfg.residual_dim
        self.is_maxpool = cfg.is_maxpool
        self.is_nearest = cfg.is_nearest

        # PATH
        self.dataset_dir = cfg.dataset_dir
        self.train_image_dir = cfg.train_image_dir
        self.val_image_dir = cfg.val_image_dir
        self.train_list_path = cfg.train_list_path
        self.val_list_path = cfg.val_list_path

        # INPUTS
        self.batch_size = cfg.batch_size
        self.inputs_x = None
        self.inputs_y = None
        self.is_training = None

        # DATASET
        self.dataset_class = dataset_class
        self.val_dataset = None
        self.val_iterator = None

        # MODEL
        self.model_class = model_class
        self.model = None
        self.features = None
        self.loss = None

        # SAVER LOADER SUMMARY
        self.loader = None

        # SESSION
        self.sess = None

    def init_inputs(self):
        with tf.variable_scope('Placeholder'):
            self.inputs_x = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3],
                                           'inputs_x')
            self.inputs_y = tf.placeholder(tf.float32, [None, self.heatmap_size[0], self.heatmap_size[0],
                                                        self.val_dataset.num_class], 'inputs_y')
            self.is_training = tf.placeholder(tf.bool, name='is_training')

    def init_dataset(self):
        start_time = time.time()

        # VAL DATASET
        self.val_dataset = self.dataset_class(image_dir=self.val_image_dir,
                                              gt_path=self.val_list_path,
                                              batch_size=self.batch_size,
                                              image_size=self.image_size,
                                              heatmap_size=self.heatmap_size
                                              )
        self.val_iterator = self.val_dataset.iterator_make_one_shot(self.MULTI_THREAD_NUM)
        print('-Creat dataset in %.3f' % (time.time() - start_time))

    def init_model(self):
        self.model = self.model_class(self.inputs_x, self.val_dataset.num_class,
                                      num_block=self.num_block,
                                      num_depth=self.num_depth,
                                      residual_dim=self.residual_dim,
                                      is_training=self.is_training,
                                      is_maxpool=self.is_maxpool,
                                      is_nearest=self.is_nearest
                                      )
        self.features = self.model.features[0][-1]

    def init_loss(self):
        start_time = time.time()

        # LOSS
        with tf.variable_scope('Loss'):
            self.loss = focal_loss(self.features, self.inputs_y)
        print('-Creat loss in %.3f' % (time.time() - start_time))

    def init_session(self, ckpt):
        start_time = time.time()
        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)  # 是否自动选择GPU
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        print('-Initializing session in %.3f' % (time.time() - start_time))
        self._load_ckpt(ckpt)

    def _load_ckpt(self, ckpt):
        t0 = time.time()
        self.loader = tf.train.Saver(var_list=tf.global_variables())
        try:
            self.loader.restore(self.sess, ckpt)
            print('Successful restore from %s in time %.2f' %
                  (ckpt, time.time() - t0))
        except Exception as e:
            raise ValueError('Failed restore from %s in time %.2f' % (ckpt, time.time() - t0))

    def validation(self):
        t0 = time.time()
        losses = 0
        features = []
        print("Begin Validation !!!")
        try:
            count = 0
            while True:
                count += 1
                imgs, hms = next(self.val_iterator)
                imgs = (imgs / 127.5) - 2
                feed_dict = {
                    self.inputs_x: imgs,
                    self.inputs_y: hms,
                    self.is_training: False}

                loss, feature = self.sess.run([self.loss, self.features], feed_dict)
                losses += loss[0] * len(imgs)
                print('Batch %d loss is %.3f' % (count, loss[0]))
                features.append(feature)
        except StopIteration:
            print('Validation Done')
        finally:
            mean_loss = losses / self.val_dataset.num_data
            print('mean_loss is %.3f in time %.3f' % (mean_loss, time.time() - t0))
        self.sess.close()
        features = np.row_stack(features)
        np.save('val_features.npy', features)
        print('Done')
        return features


    def infer_launch(self, ckpt):
        self.init_dataset()
        self.init_inputs()
        self.init_model()
        self.init_loss()
        self.init_session(ckpt)
        self.validation()




