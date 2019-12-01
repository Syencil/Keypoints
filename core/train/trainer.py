#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-20
"""
import os
import time

import tensorflow as tf
import tensorflow.contrib.slim as slim
from core.loss.loss import focal_loss, cross_entropy, softmax_cross_entropy, mean_square_loss


class Trainer():
    def __init__(self, model_class, dataset_class, cfg):

        start_time = time.time()
        # HARDWARE
        self.CUDA_VISIBLE_DEVICES = cfg.CUDA_VISIBLE_DEVICES
        if self.CUDA_VISIBLE_DEVICES is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.CUDA_VISIBLE_DEVICES
        self.MULTI_THREAD_NUM = cfg.MULTI_THREAD_NUM
        # self.MULTI_GPU = cfg.MULTI_GPU
        # self.NUM_GPU = len(self.MULTI_GPU)

        # NETWORK
        self.backbone = cfg.backbone
        self.image_size = cfg.image_size
        self.heatmap_size = cfg.heatmap_size
        self.stride = cfg.stride
        self.num_block = cfg.num_block
        self.num_depth = cfg.num_depth
        self.residual_dim = cfg.residual_dim
        self.is_maxpool = cfg.is_maxpool
        self.is_nearest = cfg.is_nearest

        # TRAINING
        self.batch_size = cfg.batch_size
        self.learning_rate_init = cfg.learning_rate_init
        self.learning_rate_warmup = cfg.learning_rate_warmup
        self.exp_decay = cfg.exp_decay

        self.warmup_epoch_size = cfg.warmup_epoch_size
        self.epoch_size = cfg.epoch_size
        self.summary_per = cfg.summary_per
        self.save_per = cfg.save_per

        self.regularization_weight = cfg.regularization_weight

        # VALIDATION
        self.val_per = cfg.val_per
        self.val_time = cfg.val_time

        # PATH
        self.dataset_dir = cfg.dataset_dir
        self.train_image_dir = cfg.train_image_dir
        self.val_image_dir = cfg.val_image_dir
        self.train_list_path = cfg.train_list_path
        self.val_list_path = cfg.val_list_path

        self.log_dir = cfg.log_dir
        self.ckpt_path = cfg.ckpt_dir

        # SAVER AND LOADER
        self.pre_trained_ckpt = cfg.pre_trained_ckpt
        self.ckpt_name = cfg.ckpt_name
        self.max_keep = cfg.max_keep

        print('-Load config in %.3f' % (time.time() - start_time))

        # DATASET
        self.dataset_class = dataset_class
        self.train_dataset = None
        self.val_dataset = None
        self.train_iterator = None
        self.val_iterator = None

        # cal option
        self.time = time.strftime(
            '%Y_%m_%d_%H_%M_%S',
            time.localtime(
                time.time()))
        self.steps_per_period = None

        # PLACE HOLDER
        self.inputs_x = None
        self.inputs_y = None
        self.is_training = None

        # MODEL
        self.model_class = model_class
        self.model = None
        self.features = None

        self.val_model = None
        self.val_features = None

        # LOSS
        self.loss_mode = cfg.loss_mode
        self.model_losses = None
        self.model_loss = None
        self.val_model_loss = None
        self.trainable_variables = None
        self.regularization_loss = None
        self.loss = None

        # LEARNING RATE
        self.global_step = None
        self.learning_rate = None

        # TRAIN OP
        self.train_op = None

        # SAVER LOADER SUMMARY
        self.loader = None
        self.saver = None
        self.summary_writer = None
        self.write_op = None

        # DEBUG
        self.is_debug = False
        self.gradient = None
        self.mean_gradient = None

        # SESSION
        self.sess = None
        #################################################################

    def init_inputs(self):
        with tf.variable_scope('Placeholder'):
            self.inputs_x = tf.placeholder(tf.float32, [None, self.image_size[0], self.image_size[1], 3],
                                           'inputs_x')
            self.inputs_y = tf.placeholder(tf.float32, [None, self.heatmap_size[0], self.heatmap_size[0],
                                                        self.train_dataset.num_class], 'inputs_y')
            # 如果使用placeholder为BN层的trainable参数,BN层中会处于一种使用tf.cond,tf.switch流控制节点(此处可以在tensorRT以及模型图中得到验证)
            # 这样的话每一个BN层都会有两条路径出来，训练太占显存，infer部署的时候还要单独进行剪枝
            # 此处直接设置为True的话，训练是没问题的。做val的时候，不调用train_op那么BN的gamma和beta不会更新
            # 并且由于mean和var设置为依赖于train_op更新，所以BN在val时所有参数都没有更新，相当于trainable=False
            # 然而在tf1.x版本中，trainable=False是让BN处于freeze状态。
            # 和infer不同的时，freeze仍然是使用当前batch的mean和var进行处理。
            # 在tf2.x版本中，bn已经改成了当trainable为False的时候是infer状态
            self.is_training = True

    def init_dataset(self):
        start_time = time.time()

        # TRAIN DATASET
        self.train_dataset = self.dataset_class(image_dir=self.train_image_dir,
                                                gt_path=self.train_list_path,
                                                batch_size=self.batch_size,
                                                image_size=self.image_size,
                                                heatmap_size=self.heatmap_size)
        self.train_iterator = self.train_dataset.iterator(
            self.MULTI_THREAD_NUM)

        # VAL DATASET
        self.val_dataset = self.dataset_class(image_dir=self.val_image_dir,
                                              gt_path=self.val_list_path,
                                              batch_size=self.batch_size,
                                              image_size=self.image_size,
                                              heatmap_size=self.heatmap_size
                                              )
        self.val_iterator = self.val_dataset.iterator(self.MULTI_THREAD_NUM)
        self.steps_per_period = int(
            self.train_dataset.num_data /
            self.batch_size)
        print('-Creat dataset in %.3f' % (time.time() - start_time))

    def init_model(self):
        print("-Creat Train model")
        self.model = self.model_class(self.inputs_x, self.train_dataset.num_class,
                                      backbone=self.backbone,
                                      num_block=self.num_block,
                                      num_depth=self.num_depth,
                                      residual_dim=self.residual_dim,
                                      is_training=True,
                                      is_maxpool=self.is_maxpool,
                                      is_nearest=self.is_nearest,
                                      reuse=False
                                      )
        self.features = self.model.features[0]

        print("-Creat Val model")
        self.val_model = self.model_class(self.inputs_x, self.train_dataset.num_class,
                                          backbone=self.backbone,
                                          num_block=self.num_block,
                                          num_depth=self.num_depth,
                                          residual_dim=self.residual_dim,
                                          is_training=False,
                                          is_maxpool=self.is_maxpool,
                                          is_nearest=self.is_nearest,
                                          reuse=True
                                          )
        self.val_features = self.val_model.features[0]

    def init_learning_rate(self):
        start_time = time.time()
        # LEARNING RATE
        with tf.variable_scope('Learning_rate'):
            self.global_step = tf.train.get_or_create_global_step()
            warmup_steps = tf.constant(self.warmup_epoch_size * self.steps_per_period,
                                       dtype=tf.int64, name='warmup_steps')
            self.learning_rate = tf.cond(
                pred=tf.less(self.global_step, warmup_steps),
                true_fn=lambda: self.learning_rate_warmup + (self.learning_rate_init - self.learning_rate_warmup)
                                * tf.cast(self.global_step, tf.float32) / tf.cast(warmup_steps, tf.float32),
                false_fn=lambda: tf.train.exponential_decay(
                    self.learning_rate_init, self.global_step, self.steps_per_period, self.exp_decay, staircase=True)
            )
        print('-Creat learning rate in %.3f' % (time.time() - start_time))

    def init_loss(self):
        start_time = time.time()

        # LOSS
        with tf.variable_scope('Loss'):
            self.trainable_variables = tf.trainable_variables()
            if self.loss_mode == 'focal':
                loss_fn = focal_loss
            elif self.loss_mode == 'sigmoid':
                loss_fn = cross_entropy
            elif self.loss_mode == 'softmax':
                loss_fn = softmax_cross_entropy
            elif self.loss_mode == 'mse':
                loss_fn = mean_square_loss
            else:
                raise ValueError('Unsupported loss mode: %s' % self.loss_mode)
            self.model_losses = loss_fn(self.features, self.inputs_y)
            self.model_loss = tf.add_n(self.model_losses)
            self.val_model_loss = loss_fn(self.val_features, self.inputs_y)[-1]
            self.regularization_loss = tf.add_n(
                [tf.nn.l2_loss(var) for var in self.trainable_variables])
            self.regularization_loss = self.regularization_weight * self.regularization_loss
            self.loss = self.model_loss + self.regularization_loss

        print('-Creat loss in %.3f' % (time.time() - start_time))

    def init_train_op(self):
        start_time = time.time()
        # TRAIN_OP
        with tf.name_scope("Train_op"):
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            clip_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
            if self.is_debug:
                self.mean_gradient = tf.reduce_mean([tf.reduce_mean(g) for g, v in gvs])
                tf.summary.scalar("mean_gradient", self.mean_gradient)
                print('Debug mode is on !!!')
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # It's important!
            # Update moving-average in BN
            self.train_op = optimizer.apply_gradients(clip_gvs, global_step=self.global_step)
        print('-Creat train op in %.3f' % (time.time() - start_time))

    def init_loader_saver_summary(self):
        start_time = time.time()
        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(var_list=tf.global_variables())
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            bn_moving_var = [g for g in g_list if 'moving_mean' in g.name]
            bn_moving_var += [g for g in g_list if 'moving_variance' in g.name]
            if len(bn_moving_var) < 1:
                print('Warning! BatchNorm layer parameters have not been saved!')
            var_list += bn_moving_var
            self.saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=self.max_keep)

        with tf.name_scope('summary'):

            tf.summary.image('input_image', self.inputs_x, max_outputs=3)
            tf.summary.image('input_hm', tf.reduce_sum(self.inputs_y,axis=-1,keepdims=True), max_outputs=3)
            tf.summary.image('output_hm', tf.reduce_sum(self.features[-1],axis=-1,keepdims=True), max_outputs=3)

            tf.summary.scalar("learning_rate", self.learning_rate)
            for i in range(len(self.model_losses)):
                tf.summary.scalar("block_%d_loss" % i, self.model_losses[i])
            tf.summary.scalar("model_loss", self.model_loss)
            tf.summary.scalar("regularization_loss", self.regularization_loss)
            tf.summary.scalar("total_loss", self.loss)
            # # Optional
            # tf.summary.scalar('keypoint_bn_moving_mean',
            #                   tf.reduce_mean(slim.get_variables_by_name('HourglassNet/keypoint_1/pre_bn/moving_mean')))
            # tf.summary.scalar('keypoint_bn_moving_var', tf.reduce_mean(
            #     slim.get_variables_by_name('HourglassNet/keypoint_1/pre_bn/moving_variance')))

            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)
            self.write_op = tf.summary.merge_all()

        print(
            '-Creat loader saver and summary in %.3f' %
            (time.time() - start_time))

    def init_session(self):
        start_time = time.time()
        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)  # 是否自动选择GPU
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.summary_writer = tf.summary.FileWriter(
            os.path.join(self.log_dir, self.time), graph=self.sess.graph)
        print('-Initializing session in %.3f' % (time.time() - start_time))

        # self.train_launch()
        ################################################################
    def _load_ckpt(self):
        t0 = time.time()
        try:
            self.loader.restore(self.sess, self.pre_trained_ckpt)
            print('Successful restore from %s in time %.2f' %
                  (self.pre_trained_ckpt, time.time() - t0))
        except Exception as e:
            print(e)
            print('Failed restore from %s in time %.2f' %
                  (self.pre_trained_ckpt, time.time() - t0))

    def train(self):
        t0 = time.time()
        self.sess.run(tf.global_variables_initializer())
        print('-Model has beed initialized in %.3f' % (time.time() - t0))
        if self.pre_trained_ckpt is not None:
            self._load_ckpt()

        print('Begin to train!')
        total_step = self.epoch_size * self.steps_per_period
        step = 0
        while step < total_step:
            # try:
            step = self.sess.run(self.global_step)
            ite = step % self.steps_per_period + 1
            epoch = step // self.steps_per_period + 1
            imgs, hms = next(self.train_iterator)
            imgs = (imgs / 127.5) - 1
            feed_dict = {
                self.inputs_x: imgs,
                self.inputs_y: hms,
                }

            if step % self.summary_per == 0:
                if self.is_debug:
                    mean_gradient = self.sess.run(self.mean_gradient, feed_dict=feed_dict)
                    print('mean_gradient: %.6f ' % mean_gradient)
                summary, _, lr, loss, model_ls, reg_ls = self.sess.run(
                    [self.write_op, self.train_op, self.learning_rate, self.loss, self.model_loss, self.regularization_loss], feed_dict=feed_dict)
                print(
                    'Epoch: %d / %d Iter: %d / %d Step: %d Loss: %.4f Model Loss: %.4f Reg Loss: %.4f Lr: %f' %
                    (epoch, self.epoch_size, ite, self.steps_per_period, step, loss, model_ls, reg_ls, lr))
                self.summary_writer.add_summary(summary, step)
            else:
                _, lr, loss, model_ls, reg_ls = self.sess.run(
                    [self.train_op, self.learning_rate, self.loss, self.model_loss, self.regularization_loss], feed_dict=feed_dict)

            if step % self.save_per == 0:
                self.saver.save(
                    self.sess,
                    os.path.join(
                        self.ckpt_path,
                        self.ckpt_name),
                    global_step=step)
            if step % self.val_per == 0 and step != 0:
                # Validation
                losses = []
                start_time = time.time()
                for s in range(self.val_time):
                    # TODO 计算loss 不更新梯度 保存每一次loss 最后打印平均loss
                    # TODO 保存几个图片输出的结果 可以用cv2.circle渲染 cv2.imwrite 存在本地
                    imgs_v, hms_v = next(self.val_iterator)
                    imgs_v = (imgs_v / 127.5) - 1
                    feed_dict = {
                        self.inputs_x: imgs_v,
                        self.inputs_y: hms_v,
                        }
                    loss = self.sess.run(self.val_model_loss, feed_dict=feed_dict)
                    losses.append(loss)
                print('Validation %d times in %.3fs mean loss is %f'
                      % (self.val_time, time.time() - start_time, sum(losses) / len(losses)))
            # except Exception as e:
            #     print(e)
        self.saver.save(
            self.sess,
            os.path.join(
                self.ckpt_path,
                self.ckpt_name),
            global_step=step)
        self.summary_writer.close()
        self.sess.close()

    def train_launch(self):
        # must in order
        self.init_dataset()
        self.init_inputs()
        self.init_model()

        # optional override
        self.init_loss()
        self.init_learning_rate()
        self.init_train_op()
        self.init_loader_saver_summary()
        self.init_session()
        self.train()
