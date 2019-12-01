#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-11-29
"""

import tensorflow as tf
from core.network.keypoints import Keypoints
from tensorflow.python import pywrap_tensorflow
import config.config_hourglass_coco as cfg
import tensorflow.contrib.slim as slim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

ori_ckpt = '/data/checkpoints/pre_train/resnet_v2_101.ckpt'
new_ckpt = os.path.join(cfg.ckpt_dir, "Keypoints_coco_resnet_v2_101.ckpt")


def change_name(name):
    # 自己的网络名称有前缀
    return name[19:]

def restore_name(name):
    return "Keypoints/backbone/" + name

def map_var(name_list):
    # {pretrain文件中的名字 ：自己模型中的tensor}
    var_list = {}
    for name in name_list:
        new_name = restore_name(name)
        var_list[name] = slim.get_variables_by_name(new_name)[0]
    return var_list

# origin
reader = pywrap_tensorflow.NewCheckpointReader(ori_ckpt)
var_ori = reader.get_variable_to_shape_map()
# network
inputs = tf.placeholder(tf.float32, [1, 512, 512, 3])
centernet = Keypoints(inputs, 80,
                      num_block=cfg.num_block,
                      backbone="resnet_v2_101",
                      num_depth=cfg.num_depth,
                      residual_dim=cfg.residual_dim,
                      is_training=True,
                      is_maxpool=cfg.is_maxpool,
                      is_nearest=cfg.is_nearest,
                      reuse=False
                      )
var_new = slim.get_variables_to_restore()

# search common
count = 0
ommit = 0
all_var = set()
restore_list = []
for key in var_new:
    # 命名改变了 改成了"CenterNet/作为前缀, 需要去掉"
    all_var.add(change_name(key.name.strip(':0')))
for key in var_ori:
    if key in all_var:
        ori_var = reader.get_tensor(key)
        new_var = slim.get_variables_by_name(restore_name(key))[0]
        s1 = list(ori_var.shape)
        s2 = new_var.get_shape().as_list()
        if s1 == s2:
            count += 1
            restore_list.append(key)
        else:
            ommit += 1
    else:
        ommit += 1
print('restore ', count)
print('ommit', ommit)
print('all', count + ommit)
var_list = map_var(restore_list)
# loader = tf.train.Saver(
#     var_list=slim.get_variables_to_restore(
#         include=restore_list,
#         exclude=['logits']))
loader = tf.train.Saver(
    var_list=var_list)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    loader.restore(sess, ori_ckpt)
    saver.save(sess, new_ckpt)
