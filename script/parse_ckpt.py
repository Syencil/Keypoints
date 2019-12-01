#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-20
"""

import os
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np

# Read data from checkpoint file
# 检查模型变量的var和mean


def parse_ckpt(checkpoint_path):
    reader =pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    # Print tensor name and values
    key2val = {}
    keys = []
    for key in var_to_shape_map:

        if key.split('/')[-1] in ['weights', 'biases']:
            print("tensor_name: ", key)
            keys.append(key)
            val = reader.get_tensor(key)
            key2val[key] = np.array(val)
            print(np.sum(reader.get_tensor(key)))
            print(np.var(reader.get_tensor(key)))
    return keys, key2val


def read_origin(path):
    org_weights_mess = []
    load = tf.train.import_meta_graph(path + '.meta')
    with tf.Session() as sess:
        load.restore(sess, path)
        for var in tf.global_variables():
            var_name = var.op.name
            var_name_mess = str(var_name).split('/')
            var_shape = var.shape
            if (var_name_mess[-1] not in ['weights', 'gamma', 'beta', 'moving_mean', 'moving_variance']):
                continue
            org_weights_mess.append([var_name, var_shape])
            print("=> " + str(var_name).ljust(50), var_shape)

def transform(key, key2val):
    for k in key:
        print(k)
        try:
            name=k.replace('HourglassNet','model').replace('backbone','stacks').replace('hourglass','stage')
            print(name)

        except Exception:
            pass


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # ckpt_name = 'hg_refined_200.ckpt-0'
    # checkpoint_path = os.path.join('../checkpoints', 'pretrained', ckpt_name)
    # _, key2val = parse_ckpt(checkpoint_path)
    ckpt_name ='mpii/Hourglass_mpii.ckpt-39000'
    checkpoint_path = os.path.join('../checkpoints',  ckpt_name)
    key = parse_ckpt(checkpoint_path)
    # transform(key,key2val)
