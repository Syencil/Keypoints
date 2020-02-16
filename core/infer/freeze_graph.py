#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-24
"""

import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util
# from tensorflow.contrib.tensorrt import trt_convert as trt
import os
import sys

sys.path.append('.')


def parse_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '-C',
        '--CUDA',
        dest='CUDA',
        default=None,
        help='CUDA_VISIBLE_DEVICE')
    parse.add_argument(
        '-c',
        '--ckpt',
        dest='ckpt',
        default=None,
        help='Freeze ckpt path')
    parse.add_argument(
        '-o',
        '--output',
        dest='output_graph',
        default=None,
        help='Output graph path')
    parse.add_argument(
        '-t',
        '--is_training',
        dest='output_graph',
        default=False,
        help='Output graph path')
    return parse.parse_args()


def freeze_graph(input_checkpoint, output_graph, is_training=False):
    '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :param is_training: Is BN using moving-mean and moving-var
    :return:
    '''
    # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
    # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

    if is_training:
        saver = tf.train.import_meta_graph(
            input_checkpoint + '.meta', clear_devices=True)
    else:
        from core.network.keypoints import Keypoints
        import config.config_hourglass_coco as config
        model = Keypoints(tf.placeholder(name="Placeholder/inputs_x", dtype=tf.float32, shape=[None, 512, 512, 3]),
                          17,
                          num_block=config.num_block,
                          num_depth=config.num_depth,
                          residual_dim=config.residual_dim,
                          is_training=False,
                          is_maxpool=config.is_maxpool,
                          is_nearest=config.is_nearest
                          )
        saver = tf.train.Saver(var_list=tf.global_variables())

    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    print('Freeze graph')
    output_node_names = ["Keypoints/keypoint_1/conv/Sigmoid"]
    print(output_node_names)

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
            sess=sess,
            input_graph_def=sess.graph_def,  # 等于:sess.graph_def
            output_node_names=output_node_names)

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." %
              len(output_graph_def.node))  # 得到当前图有几个操作节点


if __name__ == '__main__':
    args = parse_arg()
    if args.CUDA is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA
    freeze_graph(args.ckpt, args.output_graph)
