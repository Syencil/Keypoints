#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-29
"""
import tensorflow as tf
from tensorflow.python.platform import gfile


def readpb2graph(pb_path, log_dir):
    """
    transfer one pb file to visible graph in tensorboard
    You can build a model by tensorRT C++ API more easily!
    :param pb_path: pb_path
    :param log_dir: log_dir
    :return: None
    """
    with tf.Session() as sess:
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def)
        train_writer = tf.summary.FileWriter(log_dir)
        train_writer.add_graph(sess.graph)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    pb_path = '../Hourglass.pb'
