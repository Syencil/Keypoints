#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-27
"""

import tensorflow as tf
import tensorflow.contrib.tensorrt as trt

def tfpb2trtpb(pb_path, output_pb, output_node_name):
    # Inference with TF-TRT frozen graph workflow:
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # First deserialize your frozen graph:
            with tf.gfile.GFile(pb_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            # Now you can create a TensorRT inference graph from your
            # frozen graph:
            trt_graph = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=output_node_name,
                max_batch_size=1,
                max_workspace_size_bytes=2 << 20,
                precision_mode='FLOAT32')

            with tf.gfile.GFile(output_pb, "wb") as f:  # 保存模型
                f.write(trt_graph.SerializeToString())
            # Import the TensorRT graph into a new graph and run:
            # output_node = tf.import_graph_def(
            #     trt_graph,
            #     return_elements=[“your_outputs”])
            # sess.run(output_node)

pb_path = '../Hourglass.pb'
output_node_name=['HourglassNet/keypoint_1/conv/BiasAdd']