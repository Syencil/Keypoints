#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-24
"""
import uff
from tensorflow.contrib.tensorrt import trt_convert as trt
# from tensorflow.python.compiler.tensorrt import trt_convert as trt
# TODO tf==1.12.0 只支持trt4
pb_path = '../Hourglass.pb'
output_nodes = ["HourglassNet/keypoint_1/conv/BiasAdd"]
output_filename = 'Hourglass.uff'

serialized=uff.from_tensorflow_frozen_model(pb_path, output_nodes, output_filename=output_filename)
# print(serialized)

convert = trt.TrtGraphConverter(
    input_graph_def=pb_path,
    nodes_blacklist=output_nodes
)
frozen_graph = convert.convert()