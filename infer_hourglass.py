#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-21
"""
from core.infer.infer_utils import read_pb, pred_one_image
from core.infer.visual_utils import draw_point, draw_bbx, draw_skeleton
# image = cv2.imread(img_path)
# # 1.实例化模型
# sess, input_tensor, output_tensor = \
#     read_pb(pb_path, ['Placeholder/inputs_x:0'], ['HourglassNet/keypoint_1/conv/Sigmoid:0'])
# # 2.处理图片 每次处理一个图里面的数据作为batch
# #   bbxes 是提前知道的信息 bbxes = [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]
# points = pred_one_image(image, bbxes, sess, input_tensor, output_tensor)
# print(points)
# for point in points:
#     image = draw_point(image, point)