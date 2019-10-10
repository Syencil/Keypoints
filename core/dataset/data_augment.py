#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-10-08
"""
# import copy
import numpy as np
# import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage


def image_augment_with_keypoints(image, keypoints, color_jitter=0.5, crop=(
        0, 0.5), rotate=(1, -45, 45), flip=(1, 1)):

    points = np.copy(keypoints)
    transform_op_list = []

    image_h, image_w = image.shape[0:2]

    # # random crop
    # if np.random.random() < crop[0]:
    #     new_h = int(image_h * (1 - crop[1]))
    #     new_w = int(image_w * (1 - crop[1]))
    #     crop_op = iaa.CropToFixedSize(height=new_h, width=new_w)
    #     transform_op_list.append(crop_op)

    # random rotate:
    if np.random.random() < rotate[0]:
        rotate_op = iaa.Affine(
            rotate=(rotate[1],rotate[2]))
        transform_op_list.append(rotate_op)

    # random flip (horizon /vertical)
    if np.random.random() < flip[0]:
        flip_h_op = iaa.Fliplr()
        transform_op_list.append(flip_h_op)
    if np.random.random() < flip[1]:
        flip_v_op = iaa.Flipud()
        transform_op_list.append(flip_v_op)

    # color jitter
    if np.random.random() < color_jitter:
        color_jitter_op = iaa.MultiplyHueAndSaturation(
            (0.5, 1.5), per_channel=True)
        transform_op_list.append(color_jitter_op)

    kps = KeypointsOnImage([Keypoint(x=keypoints[i][0], y=keypoints[i][1]) for i in range(
        len(keypoints))], shape=image.shape)

    seq = iaa.Sequential(transform_op_list)
    img_aug, kps_aug = seq(image=image, keypoints=kps,)

    for i in range(len(kps_aug.keypoints)):
        if points[i][0] == -1:
            continue
        points[i][0]=round(kps_aug.keypoints[i].x)
        points[i][1]=round(kps_aug.keypoints[i].y)

    return img_aug.astype(np.float32), points
