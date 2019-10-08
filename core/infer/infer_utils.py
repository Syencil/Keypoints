#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-23
"""
import cv2
import numpy as np

bbx_color = (28, 255, 147)
pointer_color = (255, 204, 0)
txt_color = (0, 240, 78)


def get_results(hms, threshold=0.6):
    if len(hms.shape) == 3:
        hms = np.expand_dims(hms, axis=0)
    num_class = hms.shape[-1]
    results = []
    for b in range(len(hms)):
        joints = -1 * np.ones([num_class, 3], dtype=np.float32)
        hm = hms[b]
        for c in range(num_class):
            index = np.unravel_index(
                np.argmax(hm[:, :, c]), hm[:, :, c].shape)
            # tmp = list(index)
            tmp = [index[1], index[0]]
            score = hm[index[0], index[1], c]
            tmp.append(score)
            if score >= threshold:
                joints[c] = np.array(tmp)
        results.append(joints.tolist())
    return results


def draw_bbx(image, bbx):
    image = cv2.rectangle(
        image, (bbx[0], bbx[1]), (bbx[2], bbx[3]), bbx_color, 3)
    return image


def draw_point(image, points):
    for point in points:
        if point[0] != -1 and point[1] != -1:
            image = cv2.circle(
                image, (point[0], point[1]), 5, pointer_color, 3)
    return image


def draw_skeleton(image, points, dataset='mpii'):
    for point in points:
        if point[0] != -1 and point[1] != -1:
            image = cv2.circle(
                image, (int(point[0]), int(point[1])), 5, pointer_color, 3)
    if dataset is 'mpii':
        LINKS = [(0, 1), (1, 2), (2, 6), (6, 3), (3, 4), (4, 5), (6, 8),
                 (8, 13), (13, 14), (14, 15), (8, 12), (12, 11), (11, 10)]
        for link in LINKS:
            if points[link[0]][:2] != [-1,-1] and points[link[1]][:2] != [-1,-1]:
                image = cv2.line(image, (int(points[link[0]][0]),int(points[link[0]][1])), (int(points[link[1]][0]),int(points[link[1]][1])), bbx_color)
        return image
    elif dataset is 'coco':
        LINKS = [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
        for link in LINKS:
            if points[link[0]-1][:2] != [-1,-1] and points[link[1]-1][:2] != [-1,-1]:
                image = cv2.line(image, (int(points[link[0]-1][0]),int(points[link[0]-1][1])), (int(points[link[1]-1][0]),int(points[link[1]-1][1])), bbx_color)
        return image


def visiual_image_with_hm(img, hm):
    hm = np.sum(hm, axis=-1) * 255
    hm = np.expand_dims(hm, axis=-1)
    hm = np.tile(hm, (1, 1, 3))
    hm = cv2.resize(hm, (img.shape[1], img.shape[0]))
    img = img + hm
    # img = np.clip(img, 0, 255)
    return img