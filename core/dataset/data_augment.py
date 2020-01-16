#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-10-08
"""
import numpy as np
from albumentations import (
    KeypointParams,
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    Compose,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    HueSaturationValue,
    Resize
)


def image_augment_with_keypoint(image, keypoints, color_jitter=0.5, crop=(
        0.5, 0.8), rotate=(0.5, 30), ver_flip=0, hor_flop=0.5):

    image_h, image_w = image.shape[0:2]
    keypoints = np.clip(keypoints, None, max(image_w - 1, image_h - 1))
    points_ = []
    idx_ = []
    for i, ps in enumerate(keypoints):
        for j, p in enumerate(ps):
            if p[0] >= 0 and p[1] >= 0:
                points_.append(p)
                idx_.append([i, j])

    def get_aug(aug):
        return Compose(aug, keypoint_params=KeypointParams(format="xy"))

    aug = get_aug([VerticalFlip(p=ver_flip),
                   HorizontalFlip(p=hor_flop),
                   RandomCrop(
        p=crop[0],
        height=int(
            image_h *
            crop[1]),
        width=int(
            image_w *
            crop[1])),
        ShiftScaleRotate(p=rotate[0], rotate_limit=rotate[1]),
        RandomBrightnessContrast(p=color_jitter),
        HueSaturationValue(p=color_jitter),
        Resize(p=1, height=image_h, width=image_w)
    ]
    )
    augmented = aug(image=image, keypoints=points_)

    for i in range(len(augmented["keypoints"])):
        keypoints[idx_[i][0]][idx_[i][1]] = list(augmented["keypoints"][i])

    return augmented["image"], keypoints
