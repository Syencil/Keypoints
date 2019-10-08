#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-21
"""
from core.infer.infer import Infer
from core.network.hourglass import Hourglass
from core.dataset.keypoints import Keypoints
# import core.config.config_hourglass_coco as cfg
import config.config_hourglass_mpii as cfg

import sys
sys.path.append('.')


class InferHourglass(Infer):
    def __init__(self, model, dataset, cfg):
        super(InferHourglass, self).__init__(model, dataset, cfg)


if __name__ == '__main__':
    ckpt = 'checkpoints/mpii/Hourglass_mpii.ckpt-45500'
    infer = Infer(Hourglass, Keypoints, cfg)
    infer.infer_launch(ckpt)
