#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-12
"""
from core.train.trainer import Trainer
from core.network.hourglass import Hourglass
from core.dataset.keypoints import Keypoints
import config.config_hourglass_coco as cfg

import sys

sys.path.append('.')


class TrainHourglass(Trainer):
    def __init__(self, model, dataset, cfg):
        super(TrainHourglass, self).__init__(model, dataset, cfg)


if __name__ == '__main__':
    trainer = TrainHourglass(Hourglass, Keypoints, cfg)
    trainer.train_launch()
