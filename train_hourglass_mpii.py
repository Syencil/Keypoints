#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-26
"""
from core.train.trainer import Trainer
from core.network.hourglass import Hourglass
from core.dataset.keypoints import Keypoints
import config.config_hourglass_mpii as cfg

import sys
sys.path.append('.')


class TrainHourglass(Trainer):
    def __init__(self, model, dataset, cfg):
        super(TrainHourglass, self).__init__(model, dataset, cfg)

    def train_launch(self):
        self.is_debug = False
        # must in order
        self.init_dataset()
        self.init_inputs()
        self.init_model()

        # optional override
        self.init_loss()
        self.init_learning_rate()
        self.init_train_op()
        self.init_loader_saver_summary()
        self.init_session()
        self.train()

if __name__ == '__main__':
    trainer = TrainHourglass(Hourglass, Keypoints, cfg)
    trainer.train_launch()
