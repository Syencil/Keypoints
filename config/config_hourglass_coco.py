#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-11
"""

# HARDWARE
CUDA_VISIBLE_DEVICES = '2'
CUDA_VISIBLE_DEVICES_INFER = '1'
MULTI_THREAD_NUM = 4
MULTI_GPU = [0]

# PATH
dataset_dir = '/data/dataset/coco'
train_image_dir = '/data/dataset/coco/images/train2017'
val_image_dir = '/data/dataset/coco/images/val2017'
train_list_path = 'data/dataset/coco/coco_train.txt'
val_list_path = 'data/dataset/coco/coco_val.txt'

log_dir = 'output/coco'
ckpt_dir = '/data/checkpoints/coco'

# AUGMENT
augment = {
    "color_jitter": 0.5,
    "crop": (0.5, 0.9),
    "rotate": (0.5, 15),
    "ver_flip": 0,
    "hor_flop": 0,
}

# NETWORK
backbone = "hourglass"
loss_mode = 'focal' # focal, sigmoid, softmax, mse
image_size = (512, 512)
stride = 4
heatmap_size = (128, 128)
num_block = 1
num_depth = 5
residual_dim = [256, 384, 384, 384, 512]

is_maxpool = False
is_nearest = True

# SAVER AND LOADER
max_keep = 30
pre_trained_ckpt = None
ckpt_name = backbone + "_coco" + '.ckpt'

# TRAINING
batch_size = 16
learning_rate_init = 1e-3
learning_rate_warmup = 2.5e-4
exp_decay = 0.97

warmup_epoch_size = 0
epoch_size = 40
summary_per = 20
save_per = 2500

regularization_weight = 5e-4




# VAL
val_per = 2500
val_time = 20
val_rate = 0.1

# TEST

