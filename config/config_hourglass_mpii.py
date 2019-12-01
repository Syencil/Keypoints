#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-11
"""

# HARDWARE
CUDA_VISIBLE_DEVICES = '3'
CUDA_VISIBLE_DEVICES_INFER = '0'
MULTI_THREAD_NUM = 4
MULTI_GPU = [0]

# PATH
dataset_dir = 'data/dataset/mpii'
train_image_dir = 'data/dataset/mpii/images'
val_image_dir = 'data/dataset/mpii/images'
train_list_path = 'data/dataset/mpii/mpii_train.txt'
val_list_path = 'data/dataset/mpii/mpii_train.txt'

log_dir = 'output/mpii'
ckpt_dir = 'checkpoints/mpii'


# AUGMENT
rotate = [0.5, 30, 30]
flip = [0.5, 0.5]
random_crop = True
keep_align = 0.1
data_cleaning = True

# NETWORK
backbone = "hourglass"
loss_mode = 'focal'  # focal, sigmoid, softmax, mse
image_size = (512, 512)
stride = 4
heatmap_size = (128, 128)
num_block = 2
num_depth = 5
residual_dim = [256, 384, 384, 384, 512]

is_maxpool = False
is_nearest = True

# SAVER AND LOADER
max_keep = 30
pre_trained_ckpt = None
ckpt_name = backbone + "_voc" + ".ckpt"

# TRAINING
batch_size = 8
learning_rate_init = 2.5e-4
learning_rate_warmup = 1e-4
momentum = 0.9

warmup_epoch_size = 1
epoch_size = 60
summary_per = 20
save_per = 5000

regularization_weight = 5e-4

# VAL
val_per = 200
val_time = 20
val_rate = 0.1

# TEST

