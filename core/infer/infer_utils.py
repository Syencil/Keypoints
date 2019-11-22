#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-21
"""
import os
import cv2
import time
import numpy as np
import tensorflow as tf


def read_pb_infer(pb_path, input_node_name_and_val, output_node_name):
    """
    [xmin, ymin, xmax, ymax, score, cid]
    :param pb_path:
    :param input_node_name_and_val: {(str) input_node_name: (any) input_node_val}
    :param output_node_name: [(str) output_node_name]
    :return: [output] B x Num_bbx x 6
    """
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name='')
            config = tf.ConfigProto(allow_soft_placement=True)  # 是否自动选择GPU
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                # sess.run(tf.global_variables_initializer())
                # 定义输入的张量名称,对应网络结构的输入张量
                # input:0作为输入图像,keep_prob:0作为dropout的参数,测试时值为1,is_training:0训练参数
                feed_dict = {}
                for key in input_node_name_and_val:
                    input_tensor = sess.graph.get_tensor_by_name(key)
                    feed_dict[input_tensor] = input_node_name_and_val[key]

                # 定义输出的张量名称
                output_tensor = []
                for name in output_node_name:
                    output_tensor.append(sess.graph.get_tensor_by_name(name))

                # 测试读出来的模型是否正确，注意这里传入的是输出和输入节点的tensor的名字，不是操作节点的名字
                start_time = time.time()
                output = sess.run(output_tensor, feed_dict=feed_dict)
                print('Infer time is %.4f' % (time.time() - start_time))
    return output


def read_pb(pb_path, input_name, output_name):
    """
    Instantiation Session
    :param pb_path: (str) pb file path
    :param input_name: [(str)] input tensor names
    :param output_name: [(str)] output tensor names
    :return: (tf.Session) sess, (Tensor) input, (Tensor) output
    """
    # return sess
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name='')
            config = tf.ConfigProto(allow_soft_placement=True)  # 是否自动选择GPU
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            if not isinstance(input_name, list) or isinstance(input_name, tuple):
                input_name = [input_name]
            input_tensor = []
            output_tensor = []
            for i in range(len(input_name)):
                input_tensor.append(sess.graph.get_tensor_by_name(input_name[i]))
            for i in range(len(output_name)):
                output_tensor.append(sess.graph.get_tensor_by_name(output_name[i]))

            return sess, input_tensor, output_tensor


def pb_infer(sess, output_tensor, input_tensor=None, input_val=None):
    """
    get output
    :param sess: (tf.Session) sess
    :param output_tensor: [(Tensor)]
    :param input_tensor: [(Tensor)]
    :param input_val: [(np.array)]
    :return:
    """
    feed_dict = {}
    if input_tensor is not None and input_val is not None:
        for i in range(len(input_tensor)):
            feed_dict[input_tensor[i]] = input_val[i]

    return sess.run(output_tensor, feed_dict)


def image_process(image, bbx):
    """
    image pre-process
    :param image: (str) image_path / (np.array) image in BGR
    :param bbx: [(int) xmin, (int) ymin, (int) xmax, (int) ymax]
    :return: input_image, bbx
    """
    if type(image) == str:
        image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    crop_image, crop_bbx = crop_with_padding_and_resize(image, bbx)
    cv2.imwrite("/work/meter_recognition/render_img/crop.jpg", crop_image)
    image_norm = crop_image / 127 - 1
    return image_norm, crop_bbx


def crop_with_padding_and_resize(image, bbx, shape=(512, 512), ratio=0.2):
    """
    image pre-process
    :param image: image path or BGR image
    :param bbx: [xmin, ymin, xmax, ymax]
    :param shape: output image shape
    :param ratio: keep blank for edge
    :return: resized and padded image
    """
    image_h, image_w = image.shape[0:2]
    crop_bbx = np.copy(bbx)

    w = bbx[2] - bbx[0] + 1
    h = bbx[3] - bbx[1] + 1
    # keep 0.2 blank for edge
    crop_bbx[0] = round(bbx[0] - w * ratio)
    crop_bbx[1] = round(bbx[1] - h * ratio)
    crop_bbx[2] = round(bbx[2] + w * ratio)
    crop_bbx[3] = round(bbx[3] + h * ratio)
    # clip value from 0 to len-1
    crop_bbx[0] = 0 if crop_bbx[0] < 0 else crop_bbx[0]
    crop_bbx[1] = 0 if crop_bbx[1] < 0 else crop_bbx[1]
    crop_bbx[2] = image_w - 1 if crop_bbx[2] > image_w - 1 else crop_bbx[2]
    crop_bbx[3] = image_h - 1 if crop_bbx[3] > image_h - 1 else crop_bbx[3]
    # crop the image
    crop_image = image[crop_bbx[1]: crop_bbx[3] + 1, crop_bbx[0]: crop_bbx[2] + 1, :]
    # update width and height
    w = crop_bbx[2] - crop_bbx[0] + 1
    h = crop_bbx[3] - crop_bbx[1] + 1
    # keep aspect ratio
    # padding
    if h < w:
        pad = int(w - h)
        pad_t = pad // 2
        pad_d = pad - pad_t
        pad_image = np.pad(crop_image, ((pad_t, pad_d), (0, 0), (0, 0)), constant_values=128)
    else:
        pad = int(h - w)
        pad_l = pad // 2
        pad_r = pad - pad_l
        pad_image = np.pad(crop_image, ((0, 0), (pad_l, pad_r), (0, 0)), constant_values=128)
    crop_image = cv2.resize(pad_image, shape)
    return crop_image, crop_bbx


def rel2abs(bbx, points):
    """
    transform points location into original location
    :param bbx: [xmin, ymin, xmax, ymax] cropped bbx
    :param points: [[x, y, score]] points location in heatmap
    :return: [[x, y, score]] points location in original image
    """
    bbx = bbx.copy()
    h, w = bbx[3] - bbx[1], bbx[2] - bbx[0]
    max_len = max(h, w)
    pad_t = (max_len - h) // 2
    pad_d = (max_len - h) - (max_len - h) // 2
    pad_l = (max_len - w) // 2
    pad_r = (max_len - w) - (max_len - w) // 2
    bbx[0] -= pad_l
    bbx[1] -= pad_t
    bbx[2] += pad_r
    bbx[3] += pad_d
    for point in points:
        point[0] = bbx[0] + point[0] * max_len / 128
        point[1] = bbx[1] + point[1] * max_len / 128
    return points


def draw_point(image, points):
    for point in points:
        if int(point[0]) != -1 and int(point[1]) != -1:
            image = cv2.circle(
                image, (int(point[0]), int(point[1])), 5, (255, 204, 0), 3)
    return image


def pred_one_image(image, bbxes, sess, input_tensor, output_tensor):
    processed_images = []
    processed_bbxes = []
    for bbx in bbxes:
        input_image, croped_bbxes = image_process(image, bbx)
        processed_images.append(input_image)
        processed_bbxes.append(croped_bbxes)
    batch_image = np.stack(processed_images, axis=0)
    batch_hm = pb_infer(sess, output_tensor, input_tensor, [batch_image])[0]
    final_point = []
    for i in range(len(batch_image)):
        hm = batch_hm[i]
        img = batch_image[i]
        point = get_results(hm, threshold=0.01)[0]

        point = rel2abs(processed_bbxes[i], point)
        final_point.append(point)
    return final_point


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
            if score > threshold:
                joints[c] = np.array(tmp)
        results.append(joints.tolist())
    return results








