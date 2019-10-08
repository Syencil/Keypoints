#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-24
"""
import os
import cv2
import time
import tensorflow as tf
from core.infer.infer_utils import get_results, draw_point, draw_skeleton


def read_pb(pb_path, input_node_name_and_val, output_node_name):
    """
    :param pb_path:
    :param input_node_name_and_val: {(str) input_node_name: (any) input_node_val}
    :param output_node_name: [(str) output_node_name]
    :return: [output]
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


def loss(fe, hm):
    zeros = np.zeros_like(fe)
    ones = np.ones_like(fe)

    mask = np.where(np.equal(hm, 1.0), ones, zeros)
    inv_mask = 1.0 - mask

    num_pos = np.sum(mask)
    num_pos = np.maximum(1.0, num_pos)

    pos = np.multiply(fe, mask)
    neg = np.multiply(1.0 - fe, inv_mask)
    pre = np.log(np.add(pos, neg)+1e-9)

    pos_weight_alpha = np.multiply(1.0 - fe, mask)
    neg_weight_alpha = np.multiply(fe, inv_mask)
    weight_alpha = np.power(np.add(pos_weight_alpha, neg_weight_alpha), 2)

    pos_weight_beta = mask
    neg_weight_beta = np.multiply(1.0 - hm, inv_mask)
    weight_beta = np.power(np.add(pos_weight_beta, neg_weight_beta), 4)

    # cal loss
    loss = np.sum(- weight_beta * weight_alpha * pre) / num_pos
    return loss


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


if __name__ == '__main__':
    import numpy as np
    from core.dataset.keypoints import Keypoints
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    pb_path = 'Hourglass.pb'
    img_dir = 'data/dataset/coco/images/val2017'
    gt_path = 'data/dataset/coco/coco_val.txt'
    batch_size = 8
    img_size = (256,256)
    hm_size = (64,64)
    dataset = Keypoints(img_dir, gt_path, batch_size, img_size, hm_size)
    it = dataset.iterator(4, True)
    image, hm = next(it)
    image_norm = (image / 127.5) - 1
    input_dict = {'Placeholder/inputs_x:0': image_norm, 'Placeholder/is_training:0':False}
    output_node_name=['HourglassNet/keypoint_1/conv/Sigmoid:0']
    outputs = read_pb(pb_path, input_dict, output_node_name)
    for k in range(len(outputs)):
        # outputs[k] = sigmoid(outputs[k])
        points = get_results(outputs[k], 0.0)
        gt_points = get_results(hm, 0.0)
        print(points)
        print(gt_points)
        for i in range(len(points)):
            img = image[i][:, :, ::-1]
            for j in range(len(points[i])):
                if points[i][j][0] != -1:
                    points[i][j][0] = int(points[i][j][0]/64*img.shape[1])
                if points[i][j][1] != -1:
                    points[i][j][1] = int(points[i][j][1]/64*img.shape[0])

            one_ouput = np.sum(outputs[k][i], axis=-1, keepdims=True) * 255
            tile_output = np.tile(one_ouput, (1, 1, 3))
            tile_img =cv2.resize(tile_output, (256, 256)) + img

            cv2.imwrite('render_img/'+str(i)+'_'+str(k)+'_visible.jpg', tile_img)

            sk_img = draw_skeleton(img, points[i],'coco')
            cv2.imwrite('render_img/' + str(i) + '_' + str(k) + '_skeleton.jpg', sk_img)

            img = draw_point(img, points[i])
            cv2.imwrite('render_img/'+str(i)+'_'+str(k)+'.jpg', img)
        # outputs[k]

        print(loss(outputs[k],hm))




