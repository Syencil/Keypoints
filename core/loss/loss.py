#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-10
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim


def cross_entropy(features, heatmap):
    print('-Utilize Sigmoid-Cross-Entropy-Loss')
    """
    loss for point locating.
    B batch size
    H, W Tensor shape
    C num of classes
    CELoss
    :param features: (Tensor) without actived BxHxWxC
    :param heatmap: (Tensor) labels BxHxWxC
    :return: (List(Tensor))
    """
    if not isinstance(features, list):
        features = [features]
    losses = []
    for i in range(len(features)):
        loss = - heatmap * tf.log(features[i])
        losses.append(tf.reduce_mean(loss))
    return losses


def softmax_cross_entropy(features, heatmap):
    print('-Utilize Softmax-Cross-Entropy-Loss')
    if not isinstance(features, list):
        features = [features]
    losses = []
    for i in range(len(features)):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=features[i],
            labels=heatmap
        )
        losses.append(tf.reduce_mean(loss))
    return losses


def focal_loss(features, heatmap, alpha=2, beta=4):
    """
    Focal Loss in "CornerNet"
    Loss = -1/N * (1-p)**alpha*log(p) if y=1 or (1-y)**beta*p**alpha*log(1-p)
    add loss into Graph
    :param features: (List(Tensor)) [BxHxWxC]
    :param heatmap: (Tensor) BxHxWxC
    :param alpha: (int)
    :param beta: (int)
    :return: (List(Tensor))
    """
    eps = 1e-9
    print('-Utilize Focal-Loss')
    if type(features) is not list:
        features = [features]
    losses = []
    for i in range(len(features)):
        # feature = tf.nn.sigmoid(features[i])
        feature = tf.clip_by_value(features[i], eps, 1 - eps)
        zeros = tf.zeros_like(heatmap)
        ones = tf.ones_like(heatmap)

        # mask
        mask = tf.where(tf.equal(heatmap, 1.0), ones, zeros)
        inv_mask = tf.subtract(1.0, mask)

        # num_pos
        num_pos = tf.reduce_sum(mask)
        num_pos = tf.maximum(num_pos, 1)

        # pre
        pos = tf.multiply(feature, mask)
        neg = tf.multiply(1.0 - feature, inv_mask)
        pre = tf.log(tf.add(pos, neg) + eps)

        # weight alpha
        pos_weight_alpha = tf.multiply(1.0 - feature, mask)
        neg_weight_alpha = tf.multiply(feature, inv_mask)
        weight_alpha = tf.pow(tf.add(pos_weight_alpha, neg_weight_alpha), alpha)

        # weight beta
        pos_weight_beta = mask
        neg_weight_beta = tf.multiply(1.0 - heatmap, inv_mask)
        weight_beta = tf.pow(tf.add(pos_weight_beta, neg_weight_beta), beta)

        # cal loss
        loss = tf.reduce_sum(- weight_beta * weight_alpha * pre) / num_pos

        losses.append(loss)
    return losses

# def focal_loss(preds, gt):
#     eps = 1e-9
#     print(gt.get_shape().as_list())
#     zeros = tf.zeros_like(gt)
#     ones = tf.ones_like(gt)
#     num_pos = tf.reduce_sum(tf.where(tf.equal(gt, 1), ones, zeros))
#     loss = []
#     # loss=tf.reduce_mean(tf.log(preds))
#     for pre in preds:
#         pre = tf.nn.sigmoid(pre)
#         pos_weight = tf.where(tf.equal(gt, 1), ones - pre, zeros)
#         neg_weight = tf.where(tf.less(gt, 1), pre, zeros)
#         pos_loss = tf.reduce_sum(
#             tf.log(
#                 pre +
#                 eps) *
#             tf.pow(
#                 pos_weight +
#                 eps,
#                 2))
#         neg_loss = tf.reduce_sum(
#             tf.pow(
#                 (1 -
#                  gt +
#                  eps),
#                 4) *
#             tf.pow(
#                 neg_weight +
#                 eps,
#                 2) *
#             tf.log(
#                 (1 -
#                  pre +
#                  eps)))
#         if num_pos == 0:
#             loss.append(-neg_loss)
#         else:
#             loss.append(-(pos_loss + neg_loss) / (num_pos + eps))
#     return loss


def mean_square_loss(features, heatmap):
    print('-Utilize Mse-Loss')
    if not isinstance(features, list):
        features = [features]
    losses = []
    for i in range(len(features)):
        loss = tf.losses.mean_squared_error(
            heatmap, features[i])
        losses.append(tf.reduce_mean(loss))
    return losses
