#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-16
"""
import random
import time
import cv2
import numpy as np
import os
import copy


class Dataset():
    def __init__(self, image_dir, gt_path, batch_size,
                 image_size=(512, 512), heatmap_size=(128, 128)):
        """
        Wrapper for key-points detection dataset
        :param image_dir: (str) image dir
        :param gt_path: (str) data file eg. train.txt or val.txt, etc
        :param batch_size: (int) batch size
        :param image_size: (int, int) height, width
        :param heatmap_size: (int, int) height, width. can be divided by image_size
        """
        # 数据量太大 不能直接读到内存 tf.data.dataset 不好使用
        # 读取info支持使用多线程加速
        self.gt_path = gt_path
        self.image_dir = image_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.batch_size = batch_size

        self.data_set = self.creat_set_from_txt()
        # self.transform_image_set_abs_to_rel()

        self.num_data = len(self.data_set)
        self.num_class = len(self.data_set[0][2])
        self.stride = self.image_size[0] // self.heatmap_size[0]
        self.ratio = self.image_size[0] / self.image_size[1]

        self._pre = -self.batch_size

    def creat_set_from_txt(self):
        """
        read image info and gt into memory
        :return: [[(str) image_name, [(int) xmin, (int) ymin, (int) xmax, (int) ymax], [(int) px, (int) py]]]
        """
        image_set = []
        t0 = time.time()
        count = 0

        for line in open(self.gt_path, 'r').readlines():
            if line == '':
                continue
            count += 1
            if count % 5000 == 0:
                print("--parse %d " % count)
            b = line.split()[1].split(',')
            points = line.split()[2:]
            tmp = []
            for point in points:
                tmp.append([[round(float(x)) for x in y.split(",")]
                            for y in point.split('|')])
            image_set.append(
                (line.split()[0], [round(float(x)) for x in b], tmp))
        print('-Set has been created in %.3fs' % (time.time() - t0))
        return image_set

    def sample_batch_image_random(self):
        """
        sample data (infinitely)
        :return: list
        """
        return random.sample(self.data_set, self.batch_size)
        # return self.data_set[:self.batch_size]

    def sample_batch_image_order(self):
        """
        sample data in order (one shot)
        :return: list
        """
        self._pre += self.batch_size
        if self._pre >= self.num_data:
            raise StopIteration
        _last = self._pre + self.batch_size
        if _last > self.num_data:
            _last = self.num_data
        return self.data_set[self._pre:_last]

    def make_guassian(self, height, width, sigma=3, center=None):
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4. * np.log(2.) * ((x - x0) **
                                          2 + (y - y0) ** 2) / sigma ** 2)

    def generate_hm(self, joints, heatmap_h_w):
        num_joints = len(joints)
        hm = np.zeros([heatmap_h_w[0], heatmap_h_w[1],
                       num_joints], dtype=np.float32)
        for i in range(num_joints):
            for joint in joints[i]:
                if joint[0] != -1 and joint[1] != -1:
                    s = int(
                        np.sqrt(
                            heatmap_h_w[0]) * heatmap_h_w[1] * 10 / 4096) + 2
                    gen_hm = self.make_guassian(heatmap_h_w[0], heatmap_h_w[1], sigma=s,
                                                center=[joint[0], joint[1]])
                    hm[:, :, i] = np.maximum(hm[:, :, i], gen_hm)
        return hm

    def _crop_image_with_pad_and_resize(self, image, bbx, points, ratio=0.05):
        image_h, image_w = image.shape[0:2]
        crop_bbx = copy.deepcopy(bbx)
        crop_points = copy.deepcopy(points)

        w = bbx[2] - bbx[0] + 1
        h = bbx[3] - bbx[1] + 1
        # keep 5% blank for edge
        crop_bbx[0] = int(bbx[0] - w * ratio)
        crop_bbx[1] = int(bbx[1] - h * ratio)
        crop_bbx[2] = int(bbx[2] + w * ratio)
        crop_bbx[3] = int(bbx[3] + h * ratio)
        # clip value from 0 to len-1
        crop_bbx[0] = 0 if crop_bbx[0] < 0 else crop_bbx[0]
        crop_bbx[1] = 0 if crop_bbx[1] < 0 else crop_bbx[1]
        crop_bbx[2] = image_w - 1 if crop_bbx[2] > image_w - 1 else crop_bbx[2]
        crop_bbx[3] = image_h - 1 if crop_bbx[3] > image_h - 1 else crop_bbx[3]
        # crop the image
        crop_image = image[crop_bbx[1]: crop_bbx[3] +
                           1, crop_bbx[0]: crop_bbx[2] + 1, :]
        # update width and height
        w = crop_bbx[2] - crop_bbx[0] + 1
        h = crop_bbx[3] - crop_bbx[1] + 1
        # keep aspect ratio

        ih, iw = self.image_size

        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(crop_image, (nw, nh))

        image_paded = np.full(shape=[ih, iw, 3], fill_value=127.5)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        for i in range(len(points)):
            for j, point in enumerate(points[i]):
                if point[0] != -1 and point[1] != -1:
                    crop_points[i][j][0] = int(
                        ((point[0] - crop_bbx[0]) * scale + dw) / self.stride)
                    crop_points[i][j][1] = int(
                        ((point[1] - crop_bbx[1]) * scale + dh) / self.stride)

        return image_paded, crop_points

    def _augment(self, image, hm):
        # flip
        # if np.random.choice((0, 1)):
        #     image = image[:, ::-1, :]
        #     hm = hm[:, ::-1, :]
        # rotate
        if np.random.choice((0, 1)):
            rotate_angle = np.random.uniform(-90, 90)
            image_rotate_matrix = cv2.getRotationMatrix2D(
                (self.image_size[1] // 2, self.image_size[0] // 2), rotate_angle, 1)
            hm_rotate_matrix = cv2.getRotationMatrix2D(
                (self.heatmap_size[1] // 2, self.heatmap_size[0] // 2), rotate_angle, 1)
            image = cv2.warpAffine(
                image,
                image_rotate_matrix,
                self.image_size,
                borderValue=(
                    127.5,
                    127.5,
                    127.5))
            hm = cv2.warpAffine(hm, hm_rotate_matrix, self.heatmap_size)
            for c in range(hm.shape[2]):
                pos = np.unravel_index(np.argmax(hm[..., c]), hm[..., c].shape)
                if hm[pos[0], pos[1], c] != 0:
                    hm[pos[0], pos[1], c] = 1.0
        if np.random.choice((1, 1)):
            image = image.astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            random_bright = .25 + np.random.uniform(0.5, 1)
            # print(random_bright)
            image[:, :, 2] = image[:, :, 2] * random_bright
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image, hm

    def _one_image_and_heatmap(self, image_set):
        """
        process only one image
        :param image_set: [image_name, bbx, [points]]
        :return: (narray) image_h_w x C, (narray) heatmap_h_w x C'
        """
        image_name, bbx, point = image_set
        image_path = os.path.join(self.image_dir, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, point = self._crop_image_with_pad_and_resize(img, bbx, point)
        # img = cv2.resize(img, self.image_size)
        # img, point = image_augment_with_keypoints(img, point)
        hm = self.generate_hm(point, self.heatmap_size)
        img, hm = self._augment(img, hm)
        return img, hm

    def iterator(self, max_worker=None, is_oneshot=False):
        """
        Wrapper for batch_data processing
        transform data from txt to imgs and hms
        (Option) utilize multi thread acceleration
        generator images and heatmaps infinitely or make oneshot
        :param max_worker: (optional) (int) max worker for multi-thread
        :param is_oneshot: (optional) (bool) if False, generator will sample infinitely.
        :return: iterator. imgs, hms = next(iterator)
        """
        if is_oneshot:
            sample_fn = self.sample_batch_image_order
        else:
            sample_fn = self.sample_batch_image_random
        if max_worker is not 0:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_worker) as executor:
                while True:
                    image_set = sample_fn()
                    imgs = []
                    hms = []
                    if executor is None:
                        for i in range(len(image_set)):
                            img, hm = self._one_image_and_heatmap(image_set[i])
                            imgs.append(img)
                            hms.append(hm)
                    else:
                        all_task = [
                            executor.submit(
                                self._one_image_and_heatmap,
                                image_set[i]) for i in range(
                                len(image_set))]
                        for future in as_completed(all_task):
                            imgs.append(future.result()[0])
                            hms.append(future.result()[1])
                    final_imgs = np.stack(imgs, axis=0)
                    final_hms = np.stack(hms, axis=0)
                    yield final_imgs, final_hms
        else:
            while True:
                image_set = sample_fn()
                imgs = []
                hms = []
                for i in range(len(image_set)):
                    img, hm = self._one_image_and_heatmap(image_set[i])
                    imgs.append(img)
                    hms.append(hm)
                final_imgs = np.stack(imgs, axis=0)
                final_hms = np.stack(hms, axis=0)
                yield final_imgs, final_hms


if __name__ == '__main__':

    from core.infer.visual_utils import visiual_image_with_hm

    dataset_dir = '/data/dataset/meter'
    image_dir = '/data/dataset/meter'
    gt_path = '../../data/dataset/multi_pointer/train_list.txt'
    render_path = '../../render_img'

    ite = 3
    batch_size = 16

    coco = Dataset(image_dir, gt_path, batch_size)
    it = coco.iterator(0, False)

    t0 = time.time()
    for i in range(ite):
        b_img, b_hm = next(it)
        for j in range(batch_size):
            img = b_img[j][:, :, ::-1]
            hm = b_hm[j]
            img_hm = visiual_image_with_hm(img, hm)
            cv2.imwrite(
                '../../render_img/' +
                str(i) +
                '_' +
                str(j) +
                '_img_hm.jpg',
                img_hm)

    print(time.time() - t0)
