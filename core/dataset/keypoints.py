#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-16
"""
import time
import cv2
import numpy as np
import os


class Keypoints():
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
        self.transform_image_set_abs_to_rel()

        self.num_data = len(self.data_set)
        self.num_class = len(self.data_set[0][2])

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
                tmp.append([int(float(x)) for x in point.split(',')])
            image_set.append([line.split()[0], [int(float(x))
                                                for x in b], tmp])
        print('-Set has been created in %.3fs' % (time.time() - t0))
        return image_set

    def transform_image_set_abs_to_rel(self):
        for data in self.data_set:
            name, bbx, points = data
            w = bbx[2] - bbx[0]
            h = bbx[3] - bbx[1]
            ratio_w = self.heatmap_size[1] / w
            ratio_h = self.heatmap_size[0] / h
            for i in range(len(points)):
                points[i][0] = int((points[i][0] - bbx[0]) * ratio_w)
                points[i][1] = int((points[i][1] - bbx[1]) * ratio_h)

    def sample_batch_image_random(self):
        """
        sample data (infinitely)
        :return: list
        """
        # return random.sample(self.data_set, self.batch_size)
        return self.data_set[:self.batch_size]

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
        return np.exp(-4. * np.log(2.) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def generate_hm(self,joints, heatmap_h_w):
        num_joints = len(joints)
        hm = np.zeros([heatmap_h_w[0], heatmap_h_w[1],
                       num_joints], dtype=np.float32)
        for i in range(num_joints):
            if joints[i] != [-1, -1]:
                s = int(
                    np.sqrt(
                        heatmap_h_w[0]) * heatmap_h_w[1] * 10 / 4096) + 2
                hm[:, :, i] = self.make_guassian(heatmap_h_w[0], heatmap_h_w[1],
                                            sigma=s, center=[
                        joints[i][0], joints[i][1]]
                                            )
            else:
                hm[:, :, i] = np.zeros(
                    (heatmap_h_w[0], heatmap_h_w[1]))
        return hm

    def _one_image_and_heatmap(self, image_set):
        """
        process only one image
        :param image_set: [image_name, bbx, [points]]
        :return: (narray) image_h_w x C, (narray) heatmap_h_w x C'
        """
        image_name, bbx, point = image_set
        image_path = os.path.join(self.image_dir, image_name)
        img = cv2.imread(image_path)
        assert img is not None
        cropped_img = img[bbx[1]: bbx[3], bbx[0]:bbx[2], :]
        img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.image_size)
        img_final = img_resized.astype(np.float32)
        hm = self.generate_hm(point, self.heatmap_size)
        return img_final, hm

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

    from core.infer.infer_utils import visiual_image_with_hm

    dataset_dir = '../../data/dataset/coco'
    image_dir = '../../data/dataset/coco/images/val2017'
    gt_path = '../../data/dataset/coco/coco_val.txt'
    render_path = '../../render_img'

    ite = 2
    batch_size = 4

    coco = Keypoints(image_dir, gt_path, batch_size)
    it = coco.iterator(4)

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
