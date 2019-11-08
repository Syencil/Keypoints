#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2019 luozw, Inc. All Rights Reserved

Authors: luozhiwang(luozw1994@outlook.com)
Date: 2019-09-12
"""
import os
import json
def coco_keypoint2txt(file, txt_path, thre=1):
    with open(txt_path, 'w') as writer:
        all_count = 0
        good_count = 0
        print('Transform %s' % (file))
        index = {}
        data = json.load(open(file))
        for ann in data['annotations']:
            all_count+=1
            st = str(int(ann['bbox'][0])) + ',' + str(int(ann['bbox'][1])) + ',' + str(int(ann['bbox'][0]+ann['bbox'][2])) + ',' + str(int(ann['bbox'][1]+ann['bbox'][3]))+' '
            gt = index.get(ann['image_id'], [])
            keypoints = ann['keypoints']
            # key = []
            for i in range(len(keypoints) // 3):
                # 不存在
                if keypoints[i * 3 + 2] == 0:
                    st += '-1,-1' + ' '
                    # key.append([-1, -1])
                # 标注 但不可见
                elif keypoints[i * 3 + 2] == 1:
                    # st += str(int(keypoints[i * 3])) + ',' + \
                    #     str(int(keypoints[i * 3 + 1])) + ' '
                    st += '-1,-1' + ' '
                    # key.append([keypoints[i * 3], keypoints[i * 3 + 1]])
                # 标注 可见
                elif keypoints[i * 3 + 2] == 2:
                    st += str(int(keypoints[i * 3])) + ',' + \
                          str(int(keypoints[i * 3 + 1])) + ' '
                    # key.append([keypoints[i * 3], keypoints[i * 3 + 1]])
                else:
                    st += '-1,-1' + ' '
                    print('Unsupported keypoints val')
                    # key.append([-1, -1])
            if st.count('-1,-1') <= thre:
                good_count += 1
                # data cleaning
                gt.append(st)
                index[ann['image_id']] = gt
            # writer.write(ann['image_id']+' '+st+'\n')
        for image in data['images']:
            if image['id'] in index:
                for i in range(len(index[image['id']])):
                    writer.write(image['file_name'] + ' ' + index[image['id']][i] + '\n')
        print('total data are %d, write data are %d' % (all_count, good_count))


if __name__ == '__main__':
    dataset = 'coco'

    if dataset == 'coco':
        coco_dir = '/data/dataset/coco'
        annotations_dir = os.path.join(coco_dir, 'annotations')
        annotation_train = os.path.join(
            annotations_dir,
            'person_keypoints_train2017.json')
        annotation_val = os.path.join(
            annotations_dir,
            'person_keypoints_val2017.json')
        coco_keypoint2txt(annotation_train, '../data/dataset/coco/coco_train.txt', 10)
        coco_keypoint2txt(annotation_val, '../data/dataset/coco/coco_val.txt', 10)

    if dataset == 'mpii':
        mpii_dir = '/data/dataset/mpii'
        annotations_dir = os.path.join(mpii_dir, 'annotations')
        annotation_train = os.path.join(
            annotations_dir,
            'train.json')
        annotation_val = os.path.join(
            annotations_dir,
            'test.json')
        coco_keypoint2txt(annotation_train, '../data/dataset/mpii/mpii_train.txt', 1)
        coco_keypoint2txt(annotation_val, '../data/dataset/mpii/mpii_val.txt', 1)

