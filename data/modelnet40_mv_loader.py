#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pickle
import torchvision.transforms as tfs
import torch

from PIL import Image
from torch.utils.data import Dataset
from utils.pc_utils import translate_pointcloud


def load_mv_data(root, partition):
    with open(os.path.join(root, 'modelnet40_%s_2048pts_20views.dat' % partition), 'rb') as f:
        all_pc, all_mv, all_label = pickle.load(f)
    print('load mv data')
    print('The size of %s data is %d' % (partition, len(all_pc)))
    return all_pc, all_mv, all_label


class ModelNet40(Dataset):
    def __init__(self, data_path, num_points=1024, partition='train', generate=False):
        self.pc_data, self.mv_data, self.mv_label = load_mv_data(data_path, partition)
        self.num_views = 20
        self.num_points = num_points
        self.partition = partition
        self.generate = generate
        self.transform_mv = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item):
        data_dict = {}

        _views = []
        views = self.mv_data[item]
        label = self.mv_label[item]
        pointcloud = np.array(self.pc_data[item][:self.num_points, 0:3])

        for i in range(self.num_views):
            _views.append(self.transform_mv(Image.fromarray(views[i], "RGB")))
        views = torch.stack(_views, 0)

        if self.partition == 'train' and not self.generate:
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        pointcloud = pointcloud.astype('float32')

        data_dict['pointcloud'] = pointcloud
        data_dict['multiview'] = views
        data_dict['label'] = label

        return data_dict

    def __len__(self):
        return len(self.pc_data)


class ModelNet40_OfflineFeatures(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        feature_path = root + r'/modelnet40_%s_mvf.pth' % self.split
        self.dataset = torch.load(feature_path)

    def __len__(self):
        return len(self.dataset)

    def _get_item(self, index):
        points, label, mvf = self.dataset[index]

        if self.split == 'train':
            points = translate_pointcloud(points.numpy())
            np.random.shuffle(points)

        data_dict = {}
        data_dict['pointcloud'] = points
        data_dict['multiview'] = mvf
        data_dict['label'] = label.item()
        return data_dict

    def __getitem__(self, index):
        return self._get_item(index)