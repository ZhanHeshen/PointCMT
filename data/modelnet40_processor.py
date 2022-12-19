#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Author: Xu Yan
@File: modelnet40_processor.py
@Time: 2021/4/4 14:19
'''

import os
import argparse
import pickle

import numpy as np

from tqdm import tqdm
from PIL import Image

label_list = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk',
              'door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
              'person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet',
              'tv_stand','vase','wardrobe','xbox']
convert_name_to_label = lambda name: label_list.index(name)
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('ProcessData')

    parser.add_argument('--num_points',
                        type=int,
                        default=2048,
                        help='number of points')

    parser.add_argument('--num_views',
                        type=int,
                        default=20,
                        help='number of views')

    parser.add_argument('--mv_root',
                        type=str,
                        default='../dataset/ModelNet40/ModelNet40_mv_20view/',
                        help='root for multi-view data')

    parser.add_argument('--pc_root',
                        type=str,
                        default='../dataset/ModelNet40/modelnet40_normal_resampled/',
                        help='root for point data')

    parser.add_argument('--out_root',
                        type=str,
                        default='../dataset/ModelNet40/data/',
                        help='root for output data')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_root, exist_ok=True)
    shape_ids = {}
    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(args.pc_root, 'modelnet40_train.txt'))]
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(args.pc_root, 'modelnet40_test.txt'))]

    shape_names_train = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['train']]
    datapath_train = [
        (shape_names_train[i], os.path.join(args.pc_root, shape_names_train[i], shape_ids['train'][i]) + '.txt')
        for i in range(len(shape_ids['train']))]
    shape_names_test = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['test']]
    datapath_test = [
        (shape_names_test[i], os.path.join(args.pc_root, shape_names_test[i], shape_ids['test'][i]) + '.txt') for i
        in range(len(shape_ids['test']))]

    datapath = {
        'train': datapath_train,
        'test': datapath_test
    }

    for split in ['train', 'test']:
        print('Start process %s data...' % split)
        labels = []
        pts = []
        mvs = []
        save_path = os.path.join(args.out_root, 'modelnet40_%s_%dpts_%dviews.dat' % (split, args.num_points, args.num_views))

        for idx, (cat, path) in tqdm(enumerate(datapath[split]), total=len(datapath[split])):
            views = []
            item = path.split('/')[-1]
            mv_path = os.path.join(args.mv_root, cat, split)
            for view in range(args.num_views):
                mv_item = os.path.join(mv_path, '%s_%003d.png' % (item[:-4], view + 1))
                im = Image.open(mv_item)
                im = im.convert('RGB')
                views.append(im)
            views = np.stack(views, axis=0)
            mvs.append(views)
            point_set = np.loadtxt(path, delimiter=',').astype(np.float32)[:args.num_points, :]
            pts.append(point_set)
            label = np.array(convert_name_to_label(cat))
            labels.append(label)

        with open(save_path, 'wb') as f:
            pickle.dump([pts, mvs, labels], f)