#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Xu Yan
@File: main.py
@Time: 2021/05/13 10:39 PM
"""

from __future__ import print_function
import os
import sys
import time

import argparse
import torch
import random
import importlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics

from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from data.modelnet40_mv_loader import ModelNet40
from torch.utils.data import DataLoader
from utils.all_utils import cal_loss, IOStream

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
    os.system('cp main_mvcnn_modelnet.py checkpoints' + '/' + args.exp_name + '/' + 'main_mvcnn_modelnet.py.backup')
    os.system('cp models/%s.py checkpoints' % args.model + '/' + args.exp_name + '/' + '%s.py.backup' % args.model)
    os.system('cp data/modelnet40_mv_loader.py checkpoints' + '/' + args.exp_name + '/' + 'modelnet40_mv_loader.py.backup')


def put_cuda(data_dict):
    for k in data_dict.keys():
        data_dict[k] = data_dict[k].cuda()
    return data_dict


def train(args, io):
    train_loader = DataLoader(
        ModelNet40(
            data_path=args.data_root,
            partition='train',
        ),
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    test_loader = DataLoader(
        ModelNet40(
            data_path=args.data_root,
            partition='test',
        ),
        num_workers=8,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = importlib.import_module(args.model)
    model = model.get_model(args).to(device)
    print(str(model))

    model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.use_plateau:
        scheduler = ReduceLROnPlateau(opt, patience=10, factor=0.5, mode='max')
    else:
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss

    best_test_acc = 0

    for epoch in range(args.epochs):
        start = time.time()
        if args.use_plateau:
            scheduler.step(best_test_acc)
        else:
            scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        with tqdm(total=len(train_loader)) as pbar:
            for i, data_dict in enumerate(train_loader):
                batch_size = data_dict['pointcloud'].size()[0]
                data_dict = put_cuda(data_dict)

                logits, _ = model(data_dict)
                logits = logits['logit']
                loss = criterion(logits, data_dict['label'])
                loss.backward()

                if ((i + 1) % args.accumulation_step) == 0:
                    opt.step()
                    opt.zero_grad()

                preds = logits.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                tmp_true = data_dict['label'].cpu().numpy()
                tmp_pred = preds.detach().cpu().numpy()
                train_true.append(tmp_true)
                train_pred.append(tmp_pred)

                pbar.set_description('Epoch %d, CE LOSS: %.4f, Acc: %.4f' % (
                    epoch,
                    loss.item(),
                    metrics.accuracy_score(tmp_true, tmp_pred)
                ))
                pbar.update(1)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (
            epoch,
            train_loss * 1.0 / count,
            metrics.accuracy_score(train_true, train_pred),
            metrics.balanced_accuracy_score(train_true, train_pred))

        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data_dict in test_loader:
            data_dict = data_dict
            batch_size = data_dict['pointcloud'].size()[0]
            data_dict = put_cuda(data_dict)

            logits, _ = model(data_dict)
            logits = logits['logit']
            loss = criterion(logits, data_dict['label'])
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(data_dict['label'].cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            print('Model saved!')

        outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, best acc: %.6f' % (epoch,
                                                                                              test_loss * 1.0 / count,
                                                                                              test_acc,
                                                                                              avg_per_class_acc,
                                                                                              best_test_acc)
        io.cprint(outstr)
        end = time.time()
        last = end - start
        print(last)


def test(args, io):
    test_loader = DataLoader(
        ModelNet40(
            data_path=args.data_root,
            partition='test',
        ),
        num_workers=8,
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True)
    device = torch.device("cuda" if args.cuda else "cpu")

    model = importlib.import_module(args.model)
    model = model.get_model(args).to(device)
    print(str(model))

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []
    for data_dict in test_loader:
        data_dict = put_cuda(data_dict)

        logits, _ = model(data_dict)
        logits = logits['logit']
        preds = logits.max(dim=1)[1]
        test_true.append(data_dict['label'].cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Multiview-based Recognition')
    parser.add_argument('--exp_name', type=str, default='mvcnn_default', help='Name of the experiment')
    parser.add_argument('--data_root', type=str, default='dataset/ModelNet40/data/', help='Name of the data root')
    parser.add_argument('--mv_backbone', type=str, default='resnet18', help='Backbone model in MVCNN')
    parser.add_argument('--model', type=str, default='mvcnn', help='Model to use in `model` folder')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch)')
    parser.add_argument('--accumulation_step', type=int, default=4, help='step to return gredient)')
    parser.add_argument('--test_batch_size', type=int, default=16, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False, help='Use SGD')
    parser.add_argument('--use_plateau', type=bool, default=False, help='reduce learning rate in plateau')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=2021,  help='random seed (default: 1)')
    parser.add_argument('--eval', default=False, action='store_true', help='evaluate the model')
    parser.add_argument('--model_path', type=str, default='', help='Pretrained model path')
    parser.add_argument('--pretraining', type=bool, default=True)
    parser.add_argument('--num_class', type=int, default=40)
    args = parser.parse_args()

    _init_()
    torch.backends.cudnn.deterministic = True
    np.random.seed(args.seed)
    random.seed(args.seed)

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
