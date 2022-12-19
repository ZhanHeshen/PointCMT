import os
import numpy as np
import argparse
import random
import pprint
import importlib

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

from data.modelnet40_mv_loader import ModelNet40
from torch.utils.data import DataLoader
from time import time
from emdloss import emd_module
from utils.all_utils import (PerfTrackTrain, PerfTrackVal, TrackTrain)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate(loader, model, teacher_model, EmdLoss, task='cls'):
    model.eval()
    teacher_model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0
    test_Emdloss = 0.0

    with torch.no_grad():
        time5 = time()
        for i, data_batch in enumerate(loader):
            batch_size = data_batch['pointcloud'].shape[0]
            time1 = time()
            time2 = time()

            out, mv_feature = teacher_model(data_batch)
            mv_dec_pc = model(mv_feature)

            gt_scaled, pr_scaled = scale(data_batch['pointcloud'].cuda(), mv_dec_pc)
            lossEmd, _ = EmdLoss(pr_scaled, gt_scaled, 0.05, 3000)
            lossEmd = torch.sqrt(lossEmd).mean(1).mean()

            test_Emdloss += lossEmd.item() * batch_size

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()

    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    print('test Emdloss is: ', test_Emdloss)
    return test_Emdloss


def scale(gt_pc, pr_pc):
    B = gt_pc.shape[0]
    min_gt = gt_pc.min(axis=1)[0]
    max_gt = gt_pc.max(axis=1)[0]
    min_pr = pr_pc.min(axis=1)[0]
    max_pr = pr_pc.max(axis=1)[0]
    length_gt = torch.abs(max_gt - min_gt)
    length_pr = torch.abs(max_pr - min_pr)
    diff_gt = length_gt.max(axis=1, keepdim=True)[0] - length_gt
    diff_pr = length_pr.max(axis=1, keepdim=True)[0] - length_pr
    size_pr = length_pr.max(axis=1)[0]
    size_gt = length_gt.max(axis=1)[0]
    scaling_factor_gt = 1. / size_gt
    scaling_factor_pr = 1. / size_pr
    new_min_gt = (min_gt - diff_gt) / 2.
    new_min_pr = (min_pr - diff_pr) / 2.
    box_min = torch.ones_like(new_min_gt) * -0.5
    adjustment_factor_gt = box_min - (scaling_factor_gt * new_min_gt.permute((1, 0))).permute((1, 0))
    adjustment_factor_pr = box_min - (scaling_factor_pr * new_min_pr.permute((1, 0))).permute((1, 0))
    pred_scaled = (pr_pc.permute(2, 1, 0) * scaling_factor_pr).permute(2, 1, 0) + adjustment_factor_pr.reshape(B, -1, 3)
    gt_scaled = (gt_pc.permute(2, 1, 0) * scaling_factor_gt).permute(2, 1, 0) + adjustment_factor_gt.reshape(B, -1, 3)
    return gt_scaled, pred_scaled


def train(loader, model, teacher_model, optimizer, EmdLoss, task='cls'):
    teacher_model.eval()
    model.train()

    def get_extra_param():
        return None

    mvperf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0
    train_Emdloss = 0.0
    time3 = time()

    for i, data_batch in enumerate(loader):
        time1 = time()
        batch_size = data_batch['pointcloud'].shape[0]
        mv_out, mv_feature = teacher_model(data_batch)
        mv_dec_pc = model(mv_feature)

        gt_scaled, pr_scaled = scale(data_batch['pointcloud'].cuda(), mv_dec_pc)
        lossEmd, _ = EmdLoss(pr_scaled, gt_scaled, 0.05, 3000)
        lossEmd = torch.sqrt(lossEmd).mean(1).mean()

        loss = 30 * lossEmd
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_Emdloss += lossEmd.item() * batch_size

        mvperf.update_all(data_batch=data_batch, out=mv_out, loss=loss)
        time2 = time()

        time_data_loading += (time1 - time3)
        time_forward += (time2 - time1)
        time3 = time()
        time_backward += (time3 - time2)

        if i % 100 == 0:
            print(
                f"[{i}/{len(loader)}] avg_loss: {mvperf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")

    print('Emdloss is ', train_Emdloss * 1.0 / 9840)

    return train_Emdloss


def save_checkpoint(id, epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    path = f"checkpoints/{cfg.exp_name}/model_{id}.pth"
    torch.save({
        'cfg': vars(cfg),
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_sched_state': lr_sched.state_dict(),
        'bnm_sched_state': bnm_sched.state_dict() if bnm_sched is not None else None,
        'test_perf': test_perf,
    }, path)
    print('Checkpoint saved to %s' % path)
    model.to(DEVICE)


def get_optimizer(optim_name, params):
    if optim_name == 'vanilla':
        optimizer = torch.optim.Adam(
            params,
            lr=0.001,
            weight_decay=1e-4)
        lr_sched = lr_scheduler.CosineAnnealingLR(
            optimizer,
            250,
            eta_min=0.001)
        bnm_sched = None
    else:
        assert False

    return optimizer, lr_sched, bnm_sched


def entry_train(cfg):
    loader_train = DataLoader(
        ModelNet40(
            data_path=cfg.data_root,
            partition='train',
        ),
        num_workers=8,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True)

    loader_test = DataLoader(
        ModelNet40(
            data_path=cfg.data_root,
            partition='test',
        ),
        num_workers=8,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    teacher_model = importlib.import_module('models.mvcnn')
    teacher_model = teacher_model.get_model(cfg).to(DEVICE)
    teacher_model = nn.DataParallel(teacher_model)

    model = importlib.import_module('models.cmpg')
    model = model.get_model().to(DEVICE)
    model = nn.DataParallel(model)

    params = list(model.parameters())
    optimizer, lr_sched, bnm_sched = get_optimizer('vanilla', params)

    checkpoint = torch.load(cfg.teacher_path)
    teacher_model.load_state_dict(checkpoint)
    print('Load pretrained multiview model from %s' % cfg.teacher_path)
    teacher_model.eval()

    log_dir = f"./checkpoints/{cfg.exp_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    track_train = TrackTrain(early_stop_patience=1000)

    EmdLoss = emd_module.emdModule()
    best_test_emd = 1000.0

    for epoch in range(cfg.epochs):
        print(f'\nEpoch {epoch}')
        start = time()

        print('Training..')
        train_perf = train(loader_train, model, teacher_model, optimizer, EmdLoss)
        pprint.pprint(train_perf, width=80)

        print('\nTesting..')
        test_perf = validate(loader_test, model, teacher_model, EmdLoss)
        pprint.pprint(test_perf, width=80)

        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=train_perf,
            test_metric=test_perf
        )

        if test_perf < best_test_emd:
            best_test_emd = test_perf
            save_checkpoint('best_test', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        lr_sched.step()

        end = time()
        last = end - start
        print('every epoch lasts for ', last)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='cmpg_default', help='Name of the experiment')
    parser.add_argument('--data_root', type=str, default='dataset/ModelNet40/data/', help='Name of the data root')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, help='number of episode to train ')
    parser.add_argument('--mv_backbone', type=str, default='resnet18')
    parser.add_argument('--teacher_path', type=str, default='./checkpoint/mvcnn_default/model.t7', help='Pretrained weight for teacher network')
    parser.add_argument('--seed', type=int, default=2021, help='random seed (default: 1)')
    parser.add_argument('--num_class', type=int, default=40)
    parser.add_argument('--pretraining', type=bool, default=False)
    cfg = parser.parse_args()
    print(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    entry_train(cfg)
