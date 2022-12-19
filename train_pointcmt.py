import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import random

import os
import numpy as np
import argparse
import pprint
import importlib
import models

from time import time
from emdloss import emd_module
from data.modelnet40_mv_loader import ModelNet40_OfflineFeatures, ModelNet40
from utils.all_utils import PerfTrackTrain, PerfTrackVal, TrackTrain, smooth_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_loss(task, loss_name, data_batch, out):
    """
    Returns the tensor loss function
    :param task:
    :param loss_name:
    :param data_batch: batched data; note not applied data_batch
    :param out: output from the model
    :param dataset_name:
    :return: tensor
    """
    if task == 'cls':
        label = data_batch['label'].to(out['logit'].device)
        if loss_name == 'cross_entropy':
            loss = F.cross_entropy(out['logit'], label)
        elif loss_name == 'smooth':
            loss = smooth_loss(out['logit'], label)
        else:
            assert False
    return loss


def validate(loader, model, task='cls'):
    model.eval()

    def get_extra_param():
        return None

    perf = PerfTrackVal(task, extra_param=get_extra_param())
    time_dl = 0
    time_gi = 0
    time_model = 0
    time_upd = 0

    with torch.no_grad():
        time5 = time()
        for i, data_batch in enumerate(loader):
            time1 = time()
            time2 = time()

            out, _ = model(data_batch['pointcloud'].cuda())

            time3 = time()
            perf.update(data_batch=data_batch, out=out)
            time4 = time()

            time_dl += (time1 - time5)
            time_gi += (time2 - time1)
            time_model += (time3 - time2)
            time_upd += (time4 - time3)

            time5 = time()

    print(f"Time DL: {time_dl}, Time Get Inp: {time_gi}, Time Model: {time_model}, Time Update: {time_upd}")
    return perf.agg()


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


def train(loader, model, decoder_model, optimizer, EmdLoss, task='cls'):
    decoder_model.eval()
    model.train()

    def get_extra_param():
        return None

    perf = PerfTrackTrain(task, extra_param=get_extra_param())
    time_forward = 0
    time_backward = 0
    time_data_loading = 0

    train_fe_loss = 0.0
    train_cle_loss = 0.0

    time3 = time()
    for i, data_batch in enumerate(loader):
        time1 = time()
        batch_size = data_batch['pointcloud'].shape[0]
        mv_feature = data_batch['multiview']
        out, pc_feature = model(data_batch['pointcloud'])
        loss = get_loss(task, 'smooth', data_batch, out)

        if not cfg.no_pointcmt:
            mv2pc_logits = model(mvf=mv_feature, fc_only=True)
            pc_dec_pc = decoder_model(pc_feature)
            mv_dec_pc = decoder_model(mv_feature)
            gt_scaled, pr_scaled = scale(mv_dec_pc, pc_dec_pc)

            cleloss = F.kl_div(out['logit'].softmax(dim=1).log(), (mv2pc_logits['logit']).softmax(dim=1),
                               reduction='sum')
            loss += 0.3 * cleloss

            fe_loss, _ = EmdLoss(pr_scaled, gt_scaled, 0.05, 3000)
            fe_loss = torch.sqrt(fe_loss).mean(1).mean()
            loss += 30 * fe_loss
        else:
            cleloss = torch.Tensor([0])
            fe_loss = torch.Tensor([0])

        train_fe_loss += fe_loss.item() * batch_size
        train_cle_loss += cleloss.item() * batch_size

        perf.update_all(data_batch=data_batch, out=out, loss=loss)
        time2 = time()

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        time_data_loading += (time1 - time3)
        time_forward += (time2 - time1)
        time3 = time()
        time_backward += (time3 - time2)

        if i % 100 == 0:
            print(
                f"[{i}/{len(loader)}] avg_loss: {perf.agg_loss()}, FW time = {round(time_forward, 2)}, "
                f"BW time = {round(time_backward, 2)}, DL time = {round(time_data_loading, 2)}")

    print('Feature enhancement loss is ', train_fe_loss * 1.0 / 9840)
    print('Classifier enhencement loss is ', train_cle_loss * 1.0 / 9840)

    return perf.agg(), perf.agg_loss()


def save_checkpoint(id, epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg):
    model.cpu()
    path = f"./checkpoints/{cfg.exp_name}/model_{id}.pth"
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


def load_model_opt_sched(model, optimizer, lr_sched, bnm_sched, model_path):
    print(f'Recovering model and checkpoint from {model_path}')
    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint['model_state'])
    except:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state'])
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state'])
            model = model.module

    optimizer.load_state_dict(checkpoint['optimizer_state'])

    # for backward compatibility with saved models
    if 'lr_sched_state' in checkpoint:
        lr_sched.load_state_dict(checkpoint['lr_sched_state'])
        if checkpoint['bnm_sched_state'] is not None:
            bnm_sched.load_state_dict(checkpoint['bnm_sched_state'])
    else:
        print("WARNING: lr scheduler and bnm scheduler states are not loaded.")

    return model


def get_model(cfg):
    if cfg.model_name == 'pointnet2':
        model = models.PointNet2(
            num_class=cfg.num_class)
    else:
        raise NotImplementedError

    return model


def get_metric_from_perf(task, perf, metric_name):
    if task in ['cls', 'cls_trans']:
        assert metric_name in ['acc']
        metric = perf[metric_name]
    else:
        assert False
    return metric


def get_optimizer(params):
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=5e-2)
    lr_sched = lr_scheduler.CosineAnnealingLR(
        optimizer,
        1000,
        eta_min=0,
        last_epoch=-1)
    bnm_sched = None

    return optimizer, lr_sched, bnm_sched


def entry_train(cfg):
    dataset_train = ModelNet40_OfflineFeatures(cfg.data_root, split='train')
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=cfg.batch_size,
                                               num_workers=8, shuffle=True, drop_last=True,
                                               pin_memory=(torch.cuda.is_available()))
    loader_test = torch.utils.data.DataLoader(
        ModelNet40(
            data_path=cfg.data_root,
            partition='test',
        ),
        num_workers=8,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    model = get_model(cfg)
    model.to(DEVICE)
    model = nn.DataParallel(model)

    decoder_model = importlib.import_module('models.cmpg')
    decoder_model = decoder_model.get_model().to(DEVICE)
    decoder_model = nn.DataParallel(decoder_model)

    params = list(model.parameters())
    optimizer, lr_sched, bnm_sched = get_optimizer(params)

    deccheckpoint = torch.load(cfg.cmpg_checkpoint)
    decoder_model.load_state_dict(deccheckpoint['model_state'])
    decoder_model.eval()

    log_dir = f"./checkpoints/{cfg.exp_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    track_train = TrackTrain(early_stop_patience=1000)

    EmdLoss = emd_module.emdModule()
    print(str(model))
    for epoch in range(cfg.epochs):
        print(f'\nEpoch {epoch}')
        start = time()

        print('Training..')
        train_perf, train_loss = train(loader_train, model, decoder_model, optimizer, EmdLoss)
        pprint.pprint(train_perf, width=80)
        print('\nTesting..')
        test_perf = validate(loader_test, model)
        pprint.pprint(test_perf, width=80)
        track_train.record_epoch(
            epoch_id=epoch,
            train_metric=get_metric_from_perf('cls', train_perf, 'acc'),
            test_metric=get_metric_from_perf('cls', test_perf, 'acc'))

        if track_train.save_model(epoch, 'test'):
            print('Saving best model on the test set')
            save_checkpoint('best_test', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        if epoch % 25 == 0:
            save_checkpoint(f'{epoch}', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)

        lr_sched.step(epoch)

        end = time()
        last = end - start
        print('every epoch lasts for ', last)

    print('Saving the final model')
    save_checkpoint('final', epoch, model, optimizer, lr_sched, bnm_sched, test_perf, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='pointnet2_pointcmt', help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--data_root', type=str, default='dataset/ModelNet40/data/', help='Name of the data root')
    parser.add_argument('--model_name', type=str, default='pointnet2', help='Name of the model')
    parser.add_argument('--batch_size', type=int, default=32, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of episode to train ')
    parser.add_argument('--cmpg_checkpoint', type=str, default="pretrained/modelnet40/cmpg.pth", help='decoder model of multiview')
    parser.add_argument('--num_class', type=int, default=40)
    parser.add_argument('--no_pointcmt', default=False, action='store_true')
    cfg = parser.parse_args()

    print(cfg)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    entry_train(cfg)
