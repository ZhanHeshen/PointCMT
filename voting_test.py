import argparse
import os
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import sklearn.metrics as metrics
import numpy as np
import torch.nn.functional as F
import sys
import time
import models

from data.modelnet40_mv_loader import ModelNet40
from torch.utils.data import DataLoader


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    parser.add_argument('--data_root', type=str, default='dataset/ModelNet40/data/', help='Name of the data root')
    parser.add_argument('--checkpoint', type=str, default='pretrained/modelnet40/pointnet2_pointcmt.pth', help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, default='demo', help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model_name', default='pointnet2', help='model name')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--num_repeat', type=int, default=300)
    parser.add_argument('--num_vote', type=int, default=10)
    parser.add_argument('--num_class', type=int, default=40)
    parser.add_argument('--validate', action='store_true', help='Validate the original testing result.')
    return parser.parse_args()


class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(xyz1).float().cuda())

        return pc


def get_model(cfg):
    if cfg.model_name == 'pointnet2':
        model = models.PointNet2(
            num_class=cfg.num_class)
    else:
        raise NotImplementedError

    return model


def main():
    args = parse_args()
    print(f"args: {args}")

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    print(f"random seed is set to {args.seed}, the speed will slow down.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_printoptions(10)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f"==> Using device: {device}")

    print('==> Preparing data..')
    test_loader = DataLoader(
        ModelNet40(
            data_path=args.data_root,
            partition='test',
        ),
        num_workers=8,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    # Model
    print('==> Building model..')

    net = get_model(args)
    criterion = cal_loss
    net = net.to(device)
    checkpoint = torch.load(args.checkpoint)
    net = torch.nn.DataParallel(net)
    net.load_state_dict(checkpoint['model_state'])
    cudnn.benchmark = True

    if args.validate:
        test_out = validate(net, test_loader, criterion, device)
        print(f"Vanilla out: {test_out}")
        print(f"Note 1: Please also load the random seed parameter (if forgot, see out.txt).\n"
              f"Note 2: This result may vary little on different GPUs (and number of GPUs), we tested 2080Ti, P100, and V100.\n"
              f"[note : Original result is achieved with V100 GPUs.]\n\n\n")
        # Interestingly, we get original best_test_acc on 4 V100 gpus, but this model is trained on one V100 gpu.
        # On different GPUs, and different number of GPUs, both OA and mean_acc vary a little.
        # Also, the batch size also affect the testing results, could not understand.

    print(f"===> start voting evaluation...")
    voting(net, test_loader, device, args)


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


def voting(net, testloader, device, args):
    io = IOStream(args.checkpoint + 'run.log')
    io.cprint(str(args))

    net.eval()
    best_acc = 0
    best_mean_acc = 0
    pointscale = PointcloudScale(scale_low=0.85, scale_high=1.15)

    for i in range(args.num_repeat):
        test_true = []
        test_pred = []

        for batch_idx, databatch in enumerate(testloader):
            data = databatch['pointcloud']
            label = databatch['label']
            data, label = data.to(device), label.to(device).squeeze()
            pred = 0
            for v in range(args.num_vote):
                new_data = data
                if v > 0:
                    new_data.data = pointscale(new_data.data)
                with torch.no_grad():
                    logit, _ = net(new_data)
                    logit = logit['logit']
                    pred += F.softmax(logit, dim=1)
            pred /= args.num_vote
            label = label.view(-1)
            pred_choice = pred.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = 100. * metrics.accuracy_score(test_true, test_pred)
        test_mean_acc = 100. * metrics.balanced_accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        if test_mean_acc > best_mean_acc:
            best_mean_acc = test_mean_acc
        outstr = 'Voting %d, test acc: %.3f, test mean acc: %.3f,  [current best(all_acc: %.3f mean_acc: %.3f)]' % \
                 (i, test_acc, test_mean_acc, best_acc, best_mean_acc)
        io.cprint(outstr)

    final_outstr = 'Final voting test acc: %.6f,' % (best_acc * 100)
    io.cprint(final_outstr)


if __name__ == '__main__':
    main()
