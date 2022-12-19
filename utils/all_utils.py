import tensorboardX
import torch
import torch.nn.functional as F
import numpy as np
import sys


# Additional information that might be necessary to get the model
DATASET_NUM_CLASS = {
    'modelnet40': 40,
    'modelnet40_rscnn': 40,
    'modelnet40_pn2': 40,
    'modelnet40_dgcnn': 40,
    'modelnet10': 10,
}

class TensorboardManager:
    def __init__(self, path):
        self.writer = tensorboardX.SummaryWriter(path)

    def update(self, split, step, vals):
        for k, v in vals.items():
            self.writer.add_scalar('%s_%s' % (split, k), v, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


class TrackTrain:
    def __init__(self, early_stop_patience):
        self.early_stop_patience = early_stop_patience
        self.counter = -1
        self.best_epoch_val = -1
        self.best_epoch_train = -1
        self.best_epoch_test = -1
        self.best_test = float("-inf")
        self.best_train = float("-inf")
        self.test_best_val = float("-inf")

    def record_epoch(self, epoch_id, train_metric, test_metric):
        assert epoch_id == (self.counter + 1)
        self.counter += 1

        if test_metric >= self.best_test:
            self.best_test = test_metric
            self.best_epoch_test = epoch_id

        if train_metric >= self.best_train:
            self.best_train = train_metric
            self.best_epoch_train = epoch_id

        print('best test acc is: %f' % self.best_test)
        print('best test acc is achieved in: %f' % self.best_epoch_test)


    def save_model(self, epoch_id, split):
        """
        Whether to save the current model or not
        :param epoch_id:
        :param split:
        :return:
        """
        assert epoch_id == self.counter
        if split == 'val':
            if self.best_epoch_val == epoch_id:
                _save_model = True
            else:
                _save_model = False
        elif split == 'test':
            if self.best_epoch_test == epoch_id:
                _save_model = True
            else:
                _save_model = False
        elif split == 'train':
            print('best_epoch_train:%d'%self.best_epoch_train)
            if self.best_epoch_train == epoch_id:
                _save_model = True
            else:
                _save_model = False
        else:
            assert False

        return _save_model

    def early_stop(self, epoch_id):
        assert epoch_id == self.counter
        if (epoch_id - self.best_epoch_val) > self.early_stop_patience:
            return True
        else:
            return False


class PerfTrackVal:
    """
    Records epoch wise performance for validation
    """
    def __init__(self, task, extra_param=None):
        self.task = task
        if task in ['cls', 'cls_trans']:
            assert extra_param is None
            self.all = []
            self.class_seen = None
            self.class_corr = None
        else:
            assert False
    def update(self, data_batch, out):
        if self.task in ['cls', 'cls_trans']:
            correct = self.get_correct_list(out['logit'], data_batch['label'])
            self.all.extend(correct)
            self.update_class_see_corr(out['logit'], data_batch['label'])
        else:
            assert False
    def agg(self):
        if self.task in ['cls', 'cls_trans']:
            perf = {
                'acc': self.get_avg_list(self.all),
                'class_acc': np.mean(np.array(self.class_corr) / np.array(self.class_seen,dtype=np.float))
            }
        else:
            assert False
        return perf

    def update_class_see_corr(self, logit, label):
        if self.class_seen is None:
            num_class = logit.shape[1]
            self.class_seen = [0] * num_class
            self.class_corr = [0] * num_class

        pred_label = logit.argmax(axis=1).to('cpu').tolist()
        for _pred_label, _label in zip(pred_label, label):
            self.class_seen[_label] += 1
            if _pred_label == _label:
                self.class_corr[_pred_label] += 1

    @staticmethod
    def get_correct_list(logit, label):
        label = label.to(logit.device)
        pred_class = logit.argmax(axis=1)
        return (label == pred_class).to('cpu').tolist()
    @staticmethod
    def get_avg_list(all_list):
        for x in all_list:
            assert isinstance(x, bool)
        return sum(all_list) / len(all_list)


class PerfTrackTrain(PerfTrackVal):
    """
    Records epoch wise performance during training
    """
    def __init__(self, task, extra_param=None):
        super().__init__(task, extra_param)
        # add a list to track loss
        self.all_loss = []

    def update_loss(self, loss):
        self.all_loss.append(loss.item())

    def agg_loss(self):
        # print(self.all_loss)
        return sum(self.all_loss) / len(self.all_loss)

    def update_all(self, data_batch, out, loss):
        self.update(data_batch, out)
        self.update_loss(loss)


# source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
def smooth_loss(pred, gold):
    eps = 0.2

    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def cal_loss(pred, goal, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    goal = goal.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, goal.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, goal, reduction='mean')

    return loss

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass