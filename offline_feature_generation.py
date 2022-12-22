import os
import torch
import importlib
import argparse

from tqdm import tqdm
from data.modelnet40_mv_loader import ModelNet40
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='dataset/ModelNet40/data/',  help='Name of the data root')
parser.add_argument('--model_path', type=str, default='./checkpoint/mvcnn_default/models/model.t7', help='Pretrained model path')
parser.add_argument('--mv_backbone', type=str, default="resnet18")
parser.add_argument('--num_class', type=int, default=40)
parser.add_argument('--pretraining', type=bool, default=False)
cfg = parser.parse_args()

def generate_data(split):
    loader = DataLoader(
        ModelNet40(
            data_path=cfg.data_root,
            partition=split,
            generate=True,
        ),
        num_workers=8,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        pin_memory=True)

    features = []
    for i, data_batch in tqdm(enumerate(loader), total=(len(loader))):
        data_pc = data_batch['pointcloud']
        data_label = data_batch['label']
        _, mvf = model(data_batch)
        for j in range(data_pc.shape[0]):
            features.append((data_pc[j], data_label[j], mvf[j].detach().cpu()))
    save_path = os.path.join(cfg.data_root, 'modelnet40_%s_mvf.pth' % split)
    torch.save(features, save_path)


if __name__ == '__main__':
    model = importlib.import_module('models.mvcnn')
    model = model.get_model(cfg).to(DEVICE)
    pn_param = torch.load(cfg.model_path)
    for i in list(pn_param.keys()):
        j = i[7:]
        pn_param[j] = pn_param.pop(i)
    model.load_state_dict(pn_param)
    model.eval()

    generate_data('train')
