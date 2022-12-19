#!/usr/bin/env python
# encoding: utf-8
'''
@author: Xu Yan
@file: mvcnn.py
@time: 2021/4/2 16:57
'''
import torch.nn as nn
import torchvision.models as models


def drop_last_layer(model):
    new_model = nn.Sequential(*list(model.children())[:-1])
    return new_model


class get_model(nn.Module):

    def __init__(self, args):
        super(get_model, self).__init__()

        self.nclasses = args.num_class
        self.pretraining = args.pretraining
        self.cnn_name = args.mv_backbone
        self.use_resnet = self.cnn_name.startswith('resnet')

        if self.use_resnet:
            if self.cnn_name == 'resnet18':
                self.net = models.resnet18(pretrained=self.pretraining)
                self.net = drop_last_layer(self.net)
                self.fc = nn.Sequential(nn.Linear(512, self.nclasses))
            elif self.cnn_name == 'resnet34':
                self.net = models.resnet34(pretrained=self.pretraining)
                self.net = drop_last_layer(self.net)
                self.fc = nn.Sequential(nn.Linear(512, self.nclasses))
            elif self.cnn_name == 'resnet50':
                self.net = models.resnet50(pretrained=self.pretraining)
                self.net = drop_last_layer(self.net)
                self.fc = nn.Sequential(nn.Linear(2048, self.nclasses))
        else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier
                self.net_2 = drop_last_layer(self.net_2)
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier
                self.net_2 = drop_last_layer(self.net_2)
            elif self.cnn_name == 'vgg11_bn':
                self.net_1 = models.vgg11_bn(pretrained=self.pretraining).features
                self.net_2 = models.vgg11_bn(pretrained=self.pretraining).classifier
                self.net_2 = drop_last_layer(self.net_2)
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier
                self.net_2 = drop_last_layer(self.net_2)

            self.fc = nn.Sequential(nn.Linear(512, self.nclasses))

    def forward(self, data_dict, fc_only=False):
        if fc_only:
            pc_feature = data_dict
            logit = self.fc(pc_feature)
            out = {'logit': logit}
            return out

        x = data_dict['multiview'].float()  # [B, V, C, W, H]
        b, v = x.shape[:2]
        x = x.flatten(0, 1)  # [B*V, C, W, H]
        x = x.cuda()

        if self.use_resnet:
            x = self.net(x)
        else:
            y = self.net_1(x)
            x = self.net_2(y.reshape(y.shape[0], -1))
        x = x.view(b, -1)
        x = x.view(b, v, -1)  # [B, V, D]
        x = x.max(1)[0]  # [B, D]
        logit = self.fc(x)  # [B, C]
        out = {'logit': logit}

        return out, x
