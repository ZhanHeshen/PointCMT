import torch.nn as nn
from pointnet2.models.pointnet2_msg_cls import Pointnet2MSG


class PointNet2(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.model = Pointnet2MSG(num_classes=num_class, input_channels=0, use_xyz=True)

    def forward(self, pc=None, fc_only=False, mvf=0):
        if fc_only:
            logit = self.model(mvf, fc_only)
            return {'logit': logit}

        logit, pc_feat = self.model(pc)
        out = {'logit': logit}

        return out, pc_feat
