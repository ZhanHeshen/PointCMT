import torch.nn as nn


class get_model(nn.Module):
    def __init__(self, num_points=1024):
        super(get_model, self).__init__()
        self.num_points = num_points
        self.fc_l = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 3 * self.num_points, bias=False),
        )

    def forward(self, mv_feature):
        x = mv_feature  # [B, V, C, W, H]
        b, v = x.shape[:2]

        x = self.fc_l(x)

        return x.reshape(b, self.num_points, 3)
