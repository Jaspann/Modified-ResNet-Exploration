import torch.nn as nn


class sSE_Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial_se = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = self.sigmoid(self.spatial_se(x))
        return x * se
