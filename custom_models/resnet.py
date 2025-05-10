import torch.nn as nn
from torchvision import models


def get_standard_resnet18(num_classes, num_channels):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
