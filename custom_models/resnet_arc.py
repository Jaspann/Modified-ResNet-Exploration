import torch
import torch.nn as nn
from torchvision import models

from custom_models.components.arc_face_head import ArcFaceHead


class ResNet18_ArcFace(nn.Module):
    def __init__(self, num_classes, num_channels):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        self.arcface = ArcFaceHead(512, num_classes)

    def forward(self, x, labels):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.arcface(x, labels)
        return x
