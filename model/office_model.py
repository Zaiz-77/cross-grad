import torch.nn as nn
from torchvision import models


class OfficeModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        num_last = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_last, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)
