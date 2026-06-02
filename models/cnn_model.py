from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18


def stabilize_torch_backend() -> None:
    """Work around an intermittent native MaxPool2d access violation on Windows.

    Some Windows PyTorch builds crash (segfault) inside the MKL-DNN max-pool
    kernel during repeated forward passes. Disabling the MKL-DNN backend forces
    the reference kernel, which is slower but stable. Safe to call repeatedly.
    """
    try:
        torch.backends.mkldnn.enabled = False
    except Exception:  # pragma: no cover - backend not always present
        pass


class CricketCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def create_resnet18_model(num_classes: int, dropout: float) -> nn.Module:
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(
        1,
        model.conv1.out_channels,
        kernel_size=model.conv1.kernel_size,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        bias=False,
    )
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model
