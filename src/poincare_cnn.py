"""2D CNN for Poincare-image stress classification.

Architecture (exactly as specified):

    Input: 64x64x1
    Conv2D(16, 3x3, padding=same) - BatchNorm - ReLU - MaxPool2D(2x2)
    Conv2D(32, 3x3, padding=same) - BatchNorm - ReLU - MaxPool2D(2x2)
    Conv2D(64, 3x3, padding=same) - BatchNorm - ReLU - GlobalAveragePooling2D
    Dense(64) - ReLU - Dropout(0.3)
    Output
"""
from __future__ import annotations

import torch
from torch import nn


class PoincareCNN(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),   # same
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                          # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                                  # GAP -> (B,64,1,1)
            nn.Flatten(),                                             # (B, 64)
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
