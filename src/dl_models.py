"""1D CNN for biosignal stress classification.

`BiosignalCNN1D` is a 5-block 1D conv stack with decreasing kernel sizes (11
-> 3), ending in adaptive average pool + 2-layer MLP head. ~250 k parameters
— deliberately small for the ~860-window dataset.

The architecture mirrors the standard WESAD 1D-CNN recipe (Bobade & Vani 2020,
Lai et al. 2023, and the Kaggle "stress-detection-with-1d-cnn-99-accuracy"
reference) but is sized down for our smaller dataset and uses
`AdaptiveAvgPool1d(1)` so window length can change later without
re-architecting.
"""
from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class BiosignalCNN1D(nn.Module):
    """3-channel (ECG / Resp / Temp) 1D CNN classifier.

    Input shape: (batch, 3, 7500). Output: (batch, n_classes) logits.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            _ConvBlock(in_channels, 64, kernel=11, pool=2),    # (B,  64, 3750)
            _ConvBlock(64,         128, kernel=7,  pool=2),    # (B, 128, 1875)
            _ConvBlock(128,        256, kernel=5,  pool=2),    # (B, 256,  937)
            _ConvBlock(256,        256, kernel=3,  pool=2),    # (B, 256,  468)
            _ConvBlock(256,        256, kernel=3,  pool=1),    # (B, 256,  468)
            nn.AdaptiveAvgPool1d(1),                            # (B, 256, 1)
            nn.Flatten(),                                       # (B, 256)
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature_extractor(x)
        return self.head(z)

    @torch.no_grad()
    def freeze_features(self) -> None:
        """Disable grads on the conv stack — used for the C-head transfer condition."""
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def unfreeze_features(self) -> None:
        for p in self.feature_extractor.parameters():
            p.requires_grad = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
