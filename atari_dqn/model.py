"""
model.py — DQN network (DeepMind Nature 2015)
==============================================
Input:  (batch, 4, 84, 84)   — 4 stacked grayscale frames
Output: (batch, n_actions)   — Q-value for every possible action

Architecture:
    Conv 8×8 stride 4 → 32 filters  → ReLU
    Conv 4×4 stride 2 → 64 filters  → ReLU
    Conv 3×3 stride 1 → 64 filters  → ReLU
    Flatten
    FC 512                           → ReLU
    FC n_actions                     → Q-values (no activation)
"""

import numpy as np
import torch
import torch.nn as nn

FRAME_SIZE  = 84
FRAME_STACK = 4


class DQN(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(FRAME_STACK, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out = self._conv_out_size()

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 4, 84, 84) float32, values in [0, 1]
        returns: (batch, n_actions) Q-values
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)   # flatten conv output
        return self.fc(x)

    def _conv_out_size(self) -> int:
        dummy = torch.zeros(1, FRAME_STACK, FRAME_SIZE, FRAME_SIZE)
        return int(np.prod(self.conv(dummy).shape))
