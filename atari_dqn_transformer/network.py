"""
network.py — Transformer Q-Network
=====================================
Replaces the CNN from the DQN project with a Vision Transformer (ViT).

How it works:
  1. Each frame (84×84) is divided into non-overlapping patches (7×7 = 12×12 patches)
  2. Each patch is linearly projected to a D_MODEL vector (patch embedding)
  3. FRAME_STACK frames × patches_per_frame = sequence of tokens
  4. Positional embeddings added so transformer knows where each patch came from
  5. Transformer encoder — self-attention across all patches across all frames
  6. Global average pool over all tokens → single vector
  7. FC head → Q-value per action

Input:  (batch, FRAME_STACK, 84, 84)   — same as DQN
Output: (batch, n_actions)             — Q-values, same as DQN
"""

import math

import torch
import torch.nn as nn

from config import FRAME_SIZE, FRAME_STACK, PATCH_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT


class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        assert FRAME_SIZE % PATCH_SIZE == 0, \
            f"FRAME_SIZE {FRAME_SIZE} must be divisible by PATCH_SIZE {PATCH_SIZE}"

        self.patches_per_side = FRAME_SIZE // PATCH_SIZE
        self.num_patches      = self.patches_per_side ** 2
        self.patch_dim        = PATCH_SIZE * PATCH_SIZE   # 49 pixels per patch

        self.projection = nn.Linear(self.patch_dim, D_MODEL)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        """
        frame: (batch, H, W)  — single grayscale frame
        returns: (batch, num_patches, D_MODEL)
        """
        B, H, W = frame.shape
        p = self.patches_per_side

        x = frame.view(B, p, PATCH_SIZE, p, PATCH_SIZE)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(B, self.num_patches, self.patch_dim)

        return self.projection(x)              # (B, num_patches, D_MODEL)


class TransformerQNetwork(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()

        self.patch_embed    = PatchEmbedding()
        patches_per_frame   = self.patch_embed.num_patches
        total_tokens        = FRAME_STACK * patches_per_frame

        self.pos_embedding  = nn.Parameter(
            torch.zeros(1, total_tokens, D_MODEL)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS, enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(D_MODEL)

        # Dueling architecture: separate value and advantage streams
        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, n_actions),
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, FRAME_STACK, 84, 84)  float32 in [0,1]
        returns: (batch, n_actions)  Q-values
        """
        B, T, H, W = x.shape

        frames  = x.view(B * T, H, W)
        patches = self.patch_embed(frames)         # (B*T, num_patches, D_MODEL)
        patches = patches.view(B, T * self.patch_embed.num_patches, D_MODEL)

        tokens  = patches + self.pos_embedding     # (B, total_tokens, D_MODEL)
        encoded = self.transformer(tokens)         # (B, total_tokens, D_MODEL)
        pooled  = self.norm(encoded.mean(dim=1))   # (B, D_MODEL)

        # Dueling Q = V(s) + A(s,a) - mean(A(s,·))
        value     = self.value_head(pooled)                          # (B, 1)
        advantage = self.advantage_head(pooled)                      # (B, n_actions)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
