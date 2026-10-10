from __future__ import annotations
import torch
from torch import nn
from .bsrbf_kan import BSRBF_KANLayer

GRID_SIZE = 4
SPLINE_ORDER = 3
MIXER_BLOCKS = 2

class SmallKANMixerBlock(nn.Module):
    def __init__(self, tokens=16, channels=8):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_net = nn.Sequential(
            BSRBF_KANLayer(tokens, 32, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
            BSRBF_KANLayer(32, tokens, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER))
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_net = nn.Sequential(
            BSRBF_KANLayer(channels, 16, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER),
            BSRBF_KANLayer(16, channels, grid_size=GRID_SIZE, spline_order=SPLINE_ORDER))

    def forward(self, x):
        batch, tokens, channels = x.shape
        t = self.token_norm(x).transpose(1, 2).reshape(-1, tokens)
        x = x + self.token_net(t).reshape(batch, channels, tokens).transpose(1, 2)
        c = self.channel_norm(x).reshape(-1, channels)
        return x + self.channel_net(c).reshape(batch, tokens, channels)


class ResidualKANMixer(nn.Module):
    def __init__(self, n_blocks=MIXER_BLOCKS):
        super().__init__()
        self.rss_projection = nn.Sequential(nn.Linear(520, 64), nn.LayerNorm(64))
        self.blocks = nn.Sequential(*[SmallKANMixerBlock() for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64),
                                  nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, rssi, features):
        tokens = torch.cat([features, self.rss_projection(rssi)], dim=1).reshape(-1, 16, 8)
        return self.head(self.blocks(tokens).flatten(1))


