from __future__ import annotations
import torch
from torch import nn
from .kan_layers import make_kan_layer

GRID_SIZE = 4
SPLINE_ORDER = 3
MIXER_BLOCKS = 2

class SmallKANMixerBlock(nn.Module):
    def __init__(self, tokens=16, channels=8, family="bsrbf_kan"):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_net = nn.Sequential(
            make_kan_layer(family, tokens, 32),
            make_kan_layer(family, 32, tokens))
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_net = nn.Sequential(
            make_kan_layer(family, channels, 16),
            make_kan_layer(family, 16, channels))

    def forward(self, x):
        batch, tokens, channels = x.shape
        t = self.token_norm(x).transpose(1, 2).reshape(-1, tokens)
        x = x + self.token_net(t).reshape(batch, channels, tokens).transpose(1, 2)
        c = self.channel_norm(x).reshape(-1, channels)
        return x + self.channel_net(c).reshape(batch, tokens, channels)


class ResidualKANMixer(nn.Module):
    def __init__(self, n_blocks=MIXER_BLOCKS, family="bsrbf_kan"):
        super().__init__()
        self.rss_projection = nn.Sequential(nn.Linear(520, 64), nn.LayerNorm(64))
        self.blocks = nn.Sequential(*[SmallKANMixerBlock(family=family) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64),
                                  nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, rssi, features):
        tokens = torch.cat([features, self.rss_projection(rssi)], dim=1).reshape(-1, 16, 8)
        return self.head(self.blocks(tokens).flatten(1))


