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
    def __init__(self, n_blocks=MIXER_BLOCKS, family="bsrbf_kan", rss_dim=520,
                 arch="legacy_16x8", ap_indices=None):
        super().__init__()
        self.arch = arch
        if arch == "legacy_16x8":
            self.rss_projection = nn.Sequential(nn.Linear(rss_dim, 64), nn.LayerNorm(64))
            self.blocks = nn.Sequential(*[SmallKANMixerBlock(family=family) for _ in range(n_blocks)])
            self.head = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64),
                                      nn.SiLU(), nn.Linear(64, 2))
        elif arch == "ap_token":
            # Keep each AP identity as a token. Two input channels describe its
            # normalized RSS and whether the AP was observed in this scan.
            indices = list(range(rss_dim)) if ap_indices is None else list(ap_indices)
            if len(indices) != rss_dim or not indices or min(indices) < 0:
                raise ValueError("ap_indices must contain one non-negative ID per RSS feature")
            self.register_buffer("ap_ids", torch.as_tensor(indices, dtype=torch.long))
            self.rss_encoder = nn.Linear(2, 8)
            self.ap_embedding = nn.Embedding(max(indices) + 1, 8)
            self.feature_context = nn.Linear(64, 8)
            self.blocks = nn.Sequential(*[
                SmallKANMixerBlock(tokens=rss_dim, channels=8, family=family)
                for _ in range(n_blocks)
            ])
            self.head = nn.Sequential(nn.LayerNorm(rss_dim * 8), nn.Linear(rss_dim * 8, 128),
                                      nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 2))
        else:
            raise ValueError(f"Unknown KAN-Mixer architecture: {arch}")
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, rssi, features):
        if self.arch == "ap_token":
            observed = (rssi > -0.999).to(rssi.dtype)
            rss_and_mask = torch.stack((rssi, observed), dim=-1)
            tokens = self.rss_encoder(rss_and_mask)
            tokens = tokens + self.ap_embedding(self.ap_ids).unsqueeze(0)
            tokens = tokens + self.feature_context(features).unsqueeze(1)
        else:
            tokens = torch.cat([features, self.rss_projection(rssi)], dim=1).reshape(-1, 16, 8)
        return self.head(self.blocks(tokens).flatten(1))


