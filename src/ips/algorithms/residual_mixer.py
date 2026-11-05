from __future__ import annotations
import torch
from torch import nn
from .kan_layers import make_kan_layer

GRID_SIZE = 4
SPLINE_ORDER = 3
MIXER_BLOCKS = 2

def mixing_network(kind, family, width, hidden):
    """Build one axis mixer; KAN keeps the original state-dict layout."""
    if kind == "kan":
        return nn.Sequential(make_kan_layer(family, width, hidden),
                             make_kan_layer(family, hidden, width))
    if kind == "mlp":
        return nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                             nn.Linear(hidden, width))
    raise ValueError(f"Mixer type must be 'kan' or 'mlp', got {kind!r}")


class SmallKANMixerBlock(nn.Module):
    def __init__(self, tokens=16, channels=8, family="bsrbf_kan",
                 token_type="kan", channel_type="kan"):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_net = mixing_network(token_type, family, tokens, 32)
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_net = mixing_network(channel_type, family, channels, 16)

    def forward(self, x):
        batch, tokens, channels = x.shape
        t = self.token_norm(x).transpose(1, 2).reshape(-1, tokens)
        x = x + self.token_net(t).reshape(batch, channels, tokens).transpose(1, 2)
        c = self.channel_norm(x).reshape(-1, channels)
        return x + self.channel_net(c).reshape(batch, tokens, channels)


class ResidualKANMixer(nn.Module):
    def __init__(self, n_blocks=MIXER_BLOCKS, family="bsrbf_kan", rss_dim=520,
                 arch="legacy_16x8", ap_indices=None,
                 token_type="kan", channel_type="kan", n_spatial_clusters=0):
        super().__init__()
        self.arch = arch
        self.n_spatial_clusters = int(n_spatial_clusters)
        if self.n_spatial_clusters < 0:
            raise ValueError("n_spatial_clusters must be non-negative")
        if self.n_spatial_clusters and arch != "ap_token":
            raise ValueError("spatial-cluster supervision requires arch='ap_token'")
        if arch == "legacy_16x8":
            self.rss_projection = nn.Sequential(nn.Linear(rss_dim, 64), nn.LayerNorm(64))
            self.blocks = nn.Sequential(*[SmallKANMixerBlock(
                family=family, token_type=token_type, channel_type=channel_type)
                for _ in range(n_blocks)])
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
                SmallKANMixerBlock(tokens=rss_dim, channels=8, family=family,
                                   token_type=token_type, channel_type=channel_type)
                for _ in range(n_blocks)
            ])
            head_input = rss_dim * 8 + self.n_spatial_clusters
            self.head = nn.Sequential(nn.LayerNorm(head_input), nn.Linear(head_input, 128),
                                      nn.SiLU(), nn.Dropout(0.1), nn.Linear(128, 2))
            if self.n_spatial_clusters:
                # The cluster task teaches the AP-token representation to retain
                # coarse spatial information. Its soft probabilities also
                # condition the coordinate correction.
                self.spatial_head = nn.Sequential(nn.Linear(72, 32), nn.GELU(),
                                                  nn.Linear(32, self.n_spatial_clusters))
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
        mixed = self.blocks(tokens)
        flat = mixed.flatten(1)
        if self.n_spatial_clusters:
            context = torch.cat((features, mixed.mean(dim=1)), dim=1)
            cluster_logits = self.spatial_head(context)
            correction = self.head(torch.cat((flat, cluster_logits.softmax(dim=1)), dim=1))
            return correction, cluster_logits
        return self.head(flat)


