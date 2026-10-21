"""Direct RSS-to-position KAN-Mixer, with trainable token and channel mixing."""
import torch
from torch import nn

from .residual_mixer import mixing_network

MODEL_NAME = "kan_mixer_e2e"


class KANMixerBlock(nn.Module):
    def __init__(self, tokens=32, channels=16, dropout=.1):
        super().__init__()
        self.token_norm = nn.LayerNorm(channels)
        self.token_mixer = mixing_network("kan", "bsrbf_kan", tokens, tokens * 2)
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mixer = mixing_network("kan", "bsrbf_kan", channels, channels * 2)
        self.token_scale = nn.Parameter(torch.full((channels,), .1))
        self.channel_scale = nn.Parameter(torch.full((channels,), .1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, channels = x.shape
        token_input = self.token_norm(x).transpose(1, 2).reshape(batch * channels, tokens)
        update = self.token_mixer(token_input).reshape(batch, channels, tokens).transpose(1, 2)
        x = x + self.dropout(update) * self.token_scale
        update = self.channel_mixer(self.channel_norm(x).reshape(batch * tokens, channels))
        return x + self.dropout(update.reshape(batch, tokens, channels)) * self.channel_scale


class DirectKANMixer(nn.Module):
    """One jointly trained regressor; no separately frozen coordinate backbone.

    Three input views retain RSS level, per-scan relative RSS and AP presence.
    A learned projection compresses AP identities into latent tokens; no AP
    adjacency or physical ordering is assumed. Both mixing axes use BSRBF-KAN.
    """

    def __init__(self, input_dim, tokens=32, channels=16, blocks=2):
        super().__init__()
        self.tokens, self.channels = tokens, channels
        self.embedding = nn.Linear(input_dim * 3, tokens * channels)
        self.input_norm = nn.LayerNorm(channels)
        self.blocks = nn.Sequential(*[KANMixerBlock(tokens, channels) for _ in range(blocks)])
        self.output_norm = nn.LayerNorm(channels * 2)
        self.head = nn.Sequential(nn.Linear(channels * 2, 64), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(64, 2))

    @staticmethod
    def input_views(rss):
        observed = (rss > -.999).to(rss.dtype)
        values = rss * observed
        mean = values.sum(dim=1, keepdim=True) / observed.sum(dim=1, keepdim=True).clamp_min(1)
        relative = (rss - mean) * observed
        return torch.cat((values, relative, observed), dim=1)

    def forward(self, rss):
        tokens = self.embedding(self.input_views(rss)).reshape(-1, self.tokens, self.channels)
        mixed = self.blocks(self.input_norm(tokens))
        representation = torch.cat((mixed.mean(dim=1), mixed.amax(dim=1)), dim=1)
        return self.head(self.output_norm(representation))


def build_model(input_dim: int):
    return DirectKANMixer(input_dim)
