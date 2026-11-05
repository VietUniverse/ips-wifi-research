"""KAN layer families used in the KSE WiFi model comparison.

The variants share a common interface so the experiment runner and residual
Mixer can use the same layer family for the backbone and token/channel blocks.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .bsrbf_kan import BSRBF_KANLayer


class SplineKANLayer(nn.Module):
    """Original KAN-style learnable B-spline edge functions."""
    def __init__(self, input_dim, output_dim, grid_size=4, spline_order=3):
        super().__init__()
        self.input_dim = input_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.norm = nn.LayerNorm(input_dim)
        self.basis_count = grid_size + spline_order
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim * self.basis_count))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        step = 3.0 / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1) * step - 1.5
        self.register_buffer("grid", knots.expand(input_dim, -1).contiguous())

    def bases(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases.contiguous()

    def forward(self, x):
        x = self.norm(x)
        return F.linear(self.bases(x).flatten(1), self.weight)


class FastKANLayer(nn.Module):
    """FastKAN-style Gaussian RBF approximation to spline bases."""
    def __init__(self, input_dim, output_dim, num_centers=8):
        super().__init__()
        self.input_dim = input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.register_buffer("centers", torch.linspace(-1.5, 1.5, num_centers))
        self.width = 3.0 / (num_centers - 1)
        self.weight = nn.Parameter(torch.empty(output_dim, input_dim * num_centers))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x):
        x = self.norm(x)
        rbf = torch.exp(-((x.unsqueeze(-1) - self.centers) / self.width) ** 2)
        return F.linear(rbf.flatten(1), self.weight)


class SKANLayer(nn.Module):
    """Selectable function spaces with a learned gate per output node."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.spline = SplineKANLayer(input_dim, output_dim)
        self.fast = FastKANLayer(input_dim, output_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.base_weight = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.kaiming_uniform_(self.base_weight, a=5 ** 0.5)
        self.gate = nn.Parameter(torch.zeros(output_dim, 3))

    def forward(self, x):
        candidates = torch.stack((self.spline(x), self.fast(x),
                                  F.linear(F.silu(self.norm(x)), self.base_weight)), dim=-1)
        weights = self.gate.softmax(dim=-1)
        return (candidates * weights.unsqueeze(0)).sum(dim=-1)


class FCKANLayer(nn.Module):
    """Spline KAN with low-dimensional product interactions as in FC-KAN."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.spline = SplineKANLayer(input_dim, output_dim)
        # The paper applies function combinations in low-dimensional layers to
        # control cost. Here, adjacent pairs among up to 64 hidden channels form
        # the multiplicative basis terms.
        self.pair_count = min(32, input_dim // 2) if input_dim <= 64 else 0
        self.interaction = (nn.Linear(self.pair_count, output_dim, bias=False)
                            if self.pair_count else None)

    def forward(self, x):
        result = self.spline(x)
        if self.interaction is not None:
            pairs = x[:, :2 * self.pair_count].reshape(x.shape[0], self.pair_count, 2)
            result = result + self.interaction(pairs[:, :, 0] * pairs[:, :, 1])
        return result


LAYER_BUILDERS = {
    "kan": lambda i, o: SplineKANLayer(i, o),
    "fastkan": lambda i, o: FastKANLayer(i, o),
    "bsrbf_kan": lambda i, o: BSRBF_KANLayer(i, o, grid_size=4, spline_order=3),
    "skan": lambda i, o: SKANLayer(i, o),
    "fckan": lambda i, o: FCKANLayer(i, o),
}


def make_kan_layer(name, input_dim, output_dim):
    try:
        return LAYER_BUILDERS[name](input_dim, output_dim)
    except KeyError:
        raise ValueError(f"Unknown KAN layer family: {name}") from None


class KANBackbone(nn.Module):
    def __init__(self, input_dim, family, hidden=(512, 512, 256, 256, 128, 128, 64, 64, 2)):
        super().__init__()
        dims = (input_dim, *hidden)
        self.family = family
        self.layers = nn.ModuleList(make_kan_layer(family, i, o)
                                    for i, o in zip(dims, dims[1:]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
