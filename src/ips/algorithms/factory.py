"""Small registry shared by the experiment runner and CLI."""

from torch import nn

from .bsrbf_kan import BSRBF_KAN


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.layers(x)


def create_model(name: str, input_dim: int) -> nn.Module:
    if name == "mlp":
        return MLP(input_dim)
    if name == "bsrbf_kan":
        return BSRBF_KAN(
            [input_dim, 512, 512, 256, 256, 128, 128, 64, 64, 2],
            grid_size=4,
            spline_order=3,
        )
    raise ValueError(f"Unknown model: {name!r}. Choose mlp or bsrbf_kan.")
