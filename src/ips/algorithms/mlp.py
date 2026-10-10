"""Simple MLP baseline."""

from torch import nn

MODEL_NAME = "mlp"


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.layers(x)


def build_model(input_dim: int) -> nn.Module:
    return MLP(input_dim)
