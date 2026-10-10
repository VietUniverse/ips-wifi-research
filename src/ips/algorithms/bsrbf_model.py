"""Factory entry for the BSRBF-KAN backbone."""

from .bsrbf_kan import BSRBF_KAN

MODEL_NAME = "bsrbf_kan"


def build_model(input_dim: int):
    return BSRBF_KAN(
        [input_dim, 512, 512, 256, 256, 128, 128, 64, 64, 2],
        grid_size=4,
        spline_order=3,
    )
