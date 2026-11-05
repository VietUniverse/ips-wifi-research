"""skan model plugin."""
from .kan_layers import KANBackbone

MODEL_NAME = "skan"

def build_model(input_dim: int):
    return KANBackbone(input_dim, "skan")
