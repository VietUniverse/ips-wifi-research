"""fckan model plugin."""
from .kan_layers import KANBackbone

MODEL_NAME = "fckan"

def build_model(input_dim: int):
    return KANBackbone(input_dim, "fckan")
