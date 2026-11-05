from .base import InversePowerWKNN

MODEL_NAME = "wknn_inv_d2"


class WKNNInvD2(InversePowerWKNN):
    display_name, color, power = "WKNN (1/d²)", "#55A868", 2


def build_model(**params):
    return WKNNInvD2()
