from .base import InversePowerWKNN

MODEL_NAME = "wknn_inv_d"


class WKNNInvD(InversePowerWKNN):
    display_name, color, power = "WKNN (1/d)", "#4C72B0", 1


def build_model(**params):
    return WKNNInvD()
