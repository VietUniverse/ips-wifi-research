from .base import BaseLocalizer
from .formulas import Formulas

MODEL_NAME = "knn"


class KNN(BaseLocalizer):
    display_name, color = "kNN", "#DD8452"

    def weights(self, dists, mask):
        return Formulas.uniform_weights(mask)


def build_model(**params):
    return KNN()
