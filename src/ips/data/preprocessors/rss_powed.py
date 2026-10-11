"""Biểu diễn 'powed' (Torres-Sospedra et al.): (RSS - min + 1)^beta / (-(min - 1))^beta.
AP không bắt được trả về -1 (quy ước repo)."""
import numpy as np
from ips.neighbors.formulas import Formulas

PREPROCESS_NAME = "rss_powed"


class RssPowed:
    def __init__(self, min_rss=-104, beta=float(np.e), not_detected=100):
        self.min_rss, self.beta, self.not_detected = min_rss, beta, not_detected
        self.feature_columns = []

    def fit(self, train_frame):
        self.feature_columns = [c for c in train_frame.columns if str(c).startswith("WAP")]
        return self

    def transform(self, frame):
        raw = frame[self.feature_columns].to_numpy()
        out = Formulas.to_powed(raw, self.min_rss, self.beta, self.not_detected)
        out[raw == self.not_detected] = -1.0
        return out.astype(np.float32)

    def state_dict(self):
        return dict(min_rss=self.min_rss, beta=self.beta, not_detected=self.not_detected,
                    feature_columns=list(self.feature_columns))


def build_preprocessor(**params):
    return RssPowed(**params)
