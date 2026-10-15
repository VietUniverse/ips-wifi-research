"""Per-fingerprint RSS ranks, invariant to monotone device calibration shifts."""

import numpy as np
from scipy.stats import rankdata

from ips.data.uji import AP_COLUMNS

PREPROCESS_NAME = "rss_row_rank"


class RSSRowRank:
    def __init__(self):
        self.feature_columns = list(AP_COLUMNS)

    def fit(self, train_frame):
        self.feature_columns = list(AP_COLUMNS)
        return self

    def transform(self, frame):
        raw = frame[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        observed = raw != 100
        out = np.full(raw.shape, -1.0, dtype=np.float32)
        for row_index, row_mask in enumerate(observed):
            indices = np.flatnonzero(row_mask)
            count = len(indices)
            if count == 1:
                out[row_index, indices[0]] = 0.0
            elif count > 1:
                ranks = rankdata(raw[row_index, indices], method="average")
                out[row_index, indices] = (1.9 * (ranks - 1.0) / (count - 1.0) - 0.95)
        return out

    def state_dict(self):
        return {"feature_columns": list(self.feature_columns)}


def build_preprocessor():
    return RSSRowRank()
