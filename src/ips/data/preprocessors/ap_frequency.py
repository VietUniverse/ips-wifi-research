"""Keep APs observed sufficiently often in the internal training split."""

import numpy as np

from ips.data.uji import AP_COLUMNS

PREPROCESS_NAME = "ap_frequency"


class APFrequency:
    def __init__(self, min_observations=20):
        if min_observations < 1:
            raise ValueError("min_observations must be positive")
        self.min_observations = int(min_observations)

    def fit(self, train_frame):
        observed = (train_frame[AP_COLUMNS].to_numpy() != 100).sum(axis=0)
        self.feature_columns = [column for column, count in zip(AP_COLUMNS, observed)
                                if count >= self.min_observations]
        if not self.feature_columns:
            raise ValueError("No AP meets min_observations on internal train")
        return self

    def transform(self, frame):
        x = frame[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        x[x == 100] = -110.0
        return (2.0 * (np.clip(x, -110.0, -30.0) + 110.0) / 80.0 - 1.0).astype(np.float32)

    def state_dict(self):
        return {"feature_columns": self.feature_columns,
                "min_observations": self.min_observations}


def build_preprocessor(min_observations=20):
    return APFrequency(min_observations)
