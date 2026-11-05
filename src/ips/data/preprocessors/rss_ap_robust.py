"""Train-only robust quantile scaling for each access point."""

import numpy as np

from ips.data.uji import AP_COLUMNS

PREPROCESS_NAME = "rss_ap_robust"


class RSSAPRobust:
    def __init__(self, min_observations=30, lower_quantile=10.0, upper_quantile=90.0,
                 min_unique_values=3, min_std_db=1.0, mask_low_information=True):
        if min_observations < 1:
            raise ValueError("min_observations must be positive")
        if not 0 <= lower_quantile < upper_quantile <= 100:
            raise ValueError("quantiles must satisfy 0 <= lower < upper <= 100")
        self.min_observations = int(min_observations)
        self.lower_quantile = float(lower_quantile)
        self.upper_quantile = float(upper_quantile)
        self.min_unique_values = int(min_unique_values)
        self.min_std_db = float(min_std_db)
        self.mask_low_information = bool(mask_low_information)
        if self.min_unique_values < 1 or self.min_std_db < 0:
            raise ValueError("min_unique_values must be positive and min_std_db non-negative")
        self.feature_columns = list(AP_COLUMNS)

    def fit(self, train_frame):
        raw = train_frame[AP_COLUMNS].to_numpy(dtype=np.float32, copy=True)
        observed = raw != 100
        pooled = raw[observed]
        if pooled.size == 0:
            raise ValueError("No detected RSS values in training data")
        fallback_low, fallback_high = np.percentile(
            pooled, [self.lower_quantile, self.upper_quantile]
        )
        if fallback_high - fallback_low < 1.0:
            fallback_low, fallback_high = -104.0, -30.0

        self.lower = np.full(len(AP_COLUMNS), fallback_low, dtype=np.float32)
        self.upper = np.full(len(AP_COLUMNS), fallback_high, dtype=np.float32)
        self.active = np.ones(len(AP_COLUMNS), dtype=bool)
        for column_index in range(len(AP_COLUMNS)):
            values = raw[observed[:, column_index], column_index]
            informative = (len(values) >= self.min_observations
                           and np.unique(values).size >= self.min_unique_values
                           and (len(values) < 2 or float(np.std(values)) >= self.min_std_db))
            if self.mask_low_information and not informative:
                self.active[column_index] = False
            if len(values) >= self.min_observations:
                low, high = np.percentile(values, [self.lower_quantile, self.upper_quantile])
                if high - low >= 1.0:
                    self.lower[column_index] = low
                    self.upper[column_index] = high
        self.feature_columns = list(AP_COLUMNS)
        return self

    def transform(self, frame):
        raw = frame[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        observed = raw != 100
        scale = np.maximum(self.upper - self.lower, 1.0)
        out = 1.9 * (raw - self.lower) / scale - 0.95
        out = np.clip(out, -0.95, 0.95).astype(np.float32)
        out[~observed | ~self.active[None, :]] = -1.0
        return out

    def state_dict(self):
        return {"feature_columns": list(self.feature_columns),
                "lower": self.lower.tolist(), "upper": self.upper.tolist(),
                "active": self.active.tolist(),
                "min_observations": self.min_observations,
                "min_unique_values": self.min_unique_values,
                "min_std_db": self.min_std_db,
                "mask_low_information": self.mask_low_information,
                "lower_quantile": self.lower_quantile,
                "upper_quantile": self.upper_quantile}


def build_preprocessor(min_observations=30, lower_quantile=10.0, upper_quantile=90.0,
                       min_unique_values=3, min_std_db=1.0, mask_low_information=True):
    return RSSAPRobust(min_observations, lower_quantile, upper_quantile,
                       min_unique_values, min_std_db, mask_low_information)
