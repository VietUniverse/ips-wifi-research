"""Blend global RSS level with within-fingerprint relative signal strengths."""

import numpy as np

from ips.data.uji import AP_COLUMNS

PREPROCESS_NAME = "rss_global_relative"


class RSSGlobalRelative:
    """Partially remove phone-wide RSS offsets without discarding absolute level.

    ``relative_weight=0`` reproduces the repository's fixed global scaling.
    ``relative_weight=1`` uses RSS differences from the row median. Missing APs
    retain the repository sentinel (-1), which is outside the observed range.
    """

    def __init__(self, relative_weight=0.5, relative_scale_db=40.0):
        if not 0.0 <= relative_weight <= 1.0:
            raise ValueError("relative_weight must be between 0 and 1")
        if relative_scale_db <= 0:
            raise ValueError("relative_scale_db must be positive")
        self.relative_weight = float(relative_weight)
        self.relative_scale_db = float(relative_scale_db)
        self.feature_columns = list(AP_COLUMNS)

    def fit(self, train_frame):
        self.feature_columns = list(AP_COLUMNS)
        return self

    def transform(self, frame):
        raw = frame[self.feature_columns].to_numpy(dtype=np.float32, copy=True)
        observed = raw != 100
        safe = raw.copy()
        safe[~observed] = -110.0
        global_scaled = 2.0 * (np.clip(safe, -110.0, -30.0) + 110.0) / 80.0 - 1.0

        counts = observed.sum(axis=1)
        medians = np.zeros(len(raw), dtype=np.float32)
        valid = counts > 0
        if valid.any():
            medians[valid] = np.nanmedian(np.where(observed[valid], raw[valid], np.nan), axis=1)
        relative = np.clip((raw - medians[:, None]) / self.relative_scale_db, -1.0, 1.0) * 0.95
        mixed = ((1.0 - self.relative_weight) * global_scaled
                 + self.relative_weight * relative)
        out = np.clip(mixed, -0.95, 0.95).astype(np.float32)
        out[~observed] = -1.0
        return out

    def state_dict(self):
        return {"feature_columns": list(self.feature_columns),
                "relative_weight": self.relative_weight,
                "relative_scale_db": self.relative_scale_db}


def build_preprocessor(relative_weight=0.5, relative_scale_db=40.0):
    return RSSGlobalRelative(relative_weight, relative_scale_db)
