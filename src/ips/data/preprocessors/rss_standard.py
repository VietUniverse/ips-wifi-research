"""Original UJI RSS mapping: 100 means absent and maps to -1."""

from ips.data.uji import AP_COLUMNS, preprocess_rss

PREPROCESS_NAME = "rss_standard"


class RSSStandard:
    def fit(self, train_frame):
        self.feature_columns = list(AP_COLUMNS)
        return self

    def transform(self, frame):
        return preprocess_rss(frame)

    def state_dict(self):
        return {"feature_columns": self.feature_columns}


def build_preprocessor():
    return RSSStandard()
