"""Giả lập API dữ liệu của repo (load_csvs / make_building_data) để test họ kNN không cần tải UJI.
Chữ ký và các trường trả về giống cách ips.experiments.runner dùng chúng."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from ips.plugins import discover


def make_frames(n_train=1500, n_val=300, n_wap=60, seed=0):
    rng = np.random.default_rng(seed)
    aps = rng.uniform(0, 100, size=(n_wap, 2))

    def make(n):
        b = rng.integers(0, 3, n); f = rng.integers(0, 3, n)
        xy = rng.uniform(0, 100, size=(n, 2))
        d = np.linalg.norm(xy[:, None, :] - aps[None], axis=2)
        rssi = -30 - 0.5 * d + 3 * f[:, None] + rng.normal(0, 2, (n, n_wap))
        rssi = np.where(rssi < -84, 100, np.round(rssi))
        df = pd.DataFrame(rssi, columns=[f"WAP{i + 1:03d}" for i in range(n_wap)])
        df["LONGITUDE"], df["LATITUDE"], df["FLOOR"], df["BUILDINGID"] = xy[:, 0], xy[:, 1], f, b
        return df

    return make(n_train), make(n_val)


def load_csvs(data_dir):
    return make_frames()


def make_building_data(train, test, building, val_size, seed, preprocess, preprocess_params):
    tr, te = train[train.BUILDINGID == building], test[test.BUILDINGID == building]
    groups = tr.LONGITUDE.round(3).astype(str) + "_" + tr.LATITUDE.round(3).astype(str) + "_" + tr.FLOOR.astype(str)
    i_tr, i_va = next(GroupShuffleSplit(1, test_size=val_size, random_state=seed).split(tr, groups=groups))
    pre = discover("ips.data.preprocessors", "PREPROCESS_NAME", "build_preprocessor")[preprocess](**preprocess_params)
    pre.fit(tr.iloc[i_tr])
    xy = lambda f: f[["LONGITUDE", "LATITUDE"]].to_numpy(float)
    return SimpleNamespace(
        x_train=pre.transform(tr.iloc[i_tr]), x_val=pre.transform(tr.iloc[i_va]), x_test=pre.transform(te),
        y_train=xy(tr.iloc[i_tr]), y_val=xy(tr.iloc[i_va]), y_test=xy(te),
        train_rows=tr.index.to_numpy()[i_tr], val_rows=tr.index.to_numpy()[i_va], test_rows=te.index.to_numpy(),
        feature_columns=pre.feature_columns, preprocess_state=pre.state_dict())
