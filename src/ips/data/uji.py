"""UJIIndoorLoc split and preprocessing without official-test leakage."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

AP_COLUMNS = [f"WAP{i:03d}" for i in range(1, 521)]
TARGET_COLUMNS = ["LONGITUDE", "LATITUDE"]


@dataclass
class BuildingData:
    building: int
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_rows: np.ndarray
    val_rows: np.ndarray
    test_rows: np.ndarray
    feature_columns: list[str]
    preprocess_state: dict


def load_csvs(data_dir: str | Path):
    base = Path(data_dir)
    train = pd.read_csv(base / "trainingData.csv")
    test = pd.read_csv(base / "validationData.csv")
    required = set(AP_COLUMNS + TARGET_COLUMNS + ["BUILDINGID", "FLOOR"])
    for name, frame in (("trainingData.csv", train), ("validationData.csv", test)):
        missing = required - set(frame)
        if missing:
            raise ValueError(f"{name} lacks columns: {sorted(missing)[:5]}")
    return train, test


def preprocess_rss(frame: pd.DataFrame) -> np.ndarray:
    x = frame[AP_COLUMNS].to_numpy(dtype=np.float32, copy=True)
    x[x == 100] = -110.0
    return (2.0 * (np.clip(x, -110.0, -30.0) + 110.0) / 80.0 - 1.0).astype(np.float32)


def make_building_data(train: pd.DataFrame, test: pd.DataFrame, building: int,
                       val_size: float = 0.15, seed: int = 42,
                       preprocess_name: str = "rss_standard",
                       preprocess_params: dict | None = None) -> BuildingData:
    selected = train.loc[train.BUILDINGID == building].copy()
    official = test.loc[test.BUILDINGID == building].copy()
    if selected.empty or official.empty:
        raise ValueError(f"Building {building} is missing from train or official test")
    groups = (selected.FLOOR.astype(str) + "_" + selected.LONGITUDE.round(3).astype(str)
              + "_" + selected.LATITUDE.round(3).astype(str)).to_numpy()
    local_train, local_val = next(GroupShuffleSplit(
        n_splits=1, test_size=val_size, random_state=seed,
    ).split(selected, groups=groups))
    assert set(groups[local_train]).isdisjoint(groups[local_val])
    tr, va = selected.iloc[local_train], selected.iloc[local_val]
    from ips.data.preprocessors.factory import create_preprocessor
    preprocessor = create_preprocessor(preprocess_name, **(preprocess_params or {}))
    preprocessor.fit(tr)
    features = list(preprocessor.feature_columns)
    state = preprocessor.state_dict()
    if not isinstance(state, dict):
        raise ValueError(f"Preprocessor {preprocess_name!r} state_dict() must return a dict")
    x_train, x_val, x_test = (preprocessor.transform(frame) for frame in (tr, va, official))
    for x in (x_train, x_val, x_test):
        if x.dtype != np.float32 or x.ndim != 2 or x.shape[1] != len(features):
            raise ValueError(f"Preprocessor {preprocess_name!r} must return float32 [rows, features]")
    return BuildingData(
        building=building,
        x_train=x_train, y_train=tr[TARGET_COLUMNS].to_numpy(np.float64),
        x_val=x_val, y_val=va[TARGET_COLUMNS].to_numpy(np.float64),
        x_test=x_test, y_test=official[TARGET_COLUMNS].to_numpy(np.float64),
        train_rows=tr.index.to_numpy(), val_rows=va.index.to_numpy(),
        test_rows=official.index.to_numpy(),
        feature_columns=features,
        preprocess_state=state,
    )
