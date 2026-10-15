"""Adapters for the additional fingerprint datasets in the user's research ZIP.

The archive stays external to the repository. Adapters convert supported sources
to the canonical UJI-style table consumed by the shared experiment runners.
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ips.data.uji import AP_COLUMNS


PACK_DATASETS = {
    "hcxy": "HCXY (SODIndoorLoc)",
    "syl": "SYL (SODIndoorLoc)",
    "tuji": "TUJI1",
    "hus_case1": "HUS (test case 1)",
    "hus_case2": "HUS (test case 2)",
    "dlulib": "DLULIB (Dalat University Library)",
}


def _member(zipped: zipfile.ZipFile, suffix: str) -> str:
    matches = [name for name in zipped.namelist() if name.lower().endswith(suffix.lower())]
    if len(matches) != 1:
        raise ValueError(f"ZIP cần đúng một file kết thúc bằng {suffix!r}; tìm thấy {len(matches)}")
    return matches[0]


def _read_table(zipped: zipfile.ZipFile, suffix: str, *, excel=False, sep=None) -> pd.DataFrame:
    name = _member(zipped, suffix)
    payload = io.BytesIO(zipped.read(name))
    if excel:
        return pd.read_excel(payload, engine="openpyxl")
    return pd.read_csv(payload, sep=sep, encoding="utf-8-sig", index_col=False)


def _signal_columns(frame: pd.DataFrame) -> list[str]:
    columns = [c for c in frame.columns if re.fullmatch(r"(?:WAP|MAC|AP)\d+", str(c), re.I)]
    if not columns:
        raise ValueError("Không tìm thấy cột RSS dạng WAP1/MAC1/AP1")
    return sorted(columns, key=lambda name: int(re.search(r"\d+", str(name)).group()))


def _canonical_pair(train: pd.DataFrame, test: pd.DataFrame, *, x_col: str, y_col: str,
                    floor_col: str | None = None, train_rp: str | None = None,
                    test_rp: str | None = None, coordinate_map: pd.DataFrame | None = None,
                    missing_value: float = 100.0, building_col: str | None = None):
    signal_cols = _signal_columns(train)
    test_signals = _signal_columns(test)
    common = [c for c in signal_cols if c in test_signals]
    if len(common) > len(AP_COLUMNS):
        # Select APs by detection frequency in training only; this keeps the
        # 520-token Mixer input while never using test observations for selection.
        present = train[common].apply(pd.to_numeric, errors="coerce").ne(missing_value).sum()
        common = sorted(present.sort_values(ascending=False, kind="stable").index[:len(AP_COLUMNS)],
                        key=lambda name: int(re.search(r"\d+", str(name)).group()))

    def attach_xy(frame: pd.DataFrame, rp_col: str | None) -> pd.DataFrame:
        part = frame.copy()
        if coordinate_map is not None:
            assert rp_col is not None
            part = part.merge(coordinate_map, left_on=rp_col, right_on="_rp", how="left", validate="many_to_one")
            part[x_col], part[y_col] = part["_map_x"], part["_map_y"]
        required = [x_col, y_col]
        if floor_col and floor_col not in part:
            raise ValueError(f"Thiếu cột tầng {floor_col}")
        part[x_col] = pd.to_numeric(part[x_col], errors="coerce")
        part[y_col] = pd.to_numeric(part[y_col], errors="coerce")
        part = part.dropna(subset=required).copy()
        return part

    train = attach_xy(train, train_rp)
    test = attach_xy(test, test_rp)

    def canonical(frame: pd.DataFrame) -> pd.DataFrame:
        rss = np.full((len(frame), len(AP_COLUMNS)), 100.0, dtype=np.float32)
        for index, source_col in enumerate(common):
            values = pd.to_numeric(frame[source_col], errors="coerce")
            absent = values.isna() | values.eq(missing_value)
            # UJI-style preprocessing reserves 100 for every source's missing AP.
            rss[:, index] = values.mask(absent, 100.0).to_numpy(dtype=np.float32)
        out = pd.DataFrame(rss, columns=AP_COLUMNS, index=frame.index)
        out["LONGITUDE"] = frame[x_col].to_numpy(dtype=np.float64)
        out["LATITUDE"] = frame[y_col].to_numpy(dtype=np.float64)
        out["FLOOR"] = (pd.to_numeric(frame[floor_col], errors="coerce").fillna(0).to_numpy(dtype=int)
                        if floor_col else 0)
        out["BUILDINGID"] = (pd.to_numeric(frame[building_col], errors="coerce").fillna(0).to_numpy(dtype=int)
                             if building_col else 0)
        return out.reset_index(drop=True)

    return canonical(train), canonical(test), len(common)


def _rp_map(zipped: zipfile.ZipFile, suffix: str, rp_column="_rp") -> pd.DataFrame:
    name = _member(zipped, suffix)
    frame = pd.read_csv(io.BytesIO(zipped.read(name)), sep="\t", header=None,
                        encoding="utf-8-sig")
    if frame.shape[1] < 3:
        raise ValueError(f"Tệp tọa độ {name} cần RP, X, Y")
    return pd.DataFrame({rp_column: pd.to_numeric(frame.iloc[:, 0], errors="coerce"),
                         "_map_x": pd.to_numeric(frame.iloc[:, 1], errors="coerce"),
                         "_map_y": pd.to_numeric(frame.iloc[:, 2], errors="coerce")})


def _prepare_hcxy(zipped):
    train = _read_table(zipped, "hcxy/train_HCXY.txt", sep="\t")
    test = _read_table(zipped, "hcxy/test_HCXY.txt", sep="\t")
    coords = _rp_map(zipped, "hcxy/mapxy_HCXY.txt")
    return _canonical_pair(train, test, x_col="LONGITUDE", y_col="LATITUDE",
                           train_rp="RP", test_rp="RP", coordinate_map=coords)


def _prepare_syl(zipped):
    train = _read_table(zipped, "syl/SYL_Train.csv", sep=",")
    test = _read_table(zipped, "syl/SYL_Test.csv", sep=",")
    return _canonical_pair(train, test, x_col="LONGITUDE", y_col="LATITUDE",
                           floor_col="FloorID", missing_value=100.0)


def _prepare_tuji(zipped):
    train = _read_table(zipped, "tuji/TUJI_training.csv", sep=",")
    test = _read_table(zipped, "tuji/TUJI_testing.csv", sep=",")
    return _canonical_pair(train, test, x_col="LONGITUDE", y_col="LATITUDE",
                           floor_col="FLOOR", missing_value=100.0)


def _prepare_hus(zipped, case: int):
    train = _read_table(zipped, "hus/hus_trainall.txt", sep="\t")
    test = _read_table(zipped, f"hus/testcase{case}.txt", sep="\t")
    coords = _rp_map(zipped, "hus/hus_mapXY.txt")
    return _canonical_pair(train, test, x_col="LONGITUDE", y_col="LATITUDE",
                           train_rp="RP", test_rp="RP", coordinate_map=coords)


def _prepare_dlulib(zipped):
    training, testing = [], []
    for floor in (1, 2, 3):
        training.append(_read_table(zipped, f"Training_DLULIB_Floor{floor}_All_20.xlsx", excel=True))
        testing.append(_read_table(zipped, f"Testing_DLULIB_Floor{floor}.xlsx", excel=True))
    train, test = pd.concat(training, ignore_index=True), pd.concat(testing, ignore_index=True)
    return _canonical_pair(train, test, x_col="X", y_col="Y", floor_col="FloorID",
                           missing_value=-100.0)


_PREPARERS = {
    "hcxy": _prepare_hcxy,
    "syl": _prepare_syl,
    "tuji": _prepare_tuji,
    "hus_case1": lambda z: _prepare_hus(z, 1),
    "hus_case2": lambda z: _prepare_hus(z, 2),
    "dlulib": _prepare_dlulib,
}


def prepare_pack_dataset(zip_source, dataset: str, output_dir: str | Path) -> dict:
    """Convert one supported dataset from the archive into runner CSV files."""
    if dataset not in _PREPARERS:
        raise ValueError(f"Dataset không hỗ trợ: {dataset!r}; chọn một trong {', '.join(PACK_DATASETS)}")
    with zipfile.ZipFile(zip_source) as zipped:
        train, test, n_aps = _PREPARERS[dataset](zipped)
    if train.empty or test.empty:
        raise ValueError(f"Dataset {dataset} có train/test rỗng sau khi đọc tọa độ")
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    train.to_csv(target / "trainingData.csv", index=False)
    test.to_csv(target / "validationData.csv", index=False)
    return {"dataset": dataset, "name": PACK_DATASETS[dataset], "n_train": len(train),
            "n_test": len(test), "n_source_aps": n_aps,
            "n_features": len(AP_COLUMNS),
            "train_missing_fraction": float(train[AP_COLUMNS].eq(100).to_numpy().mean()),
            "test_missing_fraction": float(test[AP_COLUMNS].eq(100).to_numpy().mean())}
