"""Chạy họ kNN/WKNN (python -m ips.experiments.knn_cli  hoặc lệnh `ips-knn` nếu đã thêm entry point).
    --list-options
    --config configs/knn_smoke.json --data-dir DATA --output-dir runs/knn_smoke      # chạy thẳng, in so sánh, xuất file
    --config configs/knn_full.json  --data-dir DATA --menu                           # menu tương tác
"""
import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from ips.neighbors import available_models
from ips.plugins import discover
from .knn_menu import run_menu
from .knn_runner import Experiment, KNNConfig, run


def load_config(path):
    cfg = KNNConfig()
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        bad = set(raw) - {f.name for f in fields(KNNConfig)}
        if bad: raise ValueError(f"Khóa config không hợp lệ: {sorted(bad)}")
        for k, v in raw.items():
            setattr(cfg, k, tuple(v) if k in ("models", "buildings") else v)
    return cfg


def main(argv=None, loader=None, builder=None):
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="ips-knn", description="Họ kNN/WKNN trên UJIIndoorLoc")
    ap.add_argument("--config"); ap.add_argument("--data-dir"); ap.add_argument("--output-dir", default="runs/knn")
    ap.add_argument("--menu", action="store_true"); ap.add_argument("--list-options", action="store_true")
    a = ap.parse_args(argv)
    if a.list_options:
        print("Models (ips.neighbors): " + ", ".join(sorted(available_models())))
        print("Preprocessors:          " + ", ".join(sorted(discover("ips.data.preprocessors", "PREPROCESS_NAME", "build_preprocessor"))))
        return 0
    if not a.data_dir: ap.error("cần --data-dir")
    cfg = load_config(a.config)
    if not a.menu:
        run(a.data_dir, a.output_dir, cfg, loader, builder)
        return 0
    if loader is None or builder is None:
        from ips.data.uji import load_csvs, make_building_data
        loader, builder = loader or load_csvs, builder or make_building_data
    train, test = loader(a.data_dir)
    run_menu(Experiment.from_frames(train, test, cfg, builder), a.output_dir, cfg.k_ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
