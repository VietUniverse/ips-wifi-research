"""Chạy họ kNN/WKNN theo giao thức dùng chung của repo (xem runner.py):
  - mỗi tòa một lần, biết sẵn BUILDINGID; dữ liệu do make_building_data chuẩn bị (split theo nhóm vị trí);
  - k được chọn trên internal validation; official test chỉ đọc sau khi đã chọn k;
  - sai số = khoảng cách Euclid (m) trên TẤT CẢ mẫu test; kèm median, P90 và độ chính xác tầng.
Chỉ phụ thuộc repo qua: ips.plugins.discover, ips.data.uji.load_csvs, ips.data.uji.make_building_data."""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
from ips.neighbors import available_models, build_model
from ips.neighbors.formulas import Formulas

HIDDEN = ("val_err", "test_err", "floor_ok", "preds", "test_rows", "test_building")


@dataclass
class KNNConfig:
    models: tuple = ()                    # rỗng = tất cả model trong ips.neighbors
    preprocess: str = "rss_powed"
    preprocess_params: dict = field(default_factory=dict)
    buildings: tuple = (0, 1, 2)
    seed: int = 42
    val_size: float = 0.15
    k_max: int = 20
    k_ref: int = 13
    timing_repeats: int = 3
    max_train: int | None = None          # chỉ dùng cho smoke: lấy mẫu con
    max_test: int | None = None
    wap_sweep: dict | None = None         # có khóa này thì chạy thêm luồng quét số WAP (Lần 6.5), xem knn_wap_sweep.py


@dataclass
class Block:
    """Dữ liệu đã chuẩn bị của một tòa."""
    building: int
    train_rows: np.ndarray
    val_rows: np.ndarray
    test_rows: np.ndarray
    val_d: np.ndarray; val_i: np.ndarray; val_xy: np.ndarray; val_floor: np.ndarray
    test_d: np.ndarray; test_i: np.ndarray; test_xy: np.ndarray; test_floor: np.ndarray
    t_search_val: float
    t_search_test: float
    models: dict = field(default_factory=dict)


def _xy_check(frame, rows, y, what):
    sub = frame.iloc[np.asarray(rows)][["LONGITUDE", "LATITUDE"]].to_numpy(float)
    if sub.shape != np.asarray(y).shape or not np.allclose(sub, y):
        raise ValueError(f"{what}: source_row không khớp tọa độ (LONGITUDE, LATITUDE) của make_building_data; "
                         "cần sửa cách lấy tầng trong knn_runner._floor_of()")


def _floor_of(frame, rows):
    return frame.iloc[np.asarray(rows)]["FLOOR"].to_numpy(int)


def build_blocks(train, test, cfg, builder, model_names):
    if cfg.max_train: train = train.sample(n=min(cfg.max_train, len(train)), random_state=cfg.seed).reset_index(drop=True)
    if cfg.max_test: test = test.sample(n=min(cfg.max_test, len(test)), random_state=cfg.seed).reset_index(drop=True)
    blocks = []
    for b in cfg.buildings:
        d = builder(train, test, b, cfg.val_size, cfg.seed, cfg.preprocess, cfg.preprocess_params)
        _xy_check(train, d.train_rows, d.y_train, f"tòa {b} train"); _xy_check(train, d.val_rows, d.y_val, f"tòa {b} val")
        _xy_check(test, d.test_rows, d.y_test, f"tòa {b} test")
        ref_floor = _floor_of(train, d.train_rows)
        Xtr, Xva, Xte = (Formulas.missing_to_zero(a) for a in (d.x_train, d.x_val, d.x_test))
        K = min(cfg.k_max, len(Xtr))
        t0 = time.perf_counter(); vd, vi = Formulas.sorensen_search(Xva, Xtr, K); tv = time.perf_counter() - t0
        t0 = time.perf_counter(); td, ti = Formulas.sorensen_search(Xte, Xtr, K); tt = time.perf_counter() - t0
        vf, tf = _floor_of(train, d.val_rows), _floor_of(test, d.test_rows)
        nf = int(max(ref_floor.max(), vf.max(), tf.max())) + 1
        blk = Block(b, np.asarray(d.train_rows), np.asarray(d.val_rows), np.asarray(d.test_rows),
                    vd, vi, np.asarray(d.y_val), vf, td, ti, np.asarray(d.y_test), tf, tv, tt)
        for name in model_names:
            blk.models[name] = build_model(name).fit(ref_floor, np.asarray(d.y_train)[:, 0], np.asarray(d.y_train)[:, 1], nf)
        blocks.append(blk)
        print(f"[tòa {b}] train={len(Xtr)} val={len(Xva)} test={len(Xte)} | tìm láng giềng val {tv:.2f}s, test {tt:.2f}s")
    return blocks


class Experiment:
    def __init__(self, blocks, model_names, k_max, timing_repeats=3):
        self.blocks, self.k_max, self.repeats = blocks, min([k_max] + [len(b.val_d[0]) for b in blocks]), timing_repeats
        proto = {n: build_model(n) for n in model_names}
        self.names = {n: m.display_name for n, m in proto.items()}
        self.colors = {n: m.color for n, m in proto.items()}
        self.keys = list(model_names)
        self.t_search = sum(b.t_search_test for b in blocks)           # tìm láng giềng cho tập test, dùng chung
        self.n_test = sum(len(b.test_rows) for b in blocks)
        self.rows = {}

    @classmethod
    def from_frames(cls, train, test, cfg, builder):
        names = list(cfg.models) or sorted(available_models())
        return cls(build_blocks(train, test, cfg, builder, names), names, cfg.k_max, cfg.timing_repeats)

    def run(self, key, k):
        """Chạy một (model, k) trên mọi tòa; kết quả được nhớ lại."""
        if (key, k) in self.rows:
            return self.rows[(key, k)]
        val_e, test_e, fok, preds, rows, bld, t_pred, n0 = [], [], [], [], [], [], 0.0, 0
        for blk in self.blocks:
            m = blk.models[key]
            pv = m.predict(blk.val_i, blk.val_d, k)
            val_e.append(Formulas.position_error(pv.x, pv.y, blk.val_xy[:, 0], blk.val_xy[:, 1]))
            ts = []
            for _ in range(self.repeats):
                t0 = time.perf_counter(); pt = m.predict(blk.test_i, blk.test_d, k); ts.append(time.perf_counter() - t0)
            t_pred += min(ts); n0 += pt.n_d0
            test_e.append(Formulas.position_error(pt.x, pt.y, blk.test_xy[:, 0], blk.test_xy[:, 1]))
            fok.append(pt.floor == blk.test_floor); preds.append(pt); rows.append(blk.test_rows); bld.append(np.full(len(pt.x), blk.building))
        val_e, test_e, fok = np.concatenate(val_e), np.concatenate(test_e), np.concatenate(fok)
        row = dict(method=key, k=k, val_mean=float(val_e.mean()),
                   **{f"test_{c}": v for c, v in Formulas.error_summary(test_e).items()},
                   floor_acc=float(fok.mean() * 100), t_predict_s=t_pred, total_s=self.t_search + t_pred,
                   us_per_sample=(self.t_search + t_pred) / self.n_test * 1e6, n_query_d0=n0,
                   val_err=val_e, test_err=test_e, floor_ok=fok, preds=preds,
                   test_rows=np.concatenate(rows), test_building=np.concatenate(bld))
        self.rows[(key, k)] = row
        return row

    def sweep(self, keys=None, ks=None):
        keys, ks = list(keys or self.keys), list(ks or range(1, self.k_max + 1))
        return pd.DataFrame([{c: v for c, v in self.run(m, k).items() if c not in HIDDEN} for m in keys for k in ks])

    def best_k(self, key):
        """k có sai số trung bình thấp nhất trên VALIDATION (hòa thì k nhỏ hơn)."""
        df = self.sweep([key])
        return int(df.sort_values(["val_mean", "k"]).iloc[0].k)


def run(data_dir, output_dir, cfg, loader=None, builder=None):
    """Chạy trọn bộ, in so sánh và xuất file (giống ips.experiments.runner.run về bố cục)."""
    from .knn_report import Comparator
    if loader is None or builder is None:
        from ips.data.uji import load_csvs, make_building_data
        loader, builder = loader or load_csvs, builder or make_building_data
    out = Path(output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=list), encoding="utf-8")
    train, test = loader(data_dir)
    exp = Experiment.from_frames(train, test, cfg, builder)
    cmp_ = Comparator(exp, cfg.k_ref)
    cmp_.run_all()
    files = cmp_.export(out)
    print("\nĐã lưu vào", out, ":", ", ".join(files))
    if cfg.wap_sweep:
        from .knn_wap_sweep import run_wap_sweep
        try:
            run_wap_sweep(train, test, cfg, out / "wap_sweep")
        except Exception as error:      # kết quả chính đã lưu xong; lỗi ở đây không được làm mất chúng
            import traceback
            print(f"[m-sweep] LỖI, bỏ qua quét số WAP: {error}\n{traceback.format_exc()}")
    return cmp_
