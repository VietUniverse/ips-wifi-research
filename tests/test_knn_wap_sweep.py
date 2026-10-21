import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from ips.experiments.knn_runner import KNNConfig, run
from ips.experiments.knn_wap_sweep import run_wap_sweep, sweep_settings, WapSweep
from knn_stub import load_csvs, make_building_data, make_frames

SWEEP = dict(rankings=["rf", "coverage"], m_values=[20, 30, 40], n_trees=10, n_rep=1)


def _cfg(**kw):
    return KNNConfig(buildings=(0, 1), k_max=6, k_ref=4, timing_repeats=1, wap_sweep=dict(SWEEP), **kw)


class TestWapSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.te = make_frames(900, 200)           # 60 WAP
        cls.tmp = tempfile.TemporaryDirectory()
        with redirect_stdout(io.StringIO()) as log:
            cls.sw = run_wap_sweep(cls.tr, cls.te, _cfg(), Path(cls.tmp.name) / "wap_sweep")
        cls.log = log.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_grid_has_baseline_and_all_models(self):
        self.assertEqual(self.sw.m_list, [20, 30, 40, 60])
        self.assertEqual(set(self.sw.sweep.method), {"knn", "wknn_inv_d", "wknn_inv_d2"})
        self.assertEqual(set(self.sw.sweep.rk), {"rf", "coverage"})
        self.assertEqual(len(self.sw.sweep), 2 * 4 * 3 * 6)

    def test_baseline_same_for_every_ranking(self):
        b = self.sw.sweep[self.sw.sweep.m == 60]
        for (me, k), g in b.groupby(["method", "k"]):
            self.assertEqual(g.mean_error.nunique(), 1)

    def test_each_m_feeds_models_with_matching_columns(self):
        order = self.sw.sweep  # m nhỏ hơn thì số mẫu toàn 0 không nhỏ hơn mức m lớn hơn (cùng cách xếp hạng)
        z = order[(order.rk == "rf") & (order.method == "knn") & (order.k == 1)].sort_values("m")
        self.assertTrue((np.diff(z.zero_val.to_numpy()) <= 0).all())

    def test_files_and_console_markers(self):
        out = Path(self.tmp.name) / "wap_sweep"
        for f in ("khao_sat_m_k.csv", "tong_hop_theo_m.csv", "chon_m.csv", "bang_tung_m.csv", "thu_tu_wap.csv", "summary.json"):
            self.assertTrue((out / f).exists(), f)
        self.assertEqual(json.loads((out / "summary.json").read_text(encoding="utf-8"))["m_list"], [20, 30, 40, 60])
        # không được trùng dấu hiệu mà ô báo cáo của notebook UJI dùng để cắt phần kết luận
        for line in self.log.splitlines():
            self.assertFalse(line.startswith("=" * 40) or line.startswith("Lưu ý"), line)

    def test_runner_runs_sweep_after_main_report(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "o"
            with redirect_stdout(io.StringIO()):
                run("ignored", out, _cfg(), load_csvs, make_building_data)
            self.assertTrue((out / "sweep_by_k.csv").exists())              # kết quả cũ giữ nguyên
            self.assertTrue((out / "wap_sweep" / "chon_m.csv").exists())

    def test_runner_without_key_has_no_sweep(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "o"
            cfg = KNNConfig(buildings=(0, 1), k_max=6, k_ref=4, timing_repeats=1)
            with redirect_stdout(io.StringIO()):
                run("ignored", out, cfg, load_csvs, make_building_data)
            self.assertFalse((out / "wap_sweep").exists())

    def test_bad_settings_rejected(self):
        with self.assertRaises(ValueError): sweep_settings(KNNConfig(wap_sweep={"m": [1]}))
        with self.assertRaises(ValueError): sweep_settings(KNNConfig(wap_sweep={"rankings": ["abc"]}))

    def test_rf_ranking_cache_reused(self):
        with tempfile.TemporaryDirectory() as t:
            cache = str(Path(t) / "rank.npz")
            cfg = _cfg(); cfg.wap_sweep = {**SWEEP, "rank_cache": cache, "rankings": ["rf"]}
            with redirect_stdout(io.StringIO()) as first:
                run_wap_sweep(self.tr, self.te, cfg, Path(t) / "a")
            with redirect_stdout(io.StringIO()) as second:
                run_wap_sweep(self.tr, self.te, cfg, Path(t) / "b")
            self.assertIn("Đã huấn luyện RF", first.getvalue()); self.assertIn("Đã nạp xếp hạng RF", second.getvalue())


if __name__ == "__main__":
    unittest.main()
