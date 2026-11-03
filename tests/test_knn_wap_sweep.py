import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from ips.neighbors import available_models, build_model
from ips.experiments.knn_runner import BASE_MODELS, KNNConfig, is_selector, run, split_models
from ips.experiments.knn_wap_sweep import run_selector, sweep_settings
from knn_stub import load_csvs, make_building_data, make_frames


def _huim_available():
    try:
        import PAMI.highUtilityPattern.basic  # noqa: F401
        return True
    except ImportError:
        return False


def _sparse_loader(data_dir, keep=6):
    """Dữ liệu giống UJI thật hơn: mỗi mẫu chỉ bắt được vài WAP mạnh nhất (dữ liệu dày làm HUIM bùng nổ tổ hợp)."""
    out = []
    for frame in make_frames():
        frame = frame.copy(); cols = [c for c in frame.columns if c.startswith("WAP")]
        X = frame[cols].to_numpy().copy()
        drop = np.argsort(np.where(X == 100, -np.inf, -X), axis=1)[:, keep:]
        np.put_along_axis(X, drop, 100, axis=1); frame[cols] = X; out.append(frame)
    return tuple(out)


def _cfg(model, **sweep):
    return KNNConfig(models=(model,), buildings=(0, 1), k_max=6, k_ref=4, timing_repeats=1,
                     wap_sweep=dict(percentages=[20, 40, 60], n_rep=1, **sweep))


class TestSelectorRegistration(unittest.TestCase):
    def test_menu_lists_selectors_next_to_base_models(self):
        self.assertTrue({"knn", "wknn_inv_d", "wknn_inv_d2", "rf", "huim"} <= set(available_models()))
        self.assertTrue(is_selector("rf") and is_selector("huim"))
        self.assertFalse(any(is_selector(n) for n in BASE_MODELS))

    def test_default_run_has_no_selector(self):
        base, sel = split_models(KNNConfig())
        self.assertEqual((set(base), sel), ({"knn", "wknn_inv_d", "wknn_inv_d2"}, []))
        self.assertEqual(split_models(KNNConfig(models=("wknn_inv_d2",))), (["wknn_inv_d2"], []))

    def test_selector_alone_falls_back_to_base_models(self):
        self.assertEqual(split_models(KNNConfig(models=("rf",))), (list(BASE_MODELS), ["rf"]))
        self.assertEqual(split_models(KNNConfig(models=("huim", "knn"))), (["knn"], ["huim"]))

    def test_bad_settings_rejected(self):
        rf = build_model("rf")
        with self.assertRaises(ValueError): sweep_settings(KNNConfig(wap_sweep={"abc": 1}), rf)
        with self.assertRaises(ValueError): sweep_settings(KNNConfig(wap_sweep={"percentages": [0, 50]}), rf)
        with self.assertRaises(ValueError): sweep_settings(KNNConfig(wap_sweep={"scale": 5}), rf)       # khóa của HUIM, không dùng cho RF


class TestRunnerBehaviour(unittest.TestCase):
    def _run(self, cfg, loader=load_csvs):
        t = tempfile.TemporaryDirectory(); self.addCleanup(t.cleanup)
        out = Path(t.name) / "o"
        with redirect_stdout(io.StringIO()) as log:
            run("ignored", out, cfg, loader, make_building_data)
        return out, log.getvalue()

    def test_default_models_do_not_run_sweep(self):
        out, log = self._run(KNNConfig(models=("wknn_inv_d2",), buildings=(0, 1), k_max=6, k_ref=4, timing_repeats=1))
        self.assertTrue((out / "sweep_by_k.csv").exists())
        self.assertEqual([p.name for p in out.iterdir() if p.name.startswith("wap_")], [])
        self.assertNotIn("[m-sweep]", log)

    def test_rf_entry_keeps_standard_files_and_adds_sweep(self):
        out, log = self._run(_cfg("rf", n_trees=10))
        self.assertTrue((out / "sweep_by_k.csv").exists() and (out / "summary_table.csv").exists())
        self.assertTrue((out / "wap_rf" / "chon_m.csv").exists() and (out / "wap_rf" / "rf_importance.csv").exists())
        self.assertFalse((out / "wap_huim").exists())
        for line in log.splitlines():                              # không trùng dấu hiệu cắt kết luận của ô báo cáo notebook UJI
            if "[m-sweep]" in line: self.assertFalse(line.startswith("=" * 40) or line.startswith("Lưu ý"))

    @unittest.skipUnless(_huim_available(), "cần thư viện pami")
    def test_huim_entry_runs_both_algorithms(self):
        out, _ = self._run(_cfg("huim", n_tx=600, time_limit_s=60), loader=_sparse_loader)
        d = out / "wap_huim"
        import pandas as pd
        timing = pd.read_csv(d / "huim_thoi_gian.csv")
        self.assertEqual(set(timing.thuat_toan), {"EFIM", "UP-Growth"})
        self.assertTrue(json.loads((d / "huim_thong_tin.json").read_text(encoding="utf-8"))["same_itemsets"])
        self.assertTrue((d / "chon_m.csv").exists())


class TestSweepContent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.te = make_frames(900, 200)           # 60 WAP
        cls.tmp = tempfile.TemporaryDirectory()
        with redirect_stdout(io.StringIO()):
            cls.sw = run_selector(build_model("rf"), cls.tr, cls.te, _cfg("rf", n_trees=10), list(BASE_MODELS), Path(cls.tmp.name) / "w")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_grid_is_percent_of_wap_plus_baseline(self):
        self.assertEqual(self.sw.m_list, [12, 24, 36, 60])           # 20%, 40%, 60% của 60 WAP, cộng toàn bộ
        self.assertEqual(set(self.sw.sweep.method), set(BASE_MODELS))
        self.assertEqual(len(self.sw.sweep), 4 * 3 * 6)

    def test_smaller_m_never_has_fewer_all_zero_samples(self):
        z = self.sw.sweep[(self.sw.sweep.method == "knn") & (self.sw.sweep.k == 1)].sort_values("m")
        self.assertTrue((np.diff(z.zero_val.to_numpy()) <= 0).all())

    def test_rf_ranking_cache_reused(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = _cfg("rf", n_trees=10, rank_cache=str(Path(t) / "rank.npz"))
            with redirect_stdout(io.StringIO()) as first:
                run_selector(build_model("rf"), self.tr, self.te, cfg, ["knn"], Path(t) / "a")
            with redirect_stdout(io.StringIO()) as second:
                run_selector(build_model("rf"), self.tr, self.te, cfg, ["knn"], Path(t) / "b")
            self.assertIn("Đã huấn luyện RF", first.getvalue()); self.assertIn("Đã nạp xếp hạng RF", second.getvalue())


if __name__ == "__main__":
    unittest.main()
