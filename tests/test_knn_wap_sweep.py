import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from ips.neighbors import available_models, build_model
from ips.experiments.knn_runner import BASE_MODELS, KNNConfig, is_selector, run, split_models
from ips.experiments.knn_wap_sweep import RankResult, run_selector, sweep_settings
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
        self.assertTrue({"knn", "wknn_inv_d", "wknn_inv_d2", "rf", "huim_upgrowth", "huim_efim"} <= set(available_models()))
        self.assertTrue(is_selector("rf") and is_selector("huim_upgrowth") and is_selector("huim_efim"))
        self.assertFalse(any(is_selector(n) for n in BASE_MODELS))

    def test_default_run_has_no_selector(self):
        base, sel = split_models(KNNConfig())
        self.assertEqual((set(base), sel), ({"knn", "wknn_inv_d", "wknn_inv_d2"}, []))
        self.assertEqual(split_models(KNNConfig(models=("wknn_inv_d2",))), (["wknn_inv_d2"], []))

    def test_selector_alone_falls_back_to_base_models(self):
        self.assertEqual(split_models(KNNConfig(models=("rf",))), (list(BASE_MODELS), ["rf"]))
        self.assertEqual(split_models(KNNConfig(models=("huim_efim", "knn"))), (["knn"], ["huim_efim"]))
        self.assertEqual(split_models(KNNConfig(models=("huim_upgrowth", "huim_efim"))),
                         (list(BASE_MODELS), ["huim_upgrowth", "huim_efim"]))

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
        self.assertFalse((out / "wap_huim_upgrowth").exists())
        self.assertFalse((out / "wap_huim_efim").exists())
        self.assertFalse((out / "wap_huim_compare").exists())
        for line in log.splitlines():                              # không trùng dấu hiệu cắt kết luận của ô báo cáo notebook UJI
            if "[m-sweep]" in line: self.assertFalse(line.startswith("=" * 40) or line.startswith("Lưu ý"))

    def test_both_huim_use_pair_mode_without_pami(self):
        cfg = KNNConfig(models=("huim_upgrowth", "huim_efim"), buildings=(0, 1), k_max=6, k_ref=4,
                        timing_repeats=1, wap_sweep=dict(percentages=[20, 40], n_rep=1, n_tx=50))
        calls = []

        def fake_joint(Xtr, cols, s, seed):
            calls.append((Xtr.shape[1], tuple(s["percentages"]), seed))
            order = np.arange(len(cols))
            extras = {"huim_thong_tin.json": {"min_util": 10, "shared_threshold": True}}
            ranks = {alg: RankResult({alg: order}, {alg: 0.1}, {alg: alg}, extras=extras)
                     for alg in ("upgrowth", "efim")}
            results = {alg: dict(time=0.1, mem=1.0, n=1) for alg in ranks}
            walls = {"upgrowth": 0.2, "efim": 0.3}
            return dict(ranks=ranks, attempts=pd.DataFrame(), results=results, walls=walls,
                        cumulative_walls=dict(walls), n_attempts=1, min_util=10, min_util_ratio=0.01,
                        n_transactions=Xtr.shape[0], same_itemsets=True, same_order=True)

        with patch("ips.neighbors.huim.rank_joint", side_effect=fake_joint):
            out, log = self._run(cfg)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], (20, 40))
        up, ef, comp = out / "wap_huim_upgrowth", out / "wap_huim_efim", out / "wap_huim_compare"
        self.assertTrue((up / "chon_m.csv").exists() and (ef / "chon_m.csv").exists())
        for name in ("so_sanh_tong_hop.csv", "so_sanh_thoi_gian.csv", "so_sanh_min_util.csv",
                     "so_sanh_chon_m.csv", "so_sanh_thong_tin.json"):
            self.assertTrue((comp / name).exists(), name)
        info_up = json.loads((up / "huim_thong_tin.json").read_text(encoding="utf-8"))
        info_ef = json.loads((ef / "huim_thong_tin.json").read_text(encoding="utf-8"))
        self.assertEqual(info_up["min_util"], info_ef["min_util"])
        timing = pd.read_csv(comp / "so_sanh_thoi_gian.csv")
        self.assertEqual(set(timing.thuat_toan), {"UP-Growth", "EFIM"})
        self.assertIn("HUIM cặp", log)

    @unittest.skipUnless(_huim_available(), "cần thư viện pami")
    def test_huim_upgrowth_entry_creates_only_its_folder(self):
        out, _ = self._run(_cfg("huim_upgrowth", n_tx=600, time_limit_s=60), loader=_sparse_loader)
        d = out / "wap_huim_upgrowth"
        timing = pd.read_csv(d / "huim_thoi_gian.csv")
        self.assertEqual(set(timing.thuat_toan), {"UP-Growth"})
        self.assertTrue((d / "chon_m.csv").exists())
        self.assertFalse((out / "wap_huim_efim").exists())
        self.assertFalse((out / "wap_huim_compare").exists())

    @unittest.skipUnless(_huim_available(), "cần thư viện pami")
    def test_huim_efim_entry_creates_only_its_folder(self):
        out, _ = self._run(_cfg("huim_efim", n_tx=600, time_limit_s=60), loader=_sparse_loader)
        d = out / "wap_huim_efim"
        timing = pd.read_csv(d / "huim_thoi_gian.csv")
        self.assertEqual(set(timing.thuat_toan), {"EFIM"})
        self.assertTrue((d / "chon_m.csv").exists())
        self.assertFalse((out / "wap_huim_upgrowth").exists())
        self.assertFalse((out / "wap_huim_compare").exists())

    @unittest.skipUnless(_huim_available(), "cần thư viện pami")
    def test_both_huim_entries_share_threshold_and_create_comparison(self):
        cfg = KNNConfig(models=("huim_upgrowth", "huim_efim"), buildings=(0, 1), k_max=6, k_ref=4,
                        timing_repeats=1, wap_sweep=dict(percentages=[20, 40], n_rep=1, n_tx=600, time_limit_s=60))
        out, _ = self._run(cfg, loader=_sparse_loader)
        up, ef = out / "wap_huim_upgrowth", out / "wap_huim_efim"
        comp = out / "wap_huim_compare"
        self.assertTrue((up / "chon_m.csv").exists() and (ef / "chon_m.csv").exists())
        for name in ("so_sanh_tong_hop.csv", "so_sanh_thoi_gian.csv", "so_sanh_min_util.csv",
                     "so_sanh_chon_m.csv", "so_sanh_thong_tin.json"):
            self.assertTrue((comp / name).exists(), name)
        timing = pd.read_csv(comp / "so_sanh_thoi_gian.csv")
        self.assertEqual(set(timing.thuat_toan), {"UP-Growth", "EFIM"})
        self.assertTrue({"thoi_gian_khai_pha_s", "tong_thoi_gian_s", "bo_nho_mb"} <= set(timing.columns))
        info_up = json.loads((up / "huim_thong_tin.json").read_text(encoding="utf-8"))
        info_ef = json.loads((ef / "huim_thong_tin.json").read_text(encoding="utf-8"))
        compare = json.loads((comp / "so_sanh_thong_tin.json").read_text(encoding="utf-8"))
        self.assertEqual(info_up["min_util"], info_ef["min_util"])
        self.assertEqual(info_up["min_util"], compare["min_util"])
        self.assertTrue(compare["same_itemsets"])


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
