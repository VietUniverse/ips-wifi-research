"""Kiểm tra HUIM tách độc lập và chế độ so sánh cặp, không cần cài PAMI."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from ips.neighbors import build_model
from ips.neighbors.huim import rank_joint
from ips.experiments.knn_wap_sweep import export_huim_comparison, run_huim_pair


S = dict(scale=1000, unit_profit=1, min_util_ratio=0.01, escalate=2.0, max_tries=3, n_tx=10,
         time_limit_s=1, mem_fraction=0.5)


def _res(n_wap, mine=0.1, mem=5.0, n=1, checksum=123):
    return dict(time=mine, mem=mem, n=n, checksum=checksum, score=[1.0] * n_wap,
                cnt=[1] * n_wap, lens={"1": n}, top=[(1.0, [1])])


class TestHUIMIndependence(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
        self.cols = np.array(["a", "b", "c"])

    def _rank_one(self, name):
        calls = []

        def fake(workdir, alg, db, mu, n_wap, mem_limit, time_limit):
            calls.append((alg, mu))
            return _res(n_wap), "OK", 0.3

        with patch("ips.neighbors.huim.ensure_pami"), \
             patch("ips.neighbors.huim.available_ram", return_value=10 ** 12), \
             patch("ips.neighbors.huim.run_guarded", side_effect=fake), \
             redirect_stdout(io.StringIO()):
            result = build_model(name).rank(self.X, None, self.cols, S, 42)
        return result, calls

    def test_upgrowth_only_calls_upgrowth(self):
        result, calls = self._rank_one("huim_upgrowth")
        self.assertEqual([c[0] for c in calls], ["upgrowth"])
        self.assertEqual(list(result.orders), ["upgrowth"])

    def test_efim_only_calls_efim(self):
        result, calls = self._rank_one("huim_efim")
        self.assertEqual([c[0] for c in calls], ["efim"])
        self.assertEqual(list(result.orders), ["efim"])


class TestHUIMJoint(unittest.TestCase):
    def test_one_side_failure_escalates_both(self):
        calls = []
        failed_efim = {"done": False}

        def fake(workdir, alg, db, mu, n_wap, mem_limit, time_limit):
            calls.append((alg, mu))
            if alg == "efim" and not failed_efim["done"]:
                failed_efim["done"] = True
                return None, "tràn RAM", 0.2
            return _res(n_wap), "OK", 0.3

        with patch("ips.neighbors.huim.ensure_pami"), \
             patch("ips.neighbors.huim.available_ram", return_value=10 ** 12), \
             patch("ips.neighbors.huim.run_guarded", side_effect=fake), \
             redirect_stdout(io.StringIO()):
            joint = rank_joint(np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]), np.array(["a", "b", "c"]), S, 42)

        self.assertEqual([c[0] for c in calls], ["upgrowth", "efim", "upgrowth", "efim"])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual(calls[2][1], calls[3][1])
        self.assertGreater(calls[2][1], calls[0][1])
        self.assertEqual(joint["n_attempts"], 2)
        self.assertTrue(joint["same_itemsets"])
        self.assertTrue(joint["same_order"])
        self.assertEqual(set(joint["ranks"]), {"upgrowth", "efim"})
        self.assertEqual(len(joint["attempts"]), 4)



class TestHUIMEmptyResult(unittest.TestCase):
    """Hồi quy run 20261105: 0 itemset từng bị ghi 'OK' nên điểm WAP = 0 và hai thuật toán cho kết quả y hệt."""
    X = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    cols = np.array(["a", "b", "c"])

    def _run(self, fake, s=S, fn="single"):
        with patch("ips.neighbors.huim.ensure_pami"), \
             patch("ips.neighbors.huim.available_ram", return_value=10 ** 12), \
             patch("ips.neighbors.huim.run_guarded", side_effect=fake), \
             redirect_stdout(io.StringIO()):
            if fn == "single":
                return build_model("huim_efim").rank(self.X, None, self.cols, s, 42)
            return rank_joint(self.X, self.cols, s, 42)

    def test_empty_is_not_accepted_and_bisect_finds_nonempty(self):
        calls = []

        def fake(workdir, alg, db, mu, n_wap, mem_limit, time_limit):
            calls.append(mu)                                     # tổng utility của X = 7000
            if mu < 1000: return None, "tràn RAM", 0.1           # quá nặng
            if mu > 1500: return _res(n_wap, n=0), "OK", 0.1     # quá cao: rỗng
            return _res(n_wap, n=3), "OK", 0.1

        s = dict(S, min_util_ratio=0.05, escalate=8.0, max_tries=3, max_bisect=6, n_tx=100)
        result = self._run(fake, s)
        info = result.extras["huim_thong_tin.json"]
        self.assertGreater(info["n_itemset"], 0)
        self.assertGreater(info["n_wap_co_diem"], 0)
        log = result.extras["huim_nhat_ky.csv"]
        self.assertTrue(any(str(t).startswith("rỗng") for t in log.trang_thai))
        self.assertEqual(log.trang_thai.iloc[-1], "OK")

    def test_only_empty_raises_instead_of_fake_ok(self):
        def fake(workdir, alg, db, mu, n_wap, mem_limit, time_limit):
            return _res(n_wap, n=0), "OK", 0.1

        with self.assertRaises(RuntimeError) as cm:
            self._run(fake)
        self.assertIn("rỗng", str(cm.exception))

    def test_joint_fail_on_one_and_empty_on_other_raises(self):
        def fake(workdir, alg, db, mu, n_wap, mem_limit, time_limit):
            if alg == "upgrowth": return None, "tràn RAM", 0.1
            return _res(n_wap, n=0), "OK", 0.1

        with self.assertRaises(RuntimeError):
            self._run(fake, fn="joint")

    def test_fixed_ratio_runs_once(self):
        calls = []

        def fake(workdir, alg, db, mu, n_wap, mem_limit, time_limit):
            calls.append(mu)
            return _res(n_wap, n=2), "OK", 0.1

        result = self._run(fake, dict(S, min_util_ratio_fixed=0.03))
        self.assertEqual(len(calls), 1)
        self.assertTrue(result.extras["huim_thong_tin.json"]["shared_threshold"])

    def test_max_items_per_tx_keeps_strongest(self):
        from ips.neighbors.huim import _prepare_utility
        X = np.array([[0.5, 0.1, 0.9, 0.0], [0.2, 0.3, 0.4, 0.5]])
        U, single, _ = _prepare_utility(X, dict(S, n_tx=0, max_items_per_tx=2), 0)
        self.assertEqual((U > 0).sum(1).tolist(), [2, 2])
        self.assertEqual(U[0, 2], 900)
        self.assertEqual(U[0, 1], 0)


class TestHUIMPairDispatch(unittest.TestCase):
    def test_pair_prepares_once_and_writes_both_folders_plus_comparison(self):
        selectors = [build_model("huim_upgrowth"), build_model("huim_efim")]
        cfg = SimpleNamespace(seed=42, k_max=5, preprocess="rss_powed", preprocess_params={},
                              max_train=None, max_test=None, wap_sweep={})
        ranks = {"upgrowth": object(), "efim": object()}
        joint = {"ranks": ranks}
        selector_calls, compare_calls = [], []

        prepared = ("Xtr", "Xva", "cols", "ref", "truth")

        def fake_prepare(train, test, cfg):
            selector_calls.append("prepare")
            return prepared

        def fake_run_selector(selector, train, test, cfg, base, out, prepared=None, rank=None):
            selector_calls.append((selector.algorithm, Path(out).name, prepared, rank))
            return SimpleNamespace(t_run_s=1.0, t_export_s=0.5)

        def fake_export(out, joint_arg, sweeps):
            compare_calls.append((Path(out).name, joint_arg, set(sweeps)))
            return ["so_sanh_thoi_gian.csv"]

        with tempfile.TemporaryDirectory() as temp, \
             patch("ips.experiments.knn_wap_sweep.sweep_settings", return_value={}), \
             patch("ips.experiments.knn_wap_sweep.prepare", side_effect=fake_prepare), \
             patch("ips.neighbors.huim.rank_joint", return_value=joint), \
             patch("ips.experiments.knn_wap_sweep.run_selector", side_effect=fake_run_selector), \
             patch("ips.experiments.knn_wap_sweep.export_huim_comparison", side_effect=fake_export), \
             redirect_stdout(io.StringIO()):
            sweeps = run_huim_pair(selectors, None, None, cfg, ["knn"], Path(temp) / "run")

        self.assertEqual(selector_calls[0], "prepare")
        self.assertEqual([(c[0], c[1]) for c in selector_calls[1:]],
                         [("upgrowth", "wap_huim_upgrowth"), ("efim", "wap_huim_efim")])
        self.assertIs(selector_calls[1][3], ranks["upgrowth"])
        self.assertIs(selector_calls[2][3], ranks["efim"])
        self.assertTrue(all(c[2] == prepared for c in selector_calls[1:]))
        self.assertEqual(compare_calls[0][0], "wap_huim_compare")
        self.assertIs(compare_calls[0][1], joint)
        self.assertEqual(compare_calls[0][2], {"upgrowth", "efim"})
        self.assertEqual(set(sweeps), {"upgrowth", "efim"})


class TestHUIMComparisonExport(unittest.TestCase):
    def _sw(self, alg):
        sel = pd.DataFrame([dict(rk=alg, method="knn", baseline_err=2.0, m_best=2, best_err=1.5,
                                 best_d_err_pct=-25.0, best_improved=True, m_small=2, m_small_pct=50.0)])
        all_ = pd.DataFrame([dict(rk=alg, m=2, pct_wap=50.0, method="knn", success_rate=90.0,
                                  mean_error=1.5, p_value=0.01, improved=True)])
        return SimpleNamespace(sel=sel, all=all_, m_list=[2], n_wap=4, k_fix=3, verdict="ok",
                               t_run_s=1.0, t_export_s=0.5)

    def test_writes_time_and_summary_files(self):
        n_wap = 4
        attempts = pd.DataFrame([
            dict(lan=1, ty_le_min_util=0.01, min_util=10, thuat_toan="UP-Growth", trang_thai="OK",
                 thoi_gian_wall_s=0.3, thoi_gian_khai_pha_s=0.1, loi=""),
            dict(lan=1, ty_le_min_util=0.01, min_util=10, thuat_toan="EFIM", trang_thai="OK",
                 thoi_gian_wall_s=0.4, thoi_gian_khai_pha_s=0.2, loi=""),
        ])
        joint = dict(attempts=attempts, results={"upgrowth": _res(n_wap), "efim": _res(n_wap)},
                     walls={"upgrowth": 0.3, "efim": 0.4},
                     cumulative_walls={"upgrowth": 0.3, "efim": 0.4}, n_attempts=1,
                     min_util=10, min_util_ratio=0.01, n_transactions=2,
                     same_itemsets=True, same_order=True)
        sweeps = {"upgrowth": self._sw("upgrowth"), "efim": self._sw("efim")}
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "compare"
            files = export_huim_comparison(out, joint, sweeps)
            self.assertTrue({"so_sanh_tong_hop.csv", "so_sanh_thoi_gian.csv", "so_sanh_min_util.csv",
                             "so_sanh_chon_m.csv", "so_sanh_thong_tin.json"} <= set(files))
            for name in files:
                self.assertTrue((out / name).exists(), name)
            timing = pd.read_csv(out / "so_sanh_thoi_gian.csv")
            self.assertEqual(set(timing.thuat_toan), {"UP-Growth", "EFIM"})
            info = json.loads((out / "so_sanh_thong_tin.json").read_text(encoding="utf-8"))
            self.assertTrue(info["cung_min_util"])
            self.assertTrue(info["same_itemsets"])


if __name__ == "__main__":
    unittest.main()
