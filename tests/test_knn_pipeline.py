import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from ips.experiments.knn_report import Comparator
from ips.experiments.knn_runner import Experiment, KNNConfig, run
from ips.data.preprocessors.rss_powed import RssPowed
from knn_stub import load_csvs, make_building_data, make_frames


def _cfg(**kw):
    return KNNConfig(buildings=(0, 1), k_max=8, k_ref=5, timing_repeats=1, **kw)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tr, te = make_frames(900, 200)
        cls.exp = Experiment.from_frames(tr, te, _cfg(), make_building_data)

    def test_missing_ap_is_minus_one(self):
        tr, _ = make_frames(50, 10)
        pre = RssPowed().fit(tr); X = pre.transform(tr); raw = tr[pre.feature_columns].to_numpy()
        self.assertTrue((X[raw == 100] == -1).all()); self.assertTrue((X[raw != 100] >= 0).all())
        self.assertEqual(X.dtype, np.float32)

    def test_same_floor_accuracy_all_methods(self):
        for k, g in self.exp.sweep().groupby("k"):
            self.assertEqual(g.floor_acc.nunique(), 1, f"k={k}")

    def test_k1_identical_error(self):
        sw = self.exp.sweep(ks=[1])
        self.assertTrue(np.allclose(sw.test_mean, sw.test_mean.iloc[0]))

    def test_error_over_all_test_samples(self):
        self.assertEqual(len(self.exp.run("knn", 3)["test_err"]), self.exp.n_test)

    def test_k_selected_on_validation(self):
        for m in self.exp.keys:
            df = self.exp.sweep([m]); self.assertEqual(self.exp.best_k(m), int(df.loc[df.val_mean.idxmin(), "k"]))

    def test_bad_source_row_detected(self):
        def bad(*a):
            d = make_building_data(*a); d.val_rows = d.val_rows[::-1].copy(); return d
        tr, te = make_frames(300, 60)
        with self.assertRaises(ValueError): Experiment.from_frames(tr, te, _cfg(), bad)

    def test_full_run_and_files(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "o"
            cmp_ = run("ignored", out, _cfg(), load_csvs, make_building_data)
            files = ["config.json", "sweep_by_k.csv", "wilcoxon.csv", "by_building.csv", "summary.json",
                     "building_0/split.csv", "building_0/predictions.csv"]
            if importlib.util.find_spec("matplotlib"): files.append("C2_test_mean_error.png")   # biểu đồ chỉ khi có matplotlib
            for f in files:
                self.assertTrue((out / f).exists(), f)
            with self.assertRaises(FileExistsError): run("ignored", out, _cfg(), load_csvs, make_building_data)
        self.assertEqual(len(cmp_.pairs), 3)


if __name__ == "__main__":
    unittest.main()
