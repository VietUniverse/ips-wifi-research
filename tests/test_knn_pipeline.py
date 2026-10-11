import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
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
        with redirect_stdout(io.StringIO()):
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
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            Experiment.from_frames(tr, te, _cfg(), bad)

    def test_full_run_and_files(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "o"
            with redirect_stdout(io.StringIO()):
                cmp_ = run("ignored", out, _cfg(), load_csvs, make_building_data)
            files = ["config.json", "sweep_by_k.csv", "wilcoxon.csv", "by_building.csv", "summary.json",
                     "building_0/split.csv", "building_0/predictions.csv"]
            if importlib.util.find_spec("matplotlib"): files.append("C2_test_mean_error.png")   # biểu đồ chỉ khi có matplotlib
            for f in files:
                self.assertTrue((out / f).exists(), f)
            with self.assertRaises(FileExistsError): run("ignored", out, _cfg(), load_csvs, make_building_data)
        self.assertEqual(len(cmp_.pairs), 3)

    def test_single_model_run_used_by_shared_colab(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "one_model"
            with redirect_stdout(io.StringIO()):
                run("ignored", out, _cfg(models=("knn",)), load_csvs, make_building_data)
            import json
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(set(summary["official_test"]), {"knn"})

    def test_report_figures_for_single_and_multiple_models(self):
        from ips.experiments.visualize import compare_runs, render_run
        with tempfile.TemporaryDirectory() as temp:
            single, multiple = Path(temp) / "single", Path(temp) / "multiple"
            with redirect_stdout(io.StringIO()):
                run("ignored", single, _cfg(models=("knn",)), load_csvs, make_building_data)
                run("ignored", multiple, _cfg(models=("knn", "wknn_inv_d")), load_csvs, make_building_data)
            one, one_figures = render_run(single, show=False)
            many, many_figures = render_run(multiple, show=False)
            comparison, comparison_figures = compare_runs([single, multiple], show=False)
            self.assertEqual(len(one), 1)
            self.assertEqual(len(many), 2)
            self.assertEqual(len(comparison), 3)
            self.assertTrue(any("floor_confusion" in p.name for p in one_figures))
            self.assertTrue(any("timing" in p.name for p in many_figures))
            self.assertTrue(all(p.exists() for p in one_figures + many_figures + comparison_figures))


if __name__ == "__main__":
    unittest.main()
