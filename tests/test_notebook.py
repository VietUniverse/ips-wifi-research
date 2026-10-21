"""Keep the shared Colab entry point parseable on every branch."""

import json
import io
import ast
import subprocess
import sys
import tempfile
import unittest
import urllib.request
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch
from urllib.error import HTTPError


class NotebookTests(unittest.TestCase):
    def test_bad_token_stops_with_actionable_error(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        source = "".join(json.loads(path.read_text(encoding="utf-8"))["cells"][1]["source"])
        google, colab = ModuleType("google"), ModuleType("google.colab")
        files, userdata = ModuleType("google.colab.files"), ModuleType("google.colab.userdata")
        files.upload = Mock()
        userdata.get = Mock(return_value="invalid-secret")
        colab.files, colab.userdata, google.colab = files, userdata, colab
        with tempfile.TemporaryDirectory() as temp:
            source = source.replace('root = Path("/content/repo_source")',
                                    f"root = Path({temp!r}) / 'download'")
            failure = HTTPError("https://api.github.com", 401, "Unauthorized", None, None)
            with patch.dict(sys.modules, {"google": google, "google.colab": colab,
                                          "google.colab.files": files,
                                          "google.colab.userdata": userdata}), \
                 patch("urllib.request.urlopen", side_effect=failure), \
                 patch("builtins.input", return_value="main"), \
                 redirect_stdout(io.StringIO()):
                namespace = {}
                with self.assertRaisesRegex(RuntimeError, "HTTP 401.*Contents: read"):
                    exec(compile(source, "shared bootstrap", "exec"), namespace)
            files.upload.assert_not_called()

    def test_bootstrap_scans_all_feature_branches_and_merges_new_plugins(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        setup = "".join(cells[1]["source"])
        self.assertIn('extract_branch(source_ref, root)', setup)
        self.assertIn('startswith("feature/")', setup)
        self.assertIn('src", "ips", "algorithms"', setup)
        self.assertIn('"data", "preprocessors")', setup)
        self.assertIn('"dataset_adapters.py"', setup)
        self.assertIn('target.exists()', setup)

    def test_colab_dataset_menu_offers_uji_and_zip_datasets(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        data_setup = "".join(cells[2]["source"])
        self.assertIn("UJIIndoorLoc (UCI)", data_setup)
        self.assertIn("PACK_DATASETS", data_setup)
        self.assertIn("files.upload()", data_setup)
        run_cell = "".join(cells[5]["source"])
        self.assertIn('settings["buildings"] = [0]', run_cell)
        self.assertIn("str(selected_config)", run_cell)

    def test_shared_colab_code_cells_compile_and_are_clean(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"notebook cell {index}", "exec")

    def test_floor_error_export_uses_ground_truth_floor_for_neural_and_knn(self):
        import pandas as pd

        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        run_cell = "".join(cells[5]["source"])
        tree = ast.parse(run_cell)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "export_floor_metrics")
        namespace = {"Path": Path, "display": lambda value: None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "floor metrics", "exec"), namespace)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            model_dir = root / "mlp"
            building_dir = model_dir / "building_0"
            data.mkdir()
            building_dir.mkdir(parents=True)
            pd.DataFrame({"FLOOR": [0, 0, 1, 1]}).to_csv(
                data / "validationData.csv", index=False)
            pd.DataFrame({"source_row": [0, 1, 2, 3], "error_m": [1.0, 3.0, 2.0, 10.0]}).to_csv(
                building_dir / "predictions.csv", index=False)
            with redirect_stdout(io.StringIO()):
                report = namespace["export_floor_metrics"](model_dir, data)

            self.assertEqual(report["model"].tolist(), ["mlp", "mlp"])
            self.assertEqual(report["floor"].tolist(), [0, 1])
            self.assertEqual(report["n_test"].tolist(), [2, 2])
            self.assertEqual(report["mean_m"].tolist(), [2.0, 6.0])
            self.assertEqual(report["within_5m_pct"].tolist(), [100.0, 50.0])
            self.assertTrue((model_dir / "floor_metrics.csv").exists())

    def test_dataset_comparison_allows_independent_test_sets(self):
        import ast
        import pandas as pd

        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        source = "".join(cells[5]["source"])
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "compare_dataset_runs")
        namespace = {"Path": Path, "display": lambda value: None}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "dataset comparison", "exec"), namespace)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = []
            for dataset, errors in (("UJI", [1.0, 4.0]), ("DLULIB", [2.0, 3.0, 9.0])):
                output = root / dataset
                building = output / "building_0"
                building.mkdir(parents=True)
                pd.DataFrame({"error_m": errors}).to_csv(building / "predictions.csv", index=False)
                records.append({"dataset_label": dataset, "output_dir": output,
                                "model_label": "MLP"})
            with redirect_stdout(io.StringIO()):
                report = namespace["compare_dataset_runs"](records, root / "comparison")

            self.assertEqual(report.dataset.tolist(), ["DLULIB", "UJI"])
            self.assertEqual(report.n_test.tolist(), [3, 2])
            self.assertEqual(report.mean_m.tolist(), [14 / 3, 2.5])
            self.assertTrue((root / "comparison" / "dataset_comparison_metrics.csv").exists())
            self.assertTrue((root / "comparison" / "dataset_comparison_mean_error.png").exists())
            self.assertTrue((root / "comparison" / "dataset_comparison_cdf.png").exists())

    def test_model_menu_waits_for_explicit_selection(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        namespace = {"project": path.parents[1], "Path": Path,
                     "subprocess": subprocess, "sys": sys}
        available = json.dumps({"models": ["mlp", "bsrbf_kan"],
                                "preprocessors": ["rss_standard"]})
        with patch.object(subprocess, "check_output", return_value=available), \
             patch("builtins.input", side_effect=["1", "1", "1", ""]) as ask, \
             redirect_stdout(io.StringIO()):
            exec(menu, namespace)
        self.assertEqual(namespace["selected_models"], ["bsrbf_kan"])
        self.assertEqual(namespace["selected_family"], "Neural")
        self.assertEqual(namespace["selected_preprocesses"], ["rss_standard"])
        self.assertIn(("bsrbf_kan", "rss_standard"), namespace["selected_preset_map"])
        self.assertEqual(ask.call_count, 4)

    def test_kan_mixer_is_an_explicit_menu_choice_with_mixer_presets(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        available = json.dumps({"models": ["bsrbf_kan", "mlp"],
                                "preprocessors": ["rss_standard"]})
        namespace = {"project": path.parents[1], "Path": Path,
                     "subprocess": subprocess, "sys": sys}
        available_ids = ["bsrbf_kan", "mlp", "kan", "fastkan", "skan", "fckan",
                         "kan_mixer", "kan_spline_mixer", "fastkan_mixer",
                         "skan_mixer", "fckan_mixer"]
        mixer_choice = sorted(available_ids).index("kan_mixer") + 1
        with patch.object(subprocess, "check_output", return_value=available), \
             patch("builtins.input", side_effect=["1", str(mixer_choice), "1", "1", "1", ""]) as ask, \
             redirect_stdout(io.StringIO()) as output:
            exec(menu, namespace)
        self.assertEqual(namespace["selected_models"], ["kan_mixer"])
        preset = namespace["selected_preset_map"][("kan_mixer", "rss_standard")]
        self.assertTrue(json.loads((namespace["project"] / preset).read_text())["with_mixer"])
        self.assertIn("KAN-Mixer (BSRBF-KAN + Residual Mixer)", output.getvalue())
        self.assertEqual(ask.call_count, 6)

    def test_mixer_can_be_selected_alongside_its_plain_backbone(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        available = json.dumps({"models": ["bsrbf_kan", "mlp"],
                                "preprocessors": ["rss_standard"]})
        namespace = {"project": path.parents[1], "Path": Path,
                     "subprocess": subprocess, "sys": sys}
        with patch.object(subprocess, "check_output", return_value=available), \
             patch("builtins.input", side_effect=["1", "1,7", "1", "1", "1", ""]), \
             redirect_stdout(io.StringIO()):
            exec(menu, namespace)

        self.assertEqual(namespace["selected_models"], ["bsrbf_kan", "kan_mixer"])
        mapping = namespace["selected_preset_map"]
        plain = namespace["preset_settings"][mapping[("bsrbf_kan", "rss_standard")]]
        mixed = namespace["preset_settings"][mapping[("kan_mixer", "rss_standard")]]
        self.assertFalse(plain["with_mixer"])
        self.assertTrue(mixed["with_mixer"])
        run_cell = "".join(cells[5]["source"])
        self.assertIn('settings["with_mixer"] = selected_model in mixer_model_bases', run_cell)
        self.assertIn('settings.update(mixer_arch="ap_token"', run_cell)
        self.assertIn("mixer_isolate_rng=True", run_cell)

    def test_multiple_models_and_preprocessors_get_model_specific_presets(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            config_dir = project / "configs"
            config_dir.mkdir()
            for name, config in {
                "kan_baseline.json": {"model": "kan", "with_mixer": False},
                "cdae_cnn_ccpos_baseline.json": {
                    "model": "cdae_cnn_ccpos", "with_mixer": False},
                "cdae_cnn_ccpos_robust.json": {
                    "model": "cdae_cnn_ccpos", "with_mixer": False,
                    "preprocess": "rss_ap_robust", "preprocess_params": {"trim": 1}},
                "bsrbf_mixer_ap_tokens.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust", "mixer_arch": "ap_token"},
                "bsrbf_rf128_mixer_ap_robust.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust", "rf_feature_selection": True},
            }.items():
                (config_dir / name).write_text(json.dumps(config), encoding="utf-8")

            available = json.dumps({
                "models": ["cdae_cnn_ccpos", "fastkan", "mlp"],
                "preprocessors": ["rss_standard", "rss_ap_robust"],
            })
            namespace = {"project": project, "Path": Path,
                         "subprocess": subprocess, "sys": sys}
            with patch.object(subprocess, "check_output", return_value=available), \
                 patch("builtins.input", side_effect=["1", "2,3", "1,2", "", ""]), \
                 redirect_stdout(io.StringIO()):
                exec(menu, namespace)

        self.assertEqual(namespace["selected_models"], ["cdae_cnn_ccpos", "fastkan"])
        self.assertEqual(namespace["selected_preprocesses"],
                         ["rss_standard", "rss_ap_robust"])
        mapping = namespace["selected_preset_map"]
        self.assertEqual(mapping[("cdae_cnn_ccpos", "rss_ap_robust")],
                         "configs/cdae_cnn_ccpos_robust.json")
        self.assertEqual(mapping[("fastkan", "rss_standard")], "configs/kan_baseline.json")
        self.assertEqual(mapping[("fastkan", "rss_ap_robust")], "configs/kan_baseline.json")

    def test_ap_token_mixer_preset_wins_for_robust_rss(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            config_dir = project / "configs"
            config_dir.mkdir()
            configs = {
                "bsrbf_mixer_ap_robust.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust"},
                "bsrbf_rf128_mixer_ap_robust.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust", "rf_feature_selection": True},
                "bsrbf_mixer_ap_tokens.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust", "mixer_arch": "ap_token"},
            }
            for name, config in configs.items():
                (config_dir / name).write_text(json.dumps(config), encoding="utf-8")
            available = json.dumps({"models": ["bsrbf_kan", "kan", "fastkan", "skan", "fckan"],
                                    "preprocessors": ["rss_ap_robust"]})
            namespace = {"project": project, "Path": Path,
                         "subprocess": subprocess, "sys": sys}
            available_ids = ["bsrbf_kan", "kan", "fastkan", "skan", "fckan",
                             "kan_mixer", "kan_spline_mixer", "fastkan_mixer",
                             "skan_mixer", "fckan_mixer"]
            mixer_choice = sorted(available_ids).index("kan_mixer") + 1
            with patch.object(subprocess, "check_output", return_value=available), \
                 patch("builtins.input", side_effect=["1", str(mixer_choice), "1", "1", "1", ""]), \
                 redirect_stdout(io.StringIO()):
                exec(menu, namespace)
            self.assertEqual(namespace["selected_preset_map"][("kan_mixer", "rss_ap_robust")],
                             "configs/bsrbf_mixer_ap_tokens.json")

    def test_phone_domain_preset_respects_preprocess_and_spatial_choice(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "Colab_Train_UJI.ipynb"
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        menu = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            config_dir = project / "configs"
            config_dir.mkdir()
            configs = {
                "bsrbf_mixer_ap_robust.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust", "validation_group": "location"},
                "bsrbf_mixer_uji_phone_robust_no_aux.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust", "validation_group": "phone",
                    "spatial_clusters": 0},
                "bsrbf_mixer_uji_phone_robust.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_ap_robust", "validation_group": "phone",
                    "spatial_clusters": 8},
                "bsrbf_mixer_uji_phone_relative_no_aux.json": {
                    "model": "bsrbf_kan", "with_mixer": True,
                    "preprocess": "rss_global_relative", "validation_group": "phone",
                    "spatial_clusters": 0},
            }
            for name, config in configs.items():
                (config_dir / name).write_text(json.dumps(config), encoding="utf-8")
            available = json.dumps({"models": ["bsrbf_kan", "mlp"],
                                    "preprocessors": ["rss_ap_robust", "rss_global_relative"]})
            mixer_choice = sorted(["bsrbf_kan", "mlp", "kan", "fastkan", "skan", "fckan",
                                   "kan_mixer", "kan_spline_mixer", "fastkan_mixer",
                                   "skan_mixer", "fckan_mixer"]).index("kan_mixer") + 1
            for spatial_answer, expected in [
                ("1", "configs/bsrbf_mixer_uji_phone_robust_no_aux.json"),
                ("2", "configs/bsrbf_mixer_uji_phone_robust.json"),
            ]:
                namespace = {"project": project, "Path": Path,
                             "subprocess": subprocess, "sys": sys}
                answers = ["1", str(mixer_choice), "1", "1", spatial_answer, ""]
                with patch.object(subprocess, "check_output", return_value=available), \
                     patch("builtins.input", side_effect=answers), redirect_stdout(io.StringIO()):
                    exec(menu, namespace)
                self.assertEqual(namespace["selected_preset_map"][("kan_mixer", "rss_ap_robust")],
                                 expected)

    def test_direct_mixer_uses_its_own_preset_without_residual_wrapper(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/Colab_Train_UJI.ipynb").read_text(encoding="utf-8"))
        menu = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
                    and "selected_family = choose_one" in "".join(c["source"]))
        available = json.dumps({"models": ["kan_mixer_e2e"], "preprocessors": ["rss_standard"]})
        namespace = {"project": root, "Path": Path, "subprocess": subprocess, "sys": sys}
        ids = sorted(["bsrbf_kan", "kan", "fastkan", "skan", "fckan", "kan_mixer",
                      "kan_spline_mixer", "fastkan_mixer", "skan_mixer", "fckan_mixer", "kan_mixer_e2e"])
        with patch.object(subprocess, "check_output", return_value=available), \
             patch("builtins.input", side_effect=["1", str(ids.index("kan_mixer_e2e") + 1), "1", ""]), \
             redirect_stdout(io.StringIO()):
            exec(menu, namespace)
        self.assertEqual(namespace["selected_models"], ["kan_mixer_e2e"])
        preset = namespace["selected_preset_map"][("kan_mixer_e2e", "rss_standard")]
        self.assertEqual(preset, "configs/kan_mixer_e2e.json")
        self.assertFalse(namespace["preset_settings"][preset]["with_mixer"])
