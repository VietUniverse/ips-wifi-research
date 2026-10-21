import numpy as np
import pandas as pd
import torch
import unittest
import importlib
import sys
import tempfile
from pathlib import Path

from ips.algorithms.factory import available_models, create_model
from ips.algorithms.bsrbf_kan import BSRBF_KANLayer
from ips.algorithms.residual_mixer import ResidualKANMixer
from ips.data.augmentation import DeviceAugmentedRSSDataset
from ips.data.preprocessors.factory import available_preprocessors, create_preprocessor
from ips.data.uji import AP_COLUMNS, make_building_data, preprocess_rss
from ips.experiments.runner import RunConfig, run
from ips.plugins import discover


def _frame(n, building):
    values = np.full((n, 520), 100.0, dtype=np.float32)
    values[:, 0] = -65.0
    frame = pd.DataFrame(values, columns=AP_COLUMNS)
    frame["BUILDINGID"] = building
    frame["FLOOR"] = 0
    frame["LONGITUDE"] = np.repeat(np.arange(n // 2), 2)[:n].astype(float)
    frame["LATITUDE"] = np.repeat(np.arange(n // 2), 2)[:n].astype(float)
    return frame


class ProtocolTests(unittest.TestCase):
    def test_missing_ap_stays_missing_during_augmentation(self):
        x = preprocess_rss(_frame(10, 1))
        augmented, _ = DeviceAugmentedRSSDataset(x, np.zeros((10, 2)))[0]
        self.assertTrue(bool((augmented[1:] == -1).all()))

    def test_location_split_uses_no_official_test_rows(self):
        train, official = _frame(20, 1), _frame(4, 1)
        data = make_building_data(train, official, 1, seed=42)
        self.assertTrue(set(data.train_rows).isdisjoint(data.val_rows))
        self.assertEqual(len(data.test_rows), 4)
        self.assertTrue(set(data.test_rows).issubset(official.index))

    def test_phone_holdout_validation_has_no_phone_leakage(self):
        train, official = _frame(40, 1), _frame(4, 1)
        train["PHONEID"] = np.repeat(np.arange(4), 10)
        data = make_building_data(train, official, 1, seed=42,
                                  validation_group="phone")
        self.assertTrue(set(data.train_rows).isdisjoint(data.val_rows))
        self.assertTrue(set(train.loc[data.train_rows, "PHONEID"]).isdisjoint(
            set(train.loc[data.val_rows, "PHONEID"])))
        with self.assertRaisesRegex(ValueError, "requires PHONEID"):
            make_building_data(_frame(20, 1), official, 1, validation_group="phone")

    def test_sparse_phone_groups_fall_back_to_location_validation(self):
        train, official = _frame(20, 1), _frame(4, 1)
        train["PHONEID"] = np.repeat([10, 20], 10)
        data = make_building_data(train, official, 1, seed=42,
                                  validation_group="phone")
        self.assertEqual(data.validation_group_used, "location_fallback_sparse_groups")
        self.assertLess(len(data.x_val), len(data.x_train))

    def test_model_registry_output_shape(self):
        self.assertIn("mlp", available_models())
        self.assertIn("rss_standard", available_preprocessors())
        model = create_model("mlp", 520)
        self.assertEqual(tuple(model(torch.zeros(3, 520)).shape), (3, 2))

    def test_new_plugin_file_is_discovered_without_registry_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            package = Path(temp) / "example_plugins"
            package.mkdir()
            (package / "__init__.py").write_text("")
            sys.path.insert(0, temp)
            try:
                self.assertEqual(discover("example_plugins", "MODEL_NAME", "build_model"), {})
                (package / "toy.py").write_text(
                    'MODEL_NAME = "toy"\ndef build_model(input_dim):\n    return input_dim\n')
                self.assertEqual(discover("example_plugins", "MODEL_NAME", "build_model")["toy"](7), 7)
            finally:
                sys.path.remove(temp)
                sys.modules.pop("example_plugins.toy", None)
                sys.modules.pop("example_plugins", None)
                importlib.invalidate_caches()

    def test_ap_frequency_uses_only_fitted_train_rows(self):
        train = _frame(10, 1)
        official = _frame(4, 1)
        official["WAP002"] = -70.0
        processor = create_preprocessor("ap_frequency", min_observations=2).fit(train)
        self.assertEqual(processor.feature_columns, ["WAP001"])
        self.assertEqual(processor.transform(official).shape, (4, 1))
        data = make_building_data(train, official, 1, preprocess_name="ap_frequency",
                                  preprocess_params={"min_observations": 2})
        self.assertEqual(data.feature_columns, ["WAP001"])

    def test_mixer_normalizations_preserve_520_ap_and_missing_mask(self):
        frame = _frame(12, 1)
        frame.loc[:, "WAP001"] = np.linspace(-90, -50, len(frame)).astype(np.float32)
        frame.loc[:5, "WAP002"] = -70.0  # flat/low-information AP should be maskable
        for name in ("rss_global_relative", "rss_row_rank", "rss_ap_robust"):
            processor = create_preprocessor(name).fit(frame.iloc[:8])
            result = processor.transform(frame.iloc[8:])
            self.assertEqual(result.shape, (4, 520))
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.all(result[:, 2:] == -1.0))
            self.assertTrue(np.all(np.isfinite(result)))

    def test_location_balancing_and_augmentation_are_configurable(self):
        self.assertTrue(RunConfig(balance_locations=True).balance_locations)
        dataset = DeviceAugmentedRSSDataset(np.zeros((3, 520), np.float32),
                                            np.zeros((3, 2), np.float32),
                                            gain_std=0.2, noise_std=0.1,
                                            dropout_prob=0.15)
        self.assertEqual(len(dataset), 3)

    def test_runner_accepts_discovered_preprocessor(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data_dir = base / "data"
            data_dir.mkdir()
            _frame(20, 1).to_csv(data_dir / "trainingData.csv", index=False)
            _frame(4, 1).to_csv(data_dir / "validationData.csv", index=False)
            result = run(data_dir, base / "run", RunConfig(
                model="mlp", preprocess="ap_frequency",
                preprocess_params={"min_observations": 2}, buildings=(1,),
                stage1_epochs=0, stage2_epochs=1, patience=1,
            ))
            self.assertEqual(result["n"], 4)
            import json
            summary = json.loads((base / "run" / "summary.json").read_text())
            self.assertEqual(summary["by_building"][0]["n_features"], 1)

    def test_runner_can_select_rss_features_with_random_forest(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data_dir = base / "data"
            data_dir.mkdir()
            _frame(30, 1).to_csv(data_dir / "trainingData.csv", index=False)
            _frame(4, 1).to_csv(data_dir / "validationData.csv", index=False)
            result = run(data_dir, base / "run", RunConfig(
                model="mlp", preprocess="ap_frequency",
                preprocess_params={"min_observations": 2}, buildings=(1,),
                stage1_epochs=0, stage2_epochs=1, patience=1,
                rf_feature_selection=True, rf_top_k=1, rf_n_estimators=5,
            ))
            self.assertEqual(result["n"], 4)
            import json
            summary = json.loads((base / "run" / "summary.json").read_text())
            row = summary["by_building"][0]
            self.assertEqual(row["rf_original_features"], 1)
            self.assertEqual(row["rf_selected_features"], 1)
            self.assertTrue((base / "run" / "building_1" / "rf_ap_importance.csv").exists())

    def test_runner_trains_one_block_mixer_with_metric_coordinate_loss(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data_dir = base / "data"
            data_dir.mkdir()
            _frame(30, 1).to_csv(data_dir / "trainingData.csv", index=False)
            _frame(4, 1).to_csv(data_dir / "validationData.csv", index=False)
            run(data_dir, base / "run", RunConfig(
                model="bsrbf_kan", buildings=(1,), stage1_epochs=0,
                stage2_epochs=1, patience=1, with_mixer=True,
                mixer_arch="ap_token", mixer_token_type="mlp",
                mixer_channel_type="mlp", mixer_epochs=1,
                mixer_patience=1, mixer_batch_size=64, mixer_blocks=1,
                mixer_metric_coordinate_loss=True,
            ))
            import json
            summary = json.loads((base / "run" / "summary.json").read_text())
            self.assertGreater(summary["by_building"][0]["mixer_parameters"], 0)

    def test_bsrbf_layer_and_zero_initialized_mixer(self):
        layer = BSRBF_KANLayer(4, 3, grid_size=4, spline_order=3)
        self.assertEqual(tuple(layer(torch.zeros(2, 4)).shape), (2, 3))
        mixer = ResidualKANMixer(n_blocks=1)
        out = mixer(torch.zeros(2, 520), torch.zeros(2, 64))
        self.assertTrue(bool(torch.allclose(out, torch.zeros_like(out))))

    def test_ap_token_mixer_preserves_ap_ids_and_channel_path(self):
        mixer = ResidualKANMixer(n_blocks=1, rss_dim=3, arch="ap_token",
                                 ap_indices=[0, 15, 519])
        self.assertEqual(tuple(mixer.ap_ids.tolist()), (0, 15, 519))
        self.assertEqual(mixer.blocks[0].token_net[0].input_dim, 3)
        self.assertEqual(mixer.blocks[0].channel_net[0].input_dim, 8)
        out = mixer(torch.tensor([[-0.4, -1.0, 0.8]]), torch.zeros(1, 64))
        self.assertEqual(tuple(out.shape), (1, 2))
        self.assertTrue(bool(torch.allclose(out, torch.zeros_like(out))))
