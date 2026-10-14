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
from ips.algorithms.mmj_common import CDAE, SAE, _JointMixerPlugin
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

    def test_bsrbf_layer_and_zero_initialized_mixer(self):
        layer = BSRBF_KANLayer(4, 3, grid_size=4, spline_order=3)
        self.assertEqual(tuple(layer(torch.zeros(2, 4)).shape), (2, 3))
        mixer = ResidualKANMixer(n_blocks=1)
        out = mixer(torch.zeros(2, 520), torch.zeros(2, 64))
        self.assertTrue(bool(torch.allclose(out, torch.zeros_like(out))))

    def test_sae_cdae_mmj_registered_with_correct_output_shape(self):
        self.assertIn("sae_mmj", available_models())
        self.assertIn("cdae_mmj", available_models())
        for name in ("sae_mmj", "cdae_mmj"):
            model = create_model(name, 20)
            model.eval()
            with torch.no_grad():
                out = model(torch.zeros(3, 20))
            self.assertEqual(tuple(out.shape), (3, 2))

    def test_cdae_corruption_only_touches_observed_aps_in_training(self):
        from ips.algorithms.mmj_common import corrupt_rss
        x = torch.full((50, 20), -1.0)
        x[:, 0] = 0.8  # one observed AP, every row
        corrupted = corrupt_rss(x, prob=1.0)
        # Missing APs (-1) must stay untouched; only the single observed
        # column could have been forced back to "not detected".
        self.assertTrue(bool((corrupted[:, 1:] == -1.0).all()))
        self.assertTrue(bool((corrupted[:, 0] == -1.0).all()))  # prob=1.0: always corrupted

        cdae = CDAE(20, corruption_prob=1.0)
        cdae.train()
        _, reconstruction = cdae(x)
        self.assertEqual(tuple(reconstruction.shape), x.shape)

    def test_joint_mixer_plugin_backprops_reconstruction_gradient(self):
        # Mirrors exactly what runner.py's _fit_stage does per step, to
        # confirm the reconstruction loss really contributes gradient
        # without runner.py knowing about it (see mmj_common docstring).
        torch.manual_seed(0)
        model = _JointMixerPlugin(SAE(20), name="SAE-MMJ-test")
        model.train()
        encoder_param = next(model.inner.extractor.encoder.parameters())

        xb = torch.randn(4, 20)
        yb = torch.randn(4, 2)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)  # forward() does the reconstruction-loss backward internally
        position_loss = torch.nn.functional.mse_loss(pred, yb)
        grad_after_recon_only = encoder_param.grad.clone()
        position_loss.backward()
        grad_after_both = encoder_param.grad.clone()

        # The reconstruction pass alone must already have produced a
        # nonzero gradient (proves the aux loss is wired in), and the
        # position backward must add to it rather than overwrite it.
        self.assertFalse(bool(torch.allclose(grad_after_recon_only, torch.zeros_like(grad_after_recon_only))))
        self.assertFalse(bool(torch.allclose(grad_after_both, grad_after_recon_only)))

    def test_runner_trains_sae_mmj_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data_dir = base / "data"
            data_dir.mkdir()
            _frame(20, 1).to_csv(data_dir / "trainingData.csv", index=False)
            _frame(4, 1).to_csv(data_dir / "validationData.csv", index=False)
            result = run(data_dir, base / "run", RunConfig(
                model="sae_mmj", preprocess="rss_standard", buildings=(1,),
                stage1_epochs=0, stage2_epochs=1, patience=1,
            ))
            self.assertEqual(result["n"], 4)
