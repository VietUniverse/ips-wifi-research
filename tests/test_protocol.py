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
from ips.algorithms.cdae_cnn_ccpos import CDAECNNCCpos, _JointConvPlugin, add_gwn, weighted_cdae_loss
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

    def test_cdae_cnn_ccpos_registered_with_correct_output_shape(self):
        self.assertIn("cdae_cnn_ccpos", available_models())
        model = create_model("cdae_cnn_ccpos", 520)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(3, 520))
        self.assertEqual(tuple(out.shape), (3, 2))

    def test_cdae_cnn_ccpos_rejects_too_small_input_dim(self):
        with self.assertRaises(ValueError):
            create_model("cdae_cnn_ccpos", 4)  # pooling stack reduces this to nothing

    def test_add_gwn_leaves_missing_aps_untouched(self):
        x = torch.full((10, 1, 520), -1.0)
        x[:, :, 0] = 0.6  # one observed AP
        torch.manual_seed(0)
        noisy = add_gwn(x, std=0.3)
        self.assertTrue(bool((noisy[:, :, 1:] == -1.0).all()))  # untouched missing APs
        self.assertTrue(bool((noisy >= -1.0).all() and (noisy <= 1.0).all()))  # still clamped to range

    def test_weighted_cdae_loss_upweights_detected_aps(self):
        target = torch.full((4, 1, 520), -1.0)
        target[:, :, :10] = 0.5  # 10 "detected" APs out of 520
        prediction = torch.zeros_like(target)
        prediction[:, :, :10] = -1.0  # make detected-AP error larger than missing-AP error
        plain_mse = torch.nn.functional.mse_loss(prediction, target)
        weighted = weighted_cdae_loss(prediction, target)
        # Detected APs are the overwhelming minority here, so the
        # weighted loss must emphasize them more than plain MSE does.
        self.assertGreater(float(weighted), float(plain_mse))

    def test_cdae_cnn_ccpos_backprops_reconstruction_gradient(self):
        # Mirrors what runner.py's _fit_stage does per step, to confirm
        # the reconstruction loss contributes gradient to the encoder
        # without runner.py knowing about it (see module docstring).
        torch.manual_seed(0)
        model = _JointConvPlugin(CDAECNNCCpos(520))
        model.train()
        encoder_param = next(model.inner.cdae.encoder.parameters())

        xb = torch.randn(4, 520).clamp(-1, 1)
        yb = torch.randn(4, 2)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        grad_after_recon_only = encoder_param.grad.clone()
        position_loss = torch.nn.functional.mse_loss(pred, yb)
        position_loss.backward()
        grad_after_both = encoder_param.grad.clone()

        self.assertFalse(bool(torch.allclose(grad_after_recon_only, torch.zeros_like(grad_after_recon_only))))
        self.assertFalse(bool(torch.allclose(grad_after_both, grad_after_recon_only)))

    def test_runner_trains_cdae_cnn_ccpos_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data_dir = base / "data"
            data_dir.mkdir()
            _frame(20, 1).to_csv(data_dir / "trainingData.csv", index=False)
            _frame(4, 1).to_csv(data_dir / "validationData.csv", index=False)
            result = run(data_dir, base / "run", RunConfig(
                model="cdae_cnn_ccpos", preprocess="rss_standard", buildings=(1,),
                stage1_epochs=0, stage2_epochs=1, patience=1,
            ))
            self.assertEqual(result["n"], 4)
