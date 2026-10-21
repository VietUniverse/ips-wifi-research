import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ips.algorithms.kan_mixer_e2e import DirectKANMixer
from ips.experiments.joint_mixer import JointKANMixer, cache_prefix, fit_joint, predict, snapshot, to_meters
from ips.experiments.runner import metric_coordinate_loss


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 64), nn.Linear(64, 64), nn.Linear(64, 2)])


class ToyMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(68, 2)

    def forward(self, rss, features):
        return self.linear(torch.cat((rss, features), dim=1))


class DirectJointMixerTests(unittest.TestCase):
    def test_direct_mixer_both_axes_receive_gradients(self):
        model = DirectKANMixer(5, tokens=4, channels=4, blocks=1)
        x = torch.tensor([[-.2, -.5, -1., .2, -1.], [-1., -1., -1., -1., -1.]])
        result = model(x)
        self.assertEqual(tuple(result.shape), (2, 2))
        self.assertTrue(torch.isfinite(result).all())
        result.square().sum().backward()
        for block in (model.blocks[0].token_mixer, model.blocks[0].channel_mixer):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters()))

    def test_relative_view_cancels_scan_gain_and_preserves_missing_mask(self):
        original = torch.tensor([[-.4, -.2, -1., .3]])
        shifted = torch.where(original > -.999, original + .1, original)
        a, b = DirectKANMixer.input_views(original), DirectKANMixer.input_views(shifted)
        torch.testing.assert_close(a[:, 4:], b[:, 4:])
        self.assertEqual(a[0, 2], 0)
        self.assertEqual(a[0, 6], 0)
        self.assertEqual(a[0, 10], 0)

    def test_metric_loss_respects_coordinate_scales(self):
        scale = torch.tensor([10., 2.])
        x_error = metric_coordinate_loss(torch.tensor([[1., 0.]]), torch.zeros(1, 2), scale)
        y_error = metric_coordinate_loss(torch.tensor([[0., 1.]]), torch.zeros(1, 2), scale)
        self.assertAlmostEqual(float(x_error / y_error), 5., places=5)

    def test_joint_gradients_reach_suffix_and_mixer_not_frozen_prefix(self):
        model = JointKANMixer(ToyBackbone(), ToyMixer(), .5, trainable_layers=2)
        rss = torch.randn(3, 4)
        model(rss).square().sum().backward()
        self.assertIsNone(model.backbone.layers[0].weight.grad)
        self.assertGreater(float(model.backbone.layers[1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.mixer.linear.weight.grad.abs().sum()), 0)
        model.eval()
        cached = cache_prefix(model, rss.numpy(), torch.device("cpu"))
        np.testing.assert_allclose(predict(model, rss.numpy(), cached, torch.device("cpu")),
                                   model(rss).detach().numpy(), atol=1e-6)

    def test_validation_keeps_epoch_zero_when_no_improvement(self):
        model = JointKANMixer(ToyBackbone(), ToyMixer(), .5, trainable_layers=2)
        before = snapshot(model)
        x = np.random.default_rng(42).normal(size=(4, 4)).astype(np.float32)
        y = np.zeros((4, 2))
        result = fit_joint(model, x, y, x, y, np.zeros(2), np.ones(2), torch.device("cpu"),
                           epochs=1, patience=1, batch_size=4, lr=0., backbone_lr=0.)
        self.assertEqual(result["best_epoch"], 0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_inverse_coordinates_keep_submeter_precision(self):
        coordinates = to_meters(np.array([[.01, .02]], np.float32), [0., 4864900.], [1., 1.])
        self.assertEqual(coordinates.dtype, np.float64)
        self.assertAlmostEqual(coordinates[0, 1] - 4864900., .02, places=7)

    def test_full_refit_consumes_all_training_rows_and_exports_no_validation_for_ensemble(self):
        from sklearn.preprocessing import StandardScaler
        from ips.data.uji import AP_COLUMNS, make_building_data
        from ips.experiments.refit_direct_mixer import refit_building
        train = pd.DataFrame(np.full((24, 520), -80.), columns=AP_COLUMNS)
        train["LONGITUDE"] = np.arange(24, dtype=float)
        train["LATITUDE"] = 4864900. + np.arange(24, dtype=float)
        train["BUILDINGID"], train["FLOOR"] = 0, 0
        official = train.iloc[:4].copy()
        data = make_building_data(train, official, 0)
        scaler = StandardScaler().fit(data.y_train)
        model = DirectKANMixer(520)
        checkpoint = {"model": "kan_mixer_e2e", "state_dict": snapshot(model),
                      "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                      "ap_columns": data.feature_columns, "preprocess_state": data.preprocess_state}
        config = {"model": "kan_mixer_e2e", "preprocess": "rss_standard", "batch_size": 24,
                  "augment_gain_std": 0., "augment_noise_std": 0., "augment_dropout_prob": 0.}
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp) / "source", Path(temp) / "output"
            (source / "building_0").mkdir(parents=True)
            torch.save(checkpoint, source / "building_0/checkpoint.pt")
            frame, score = refit_building(source, output, train, official, config, 0, epochs=1)
            self.assertEqual(score["n_train"], 24)
            self.assertEqual(len(frame), 4)
            self.assertTrue(np.isfinite(frame["error_m"]).all())
            self.assertFalse((output / "building_0/validation_predictions.csv").exists())
            saved = torch.load(output / "building_0/checkpoint.pt", weights_only=False)
            self.assertEqual(saved["full_train_refit"]["n"], 24)


if __name__ == "__main__":
    unittest.main()
