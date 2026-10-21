import unittest

import numpy as np
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


if __name__ == "__main__":
    unittest.main()
