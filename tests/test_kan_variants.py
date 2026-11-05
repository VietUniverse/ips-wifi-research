"""Shape and gradient checks for paper-inspired KAN layer variants."""
import unittest

import torch

from ips.algorithms.factory import available_models, create_model
from ips.algorithms.kan_layers import KANBackbone, make_kan_layer
from ips.algorithms.residual_mixer import ResidualKANMixer


class KANVariantTests(unittest.TestCase):
    def test_paper_families_are_discovered(self):
        available = set(available_models())
        self.assertTrue({"kan", "fastkan", "bsrbf_kan", "skan", "fckan"}.issubset(available))

    def test_each_variant_layer_has_finite_forward_and_gradient(self):
        for family in ("kan", "fastkan", "bsrbf_kan", "skan", "fckan"):
            with self.subTest(family=family):
                layer = make_kan_layer(family, 12, 7)
                x = torch.randn(3, 12, requires_grad=True)
                y = layer(x)
                self.assertEqual(tuple(y.shape), (3, 7))
                self.assertTrue(torch.isfinite(y).all())
                y.square().mean().backward()
                self.assertIsNotNone(x.grad)

    def test_all_backbones_and_matching_mixer_layers_run(self):
        for family in ("kan", "fastkan", "bsrbf_kan", "skan", "fckan"):
            with self.subTest(family=family):
                model = KANBackbone(12, family, hidden=(8, 6, 2))
                self.assertEqual(tuple(model(torch.randn(2, 12)).shape), (2, 2))
                mixer = ResidualKANMixer(n_blocks=1, family=family)
                output = mixer(torch.randn(2, 520), torch.randn(2, 64))
                self.assertEqual(tuple(output.shape), (2, 2))
                self.assertTrue(torch.isfinite(output).all())

    def test_all_token_channel_mixer_combinations_are_trainable(self):
        for token_type, channel_type in (("kan", "kan"), ("kan", "mlp"),
                                         ("mlp", "kan"), ("mlp", "mlp")):
            with self.subTest(token=token_type, channel=channel_type):
                mixer = ResidualKANMixer(n_blocks=1, family="bsrbf_kan", rss_dim=5,
                                         arch="ap_token", token_type=token_type,
                                         channel_type=channel_type)
                out = mixer(torch.randn(2, 5), torch.randn(2, 64))
                self.assertEqual(tuple(out.shape), (2, 2))
                self.assertTrue(torch.isfinite(out).all())
                block_input = torch.randn(2, 5, 8, requires_grad=True)
                mixer.blocks(block_input).square().mean().backward()
                self.assertIsNotNone(block_input.grad)
                self.assertTrue(torch.isfinite(block_input.grad).all())

    def test_mixer_accepts_random_forest_selected_rss_dimension(self):
        mixer = ResidualKANMixer(n_blocks=1, family="bsrbf_kan", rss_dim=128)
        output = mixer(torch.randn(2, 128), torch.randn(2, 64))
        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_plugin_backbones_use_standard_architecture_and_output(self):
        for family in ("kan", "fastkan", "skan", "fckan"):
            with self.subTest(family=family):
                model = create_model(family, 20)
                self.assertEqual(tuple(model(torch.randn(2, 20)).shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
