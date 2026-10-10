import numpy as np
import pandas as pd
import torch
import unittest

from ips.algorithms.factory import create_model
from ips.algorithms.bsrbf_kan import BSRBF_KANLayer
from ips.algorithms.residual_mixer import ResidualKANMixer
from ips.data.augmentation import DeviceAugmentedRSSDataset
from ips.data.uji import AP_COLUMNS, make_building_data, preprocess_rss


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
        model = create_model("mlp", 520)
        self.assertEqual(tuple(model(torch.zeros(3, 520)).shape), (3, 2))

    def test_bsrbf_layer_and_zero_initialized_mixer(self):
        layer = BSRBF_KANLayer(4, 3, grid_size=4, spline_order=3)
        self.assertEqual(tuple(layer(torch.zeros(2, 4)).shape), (2, 3))
        mixer = ResidualKANMixer(n_blocks=1)
        out = mixer(torch.zeros(2, 520), torch.zeros(2, 64))
        self.assertTrue(bool(torch.allclose(out, torch.zeros_like(out))))
