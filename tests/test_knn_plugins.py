import unittest
from ips.neighbors import available_models
from ips.plugins import discover


class TestDiscovery(unittest.TestCase):
    def test_models_found(self):
        # ba model gốc cộng ba mục chọn WAP (rf, huim_upgrowth, huim_efim) trong cùng danh sách của menu notebook UJI
        self.assertEqual(sorted(available_models()),
                         ["huim_efim", "huim_upgrowth", "knn", "rf", "wknn_inv_d", "wknn_inv_d2"])

    def test_preprocessor_found_by_repo_discover(self):
        found = discover("ips.data.preprocessors", "PREPROCESS_NAME", "build_preprocessor")
        self.assertIn("rss_powed", found)


if __name__ == "__main__":
    unittest.main()
