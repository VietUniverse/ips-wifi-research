import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from ips.data.dataset_adapters import prepare_pack_dataset
from ips.data.uji import AP_COLUMNS, load_csvs


class DatasetAdapterTests(unittest.TestCase):
    def test_hcxy_archive_converts_to_canonical_train_test(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("HCXY/train_HCXY.txt",
                            "RP\tMAC1\tMAC2\n1\t-50\t100\n2\t-60\t-70\n")
            zipped.writestr("HCXY/test_HCXY.txt", "RP\tMAC1\tMAC2\n2\t-55\t-65\n")
            zipped.writestr("HCXY/mapxy_HCXY.txt", "1\t10.0\t20.0\n2\t11.0\t21.0\n")
        archive.seek(0)
        with tempfile.TemporaryDirectory() as temp:
            result = prepare_pack_dataset(archive, "hcxy", temp)
            train, test = load_csvs(temp)
        self.assertEqual((result["n_train"], result["n_test"], result["n_source_aps"]), (2, 1, 2))
        self.assertEqual(len([c for c in train if c.startswith("WAP")]), len(AP_COLUMNS))
        self.assertEqual(train.loc[0, "WAP001"], -50)
        self.assertEqual(train.loc[0, "WAP002"], 100)
        self.assertEqual(train.loc[0, "LONGITUDE"], 10.0)
        self.assertEqual(test.loc[0, "LATITUDE"], 21.0)


if __name__ == "__main__":
    unittest.main()
