"""Tests for the opt-in ccpos.ipynb-faithful pipeline (ips.algorithms.ccpos_notebook),
kept in its own file so tests/test_protocol.py is untouched."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from ips.data.uji import AP_COLUMNS
from ips.experiments.runner import RunConfig, run


def _frame(n, building, floor=0):
    values = np.full((n, 520), 100.0, dtype=np.float32)
    values[:, 0] = -65.0
    frame = pd.DataFrame(values, columns=AP_COLUMNS)
    frame["BUILDINGID"] = building
    frame["FLOOR"] = floor
    frame["LONGITUDE"] = np.arange(n, dtype=float)
    frame["LATITUDE"] = np.arange(n, dtype=float)
    return frame


class CCPosNotebookPipelineTests(unittest.TestCase):
    def test_runs_end_to_end_and_matches_output_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data_dir = base / "data"
            data_dir.mkdir()
            train = pd.concat([
                _frame(16, 0, floor=0), _frame(16, 0, floor=1),
                _frame(16, 1, floor=0),
            ], ignore_index=True)
            test = pd.concat([_frame(4, 0, floor=0), _frame(4, 1, floor=0)], ignore_index=True)
            train.to_csv(data_dir / "trainingData.csv", index=False)
            test.to_csv(data_dir / "validationData.csv", index=False)

            result = run(data_dir, base / "run", RunConfig(
                model="cdae_cnn_ccpos", buildings=(0, 1), seed=0, batch_size=8, val_size=0.2,
                ccpos_notebook_pipeline=True,
                ccpos_cdae_epochs=1, ccpos_cdae_patience=1,
                ccpos_task_epochs=1, ccpos_task_patience=1, ccpos_floor_patience=1,
                ccpos_position_epochs=1, ccpos_position_patience=1,
                ccpos_kmeans_k=4,
            ))

            self.assertIn("mean_m", result)
            self.assertIn("building_accuracy_pct", result)
            self.assertIn("floor_accuracy_pct", result)
            for building in (0, 1):
                folder = base / "run" / f"building_{building}"
                self.assertTrue((folder / "predictions.csv").exists())
                self.assertTrue((folder / "history.csv").exists())
                self.assertTrue((folder / "checkpoint.pt").exists())
                preds = pd.read_csv(folder / "predictions.csv")
                self.assertTrue((preds["building"] == building).all())
            summary = pd.read_csv(base / "run" / "by_building.csv")
            self.assertEqual(len(summary), 2)

    def test_default_pipeline_untouched_for_other_models(self):
        """Every other model still takes runner.py's original per-building loop."""
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            data_dir = base / "data"
            data_dir.mkdir()
            _frame(20, 1).to_csv(data_dir / "trainingData.csv", index=False)
            _frame(4, 1).to_csv(data_dir / "validationData.csv", index=False)
            result = run(data_dir, base / "run", RunConfig(
                model="mlp", buildings=(1,), stage1_epochs=0, stage2_epochs=1, patience=1,
            ))
            self.assertEqual(result["n"], 4)

    def test_pipeline_flag_rejected_for_other_models(self):
        with self.assertRaises(ValueError):
            run("unused", "unused", RunConfig(model="mlp", ccpos_notebook_pipeline=True))


if __name__ == "__main__":
    unittest.main()
