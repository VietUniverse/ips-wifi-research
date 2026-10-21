"""Validation-weighted ensembles for runs with matching source rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _fit_weights(target: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Fit non-negative weights summing to one by validation mean distance."""
    count = len(predictions)
    initial = np.full(count, 1.0 / count)

    def score(weights):
        blended = np.einsum("m,mnd->nd", weights, predictions)
        return np.linalg.norm(target - blended, axis=1).mean()

    result = minimize(score, initial, method="SLSQP", bounds=[(0.0, 1.0)] * count,
                      constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1.0}],
                      options={"maxiter": 500, "ftol": 1e-9})
    return result.x if result.success else initial


def validation_weighted_ensemble(run_records, output_dir):
    """Blend compatible model runs, learning weights from internal validation only.

    A run record needs ``dataset_key``, ``preprocess_label``, ``output_dir`` and
    ``model_label``. Test coordinates are only read after the weights are fixed.
    """
    groups = {}
    for record in run_records:
        key = (record.get("dataset_key"), record.get("preprocess_label"))
        groups.setdefault(key, []).append(record)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for (dataset_key, preprocess), records in groups.items():
        usable = []
        for record in records:
            folder = Path(record["output_dir"])
            if (list(folder.glob("building_*/validation_predictions.csv"))
                    and list(folder.glob("building_*/predictions.csv"))):
                usable.append(record)
        if len(usable) < 2:
            continue

        building_ids = sorted({int(p.parent.name.split("_")[-1])
                               for record in usable
                               for p in Path(record["output_dir"]).glob("building_*/predictions.csv")})
        for building in building_ids:
            candidates = []
            for record in usable:
                folder = Path(record["output_dir"]) / f"building_{building}"
                val_path, test_path = folder / "validation_predictions.csv", folder / "predictions.csv"
                if val_path.exists() and test_path.exists():
                    candidates.append((record, pd.read_csv(val_path), pd.read_csv(test_path)))
            if len(candidates) < 2:
                continue

            # Align by the original row IDs. Reject runs with inconsistent labels/splits.
            val = candidates[0][1][["source_row", "true_x", "true_y"]].copy()
            test = candidates[0][2][["source_row", "true_x", "true_y"]].copy()
            valid = True
            for _, val_frame, test_frame in candidates[1:]:
                val = val.merge(val_frame[["source_row", "true_x", "true_y"]],
                                on="source_row", suffixes=("", "_other"), validate="one_to_one")
                test = test.merge(test_frame[["source_row", "true_x", "true_y"]],
                                  on="source_row", suffixes=("", "_other"), validate="one_to_one")
                for axis in ("x", "y"):
                    if not np.allclose(val[f"true_{axis}"], val[f"true_{axis}_other"], atol=1e-6):
                        valid = False
                    if not np.allclose(test[f"true_{axis}"], test[f"true_{axis}_other"], atol=1e-6):
                        valid = False
                val.drop(columns=["true_x_other", "true_y_other"], inplace=True)
                test.drop(columns=["true_x_other", "true_y_other"], inplace=True)
            if not valid:
                raise ValueError(f"Runs have inconsistent target labels for building {building}")

            val_predictions, test_predictions, labels = [], [], []
            for record, val_frame, test_frame in candidates:
                val_frame = val_frame.set_index("source_row").loc[val.source_row]
                test_frame = test_frame.set_index("source_row").loc[test.source_row]
                val_predictions.append(val_frame[["pred_x", "pred_y"]].to_numpy())
                test_predictions.append(test_frame[["pred_x", "pred_y"]].to_numpy())
                labels.append(record["model_label"])

            val_predictions = np.stack(val_predictions)
            test_predictions = np.stack(test_predictions)
            y_val = val[["true_x", "true_y"]].to_numpy()
            y_test = test[["true_x", "true_y"]].to_numpy()
            weights = _fit_weights(y_val, val_predictions)
            val_blend = np.einsum("m,mnd->nd", weights, val_predictions)
            test_blend = np.einsum("m,mnd->nd", weights, test_predictions)
            val_errors = np.linalg.norm(y_val - val_blend, axis=1)
            test_errors = np.linalg.norm(y_test - test_blend, axis=1)

            folder = output / str(dataset_key) / str(preprocess) / f"building_{building}"
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"source_row": test.source_row, "true_x": y_test[:, 0],
                          "true_y": y_test[:, 1], "pred_x": test_blend[:, 0],
                          "pred_y": test_blend[:, 1], "error_m": test_errors}).to_csv(
                              folder / "predictions.csv", index=False)
            summaries.append({
                "dataset": dataset_key, "preprocess": preprocess, "building": building,
                "n_models": len(labels), "models": labels,
                "weights": {label: float(weight) for label, weight in zip(labels, weights)},
                "validation_mean_m": float(val_errors.mean()),
                "test": {"n": int(len(test_errors)), "mean_m": float(test_errors.mean()),
                         "median_m": float(np.median(test_errors)),
                         "p90_m": float(np.quantile(test_errors, .9)),
                         "within_5m_pct": float(100 * np.mean(test_errors <= 5))},
            })

    report = {"weight_selection": "simplex weights fit on internal validation only",
              "test_use": "official test used only for final metrics",
              "by_building": summaries}
    (output / "validation_ensemble_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not summaries:
        print("Không đủ model có cả validation_predictions.csv và predictions.csv để ghép ensemble.")
        return report

    table = pd.DataFrame([{**row["test"], "dataset": row["dataset"],
                           "preprocess": row["preprocess"], "building": row["building"],
                           "validation_mean_m": row["validation_mean_m"],
                           "weights": json.dumps(row["weights"], ensure_ascii=False)}
                          for row in summaries])
    table.to_csv(output / "validation_ensemble_by_building.csv", index=False)
    for (dataset_key, preprocess), rows in table.groupby(["dataset", "preprocess"]):
        print(f"\nValidation-weighted ensemble — {dataset_key} / {preprocess}")
        display_cols = ["building", "n", "validation_mean_m", "mean_m", "median_m",
                        "p90_m", "within_5m_pct"]
        print(rows[display_cols].round(3).to_string(index=False))
        n = rows["n"].sum()
        print(f"Official test overall mean: {(rows['mean_m'] * rows['n']).sum() / n:.3f} m")
    return report
