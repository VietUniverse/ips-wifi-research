"""Figures and tables from saved UJI experiment outputs.

The official test is used only for reporting; model and k selection remain on
the internal validation split in the experiment runners.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _read(run_dir):
    root = Path(run_dir)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    buildings = pd.read_csv(root / "by_building.csv")
    predictions = pd.concat(
        [pd.read_csv(p) for p in sorted(root.glob("building_*/predictions.csv"))],
        ignore_index=True,
    )
    if predictions.empty:
        raise ValueError(f"No predictions in {root}")
    return root, config, summary, buildings, predictions


def _models(config, predictions):
    if "models" in config:
        models = list(config["models"])
        if not models:
            models = [c.removesuffix("_error_m") for c in predictions if c.endswith("_error_m")]
        return models
    return [config["model"] + ("+mixer" if config.get("with_mixer") else "")]


def _errors(config, predictions, model):
    if "models" in config:
        return predictions[f"{model}_error_m"].to_numpy(float)
    return predictions["error_m"].to_numpy(float)


def _table(config, buildings, predictions):
    rows = []
    for model in _models(config, predictions):
        err = _errors(config, predictions, model)
        row = {"model": model, "n_test": len(err), "mean_m": err.mean(),
               "median_m": np.median(err), "p90_m": np.quantile(err, .9),
               "within_5m_pct": 100 * np.mean(err <= 5)}
        if "models" in config:
            row["floor_accuracy_pct"] = 100 * np.mean(
                predictions[f"{model}_pred_floor"].to_numpy() == predictions["true_floor"].to_numpy())
            root = Path(buildings.attrs["root"])
            selected = json.loads((root / "summary.json").read_text(encoding="utf-8"))["selected_k"][model]
            record = pd.read_csv(root / "sweep_by_k.csv")
            chosen = record[(record.method == model) & (record.k == selected)]
            if not chosen.empty:
                row["selected_k"] = int(selected)
                row["prediction_time_s"] = float(chosen.iloc[0]["t_predict_s"])
                row["total_time_s"] = float(chosen.iloc[0]["total_s"])
        rows.append(row)
    return pd.DataFrame(rows)


def _save(fig, folder, name):
    fig.tight_layout()
    path = folder / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _bar(ax, labels, values, title, ylabel):
    bars = ax.bar(labels, values, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)


def render_run(run_dir, show=True):
    """Create figures for one run; handles one or several KNN models."""
    root, cfg, _, by_building, pred = _read(run_dir)
    by_building.attrs["root"] = str(root)
    folder = root / "figures"
    folder.mkdir(exist_ok=True)
    table = _table(cfg, by_building, pred)
    table.to_csv(folder / "report_metrics.csv", index=False)
    paths = []
    models = _models(cfg, pred)

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2.4), 4.5))
    _bar(ax, models, table.mean_m, "Mean positioning error | official test", "Error (m)")
    paths.append(_save(fig, folder, "mean_error.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in models:
        err = np.sort(_errors(cfg, pred, model))
        ax.step(err, np.arange(1, len(err) + 1) / len(err), where="post", label=model)
    ax.set(xlabel="Positioning error (m)", ylabel="Fraction of samples", ylim=(0, 1.02),
           title="Positioning error CDF | official test")
    ax.grid(alpha=.25); ax.legend()
    paths.append(_save(fig, folder, "cdf.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist([_errors(cfg, pred, m) for m in models], bins=30,
            label=models, alpha=.7, histtype="stepfilled")
    ax.set(xlabel="Positioning error (m)", ylabel="Number of samples",
           title="Error distribution | official test")
    ax.grid(alpha=.25); ax.legend()
    paths.append(_save(fig, folder, "error_histogram.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    building_ids = sorted(pred.building.unique())
    width = .8 / len(models)
    for j, model in enumerate(models):
        vals = [_errors(cfg, pred[pred.building == b], model).mean() for b in building_ids]
        ax.bar(np.arange(len(building_ids)) + (j - (len(models)-1)/2)*width,
               vals, width=width, label=model)
    ax.set(xticks=range(len(building_ids)), xticklabels=[str(b) for b in building_ids],
           xlabel="Building", ylabel="Mean error (m)", title="Error by building | official test")
    ax.grid(axis="y", alpha=.25); ax.legend()
    paths.append(_save(fig, folder, "by_building.png"))

    if "true_floor" in pred:
        classes = sorted(set(pred.true_floor.astype(int)).union(*[
            set(pred[f"{m}_pred_floor"].astype(int)) for m in models]))
        for model in models:
            fig, axes = plt.subplots(1, len(building_ids), figsize=(5 * len(building_ids), 4.5), squeeze=False)
            for ax, b in zip(axes[0], building_ids):
                part = pred[pred.building == b]
                matrix = pd.crosstab(part.true_floor.astype(int),
                                     part[f"{model}_pred_floor"].astype(int))
                matrix = matrix.reindex(index=classes, columns=classes, fill_value=0)
                ax.imshow(matrix, cmap="Blues")
                for i in range(len(classes)):
                    for j in range(len(classes)):
                        ax.text(j, i, str(matrix.iat[i, j]), ha="center", va="center", fontsize=8)
                ax.set(xticks=range(len(classes)), yticks=range(len(classes)),
                       xticklabels=classes, yticklabels=classes, xlabel="Predicted floor",
                       ylabel="True floor", title=f"Building {b} (n={len(part)})")
            fig.suptitle(f"Floor confusion matrix | {model}")
            paths.append(_save(fig, folder, f"floor_confusion_{model}.png"))

        if "total_time_s" in table:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            _bar(axes[0], models, table.total_time_s, "Total test search + prediction", "Seconds")
            _bar(axes[1], models, table.prediction_time_s * 1000 / table.n_test,
                 "Prediction aggregation per sample", "Milliseconds")
            paths.append(_save(fig, folder, "timing.png"))
    else:
        histories = [pd.read_csv(p) for p in sorted(root.glob("building_*/history.csv"))]
        if histories:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            for b, history in zip(building_ids, histories):
                axes[0].plot(range(len(history)), history.train_loss, label=f"Building {b}")
                axes[1].plot(range(len(history)), history.val_mean_m, label=f"Building {b}")
            axes[0].set(xlabel="Logged epoch", ylabel="Training loss", title="Training loss")
            axes[1].set(xlabel="Logged epoch", ylabel="Validation mean error (m)", title="Validation error")
            for ax in axes: ax.grid(alpha=.25); ax.legend()
            paths.append(_save(fig, folder, "training_history.png"))

    if show:
        from IPython.display import display, Image
        display(table.round(3))
        for path in paths: display(Image(filename=str(path)))
    return table, paths


def compare_runs(run_dirs, labels=None, show=True):
    """Compare saved runs on exactly the same official test rows."""
    roots = [Path(p) for p in run_dirs]
    labels = labels or [p.name for p in roots]
    if len(roots) != len(labels) or not roots:
        raise ValueError("Provide one label per run")
    frames = []
    for root, label in zip(roots, labels):
        _, cfg, _, _, pred = _read(root)
        for model in _models(cfg, pred):
            key = pred[["building", "source_row"]].copy()
            key["error_m"] = _errors(cfg, pred, model)
            if key.duplicated(["building", "source_row"]).any():
                raise ValueError(f"Duplicate test rows in {root}")
            key["label"] = f"{label}: {model}"
            frames.append(key)
    baseline = frames[0][["building", "source_row"]].sort_values(["building", "source_row"]).reset_index(drop=True)
    for frame in frames[1:]:
        actual = frame[["building", "source_row"]].sort_values(["building", "source_row"]).reset_index(drop=True)
        if not actual.equals(baseline):
            raise ValueError("Runs use different official test rows; comparison would be misleading")
    out = roots[-1] / "comparison_figures"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame([{"model": f.label.iloc[0], "n_test": len(f),
                           "mean_m": f.error_m.mean(), "median_m": f.error_m.median(),
                           "p90_m": f.error_m.quantile(.9), "within_5m_pct": 100*(f.error_m <= 5).mean()}
                          for f in frames])
    table.to_csv(out / "comparison_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(max(7, len(table)*2), 4.5))
    _bar(ax, table.model, table.mean_m, "Model comparison | same official test", "Mean error (m)")
    ax.tick_params(axis="x", labelrotation=20)
    bar = _save(fig, out, "comparison_mean_error.png")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f in frames:
        err = np.sort(f.error_m.to_numpy())
        ax.step(err, np.arange(1, len(err)+1)/len(err), where="post", label=f.label.iloc[0])
    ax.set(xlabel="Positioning error (m)", ylabel="Fraction of samples", ylim=(0, 1.02),
           title="Model comparison CDF | same official test")
    ax.grid(alpha=.25); ax.legend()
    cdf = _save(fig, out, "comparison_cdf.png")
    buildings = sorted(baseline.building.unique())
    fig, ax = plt.subplots(figsize=(max(7, len(buildings)*2.5), 4.5))
    width = .8 / len(frames)
    for j, frame in enumerate(frames):
        means = [frame.loc[frame.building == b, "error_m"].mean() for b in buildings]
        ax.bar(np.arange(len(buildings)) + (j-(len(frames)-1)/2)*width,
               means, width=width, label=frame.label.iloc[0])
    ax.set(xticks=range(len(buildings)), xticklabels=[str(b) for b in buildings],
           xlabel="Building", ylabel="Mean error (m)",
           title="Model comparison by building | same official test")
    ax.grid(axis="y", alpha=.25); ax.legend()
    per_building = _save(fig, out, "comparison_by_building.png")
    if show:
        from IPython.display import display, Image
        display(table.round(3))
        display(Image(filename=str(bar)), Image(filename=str(cdf)), Image(filename=str(per_building)))
    return table, [bar, cdf, per_building]
