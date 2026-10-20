"""Shared UJI training protocol for MLP, BSRBF-KAN and residual KAN-Mixer."""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from ips.algorithms.factory import create_model
from ips.algorithms.residual_mixer import ResidualKANMixer
from ips.data.augmentation import DeviceAugmentedRSSDataset
from ips.data.uji import load_csvs, make_building_data


@dataclass
class RunConfig:
    model: str = "bsrbf_kan"
    preprocess: str = "rss_standard"
    preprocess_params: dict = field(default_factory=dict)
    buildings: tuple[int, ...] = (0, 1, 2)
    seed: int = 42
    val_size: float = 0.15
    batch_size: int = 64
    stage1_epochs: int = 30
    stage2_epochs: int = 80
    patience: int = 15
    mixer_epochs: int = 40
    mixer_patience: int = 12
    mixer_batch_size: int = 128
    with_mixer: bool = False
    mixer_arch: str = "legacy_16x8"
    rf_feature_selection: bool = False
    rf_top_k: int = 128
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None
    balance_locations: bool = False
    augment_gain_std: float = 0.10
    augment_noise_std: float = 0.05
    augment_dropout_prob: float = 0.06
    device: str = "auto"


def mean_error(true_xy, pred_xy):
    return float(np.linalg.norm(true_xy - pred_xy, axis=1).mean())


def metrics(true_xy, pred_xy):
    errors = np.linalg.norm(true_xy - pred_xy, axis=1)
    return {"n": int(len(errors)), "mean_m": float(errors.mean()),
            "median_m": float(np.median(errors)), "p90_m": float(np.quantile(errors, .9))}


def coordinate_loss(pred, target):
    return torch.linalg.vector_norm(pred - target, dim=1).mean() + 0.25 * F.mse_loss(pred, target)


def _device(config):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)


@torch.no_grad()
def _predict(model, x, device, batch_size=256):
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[start:start + batch_size], device=device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def _fit_stage(model, loader, x_val, y_val, scaler, device, epochs, patience, lr,
               stage_name, trainable_only=False):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    history = []
    print(f"{stage_name}: start {epochs} epochs, {len(loader)} batches/epoch on {device}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        started = time.perf_counter()
        for batch_index, (xb, yb) in enumerate(loader, 1):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = coordinate_loss(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss in {stage_name}")
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            total += loss.item() * len(xb)
            if epoch == 1 and (batch_index == 1 or batch_index % max(1, len(loader) // 4) == 0):
                print(f"{stage_name} epoch 1: batch {batch_index}/{len(loader)}, "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
        pred_val = scaler.inverse_transform(_predict(model, x_val, device).astype(np.float64))
        score = mean_error(y_val, pred_val)
        scheduler.step(score)
        history.append({"stage": stage_name, "epoch": epoch,
                        "train_loss": total / len(loader.dataset), "val_mean_m": score})
        if score < best - 1e-4:
            best, best_state, best_epoch, stale = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"{stage_name} {epoch:03d}: train={total / len(loader.dataset):.4f} "
                  f"val={score:.3f} m elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError(f"No finite validation checkpoint for {stage_name}")
    model.load_state_dict(best_state)
    return best, best_epoch, history


@torch.no_grad()
def _features_and_baseline(model, x, device, batch_size=256):
    if not hasattr(model, "layers"):
        raise ValueError("Residual Mixer requires a KAN backbone with layers")
    model.eval()
    feats, base = [], []
    for start in range(0, len(x), batch_size):
        xb = torch.as_tensor(x[start:start + batch_size], device=device)
        for layer in model.layers[:-1]:
            xb = layer(xb)
        feats.append(xb.cpu())
        base.append(model.layers[-1](xb).cpu())
    return torch.cat(feats), torch.cat(base)


@torch.no_grad()
def _mixer_correction(mixer, x, features, device, batch_size=256):
    mixer.eval()
    parts = []
    for start in range(0, len(x), batch_size):
        rss = torch.as_tensor(x[start:start + batch_size], device=device)
        feat = features[start:start + batch_size].to(device)
        parts.append(mixer(rss, feat).cpu().numpy())
    return np.concatenate(parts)


def _fit_mixer(model, data, scaler, config, device):
    # Frozen backbone. Validation chooses both epoch and alpha; official test is read only after selection.
    for p in model.parameters():
        p.requires_grad_(False)
    train_feat, train_base = _features_and_baseline(model, data.x_train, device)
    val_feat, val_base = _features_and_baseline(model, data.x_val, device)
    test_feat, test_base = _features_and_baseline(model, data.x_test, device)
    target = torch.as_tensor(scaler.transform(data.y_train), dtype=torch.float32)
    loader = DataLoader(TensorDataset(torch.as_tensor(data.x_train), train_feat, train_base, target),
                        batch_size=config.mixer_batch_size, shuffle=True)
    ap_indices = None
    if config.mixer_arch == "ap_token":
        # Preserve original AP identities even when a selector keeps only a subset.
        parsed = []
        for index, column in enumerate(data.feature_columns):
            digits = "".join(ch for ch in str(column) if ch.isdigit())
            parsed.append(max(int(digits) - 1, 0) if str(column).upper().startswith("WAP") and digits else index)
        ap_indices = parsed
    mixer = ResidualKANMixer(family=config.model, rss_dim=data.x_train.shape[1],
                             arch=config.mixer_arch, ap_indices=ap_indices).to(device)
    optimizer = torch.optim.AdamW(mixer.parameters(), lr=3e-4, weight_decay=1e-5)
    base_val = scaler.inverse_transform(val_base.numpy().astype(np.float64))
    best = mean_error(data.y_val, base_val)
    best_state, best_alpha, best_epoch, stale = None, 0.0, 0, 0
    history = []
    print(f"mixer: start {config.mixer_epochs} epochs, {len(loader)} batches/epoch on {device}", flush=True)
    for epoch in range(1, config.mixer_epochs + 1):
        mixer.train()
        total = 0.0
        started = time.perf_counter()
        for batch_index, (rss, feat, base, y) in enumerate(loader, 1):
            rss, feat, base, y = (v.to(device) for v in (rss, feat, base, y))
            optimizer.zero_grad(set_to_none=True)
            loss = coordinate_loss(base + mixer(rss, feat), y)
            loss.backward()
            nn.utils.clip_grad_norm_(mixer.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(rss)
            if epoch == 1 and (batch_index == 1 or batch_index % max(1, len(loader) // 4) == 0):
                print(f"mixer epoch 1: batch {batch_index}/{len(loader)}, "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
        delta = _mixer_correction(mixer, data.x_val, val_feat, device)
        scores = {alpha: mean_error(data.y_val, scaler.inverse_transform(
            val_base.numpy().astype(np.float64) + alpha * delta))
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)}
        alpha = min(scores, key=scores.get)
        val = scores[alpha]
        history.append({"stage": "mixer", "epoch": epoch,
                        "train_loss": total / len(loader.dataset), "val_mean_m": val, "alpha": alpha})
        if val < best - 1e-3:
            best, best_alpha, best_epoch, best_state, stale = val, alpha, epoch, copy.deepcopy(mixer.state_dict()), 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"mixer {epoch:03d}: train={total / len(loader.dataset):.4f} "
                  f"val={val:.3f} m alpha={alpha} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if stale >= config.mixer_patience:
            break
    if best_state is not None:
        mixer.load_state_dict(best_state)
    delta_test = (_mixer_correction(mixer, data.x_test, test_feat, device)
                  if best_alpha else np.zeros_like(test_base.numpy()))
    pred_norm = test_base.numpy().astype(np.float64) + best_alpha * delta_test
    return scaler.inverse_transform(pred_norm), {"alpha": best_alpha, "best_epoch": best_epoch,
             "best_val_m": best, "state_dict": best_state, "history": history}


def run(data_dir: str | Path, output_dir: str | Path, config: RunConfig):
    kan_families = {"kan", "fastkan", "bsrbf_kan", "skan", "fckan"}
    if config.with_mixer and config.model not in kan_families:
        raise ValueError(f"with_mixer requires a KAN model; choose one of {sorted(kan_families)}")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = _device(config)
    train, test = load_csvs(data_dir)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    rows, preds = [], []
    for building in config.buildings:
        data = make_building_data(train, test, building, config.val_size, config.seed,
                                  config.preprocess, config.preprocess_params)
        print(f"building {building}: train={len(data.x_train)} val={len(data.x_val)} "
              f"test={len(data.x_test)} features={data.x_train.shape[1]} device={device}", flush=True)
        folder = output / f"building_{building}"
        folder.mkdir()
        feature_selection_info = None
        if config.rf_feature_selection:
            n_features = data.x_train.shape[1]
            if not 1 <= config.rf_top_k <= n_features:
                raise ValueError(f"rf_top_k must be between 1 and {n_features}, got {config.rf_top_k}")
            # Rank APs using only the training split. Validation selects
            # checkpoints/alpha; the official test remains untouched.
            selector = RandomForestRegressor(
                n_estimators=config.rf_n_estimators,
                max_depth=config.rf_max_depth,
                min_samples_leaf=2,
                max_features=0.5,
                random_state=config.seed,
                n_jobs=-1,
            )
            selector.fit(data.x_train, data.y_train)
            importance = selector.feature_importances_
            ranked = np.argsort(importance)[::-1]
            selected = np.sort(ranked[:config.rf_top_k])
            pd.DataFrame({
                "rank": np.arange(1, n_features + 1),
                "feature": np.asarray(data.feature_columns)[ranked],
                "importance": importance[ranked],
                "selected": np.isin(ranked, selected),
            }).to_csv(folder / "rf_ap_importance.csv", index=False)
            original_columns = list(data.feature_columns)
            data.x_train = data.x_train[:, selected]
            data.x_val = data.x_val[:, selected]
            data.x_test = data.x_test[:, selected]
            data.feature_columns = [original_columns[i] for i in selected]
            feature_selection_info = {
                "method": "RandomForestRegressor impurity feature importance",
                "fit_split": "train_only", "original_features": n_features,
                "selected_features": len(selected), "selected_indices": selected.tolist(),
                "selected_columns": data.feature_columns,
                "n_estimators": config.rf_n_estimators,
                "max_depth": config.rf_max_depth,
                "min_samples_leaf": 2, "max_features": 0.5,
            }
            print(f"RF AP selection: {n_features} -> {len(selected)} features; fitted on train only", flush=True)
        pd.DataFrame({"source_row": np.r_[data.train_rows, data.val_rows],
                      "split": ["train"] * len(data.train_rows) + ["internal_val"] * len(data.val_rows)})\
            .to_csv(folder / "split.csv", index=False)
        scaler = StandardScaler().fit(data.y_train)
        y_norm = scaler.transform(data.y_train).astype(np.float32)
        train_dataset = DeviceAugmentedRSSDataset(
            data.x_train, y_norm, gain_std=config.augment_gain_std,
            noise_std=config.augment_noise_std, dropout_prob=config.augment_dropout_prob)
        sampler = None
        if config.balance_locations:
            local_train = train.loc[data.train_rows]
            groups = (local_train.FLOOR.astype(str) + "_"
                      + local_train.LONGITUDE.round(1).astype(str) + "_"
                      + local_train.LATITUDE.round(1).astype(str)).to_numpy()
            _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
            weights = torch.as_tensor(1.0 / counts[inverse], dtype=torch.double)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                                  shuffle=sampler is None, sampler=sampler)
        model = create_model(config.model, data.x_train.shape[1]).to(device)
        history = []
        if config.model == "bsrbf_kan" and config.stage1_epochs:
            for p in model.parameters():
                p.requires_grad_(False)
            for layer in model.layers[-3:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            _, _, h = _fit_stage(model, train_loader, data.x_val, data.y_val, scaler,
                                 device, config.stage1_epochs, config.patience, 3e-4, "stage1")
            history += h
            for p in model.parameters():
                p.requires_grad_(True)
        best_val, epoch, h = _fit_stage(model, train_loader, data.x_val, data.y_val, scaler,
                                        device, config.stage2_epochs, config.patience, 1e-4,
                                        "stage2")
        history += h
        base_pred = scaler.inverse_transform(_predict(model, data.x_test, device).astype(np.float64))
        result = {"building": building, "model": config.model, "preprocess": config.preprocess,
                  "n_features": len(data.feature_columns), "best_val_m": best_val,
                  "best_epoch": epoch, "test_mean_m": mean_error(data.y_test, base_pred)}
        if feature_selection_info is not None:
            result["rf_original_features"] = feature_selection_info["original_features"]
            result["rf_selected_features"] = feature_selection_info["selected_features"]
        chosen_pred = base_pred
        checkpoint = {"model": config.model, "building": building,
                      "preprocess": config.preprocess, "preprocess_params": config.preprocess_params,
                      "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                      "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                      "ap_columns": data.feature_columns,
                      "preprocess_state": data.preprocess_state,
                      "mixer_arch": config.mixer_arch if config.with_mixer else None}
        if feature_selection_info is not None:
            checkpoint["rf_feature_selection"] = feature_selection_info
        if config.with_mixer:
            chosen_pred, mixed = _fit_mixer(model, data, scaler, config, device)
            result.update({"mixer_val_m": mixed["best_val_m"], "alpha": mixed["alpha"],
                           "mixer_epoch": mixed["best_epoch"], "mixer_arch": config.mixer_arch,
                           "mixer_test_mean_m": mean_error(data.y_test, chosen_pred)})
            checkpoint["mixer_state_dict"] = mixed["state_dict"]
            checkpoint["alpha"] = mixed["alpha"]
            history += mixed["history"]
        torch.save(checkpoint, folder / "checkpoint.pt")
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        per_sample = pd.DataFrame({"source_row": data.test_rows, "building": building,
                                   "true_x": data.y_test[:, 0], "true_y": data.y_test[:, 1],
                                   "baseline_x": base_pred[:, 0], "baseline_y": base_pred[:, 1],
                                   "pred_x": chosen_pred[:, 0], "pred_y": chosen_pred[:, 1],
                                   "error_m": np.linalg.norm(chosen_pred - data.y_test, axis=1)})
        per_sample.to_csv(folder / "predictions.csv", index=False)
        preds.append(per_sample)
        result.update(metrics(data.y_test, chosen_pred))
        rows.append(result)
        print(f"building {building}: {result['mean_m']:.3f} m")
    summary = pd.concat(preds, ignore_index=True)
    all_metrics = metrics(summary[["true_x", "true_y"]].to_numpy(),
                          summary[["pred_x", "pred_y"]].to_numpy())
    pd.DataFrame(rows).to_csv(output / "by_building.csv", index=False)
    (output / "summary.json").write_text(json.dumps({"by_building": rows, "official_test": all_metrics},
                                                    indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))
    return all_metrics
