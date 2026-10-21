"""Fine-tune a saved KAN-Mixer as one regressor, using internal validation only.

Example: python -m ips.experiments.joint_mixer --checkpoint-dir RUN --data-dir DATA
         --output-dir NEW_RUN --epochs 25 --backbone-layers 3
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ips.algorithms.factory import create_model
from ips.algorithms.residual_mixer import ResidualKANMixer
from ips.data.uji import load_csvs, make_building_data


class JointKANMixer(nn.Module):
    """A single coordinate predictor with gradients through KAN and Mixer."""

    def __init__(self, backbone, mixer, alpha, trainable_layers=3):
        super().__init__()
        if not 1 <= trainable_layers <= len(backbone.layers):
            raise ValueError("trainable_layers must be between 1 and the backbone depth")
        self.backbone, self.mixer, self.alpha = backbone, mixer, float(alpha)
        self.prefix_depth = len(backbone.layers) - trainable_layers
        eligible = {id(p) for p in backbone.parameters() if p.requires_grad}
        for p in backbone.parameters():
            p.requires_grad_(False)
        for layer in backbone.layers[self.prefix_depth:]:
            for p in layer.parameters():
                p.requires_grad_(id(p) in eligible)

    def prefix(self, rss):
        for layer in self.backbone.layers[:self.prefix_depth]:
            rss = layer(rss)
        return rss

    def from_prefix(self, rss, hidden):
        for layer in self.backbone.layers[self.prefix_depth:-1]:
            hidden = layer(hidden)
        base = self.backbone.layers[-1](hidden)
        correction = self.mixer(rss, hidden)
        if isinstance(correction, tuple):
            correction = correction[0]
        return base + self.alpha * correction

    def forward(self, rss):
        return self.from_prefix(rss, self.prefix(rss))


def to_meters(normalized, mean, scale):
    # UJI northings are around 4.86e6: float32 conversion loses sub-meter detail.
    return np.asarray(normalized, dtype=np.float64) * np.asarray(scale, dtype=np.float64) + np.asarray(mean, dtype=np.float64)


def snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


@torch.no_grad()
def cache_prefix(model, x, device, batch_size=256):
    model.eval()
    return torch.cat([model.prefix(torch.as_tensor(x[i:i + batch_size], device=device)).cpu()
                      for i in range(0, len(x), batch_size)])


@torch.no_grad()
def predict(model, x, hidden, device, batch_size=256):
    model.eval()
    return torch.cat([model.from_prefix(torch.as_tensor(x[i:i + batch_size], device=device),
                                      hidden[i:i + batch_size].to(device)).cpu()
                      for i in range(0, len(x), batch_size)]).numpy()


def fit_joint(model, x_train, y_train, x_val, y_val, mean, scale, device,
              epochs=25, patience=6, batch_size=128, lr=1e-4, backbone_lr=1e-5):
    """Fit using train/validation only. Restore the starting state if no improvement."""
    model.to(device)
    train_hidden = cache_prefix(model, x_train, device)
    val_hidden = cache_prefix(model, x_val, device)
    initial_val = to_meters(predict(model, x_val, val_hidden, device), mean, scale)
    initial_score = float(np.linalg.norm(initial_val - y_val, axis=1).mean())
    best_score, best_epoch, stale = initial_score, 0, 0
    best_state = snapshot(model)
    target = ((y_train - mean) / scale).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.as_tensor(x_train), train_hidden,
                                     torch.as_tensor(target)), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": backbone_lr},
        {"params": [p for p in model.mixer.parameters() if p.requires_grad], "lr": lr},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    scale_reference = scale_tensor.mean()
    history = []
    print(f"Joint start: validation={initial_score:.3f} m, frozen prefix={model.prefix_depth} layers, "
          f"{len(loader)} batches/epoch, device={device}", flush=True)
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        train_sum = 0.0
        for rss, hidden, target_batch in loader:
            rss, hidden, target_batch = rss.to(device), hidden.to(device), target_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model.from_prefix(rss, hidden)
            distance_m = torch.linalg.vector_norm((pred - target_batch) * scale_tensor, dim=1).mean()
            loss = distance_m / scale_reference
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite joint loss")
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            train_sum += distance_m.item() * len(rss)
        val_prediction = to_meters(predict(model, x_val, val_hidden, device), mean, scale)
        score = float(np.linalg.norm(val_prediction - y_val, axis=1).mean())
        scheduler.step(score)
        if score < best_score - 1e-4:
            best_score, best_epoch, best_state, stale = score, epoch, snapshot(model), 0
        else:
            stale += 1
        history.append({"stage": "joint", "epoch": epoch, "train_mean_m": train_sum / len(x_train),
                        "val_mean_m": score, "elapsed_s": time.perf_counter() - start})
        print(f"joint {epoch:03}: train={history[-1]['train_mean_m']:.3f} m "
              f"val={score:.3f} m best={best_score:.3f} m ({history[-1]['elapsed_s']:.1f}s)", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return {"initial_val_m": initial_score, "best_val_m": best_score, "best_epoch": best_epoch,
            "history": history, "val_prediction": to_meters(predict(model, x_val, val_hidden, device), mean, scale)}


def run_joint(checkpoint_dir, data_dir, output_dir, epochs=25, patience=6,
              backbone_layers=3, batch_size=128, lr=1e-4, backbone_lr=1e-5, seed=42):
    source, output = Path(checkpoint_dir), Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output directory: {output}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if not config.get("with_mixer"):
        raise ValueError("Source run must contain a trained KAN-Mixer")
    if config.get("rf_feature_selection"):
        raise ValueError("Joint checkpoint runner currently requires the full input AP set")
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(epochs=epochs, patience=patience, backbone_layers=backbone_layers,
                    batch_size=batch_size, lr=lr, backbone_lr=backbone_lr, seed=seed)
    config["joint_finetuning"] = {**settings, "source": str(source.resolve())}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    train, official = load_csvs(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records, predictions = [], []
    for building in config.get("buildings", [0, 1, 2]):
        torch.manual_seed(seed + building)
        np.random.seed(seed + building)
        checkpoint = torch.load(source / f"building_{building}" / "checkpoint.pt", map_location="cpu", weights_only=False)
        data = make_building_data(train, official, building, config.get("val_size", .15), config.get("seed", 42),
                                  config.get("preprocess", "rss_standard"), config.get("preprocess_params", {}))
        if data.feature_columns != checkpoint["ap_columns"]:
            raise ValueError("Checkpoint feature columns differ from reconstructed data")
        if "preprocess_state" in checkpoint and data.preprocess_state != checkpoint["preprocess_state"]:
            raise ValueError("Checkpoint preprocessing differs from reconstructed training split")
        mean, scale = np.asarray(checkpoint["scaler_mean"]), np.asarray(checkpoint["scaler_scale"])
        if not np.allclose(data.y_train.mean(axis=0), mean, rtol=0, atol=1e-5):
            raise ValueError("Checkpoint coordinate scaler differs from reconstructed split")
        backbone = create_model(checkpoint["model"], data.x_train.shape[1])
        backbone.load_state_dict(checkpoint["state_dict"])
        family = checkpoint.get("mixer_family", checkpoint["model"])
        arch = checkpoint.get("mixer_arch") or config.get("mixer_arch", "legacy_16x8")
        mixer = ResidualKANMixer(family=family, rss_dim=data.x_train.shape[1], arch=arch,
                                 token_type=checkpoint.get("mixer_token_type", "kan"),
                                 channel_type=checkpoint.get("mixer_channel_type", "kan"),
                                 n_spatial_clusters=checkpoint.get("spatial_clusters", 0))
        alpha = float(checkpoint.get("alpha", 0.0))
        if checkpoint.get("mixer_state_dict") is not None:
            mixer.load_state_dict(checkpoint["mixer_state_dict"])
        if alpha == 0:
            nn.init.zeros_(mixer.head[-1].weight)
            nn.init.zeros_(mixer.head[-1].bias)
            alpha = 1.0  # zero correction preserves baseline and allows gradients
        model = JointKANMixer(backbone, mixer, alpha, backbone_layers).to(device)
        starting_state = snapshot(model)
        print(f"\nBUILDING {building}: joint KAN + token/channel Mixer", flush=True)
        result = fit_joint(model, data.x_train, data.y_train, data.x_val, data.y_val, mean, scale, device,
                           epochs, patience, batch_size, lr, backbone_lr)
        # Test labels are first used below, after training and checkpoint selection.
        test_hidden = cache_prefix(model, data.x_test, device)
        final_state = snapshot(model)
        final_pred = to_meters(predict(model, data.x_test, test_hidden, device), mean, scale)
        model.load_state_dict(starting_state)
        original_pred = to_meters(predict(model, data.x_test, test_hidden, device), mean, scale)
        model.load_state_dict(final_state)
        error = np.linalg.norm(final_pred - data.y_test, axis=1)
        before = np.linalg.norm(original_pred - data.y_test, axis=1)
        record = {"building": building, "n": len(error), "baseline_val_m": result["initial_val_m"],
                  "joint_val_m": result["best_val_m"], "best_epoch": result["best_epoch"],
                  "baseline_test_m": float(before.mean()), "mean_m": float(error.mean()),
                  "median_m": float(np.median(error)), "p90_m": float(np.quantile(error, .9))}
        records.append(record)
        folder = output / f"building_{building}"
        folder.mkdir()
        frame = pd.DataFrame({"source_row": data.test_rows, "building": building,
                              "true_x": data.y_test[:, 0], "true_y": data.y_test[:, 1],
                              "pred_x": final_pred[:, 0], "pred_y": final_pred[:, 1], "error_m": error,
                              "baseline_x": original_pred[:, 0], "baseline_y": original_pred[:, 1]})
        frame.to_csv(folder / "predictions.csv", index=False)
        predictions.append(frame)
        vp = result["val_prediction"]
        pd.DataFrame({"source_row": data.val_rows, "building": building, "true_x": data.y_val[:, 0],
                      "true_y": data.y_val[:, 1], "pred_x": vp[:, 0], "pred_y": vp[:, 1],
                      "error_m": np.linalg.norm(vp - data.y_val, axis=1)}).to_csv(folder / "validation_predictions.csv", index=False)
        pd.DataFrame(result["history"]).to_csv(folder / "history.csv", index=False)
        checkpoint.update(state_dict=snapshot(model.backbone), mixer_state_dict=snapshot(model.mixer),
                          alpha=alpha, joint_finetuning=settings, joint_best_epoch=result["best_epoch"])
        torch.save(checkpoint, folder / "checkpoint.pt")
        print(f"building {building}: original={before.mean():.3f} m joint={error.mean():.3f} m "
              f"selected_epoch={result['best_epoch']}", flush=True)
    errors = pd.concat(predictions)["error_m"].to_numpy()
    summary = {"by_building": records, "official_test": {"n": len(errors), "mean_m": float(errors.mean()),
               "median_m": float(np.median(errors)), "p90_m": float(np.quantile(errors, .9))},
               "under_6m": bool(errors.mean() < 6), "selection": "internal validation only; epoch 0 allowed"}
    pd.DataFrame(records).to_csv(output / "by_building.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--backbone-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = vars(parser.parse_args())
    torch.set_num_threads(args.pop("cpu_threads"))
    run_joint(**args)


if __name__ == "__main__":
    main()
