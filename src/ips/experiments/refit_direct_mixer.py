"""Fixed, predeclared full-training refit of the direct KAN-Mixer.

The former internal-validation samples become training data. Therefore no
validation_predictions.csv is exported for fitting an ensemble on this run.
This is a separate experiment, not a checkpoint selected on official test.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ips.algorithms.factory import create_model
from ips.data.augmentation import DeviceAugmentedRSSDataset
from ips.data.uji import load_csvs, make_building_data
from ips.experiments.joint_mixer import snapshot, to_meters
from ips.experiments.runner import _predict, metric_coordinate_loss, metrics


def refit_building(source, output, train, official, config, building, epochs=5, lr=3e-5):
    """Five epochs are fixed before test evaluation, not chosen from its errors."""
    if config["model"] != "kan_mixer_e2e" or config.get("with_mixer"):
        raise ValueError("This experiment accepts only the direct kan_mixer_e2e architecture")
    checkpoint = torch.load(source / f"building_{building}/checkpoint.pt", map_location="cpu", weights_only=False)
    data = make_building_data(train, official, building, config.get("val_size", .15), config.get("seed", 42),
                              config["preprocess"], config.get("preprocess_params", {}))
    if checkpoint["ap_columns"] != data.feature_columns or checkpoint["preprocess_state"] != data.preprocess_state:
        raise ValueError("Checkpoint preprocessing does not match reconstructed data")
    mean, scale = np.asarray(checkpoint["scaler_mean"]), np.asarray(checkpoint["scaler_scale"])
    x = np.concatenate((data.x_train, data.x_val))
    y = np.concatenate((data.y_train, data.y_val))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.get("seed", 42) + building)
    model = create_model(config["model"], x.shape[1]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = DeviceAugmentedRSSDataset(x, ((y - mean) / scale).astype(np.float32),
                                        gain_std=config["augment_gain_std"],
                                        noise_std=config["augment_noise_std"],
                                        dropout_prob=config["augment_dropout_prob"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-5)
    coordinate_scale = torch.as_tensor(scale, device=device, dtype=torch.float32)
    history = []
    for epoch in range(1, epochs + 1):
        start, total = time.perf_counter(), 0.
        model.train()
        for rss, target in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = metric_coordinate_loss(model(rss.to(device)), target.to(device), coordinate_scale)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite full-training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total += loss.item() * len(rss)
        history.append({"stage": "full_train_refit", "epoch": epoch, "train_loss": total / len(x)})
        print(f"building {building} full-train {epoch}/{epochs}: loss={total / len(x):.4f} "
              f"elapsed={time.perf_counter() - start:.1f}s", flush=True)
    prediction = to_meters(_predict(model, data.x_test, device), mean, scale)
    frame = pd.DataFrame({"source_row": data.test_rows, "building": building,
                          "true_x": data.y_test[:, 0], "true_y": data.y_test[:, 1],
                          "pred_x": prediction[:, 0], "pred_y": prediction[:, 1],
                          "error_m": np.linalg.norm(prediction - data.y_test, axis=1)})
    folder = output / f"building_{building}"
    folder.mkdir(parents=True, exist_ok=False)
    frame.to_csv(folder / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    checkpoint.update(state_dict=snapshot(model), full_train_refit={"epochs": epochs, "lr": lr,
                                                                  "n": len(x), "sampling": "uniform"})
    torch.save(checkpoint, folder / "checkpoint.pt")
    score = {"building": building, "n_train": len(x), **metrics(data.y_test, prediction)}
    print(score, flush=True)
    return frame, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.cpu_threads)
    config = json.loads((args.checkpoint_dir / "config.json").read_text())
    (args.output_dir / "config.json").write_text(json.dumps({**config,
        "full_train_refit": {"epochs": args.epochs, "lr": args.lr,
                             "source": str(args.checkpoint_dir), "sampling": "uniform",
                             "validation_is_now_training": True}}, indent=2))
    train, official = load_csvs(args.data_dir)
    frames, rows = [], []
    for building in config["buildings"]:
        frame, row = refit_building(args.checkpoint_dir, args.output_dir, train, official, config,
                                   building, args.epochs, args.lr)
        frames.append(frame)
        rows.append(row)
    combined = pd.concat(frames)
    score = metrics(combined[["true_x", "true_y"]].to_numpy(), combined[["pred_x", "pred_y"]].to_numpy())
    summary = {"by_building": rows, "official_test": score, "under_6m": score["mean_m"] < 6,
               "protocol": "fixed full-training refit, no test-based epoch selection"}
    pd.DataFrame(rows).to_csv(args.output_dir / "by_building.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
