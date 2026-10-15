"""Faithful, additive port of ``ccpos.ipynb``'s full multi-stage CCpos
pipeline (CDAE pretrain -> Building classifier -> per-building Floor
classifier -> Position CNN -> frozen-feature MLP -> fine-tune -> BF-MLP
ensemble), kept entirely separate from ``ips.algorithms.cdae_cnn_ccpos``.

Why a separate module instead of editing ``cdae_cnn_ccpos.py``
-----------------------------------------------------------------
``cdae_cnn_ccpos.py`` already documents, in its own module docstring, a
deliberate, scoped-down port of ccpos.ipynb that fits this repo's
``build_model(input_dim)`` plugin contract and ``runner.py``'s single-stage,
per-building training loop (no building/floor classifiers, CDAE+position
trained jointly from epoch 1, this repo's -1..1 RSS convention instead of
the notebook's own 0..1 one). That version is unchanged by this file and
stays the DEFAULT behaviour for every existing config.

This module is the OPPOSITE tradeoff: it reproduces ccpos.ipynb's actual
multi-stage pipeline as closely as this repo's data/output conventions
allow, including the two pieces the other file explicitly leaves out
(Building/Floor auxiliary classifiers and the BF-MLP ensemble), and the
thing most responsible for ccpos.ipynb's accuracy that per-building
training cannot reproduce: the CDAE encoder (and Building/Position models)
are pretrained ONCE on every selected building pooled together, the same
way ccpos.ipynb trains on the whole UJIIndoorLoc training set rather than
one building at a time. Only the per-building Floor classifier is trained
on a single building's data, exactly as in ccpos.ipynb's Cell 2C.

It is opt-in: ``ips.experiments.runner.run()`` only calls
``run_notebook_pipeline`` when ``RunConfig.ccpos_notebook_pipeline=True``
and ``RunConfig.model == "cdae_cnn_ccpos"`` (see
``configs/cdae_cnn_ccpos_notebook.json``). Every existing config, and
every other model, is completely unaffected -- the default per-building
loop in runner.py is untouched and still runs for everyone else.

Deliberate deviations from ccpos.ipynb (all because this module must slot
into ``runner.py``'s existing output contract -- ``config.json``,
``by_building.csv``, ``summary.json``, ``building_*/predictions.csv`` and
``building_*/history.csv``, read by ``ips.experiments.visualize`` -- rather
than ccpos.ipynb's own ad hoc Colab CSV/JSON dump):

1. RSS normalization is this module's own 0..1 ``RSSINormalizer``
   (missing AP == 0), read directly from the raw ``WAP###`` columns in the
   training CSVs -- NOT this repo's ``rss_standard`` preprocessor (-1..1,
   missing == -1), which ``RunConfig.preprocess`` still controls for every
   OTHER model. This keeps the notebook's own RSS convention intact
   instead of forcing Sigmoid/Tanh or missing-value deviations the way
   ``cdae_cnn_ccpos.py`` had to.
2. One shared model (CDAE encoder, building classifier, per-building floor
   classifiers, BF-MLP) is trained across all of ``RunConfig.buildings``
   pooled together and then evaluated separately per building, so the
   SAME ``best_val_m``/``best_epoch`` and the same ``checkpoint.pt`` are
   reported/saved under every ``building_*/`` folder here -- there is no
   independent per-building model the way the default pipeline has one.
   This is intentional: ccpos.ipynb never trains a separate model per
   building either.
3. The KMeans position-validation split is cached as
   ``<output_dir>/position_kmeans_split.npz`` (per-run), not ccpos.ipynb's
   global, cross-run ``./ccpos_runtime/...`` path.
4. Epoch budgets / patience / batch size / noise / learning rate are
   ``RunConfig`` fields (``ccpos_*``, all defaulted to ccpos.ipynb's own
   CONFIG constants) instead of module-level globals, since this module
   must be callable repeatedly with different settings.

Everything else -- model architectures, the two-phase pretrain-then-freeze
schedule, the coordinate-group / KMeans splits, the per-batch-detected-AP
weighted reconstruction loss (computed ONCE globally here, exactly like
ccpos.ipynb, since this module -- unlike ``cdae_cnn_ccpos.py`` -- has
access to the full pooled training signal before training starts) -- is
transcribed as-is from ``ccpos.ipynb``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from ips.data.uji import AP_COLUMNS, TARGET_COLUMNS

NUM_WAPS = len(AP_COLUMNS)  # 520, matches ccpos.ipynb's NUM_WAPS


# ---------------------------------------------------------------------------
# Label helpers (ccpos.ipynb: label_key / encode_labels)
# ---------------------------------------------------------------------------

def _label_key(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _encode_labels(values, mapping: dict) -> np.ndarray:
    return np.asarray([mapping[_label_key(v)] for v in values], dtype=np.int64)


# ---------------------------------------------------------------------------
# RSS / coordinate normalization (ccpos.ipynb: RSSINormalizer / CoordinateMinMax)
# ---------------------------------------------------------------------------

class RSSINormalizer:
    """0..1 encoding, missing AP (raw value 100) -> 0. This is ccpos.ipynb's
    own convention -- deliberately NOT this repo's ``rss_standard``
    preprocessor. Callers must pass raw ``WAP###`` values straight from the
    training CSVs (not already preprocessed by ``ips.data.preprocessors``).
    """

    def fit(self, x: np.ndarray) -> "RSSINormalizer":
        valid = x[x != 100]
        self.min_rssi = float(valid.min()) if valid.size else -100.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).copy()
        missing = x == 100
        x = (x - self.min_rssi) / (-self.min_rssi)
        x[missing] = 0.0
        return np.clip(x, 0, 1).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


class CoordinateMinMax:
    def fit(self, x: np.ndarray) -> "CoordinateMinMax":
        self.minimum = x.min(axis=0).astype(np.float32)
        self.maximum = x.max(axis=0).astype(np.float32)
        self.range = self.maximum - self.minimum
        self.range[self.range == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.minimum) / self.range

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.range + self.minimum

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


# ---------------------------------------------------------------------------
# Splits (ccpos.ipynb: coordinate_group_split / kmeans_validation_split)
# ---------------------------------------------------------------------------

def coordinate_group_split(coordinates: np.ndarray, val_ratio: float = 0.1, seed: int = 42):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    unique_coords, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    rng = np.random.default_rng(seed)
    unique_ids = np.arange(len(unique_coords), dtype=np.int64)
    rng.shuffle(unique_ids)
    num_val = max(1, int(round(len(unique_coords) * val_ratio)))
    val_ids = set(unique_ids[:num_val].tolist())
    is_val = np.asarray([lid in val_ids for lid in inverse], dtype=bool)
    return np.where(~is_val)[0].astype(np.int64), np.where(is_val)[0].astype(np.int64)


def kmeans_validation_split(coordinates: np.ndarray, k: int = 2000, seed: int = 42):
    coordinates = np.asarray(coordinates, dtype=np.float32)
    num_unique = len(np.unique(coordinates, axis=0))
    k = max(1, min(k, num_unique))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    kmeans.fit(coordinates)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    val_indices = []
    for cid in range(k):
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        d = np.linalg.norm(coordinates[members] - centers[cid], axis=1)
        val_indices.append(int(members[np.argmin(d)]))
    val_indices = np.asarray(val_indices, dtype=np.int64)
    all_idx = np.arange(len(coordinates), dtype=np.int64)
    train_indices = np.setdiff1d(all_idx, val_indices)
    return train_indices, val_indices


# ---------------------------------------------------------------------------
# Dataset (ccpos.ipynb: CCPosDataset)
# ---------------------------------------------------------------------------

class CCPosDataset(Dataset):
    def __init__(self, signal, position, building, floor):
        self.signal = torch.tensor(signal[:, None, :], dtype=torch.float32)
        self.position = torch.tensor(position, dtype=torch.float32)
        self.building = torch.tensor(building, dtype=torch.long)
        self.floor = torch.tensor(floor, dtype=torch.long)

    def __len__(self):
        return len(self.signal)

    def __getitem__(self, i):
        return {"x": self.signal[i], "position": self.position[i],
                "building": self.building[i], "floor": self.floor[i]}


# ---------------------------------------------------------------------------
# Models (ccpos.ipynb Cell 1) -- transcribed as-is. Kept fully separate from
# ips.algorithms.cdae_cnn_ccpos's CDAE (Tanh decoder, -1..1 convention):
# this module's CDAE uses ccpos.ipynb's own Sigmoid decoder / 0..1 convention.
# ---------------------------------------------------------------------------

class Conv1dSameReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        total_padding = kernel_size - 1
        self.left = total_padding // 2
        self.right = total_padding - self.left
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.left, self.right))
        return F.relu(self.conv(x))


class CDAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 454, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(454, 388, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(388, 256, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 388, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(388, 454, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(454, 1, kernel_size=3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def cdae_loss(prediction, target, positive_weight: float):
    detected = (target > 0).float()
    weights = 1 + detected * (positive_weight - 1)
    error = (prediction - target) ** 2
    return (weights * error).sum() / weights.sum()


class BuildingHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            Conv1dSameReLU(256, 64, 3),
            Conv1dSameReLU(64, 31, 2),
            nn.MaxPool1d(2),
            Conv1dSameReLU(31, 32, 2),
            Conv1dSameReLU(32, 16, 1),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 256, NUM_WAPS)
            flat = self.features(dummy).flatten(1).shape[1]
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(flat, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return F.softmax(self.fc(x), dim=1)


class FloorHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            Conv1dSameReLU(256, 256, 3),
            Conv1dSameReLU(256, 128, 3),
            nn.MaxPool1d(2),
            Conv1dSameReLU(128, 128, 3),
            Conv1dSameReLU(128, 64, 3),
            nn.MaxPool1d(2),
            Conv1dSameReLU(64, 64, 2),
            Conv1dSameReLU(64, 32, 1),
            nn.MaxPool1d(2),
            Conv1dSameReLU(32, 32, 1),
            Conv1dSameReLU(32, 16, 1),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 256, NUM_WAPS)
            flat = self.features(dummy).flatten(1).shape[1]
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(flat, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return F.softmax(self.fc(x), dim=1)


class PositionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            Conv1dSameReLU(256, 256, 3),
            Conv1dSameReLU(256, 128, 3),
            nn.MaxPool1d(2),
            Conv1dSameReLU(128, 64, 3),
            Conv1dSameReLU(64, 32, 2),
            nn.MaxPool1d(2),
            Conv1dSameReLU(32, 32, 2),
            Conv1dSameReLU(32, 16, 1),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 256, NUM_WAPS)
            flat = self.features(dummy).flatten(1).shape[1]
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(flat, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class PositionMLPHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class PositionMLPModel(nn.Module):
    def __init__(self, encoder, features):
        super().__init__()
        self.encoder = encoder
        self.features = features
        try:
            dev = next(self.encoder.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        with torch.no_grad():
            dummy = torch.zeros(1, 1, NUM_WAPS, device=dev)
            flat = torch.flatten(self.features(self.encoder(dummy)), 1).shape[1]
        self.mlp = PositionMLPHead(flat)

    def forward(self, x):
        x = self.features(self.encoder(x))
        x = torch.flatten(x, 1)
        return self.mlp(x)


class PositionBFMLPModel(nn.Module):
    def __init__(self, encoder, features, building_model, floor_models: dict):
        super().__init__()
        self.encoder = encoder
        self.features = features
        self.building_model = building_model
        self.floor_models = nn.ModuleDict({str(k): v for k, v in floor_models.items()})
        for p in self.building_model.parameters():
            p.requires_grad = False
        for fm in self.floor_models.values():
            for p in fm.parameters():
                p.requires_grad = False
        # Shape-inference only: resolve the encoder's own device and move the
        # auxiliary building/floor models to match it, so this works whether
        # the caller already synchronized their devices beforehand or not
        # (ccpos.ipynb relies on explicit .cpu() calls between stages for
        # this; this constructor does not assume that discipline).
        try:
            dev = next(self.encoder.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        self.building_model.to(dev)
        for fm in self.floor_models.values():
            fm.to(dev)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, NUM_WAPS, device=dev)
            flat_size = torch.flatten(self.features(self.encoder(dummy)), 1).shape[1]
        self.building_model.eval()
        for fm in self.floor_models.values():
            fm.eval()
        with torch.no_grad():
            building_probs = self.building_model(dummy)
            floor_prob_size = sum(self.floor_models[k](dummy).shape[1]
                                  for k in sorted(self.floor_models.keys(), key=int))
        self.flat_size = flat_size
        self.aux_size = building_probs.shape[1] + floor_prob_size
        self.mlp = PositionMLPHead(self.flat_size + self.aux_size)

    def get_aux_features(self, x):
        with torch.no_grad():
            building_probs = self.building_model(x)
            floor_features = []
            for key in sorted(self.floor_models.keys(), key=int):
                bid = int(key)
                floor_probs = self.floor_models[key](x)
                bw = building_probs[:, bid:bid + 1]
                floor_features.append(floor_probs * bw)
            floor_features = torch.cat(floor_features, dim=1)
            return torch.cat([building_probs, floor_features], dim=1)

    def forward(self, x):
        z = torch.flatten(self.features(self.encoder(x)), 1)
        aux = self.get_aux_features(x)
        return self.mlp(torch.cat([z, aux], dim=1))


def initialize_bf_mlp_from_position_model(old_model, new_model):
    old_first = old_model.mlp.net[0]
    new_first = new_model.mlp.net[0]
    if new_first.in_features <= old_first.in_features:
        raise RuntimeError("BF MLP input size should be larger.")
    with torch.no_grad():
        new_first.weight[:, :old_first.in_features].copy_(old_first.weight)
        new_first.weight[:, old_first.in_features:].zero_()
        new_first.bias.copy_(old_first.bias)
        new_model.mlp.net[3].weight.copy_(old_model.mlp.net[3].weight)
        new_model.mlp.net[3].bias.copy_(old_model.mlp.net[3].bias)
        new_model.mlp.net[5].weight.copy_(old_model.mlp.net[5].weight)
        new_model.mlp.net[5].bias.copy_(old_model.mlp.net[5].bias)
    return new_model


class CCPosTask(nn.Module):
    def __init__(self, encoder, head, encoder_dropout=0.7, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.encoder_dropout = nn.Dropout(encoder_dropout)
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                x = self.encoder(x)
        else:
            x = self.encoder(x)
        x = self.encoder_dropout(x)
        return self.head(x)


class FrozenFeaturePositionMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Early stopping (ccpos.ipynb: EarlyStopping)
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, mode="min", patience=3):
        self.mode = mode
        self.patience = patience
        self.best = None
        self.counter = 0
        self.best_state = None
        self.best_epoch = None

    def is_better(self, value):
        if self.best is None:
            return True
        return value < self.best if self.mode == "min" else value > self.best

    def step(self, value, model, epoch=None):
        if self.is_better(value):
            self.best = value
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = True
        else:
            self.counter += 1
            improved = False
        return improved, self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# Training stages (ccpos.ipynb: train_cdae / train_task / train_position_cnn /
# train_frozen_feature_mlp / finetune_position_mlp / train_position_bf_mlp)
# ---------------------------------------------------------------------------

def validate_position_mlp(model, loader, device, coord_scaler):
    model.eval()
    val_mae, val_n = 0.0, 0
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target = batch["position"].to(device)
            out = model(x)
            mae = F.l1_loss(out, target)
            bs = x.size(0)
            val_mae += mae.item() * bs
            val_n += bs
            preds.append(out.cpu().numpy())
            targets.append(target.cpu().numpy())
    val_mae /= val_n
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    pred_real = coord_scaler.inverse_transform(preds)
    target_real = coord_scaler.inverse_transform(targets)
    errors = np.linalg.norm(pred_real - target_real, axis=1)
    return {"mae": float(val_mae), "mean": float(errors.mean()),
            "median": float(np.median(errors)), "p90": float(np.percentile(errors, 90))}


def train_cdae(model, train_loader, val_loader, device, epochs, patience, noise_std, lr,
               use_amp, positive_weight, history):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopping(mode="min", patience=patience)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            clean = batch["x"].to(device, non_blocking=True)
            noisy = clean + noise_std * torch.randn_like(clean)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                recon = model(noisy)
                loss = cdae_loss(recon, clean, positive_weight)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = clean.size(0)
            total_loss += loss.item() * bs
            total_n += bs
        train_loss = total_loss / total_n
        model.eval()
        val_mae, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                clean = batch["x"].to(device)
                noisy = clean + noise_std * torch.randn_like(clean)
                recon = model(noisy)
                val_mae += F.l1_loss(recon, clean).item() * clean.size(0)
                vn += clean.size(0)
        val_mae /= vn
        improved, should_stop = stopper.step(val_mae, model, epoch=epoch)
        history.append({"stage": "cdae", "epoch": epoch, "train_loss": train_loss, "val_mean_m": val_mae})
        if should_stop:
            break
    stopper.restore(model)
    return model


def train_classifier(model, task_name, num_classes, train_loader, val_loader, device, epochs,
                     patience, lr, use_amp, history, log_prefix=None):
    model = model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    stopper = EarlyStopping(mode="max", patience=patience)
    stage = log_prefix or task_name
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch[task_name].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = F.mse_loss(out, F.one_hot(y, num_classes=num_classes).float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs
            correct += (out.argmax(1) == y).sum().item()
        train_loss = total_loss / n
        model.eval()
        val_correct, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch[task_name].to(device, non_blocking=True)
                out = model(x)
                vn += x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
        val_acc = val_correct / vn * 100
        improved, should_stop = stopper.step(val_acc, model, epoch=epoch)
        history.append({"stage": stage, "epoch": epoch, "train_loss": train_loss, "val_mean_m": val_acc})
        if should_stop:
            break
    stopper.restore(model)
    return model


def make_building_floor_loaders(full_dataset, train_indices, val_indices, train_building,
                                building_id, batch_size, loader_opts):
    b_train = np.asarray([int(i) for i in train_indices if train_building[int(i)] == building_id], dtype=np.int64)
    b_val = np.asarray([int(i) for i in val_indices if train_building[int(i)] == building_id], dtype=np.int64)
    if len(b_train) == 0 or len(b_val) == 0:
        raise RuntimeError(f"Building {building_id} has no train/val rows for floor training")
    train_loader = DataLoader(Subset(full_dataset, b_train.tolist()), shuffle=True,
                              batch_size=batch_size, **loader_opts)
    val_loader = DataLoader(Subset(full_dataset, b_val.tolist()), shuffle=False,
                            batch_size=batch_size, **loader_opts)
    return train_loader, val_loader


def train_position_cnn(model, train_loader, val_loader, coord_scaler, device, epochs, patience,
                       lr, history, freeze_encoder=True):
    model = model.to(device)
    for p in model.encoder.parameters():
        p.requires_grad = not freeze_encoder
    for p in model.head.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    stopper = EarlyStopping(mode="min", patience=patience)
    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_encoder:
            model.encoder.eval()
        total_loss, n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["position"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
            optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            n += bs
        train_loss = total_loss / n
        val = validate_position_mlp(model, val_loader, device, coord_scaler)
        improved, should_stop = stopper.step(val["mae"], model, epoch=epoch)
        history.append({"stage": "position_cnn", "epoch": epoch, "train_loss": train_loss, "val_mean_m": val["mean"]})
        if should_stop:
            break
    stopper.restore(model)
    return model


def extract_frozen_features(encoder, features, signal_array, device, batch_size=256):
    encoder = encoder.to(device).eval()
    features = features.to(device).eval()
    signal_tensor = torch.tensor(signal_array[:, None, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(signal_tensor), batch_size=batch_size, shuffle=False)
    all_feats = []
    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            z = features(encoder(x))
            all_feats.append(torch.flatten(z, 1).cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def train_frozen_feature_mlp(train_feat, train_coord_norm, train_indices, val_indices, coord_scaler,
                             device, epochs, patience, lr, batch_size, seed, history):
    X_train = train_feat[train_indices].astype(np.float32)
    y_train = train_coord_norm[train_indices].astype(np.float32)
    X_val = train_feat[val_indices].astype(np.float32)
    y_val = train_coord_norm[val_indices].astype(np.float32)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size, shuffle=False)
    model = FrozenFeaturePositionMLP(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopping(mode="min", patience=patience)
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, n = 0.0, 0
        for x, target in train_loader:
            x, target = x.to(device, non_blocking=True), target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            train_loss += loss.item() * bs
            n += bs
        train_loss /= n
        model.eval()
        val_mae, vn = 0.0, 0
        preds, targets = [], []
        with torch.no_grad():
            for x, target in val_loader:
                x, target = x.to(device), target.to(device)
                out = model(x)
                val_mae += F.l1_loss(out, target).item() * x.size(0)
                vn += x.size(0)
                preds.append(out.cpu().numpy())
                targets.append(target.cpu().numpy())
        val_mae /= vn
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        errors = np.linalg.norm(coord_scaler.inverse_transform(preds) - coord_scaler.inverse_transform(targets), axis=1)
        improved, should_stop = stopper.step(val_mae, model, epoch=epoch)
        history.append({"stage": "frozen_mlp", "epoch": epoch, "train_loss": train_loss,
                        "val_mean_m": float(errors.mean())})
        if should_stop:
            break
    stopper.restore(model)
    return model


def finetune_position_mlp(model, train_loader, val_loader, coord_scaler, device, epochs, patience,
                          history, encoder_lr=1e-5, head_lr=1e-4, grad_clip=5.0):
    model = model.to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.features[6:].parameters():
        p.requires_grad = True
    for p in model.mlp.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam([
        {"params": model.features[6:].parameters(), "lr": encoder_lr},
        {"params": model.mlp.parameters(), "lr": head_lr},
    ])
    stopper = EarlyStopping(mode="min", patience=patience)
    initial = validate_position_mlp(model, val_loader, device, coord_scaler)
    stopper.step(initial["mae"], model, epoch=0)
    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["position"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=grad_clip)
            optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
        train_loss = total_loss / total_n
        val = validate_position_mlp(model, val_loader, device, coord_scaler)
        improved, should_stop = stopper.step(val["mae"], model, epoch=epoch)
        history.append({"stage": "finetune_mlp", "epoch": epoch, "train_loss": train_loss, "val_mean_m": val["mean"]})
        if should_stop:
            break
    stopper.restore(model)
    return model


def train_position_bf_mlp(model, train_loader, val_loader, coord_scaler, device, epochs, patience,
                          lr, history, grad_clip=5.0):
    model = model.to(device)
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.building_model.parameters():
        p.requires_grad = False
    for fm in model.floor_models.values():
        for p in fm.parameters():
            p.requires_grad = False
    for p in model.mlp.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    stopper = EarlyStopping(mode="min", patience=patience)
    initial = validate_position_mlp(model, val_loader, device, coord_scaler)
    stopper.step(initial["mae"], model.mlp, epoch=0)
    for epoch in range(1, epochs + 1):
        model.train()
        model.encoder.eval()
        model.features.eval()
        model.building_model.eval()
        for fm in model.floor_models.values():
            fm.eval()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["position"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.mlp.parameters(), max_norm=grad_clip)
            optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            total_n += bs
        train_loss = total_loss / total_n
        val = validate_position_mlp(model, val_loader, device, coord_scaler)
        improved, should_stop = stopper.step(val["mae"], model.mlp, epoch=epoch)
        history.append({"stage": "bf_mlp", "epoch": epoch, "train_loss": train_loss, "val_mean_m": val["mean"]})
        if should_stop:
            break
    stopper.restore(model.mlp)
    return model


# ---------------------------------------------------------------------------
# Orchestration -- called by ips.experiments.runner.run() when
# RunConfig.ccpos_notebook_pipeline=True. Produces the same per-building
# output contract (config.json is written by runner.py itself; this writes
# by_building.csv, summary.json, building_*/{split,predictions,history}.csv
# and building_*/checkpoint.pt) so ips.experiments.visualize keeps working
# unchanged.
# ---------------------------------------------------------------------------

def run_notebook_pipeline(train: pd.DataFrame, test: pd.DataFrame, config, output: Path, device):
    import random

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    use_amp = device.type == "cuda"

    buildings = list(config.buildings)
    sel_train = train.loc[train.BUILDINGID.isin(buildings)].copy()
    sel_test = test.loc[test.BUILDINGID.isin(buildings)].copy()
    if sel_train.empty or sel_test.empty:
        raise ValueError(f"No rows for buildings {buildings}")

    train_signal_raw = sel_train[AP_COLUMNS].to_numpy(dtype=np.float32)
    test_signal_raw = sel_test[AP_COLUMNS].to_numpy(dtype=np.float32)
    train_coord_raw = sel_train[TARGET_COLUMNS].to_numpy(dtype=np.float32)
    test_coord_raw = sel_test[TARGET_COLUMNS].to_numpy(dtype=np.float32)

    building_labels = sorted({_label_key(x) for x in sel_train.BUILDINGID})
    floor_labels = sorted({_label_key(x) for x in sel_train.FLOOR})
    building_map = {x: i for i, x in enumerate(building_labels)}
    floor_map = {x: i for i, x in enumerate(floor_labels)}
    train_building = _encode_labels(sel_train.BUILDINGID, building_map)
    test_building = _encode_labels(sel_test.BUILDINGID, building_map)
    train_floor = _encode_labels(sel_train.FLOOR, floor_map)
    test_floor = _encode_labels(sel_test.FLOOR, floor_map)
    num_buildings = len(building_map)
    num_floors = len(floor_map)

    rssi_scaler = RSSINormalizer()
    train_signal = rssi_scaler.fit_transform(train_signal_raw)
    test_signal = rssi_scaler.transform(test_signal_raw)
    coord_scaler = CoordinateMinMax()
    train_coord = coord_scaler.fit_transform(train_coord_raw)
    test_coord = coord_scaler.transform(test_coord_raw)

    train_indices, val_indices = coordinate_group_split(train_coord_raw, val_ratio=config.val_size, seed=config.seed)

    split_path = output / "position_kmeans_split.npz"
    if split_path.exists():
        split = np.load(split_path)
        position_train_indices, position_val_indices = split["train"], split["val"]
    else:
        position_train_indices, position_val_indices = kmeans_validation_split(
            train_coord_raw, k=config.ccpos_kmeans_k, seed=config.seed)
        np.savez(split_path, train=position_train_indices, val=position_val_indices)

    full_train_dataset = CCPosDataset(train_signal, train_coord, train_building, train_floor)
    test_dataset = CCPosDataset(test_signal, test_coord, test_building, test_floor)
    loader_opts = {"num_workers": 0, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(full_train_dataset, train_indices.tolist()), shuffle=True,
                              batch_size=config.batch_size, **loader_opts)
    val_loader = DataLoader(Subset(full_train_dataset, val_indices.tolist()), shuffle=False,
                            batch_size=config.batch_size, **loader_opts)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size, **loader_opts)
    pos_gen = torch.Generator()
    pos_gen.manual_seed(config.seed)
    position_train_loader = DataLoader(Subset(full_train_dataset, position_train_indices.tolist()), shuffle=True,
                                       batch_size=config.batch_size, generator=pos_gen, **loader_opts)
    position_val_loader = DataLoader(Subset(full_train_dataset, position_val_indices.tolist()), shuffle=False,
                                     batch_size=config.batch_size, **loader_opts)

    history = []

    # 2A -- CDAE pretrain, pooled across every selected building.
    zero_count = (train_signal == 0).sum()
    nonzero_count = (train_signal > 0).sum()
    positive_weight = float(np.clip(zero_count / max(nonzero_count, 1), 1, 50))
    cdae = train_cdae(CDAE(), train_loader, val_loader, device, config.ccpos_cdae_epochs,
                      config.ccpos_cdae_patience, config.ccpos_noise_factor, config.ccpos_lr,
                      use_amp, positive_weight, history)
    pretrained_encoder = copy.deepcopy(cdae.encoder).cpu()

    # 2B -- Building classifier, pooled.
    building_model = CCPosTask(copy.deepcopy(pretrained_encoder), BuildingHead(num_buildings), encoder_dropout=0.7)
    building_model = train_classifier(building_model, "building", num_buildings, train_loader, val_loader,
                                      device, config.ccpos_task_epochs, config.ccpos_task_patience,
                                      config.ccpos_lr, use_amp, history)
    building_model = building_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2C -- Floor classifier, one per building.
    floor_models = {}
    for building_id in range(num_buildings):
        floor_model = CCPosTask(copy.deepcopy(pretrained_encoder), FloorHead(num_floors), encoder_dropout=0.7)
        f_train_loader, f_val_loader = make_building_floor_loaders(
            full_train_dataset, train_indices, val_indices, train_building, building_id,
            config.batch_size, loader_opts)
        floor_model = train_classifier(floor_model, "floor", num_floors, f_train_loader, f_val_loader,
                                       device, config.ccpos_task_epochs, config.ccpos_floor_patience,
                                       config.ccpos_lr, use_amp, history, log_prefix=f"floor_{building_id}")
        floor_models[building_id] = floor_model.cpu()

    # 2D-1 -- Position CNN, frozen encoder, pooled (KMeans split).
    position_cnn_model = CCPosTask(copy.deepcopy(pretrained_encoder), PositionHead(), encoder_dropout=0.7,
                                   freeze_encoder=True)
    position_cnn_model = train_position_cnn(position_cnn_model, position_train_loader, position_val_loader,
                                            coord_scaler, device, config.ccpos_position_epochs,
                                            config.ccpos_position_patience, config.ccpos_lr, history)
    trained_position_features = copy.deepcopy(position_cnn_model.head.features).cpu()
    trained_position_encoder = copy.deepcopy(position_cnn_model.encoder).cpu()

    # 2D-2 -- frozen-feature MLP.
    train_feat = extract_frozen_features(trained_position_encoder, trained_position_features, train_signal, device)
    frozen_mlp_model = train_frozen_feature_mlp(
        train_feat, train_coord, position_train_indices, position_val_indices, coord_scaler,
        device, config.ccpos_position_epochs, config.ccpos_position_patience, config.ccpos_lr,
        config.batch_size, config.seed, history)

    # 2D-3 -- fine-tune last CNN block + MLP.
    base_position_model = PositionMLPModel(copy.deepcopy(trained_position_encoder), copy.deepcopy(trained_position_features))
    base_position_model.mlp.load_state_dict(frozen_mlp_model.state_dict())
    base_position_model = finetune_position_mlp(base_position_model, position_train_loader, position_val_loader,
                                                 coord_scaler, device, config.ccpos_position_epochs,
                                                 config.ccpos_position_patience, history)
    base_position_model = base_position_model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2E -- BF-MLP ensemble.
    position_model = PositionBFMLPModel(copy.deepcopy(base_position_model.encoder),
                                        copy.deepcopy(base_position_model.features),
                                        building_model, floor_models)
    position_model = initialize_bf_mlp_from_position_model(base_position_model, position_model)
    position_model = train_position_bf_mlp(position_model, position_train_loader, position_val_loader,
                                           coord_scaler, device, config.ccpos_position_epochs,
                                           config.ccpos_position_patience, config.ccpos_lr, history)

    bf_history = [h for h in history if h["stage"] == "bf_mlp"]
    if bf_history:
        best_entry = min(bf_history, key=lambda h: h["val_mean_m"])
        bf_best_val_m, bf_best_epoch = best_entry["val_mean_m"], best_entry["epoch"]
    else:
        bf_best_val_m, bf_best_epoch = float("nan"), None

    # Evaluate on the official test set (pooled), then split results back per building.
    building_model = building_model.to(device).eval()
    position_model = position_model.to(device).eval()
    for m in floor_models.values():
        m.to(device).eval()

    b_preds, f_preds, p_preds = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x"].to(device)
            b_out = building_model(x)
            b_pred_batch = b_out.argmax(dim=1)
            f_out = torch.zeros(x.size(0), num_floors, device=device)
            for bid in range(num_buildings):
                mask = b_pred_batch == bid
                if mask.any():
                    f_out[mask] = floor_models[bid](x[mask])
            p_out = position_model(x)
            b_preds.append(b_pred_batch.cpu().numpy())
            f_preds.append(f_out.argmax(1).cpu().numpy())
            p_preds.append(p_out.cpu().numpy())
    b_preds = np.concatenate(b_preds)
    f_preds = np.concatenate(f_preds)
    p_preds = np.concatenate(p_preds)
    pred_real = coord_scaler.inverse_transform(p_preds)
    target_real = coord_scaler.inverse_transform(test_coord)
    errors = np.linalg.norm(pred_real - target_real, axis=1)

    building_accuracy = float((b_preds == test_building).mean() * 100)
    floor_accuracy = float((f_preds == test_floor).mean() * 100)

    shared_checkpoint = {
        "model": "cdae_cnn_ccpos", "pipeline": "ccpos_notebook", "buildings": buildings,
        "cdae_encoder_state_dict": {k: v.cpu() for k, v in pretrained_encoder.state_dict().items()},
        "building_model_state_dict": {k: v.cpu() for k, v in building_model.state_dict().items()},
        "floor_models_state_dict": {bid: {k: v.cpu() for k, v in m.state_dict().items()}
                                    for bid, m in floor_models.items()},
        "position_model_state_dict": {k: v.cpu() for k, v in position_model.state_dict().items()},
        "coord_scaler_min": coord_scaler.minimum, "coord_scaler_max": coord_scaler.maximum,
        "rssi_min": rssi_scaler.min_rssi,
        "building_map": building_map, "floor_map": floor_map, "ap_columns": list(AP_COLUMNS),
    }

    rows_all, rows = [], []
    for building_id in buildings:
        local_id = building_map[_label_key(building_id)]
        mask = sel_test.BUILDINGID.to_numpy() == building_id
        rows_idx = np.where(mask)[0]
        if len(rows_idx) == 0:
            continue
        folder = output / f"building_{building_id}"
        folder.mkdir(parents=True, exist_ok=True)

        tr_mask = train_building[train_indices] == local_id
        va_mask = train_building[val_indices] == local_id
        pd.DataFrame({
            "source_row": np.r_[sel_train.index.to_numpy()[train_indices][tr_mask],
                                sel_train.index.to_numpy()[val_indices][va_mask]],
            "split": ["train"] * int(tr_mask.sum()) + ["internal_val"] * int(va_mask.sum()),
        }).to_csv(folder / "split.csv", index=False)

        per_sample = pd.DataFrame({
            "source_row": sel_test.index.to_numpy()[rows_idx], "building": building_id,
            "true_x": target_real[rows_idx, 0], "true_y": target_real[rows_idx, 1],
            "baseline_x": pred_real[rows_idx, 0], "baseline_y": pred_real[rows_idx, 1],
            "pred_x": pred_real[rows_idx, 0], "pred_y": pred_real[rows_idx, 1],
            "error_m": errors[rows_idx],
            "true_floor": test_floor[rows_idx], "pred_floor": f_preds[rows_idx],
        })
        per_sample.to_csv(folder / "predictions.csv", index=False)
        rows_all.append(per_sample)

        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        torch.save(shared_checkpoint, folder / "checkpoint.pt")

        rows.append({"building": building_id, "model": "cdae_cnn_ccpos", "preprocess": "ccpos_notebook",
                     "n_features": len(AP_COLUMNS), "best_val_m": bf_best_val_m, "best_epoch": bf_best_epoch,
                     "test_mean_m": float(per_sample.error_m.mean()), "n": int(len(per_sample)),
                     "mean_m": float(per_sample.error_m.mean()), "median_m": float(per_sample.error_m.median()),
                     "p90_m": float(per_sample.error_m.quantile(.9))})

    summary_pred = pd.concat(rows_all, ignore_index=True)
    all_metrics = {"n": int(len(summary_pred)), "mean_m": float(summary_pred.error_m.mean()),
                   "median_m": float(summary_pred.error_m.median()),
                   "p90_m": float(summary_pred.error_m.quantile(.9)),
                   "building_accuracy_pct": building_accuracy, "floor_accuracy_pct": floor_accuracy}

    pd.DataFrame(rows).to_csv(output / "by_building.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps({"by_building": rows, "official_test": all_metrics}, indent=2), encoding="utf-8")
    for row in rows:
        print(f"building {row['building']}: {row['mean_m']:.3f} m")
    print(json.dumps(all_metrics, indent=2))
    return all_metrics
