"""CDAE-CNN, ported from the author-provided ``ccpos.ipynb`` notebook,
itself an implementation of the CCpos paper (Qin, Zuo & Wang 2021,
"CCpos: WiFi Fingerprint Indoor Positioning System Based on CDAE-CNN",
Sensors 21(4):1114, https://doi.org/10.3390/s21041114).

This REPLACES an earlier version of this file that was reconstructed
from the paper's PDF alone (kernel sizes there were an approximation
due to PDF/OCR ambiguity in its tables). ``ccpos.ipynb`` gives the
actual, working implementation, so ``CDAE``, ``Conv1dSameReLU`` and
``PositionHead`` below are now transcribed from it directly rather
than reconstructed: encoder/decoder channel sizes (454/388/256),
conv kernel sizes (including the notebook's mix of kernel=3/2/1 and
its asymmetric "same" padding for even kernels), the position head's
exact conv+pool stack, and the weighted reconstruction loss are all
taken as-is. This is NOT the same model as ``cdae_mmj`` in this repo,
which is ported from a different, unrelated notebook (SAE-CDAE_MMJ)
and only shares the name "CDAE".

``ccpos.ipynb`` itself is a much larger, multi-task pipeline than a
single position-regression model: it pretrains the CDAE unsupervised
first, then FREEZES the encoder and trains building/floor classifiers
and a position head separately on top of the frozen features, with a
further fine-tuning stage and a building+floor-conditioned ensemble
(BF-MLP). None of that fits this repo's plugin contract -- building
ID is already known per the outer per-building loop in runner.py,
there is no floor-prediction concept anywhere in this repo, and
``build_model(input_dim)`` is never given the actual training data
needed to run a separate pretraining phase before runner.py's single
training loop starts. Per the explicit instruction that accompanied
this port ("phải deploy theo đúng luồng của repo và cấu trúc của file
ccpos.ipynb" -- keep this repo's plugin/runner.py flow, but match
ccpos.ipynb's own structure as closely as that allows), this file
keeps ONLY the CDAE + position-CNN half, trained with the same
"backprop the reconstruction loss inside forward()" trick already
used by ``mmj_common._JointMixerPlugin`` and the previous version of
this file -- i.e. encoder and position head are trained jointly from
the first epoch, rather than the notebook's pretrain-then-freeze
two-phase schedule. This is the closest equivalent achievable without
changing ``factory.py``/``runner.py`` to pass training data into
``build_model``, which would be a shared-code change affecting every
other model in the repo.

Three further, deliberate deviations from ccpos.ipynb, all for the
same reason already documented for ``sae_mmj``/``cdae_mmj`` and the
previous version of this file -- compatibility with this repo's
``rss_standard`` preprocessor and its unconditional
``DeviceAugmentedRSSDataset`` augmentation (missing AP == -1, range
[-1, 1]), instead of the notebook's own zero-to-one RSSI encoding
(missing AP == 0, range [0, 1]):

1. Decoder's final activation is Tanh here, not the notebook's
   Sigmoid (Sigmoid's [0, 1] range would not match a [-1, 1] target).
2. The weighted reconstruction loss's "is this AP detected" check
   uses ``x > -0.999`` (this repo's "observed" convention, also used
   by ``ips.data.augmentation``) instead of the notebook's ``x > 0``.
3. ``POSITIVE_WEIGHT`` (the loss weight given to detected APs, to
   counter how sparse WiFi RSS data is) is computed from EACH BATCH
   here, not once from the whole training set like the notebook --
   ``build_model(input_dim)`` has no access to the full dataset to
   compute a single global ratio upfront.

``PositionHead``'s output layer in ccpos.ipynb is already a plain
``nn.Linear`` (no activation) -- unlike the earlier, paper-reconstructed
version of this file, no deviation is needed here: it already matches
this repo's ``StandardScaler``-normalized (possibly negative) position
targets.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# ccpos.ipynb Table/CONFIG constants (CDAE_EPOCHS/TASK_EPOCHS/POSITION_EPOCHS
# are the notebook's own multi-stage epoch budgets and have no direct
# equivalent here; NOISE_FACTOR, the dropout rates and BATCH_SIZE/
# LEARNING_RATE below are used as-is in configs/cdae_cnn_ccpos_baseline.json).
NOISE_STD = 0.3  # ccpos.ipynb: NOISE_FACTOR
ENCODER_DROPOUT = 0.7  # ccpos.ipynb: CCPosTask's encoder_dropout / POSITION_ENCODER_DROPOUT
POSITION_HEAD_DROPOUT = 0.3  # ccpos.ipynb: PositionHead's internal dropout


def add_gwn(x, std):
    """Additive Gaussian white noise, matching ccpos.ipynb's CDAE
    corruption, restricted to currently-"observed" entries so this
    repo's -1 "AP not detected" sentinel is left alone -- see the
    module docstring for why that differs from the notebook.
    """
    if std <= 0:
        return x
    observed = x > -0.999
    noisy = (x + torch.randn_like(x) * std).clamp(-1.0, 1.0)
    return torch.where(observed, noisy, x)


class Conv1dSameReLU(nn.Module):
    """Transcribed as-is from ccpos.ipynb: "same"-length 1D conv (with
    asymmetric left/right padding for even kernel sizes, since
    ``kernel_size - 1`` does not split evenly in that case) + ReLU.
    """

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
    """Encoder/decoder channel sizes (1->454->388->256, mirrored back
    down) transcribed as-is from ccpos.ipynb. Decoder output activation
    is Tanh, not the notebook's Sigmoid -- see module docstring.
    """

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
            nn.ConvTranspose1d(454, 1, kernel_size=3, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return z, self.decoder(z)


def weighted_cdae_loss(prediction, target):
    """ccpos.ipynb's weighted reconstruction loss, adapted to this
    repo's -1-for-missing convention (see module docstring point 2)
    and computed per-batch (see point 3) instead of once globally.
    """
    detected = (target > -0.999).float()
    denom = detected.numel() - detected.sum()
    positive_weight = torch.clamp(denom / detected.sum().clamp(min=1.0), 1.0, 50.0)
    weights = 1 + detected * (positive_weight - 1)
    error = (prediction - target) ** 2
    return (weights * error).sum() / weights.sum()


class PositionHead(nn.Module):
    """Transcribed as-is from ccpos.ipynb: the conv+pool stack, the
    dropout rate, and the plain-Linear (no activation) output layer.
    ``input_dim`` replaces the notebook's hardcoded ``NUM_WAPS`` so
    this stays input-size-agnostic like every other plugin in this
    repo (the notebook only ever used it with NUM_WAPS=520).
    """

    def __init__(self, input_dim: int):
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
            dummy = torch.zeros(1, 256, input_dim)
            flat = self.features(dummy).flatten(1).shape[1]
        if flat < 1:
            raise ValueError(
                f"cdae_cnn_ccpos needs a larger input_dim than {input_dim}; "
                "its position head's pooling stack reduced the feature map to nothing."
            )
        self.dropout = nn.Dropout(POSITION_HEAD_DROPOUT)
        self.fc = nn.Linear(flat, 2)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


class CDAECNNCCpos(nn.Module):
    """forward(rss) -> (position, reconstruction). Encoder dropout
    (0.7, ccpos.ipynb's CCPosTask.encoder_dropout) is applied to the
    encoder's output before the position head, exactly as in the
    notebook's CCPosTask wrapper.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.cdae = CDAE()
        self.encoder_dropout = nn.Dropout(ENCODER_DROPOUT)
        self.head = PositionHead(input_dim)

    def forward(self, x):
        x_in = x.unsqueeze(1)  # [batch, 1, input_dim]
        noisy = add_gwn(x_in, NOISE_STD) if self.training else x_in
        z, reconstruction = self.cdae(noisy)
        position = self.head(self.encoder_dropout(z))
        return position, reconstruction.squeeze(1)


class _JointConvPlugin(nn.Module):
    """Adapts CDAECNNCCpos to the repo's `[batch, 2]`-only plugin
    interface while still training encoder + head jointly (see the
    module docstring for why this replaces ccpos.ipynb's own
    pretrain-then-freeze schedule). Same backward(retain_graph=True)
    trick as mmj_common._JointMixerPlugin.
    """

    def __init__(self, inner: CDAECNNCCpos, recon_loss_weight: float = 1.0):
        super().__init__()
        self.inner = inner
        self.recon_loss_weight = recon_loss_weight

    def forward(self, x):
        position, reconstruction = self.inner(x)
        if self.training and torch.is_grad_enabled() and reconstruction.requires_grad:
            recon_loss = self.recon_loss_weight * weighted_cdae_loss(reconstruction, x)
            if torch.isfinite(recon_loss):
                recon_loss.backward(retain_graph=True)
        return position


MODEL_NAME = "cdae_cnn_ccpos"


def build_model(input_dim: int):
    return _JointConvPlugin(CDAECNNCCpos(input_dim))
