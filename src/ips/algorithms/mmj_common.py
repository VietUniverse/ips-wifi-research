"""Shared building blocks for the SAE-MMJ and CDAE-MMJ plugins.

Ported from the SAE-CDAE_MMJ research notebook. Architecture, dimensions
and the joint (reconstruction + position) training objective are kept
as close to the notebook as the shared `runner.py` protocol allows.

This module defines no ``MODEL_NAME`` / ``build_model`` pair, so the
plugin discovery in ``ips.plugins`` skips it (same pattern already used
by ``residual_mixer.py``): it is shared code, not a model entry itself.

== Why forward() does something unusual ==

``runner.py`` is shared across every model in this repo and only knows
how to do, per training step::

    optimizer.zero_grad()
    loss = coordinate_loss(model(xb), yb)   # model(xb) must be the [batch, 2] prediction
    loss.backward()
    optimizer.step()

The notebook's models are trained on a *joint* objective instead:
``ALPHA * reconstruction_loss + BETA * position_loss``, where
``reconstruction_loss`` compares the autoencoder's output against the
input RSS vector. There is no way to return that second loss term
through the `[batch, 2]` interface `runner.py` expects, and changing
`runner.py` to know about per-model auxiliary losses would touch
shared code every other model/teammate also depends on.

Instead, ``_JointMixerPlugin.forward`` computes the reconstruction loss
itself and, only while training with gradients enabled, calls
``aux_loss.backward(retain_graph=True)`` *before returning*. This
populates `.grad` on every encoder/decoder parameter immediately.
When `runner.py` later calls `loss.backward()` on the position loss
it returned, autograd *adds* the position-loss gradients on top
(`backward()` accumulates into `.grad`, it does not overwrite it), so
every parameter ends up with the sum of both gradients -- exactly what
a single `joint_loss.backward()` call would have produced, without
`runner.py` ever knowing a second loss existed. `retain_graph=True` on
the first call keeps the graph alive for the second.

This is intentionally self-contained inside the model: no shared file
is touched, and every other plugin keeps using the plain
`loss.backward()` path unaffected.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# Architecture constants, copied from the notebook (SAE-CDAE_MMJ).
LATENT_DIM = 64
NUM_TOKENS = 20
HIDDEN_DIM = 256
NUM_MIXER_BLOCKS = 2
TOKEN_MLP_DIM = 128
CHANNEL_MLP_DIM = 512
DROPOUT = 0.1

# Joint-loss weight for the reconstruction term (ALPHA in the notebook).
# The position term's weight (BETA=2.0 in the notebook) is not applied
# here: the position side of the loss is runner.py's own
# `coordinate_loss`, which this plugin does not touch or rescale.
RECON_LOSS_WEIGHT = 1.0

# Denoising corruption rate for CDAE: fraction of currently-detected
# APs randomly set to "not detected" before encoding, training-time only.
DEFAULT_CORRUPTION_PROB = 0.3

# These models are paired with the repo's existing ``rss_standard``
# preprocessor (missing AP -> -1, range [-1, 1]) rather than the
# notebook's own [0, 1]/missing=0 encoding. Two reasons: (1) it keeps
# the comparison with MLP/BSRBF-KAN on the exact same features, and
# (2) ``runner.py`` applies ``DeviceAugmentedRSSDataset`` -- which
# already assumes missing AP == -1 -- to every model unconditionally,
# so a differently-scaled preprocessor would silently corrupt that
# augmentation's "is this AP observed" check. The decoder's final
# activation is Tanh (not the notebook's Sigmoid) to match this [-1, 1]
# range, and corruption below sets entries to -1 (not 0) for the same
# reason.


class SAE(nn.Module):
    """Plain Stacked Autoencoder front-end (no corruption)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, LATENT_DIM), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return z, reconstructed


def corrupt_rss(x, prob):
    """Randomly mark a fraction of currently-observed APs as "not
    detected" (training-time only), using the same -1 sentinel and the
    same ``x > -0.999`` observed-check as
    ``ips.data.augmentation.DeviceAugmentedRSSDataset`` -- so an
    already-missing entry is left alone and only genuinely detected APs
    are corrupted.
    """
    if prob <= 0:
        return x
    observed = x > -0.999
    drop = (torch.rand_like(x) < prob) & observed
    return torch.where(drop, torch.full_like(x, -1.0), x)


class CDAE(nn.Module):
    """Denoising Autoencoder front-end: same shape as SAE, but the
    encoder sees a randomly corrupted input during training while the
    decoder is still asked to reconstruct the ORIGINAL (clean) input.
    At eval time (``self.training == False``) no corruption is applied.
    """

    def __init__(self, input_dim: int, corruption_prob: float = DEFAULT_CORRUPTION_PROB):
        super().__init__()
        self.corruption_prob = corruption_prob
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, LATENT_DIM), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Tanh(),
        )

    def forward(self, x):
        x_in = corrupt_rss(x, self.corruption_prob) if self.training else x
        z = self.encoder(x_in)
        reconstructed = self.decoder(z)
        # Reconstruction is compared against the clean x by the caller
        # (JointMixerModel keeps the original x around for that), which
        # is the standard denoising-autoencoder objective.
        return z, reconstructed


class MixerBlockV2(nn.Module):
    def __init__(self, num_tokens=NUM_TOKENS, hidden_dim=HIDDEN_DIM,
                 token_mlp_dim=TOKEN_MLP_DIM, channel_mlp_dim=CHANNEL_MLP_DIM,
                 dropout=DROPOUT):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(token_mlp_dim, num_tokens), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(channel_mlp_dim, hidden_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        y = self.norm1(x).transpose(1, 2)
        x = x + self.token_mlp(y).transpose(1, 2)
        y = self.norm2(x)
        return x + self.channel_mlp(y)


class JointMixerModel(nn.Module):
    """Feature extractor (SAE or CDAE) + MLP-Mixer backend + position head.

    Unchanged from the notebook: ``forward(rss)`` returns
    ``(position, reconstruction)``. This is wrapped by
    ``_JointMixerPlugin`` below to match the repo's single-tensor
    plugin interface.
    """

    def __init__(self, feature_extractor: nn.Module, name: str):
        super().__init__()
        self.name = name
        self.extractor = feature_extractor
        self.latent_norm = nn.LayerNorm(LATENT_DIM)
        self.embedding = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, NUM_TOKENS * HIDDEN_DIM),
        )
        self.mixers = nn.Sequential(*[MixerBlockV2() for _ in range(NUM_MIXER_BLOCKS)])
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.position_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 2),
        )

    def forward(self, rss):
        z, reconstruction = self.extractor(rss)
        z = self.latent_norm(z)
        x = self.embedding(z).view(rss.size(0), NUM_TOKENS, HIDDEN_DIM)
        x = self.mixers(x)
        x = self.norm(x).mean(dim=1)
        position = self.position_head(x)
        return position, reconstruction


class _JointMixerPlugin(nn.Module):
    """Adapts ``JointMixerModel`` to the repo's `[batch, 2]`-only plugin
    interface while still training on the joint (reconstruction +
    position) objective -- see the module docstring for how.
    """

    def __init__(self, feature_extractor: nn.Module, name: str,
                 recon_loss_weight: float = RECON_LOSS_WEIGHT):
        super().__init__()
        self.inner = JointMixerModel(feature_extractor, name=name)
        self.recon_loss_weight = recon_loss_weight

    def forward(self, x):
        position, reconstruction = self.inner(x)
        if self.training and torch.is_grad_enabled() and reconstruction.requires_grad:
            recon_loss = self.recon_loss_weight * F.mse_loss(reconstruction, x)
            if torch.isfinite(recon_loss):
                recon_loss.backward(retain_graph=True)
        return position


# ---------------------------------------------------------------------------
# Pretrained variants (SAE-MMP / CDAE-MMP)
# ---------------------------------------------------------------------------
#
# Paper's SAE-MMP/CDAE-MMP are two genuinely separate training phases: (1) the
# autoencoder is trained on unlabeled data with reconstruction loss only, (2)
# its (now fixed) latent features are fed to the MLP-Mixer, which is trained
# with the supervised position loss. This is different from SAE-MMJ/CDAE-MMJ
# above, where both parts are updated together from the start.
#
# Like `_JointMixerPlugin`, this is implemented self-contained inside the
# model -- no edits to `runner.py` -- because `runner.py` only ever runs a
# single `_fit_stage(... epochs ...)` call for these models (its own
# `stage1_epochs` branch is hardwired to `bsrbf_kan` and does unrelated,
# label-supervised partial-layer fine-tuning, not unsupervised AE
# pretraining), and `ips.algorithms.factory.create_model(name, input_dim)`
# has no channel to pass a per-run epoch count into `build_model` anyway.
#
# Mechanism: `_PretrainedMixerPlugin` counts training forward() calls in
# `self._step`. While `self._step < self.pretrain_steps` ("phase 1"):
#   - the extractor (SAE/CDAE) is trainable and is updated *only* from the
#     reconstruction loss, via the same manual `backward(retain_graph=True)`
#     trick `_JointMixerPlugin` uses (see the module docstring above);
#   - the latent `z` fed to the Mixer/position head is `z.detach()`, so the
#     position loss that `runner.py` backward()s afterwards cannot reach the
#     extractor's parameters -- it never sees the (unsupervised) SAE/CDAE
#     objective, matching "the SAE is first trained on unlabeled data" in
#     the paper;
#   - the Mixer/position head stay trainable and *do* receive the position
#     loss gradient during phase 1 too. This is a deliberate, safe choice,
#     not an oversight: `runner.py`'s `_fit_stage` unconditionally calls
#     `coordinate_loss(model(xb), yb).backward()` every step, which requires
#     `model(xb)` to carry a gradient path with at least one `requires_grad`
#     leaf; freezing the Mixer/head during phase 1 as well (on top of the
#     `z.detach()`) would make that output carry no grad at all and crash
#     that shared call. Letting the Mixer/head warm-start on the
#     progressively-improving (but detached) features is harmless for the
#     paper's distinction, since what actually defines "pretrained" here is
#     that the extractor never sees the position signal -- which still holds.
#
# At `self._step == self.pretrain_steps` ("phase 2"), the extractor's
# parameters are frozen (`requires_grad_(False)`) and the reconstruction
# loss is no longer computed or backward()ed. `z` is no longer manually
# detached in phase 2 either -- it does not need to be: with every extractor
# parameter frozen and the input `x` itself not requiring grad, `z` already
# carries no gradient path back into the extractor, so the position loss
# naturally only updates the Mixer/position head from here on, exactly the
# "fixed pretrained features -> supervised Mixer training" the paper
# describes.
#
# `self._step` only advances on *training* forward() calls (guarded by
# `self.training and torch.is_grad_enabled()`), so validation/prediction
# passes (``_predict`` runs under ``model.eval()`` and ``torch.no_grad()``)
# never consume the pretrain budget.
#
# `PRETRAIN_STEPS` is a fixed step budget, not an epoch count, because this
# class has no visibility into the dataloader's batches-per-epoch (that is
# decided inside `runner.py`, which is intentionally not touched). At the
# repo's batch_size=64, 2000 steps is roughly 10-40 epochs depending on the
# building/dataset size -- in the same ballpark as `RunConfig`'s own
# `stage1_epochs` default (30) used elsewhere in this repo. Pass a different
# `pretrain_steps` to the constructor to override it per-instance; there is
# currently no config.json knob for this (see the module-level rationale
# above), so overriding means constructing the plugin directly rather than
# going through `create_model`/`build_model`.
PRETRAIN_STEPS = 2000


class _PretrainedMixerPlugin(nn.Module):
    """Two-phase pretrained-then-supervised adapter (SAE-MMP / CDAE-MMP).

    Phase 1 (``self._step < pretrain_steps``): extractor trained from
    reconstruction loss only; Mixer/position head warm-start on detached
    features. Phase 2: extractor frozen; Mixer/position head trained from
    the position loss alone. See the module-level comment above for why
    this is implemented this way instead of touching ``runner.py``.
    """

    def __init__(self, feature_extractor: nn.Module, name: str,
                 pretrain_steps: int = PRETRAIN_STEPS,
                 recon_loss_weight: float = RECON_LOSS_WEIGHT):
        super().__init__()
        self.inner = JointMixerModel(feature_extractor, name=name)
        self.pretrain_steps = pretrain_steps
        self.recon_loss_weight = recon_loss_weight
        self._step = 0
        self._extractor_trainable = True  # matches the fresh module's initial requires_grad=True

    def _set_extractor_trainable(self, trainable: bool):
        if trainable == self._extractor_trainable:
            return
        for p in self.inner.extractor.parameters():
            p.requires_grad_(trainable)
        self._extractor_trainable = trainable

    def forward(self, x):
        pretraining = self._step < self.pretrain_steps
        self._set_extractor_trainable(pretraining)

        z, reconstruction = self.inner.extractor(x)
        z_in = z.detach() if pretraining else z
        z_in = self.inner.latent_norm(z_in)
        h = self.inner.embedding(z_in).view(x.size(0), NUM_TOKENS, HIDDEN_DIM)
        h = self.inner.mixers(h)
        h = self.inner.norm(h).mean(dim=1)
        position = self.inner.position_head(h)

        if self.training and torch.is_grad_enabled():
            if pretraining:
                recon_loss = self.recon_loss_weight * F.mse_loss(reconstruction, x)
                if torch.isfinite(recon_loss):
                    recon_loss.backward(retain_graph=True)
            self._step += 1
        return position
