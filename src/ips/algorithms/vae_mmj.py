"""Variational-Autoencoder + MLP-Mixer joint position model (VAE-MMJ).

Replaces the feature-extractor front-end of SAE-MMJ/CDAE-MMJ (see
``mmj_common``) with a VAE: the encoder parameterizes a Gaussian
(mu, logvar) over the latent instead of emitting it directly, ``z`` is
drawn via the reparameterization trick, and training adds a
KL-divergence term next to the reconstruction loss:

    L = ALPHA * L_rec(L1) + BETA_KL * L_KL

This mirrors the VAE loss used for WiFi-fingerprint encoding in two
sources verified before writing this file:
  - "Learning-Based WiFi Fingerprint Inpainting via Generative
    Adversarial Networks" (arXiv:2407.18915), whose IAP model uses
    exactly L_VAE = alpha*L_rec + beta*L_KL, with L_rec the L1 distance
    between reconstructed and ground-truth fingerprint and L_KL the
    KL-divergence of the latent against N(0, 1).
  - US patent "System and method for WiFi-based indoor localization
    via unsupervised domain adaptation" (US11363416), whose VAE encodes
    to a Gaussian (mu, sigma), samples the latent from it, and trains
    on KL-divergence + reconstruction loss.

Deviations from those sources, chosen to stay a drop-in replacement for
SAE/CDAE inside the existing, unmodified ``mmj_common.JointMixerModel``
(see that module's docstring for the joint-loss/backward mechanism):
  - Latent dim is ``mmj_common.LATENT_DIM`` (64), not the IAP paper's
    60 -- ``JointMixerModel.embedding``/``latent_norm`` are built for
    the shared constant and are not touched by this file.
  - Encoder/decoder hidden sizes are 256/128, matching ``SAE``/``CDAE``
    in ``mmj_common.py``, not the IAP paper's 128/192.
  - Decoder's final activation is Tanh, not sigmoid/linear, to match
    the repo's ``rss_standard`` [-1, 1] encoding (same reasoning as
    ``mmj_common.SAE``/``CDAE`` -- see that module's docstring).
  - KL_LOSS_WEIGHT below is a starting value in the usual beta-VAE
    range (not taken from either source, which do not state one
    numerically); it is a hyperparameter worth tuning, not a
    reproduction of a published constant.

This file defines its own plugin wrapper (``_JointVAEMixerPlugin``)
instead of reusing ``mmj_common._JointMixerPlugin`` because the latter
hard-codes a single MSE auxiliary loss, while VAE-MMJ needs two terms
(L1 reconstruction + KL) backward'd together. ``mmj_common.py`` is not
modified by this file.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .mmj_common import LATENT_DIM, RECON_LOSS_WEIGHT, JointMixerModel

MODEL_NAME = "vae_mmj"

# Beta-VAE style weight on the KL term. Kept separate from
# mmj_common.RECON_LOSS_WEIGHT (reused here for the L1 term) since the
# two losses have different scales and literature does not fix this
# value for WiFi-RSS VAEs -- tune if KL dominates or vanishes.
KL_LOSS_WEIGHT = 0.01


class VAE(nn.Module):
    """VAE front-end: same input/output shape as ``mmj_common.SAE``/
    ``CDAE`` (encoder -> LATENT_DIM latent, Tanh-decoded reconstruction),
    but the encoder parameterizes a Gaussian (mu, logvar) and ``z`` is
    sampled from it via the reparameterization trick during training
    (eval uses the mean, the standard VAE convention for deterministic
    inference).

    ``forward`` keeps the same ``(z, reconstruction)`` 2-tuple contract
    as ``SAE``/``CDAE`` so it plugs into ``mmj_common.JointMixerModel``
    unchanged; ``mu``/``logvar`` from the most recent forward pass are
    cached on ``self.mu``/``self.logvar`` for the KL term computed in
    ``_JointVAEMixerPlugin`` below.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, LATENT_DIM)
        self.fc_logvar = nn.Linear(128, LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Tanh(),
        )
        self.mu = None
        self.logvar = None

    def forward(self, x):
        h = self.hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        self.mu, self.logvar = mu, logvar
        reconstructed = self.decoder(z)
        return z, reconstructed


class _JointVAEMixerPlugin(nn.Module):
    """Adapts ``JointMixerModel`` to the repo's `[batch, 2]`-only plugin
    interface, same role and same backward-before-returning mechanism as
    ``mmj_common._JointMixerPlugin`` (see that class's docstring / the
    module docstring of ``mmj_common.py`` for why), but backward'ing two
    auxiliary terms (L1 reconstruction + KL) instead of one MSE term.
    """

    def __init__(self, feature_extractor: VAE, name: str,
                 recon_loss_weight: float = RECON_LOSS_WEIGHT,
                 kl_loss_weight: float = KL_LOSS_WEIGHT):
        super().__init__()
        self.inner = JointMixerModel(feature_extractor, name=name)
        self.recon_loss_weight = recon_loss_weight
        self.kl_loss_weight = kl_loss_weight

    def forward(self, x):
        position, reconstruction = self.inner(x)
        if self.training and torch.is_grad_enabled() and reconstruction.requires_grad:
            recon_loss = self.recon_loss_weight * F.l1_loss(reconstruction, x)
            mu, logvar = self.inner.extractor.mu, self.inner.extractor.logvar
            kl_loss = self.kl_loss_weight * (
                -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            )
            aux_loss = recon_loss + kl_loss
            if torch.isfinite(aux_loss):
                aux_loss.backward(retain_graph=True)
        return position


def build_model(input_dim: int):
    return _JointVAEMixerPlugin(VAE(input_dim), name="VAE-MMJ")
