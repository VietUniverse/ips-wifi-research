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

Option B: matches the IAP paper's own numbers (true 60-dim latent,
128/192 encoder, 128/64 decoder) instead of reusing SAE/CDAE's
256/128/64 shape. Deviations that remain, chosen to do this without
touching the existing, unmodified ``mmj_common.JointMixerModel`` (see
that module's docstring for the joint-loss/backward mechanism):
  - ``JointMixerModel.embedding``/``latent_norm`` hard-code
    ``mmj_common.LATENT_DIM`` (64) as the feature-extractor output
    width, for every model (SAE/CDAE/VAE alike) -- not changed here.
    Since the paper's true latent is 60-dim, ``VAE`` below keeps the
    *real* 60-dim ``z``/``mu``/``logvar`` for reconstruction and KL
    (both computed in true 60-dim space, matching the paper exactly),
    and adds one extra ``mixer_projection: Linear(60, 64)`` layer that
    is NOT in the paper, used only to hand the position branch a
    64-dim vector so it fits ``JointMixerModel`` unmodified. This
    projection carries no reconstruction/KL signal of its own.
  - Decoder's final activation is Tanh, not left unspecified as in the
    paper, to match the repo's ``rss_standard`` [-1, 1] encoding (same
    reasoning as ``mmj_common.SAE``/``CDAE`` -- see that module's
    docstring).
  - KL_LOSS_WEIGHT below is a starting value in the usual beta-VAE
    range (not taken from either source, which do not state one
    numerically); it is a hyperparameter worth tuning, not a
    reproduction of a published constant.
  - Optimizer/LR/epochs/batch size the paper states (Adam, lr=1e-4,
    1000 epochs, batch 16) are training-script settings, outside the
    scope of this model-definition file.

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

# The paper's actual latent width (dim(v_i) = 60). Reconstruction and
# KL are computed in this space -- see VAE's docstring.
TRUE_LATENT_DIM = 60

# Beta-VAE style weight on the KL term. Kept separate from
# mmj_common.RECON_LOSS_WEIGHT (reused here for the L1 term) since the
# two losses have different scales and literature does not fix this
# value for WiFi-RSS VAEs -- tune if KL dominates or vanishes.
KL_LOSS_WEIGHT = 0.01


class VAE(nn.Module):
    """VAE front-end matching the IAP paper's own architecture: encoder
    Dense(128) -> Dense(192) -> (mu, logvar) each TRUE_LATENT_DIM (60),
    decoder Dense(128) -> Dense(64) -> reconstruction. The encoder
    parameterizes a Gaussian (mu, logvar) and ``z`` is sampled from it
    via the reparameterization trick during training (eval uses the
    mean, the standard VAE convention for deterministic inference).

    ``forward`` returns a 2-tuple like ``SAE``/``CDAE`` so it plugs
    into ``mmj_common.JointMixerModel`` unchanged, but what it returns
    as the first element is ``mixer_projection(z)`` (64-dim), NOT the
    true latent -- a shim needed only because ``JointMixerModel`` is
    hard-coded to ``mmj_common.LATENT_DIM`` (64) (see module docstring).
    The true 60-dim ``z``/``mu``/``logvar`` from the most recent forward
    pass are cached on ``self.z``/``self.mu``/``self.logvar`` for the
    reconstruction/KL terms computed in ``_JointVAEMixerPlugin`` below
    -- reconstruction and KL never go through ``mixer_projection``.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder_hidden = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 192), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(192, TRUE_LATENT_DIM)
        self.fc_logvar = nn.Linear(192, TRUE_LATENT_DIM)
        self.decoder = nn.Sequential(
            nn.Linear(TRUE_LATENT_DIM, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, input_dim), nn.Tanh(),
        )
        # Shim, not in the paper -- see class docstring.
        self.mixer_projection = nn.Linear(TRUE_LATENT_DIM, LATENT_DIM)

        self.z = None
        self.mu = None
        self.logvar = None

    def forward(self, x):
        h = self.encoder_hidden(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        self.z, self.mu, self.logvar = z, mu, logvar
        reconstructed = self.decoder(z)
        projected = self.mixer_projection(z)
        return projected, reconstructed


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
