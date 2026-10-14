"""Denoising-Autoencoder + MLP-Mixer joint position model (CDAE-MMJ).

Ported from the SAE-CDAE_MMJ research notebook. During training a
random subset of currently-detected APs is zeroed before encoding,
while the decoder is still asked to reconstruct the original
(uncorrupted) RSS vector -- the standard denoising-autoencoder
objective, aimed at the sparsity of WiFi RSS fingerprints. See
``mmj_common`` for the shared architecture, for how the joint
reconstruction+position objective is trained without changes to
``runner.py``, and for why this model is paired with the repo's
``rss_standard`` preprocessor instead of the notebook's own RSS
encoding.
"""

from .mmj_common import CDAE, DEFAULT_CORRUPTION_PROB, _JointMixerPlugin

MODEL_NAME = "cdae_mmj"


def build_model(input_dim: int):
    return _JointMixerPlugin(CDAE(input_dim, DEFAULT_CORRUPTION_PROB), name="CDAE-MMJ")
