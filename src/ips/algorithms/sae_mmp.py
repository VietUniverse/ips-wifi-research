"""Stacked-Autoencoder + MLP-Mixer PRETRAINED position model (SAE-MMP).

Companion to ``sae_mmj.py``: same ``SAE`` feature extractor and the same
Mixer/position-head architecture, but trained in the paper's two separate
phases (SAE first, unsupervised, on reconstruction loss alone; then the
Mixer/position head on the supervised position loss with the SAE's features
fixed) instead of jointly from the start. See ``mmj_common``'s
"Pretrained variants" section for the full mechanism and for why this is
self-contained in the model rather than implemented as changes to
``runner.py``, and for why this model is paired with the repo's
``rss_standard`` preprocessor instead of the notebook's own RSS encoding.
"""

from .mmj_common import SAE, _PretrainedMixerPlugin

MODEL_NAME = "sae_mmp"


def build_model(input_dim: int):
    return _PretrainedMixerPlugin(SAE(input_dim), name="SAE-MMP")
