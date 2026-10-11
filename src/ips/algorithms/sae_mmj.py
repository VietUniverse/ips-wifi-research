"""Stacked-Autoencoder + MLP-Mixer joint position model (SAE-MMJ).

Ported from the SAE-CDAE_MMJ research notebook. See ``mmj_common`` for
the shared architecture, for how the joint reconstruction+position
objective is trained without changes to ``runner.py``, and for why this
model is paired with the repo's ``rss_standard`` preprocessor instead
of the notebook's own RSS encoding.
"""

from .mmj_common import SAE, _JointMixerPlugin

MODEL_NAME = "sae_mmj"


def build_model(input_dim: int):
    return _JointMixerPlugin(SAE(input_dim), name="SAE-MMJ")
