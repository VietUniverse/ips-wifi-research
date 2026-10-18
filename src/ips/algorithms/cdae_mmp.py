"""Denoising-Autoencoder + MLP-Mixer PRETRAINED position model (CDAE-MMP).

Companion to ``cdae_mmj.py``: same ``CDAE`` feature extractor (random
currently-detected APs zeroed before encoding during training, decoder
still reconstructs the original RSS vector) and the same Mixer/position-head
architecture, but trained in the paper's two separate phases (CDAE first,
unsupervised, on reconstruction loss alone; then the Mixer/position head on
the supervised position loss with the CDAE's features fixed) instead of
jointly from the start. See ``mmj_common``'s "Pretrained variants" section
for the full mechanism and for why this is self-contained in the model
rather than implemented as changes to ``runner.py``, and for why this model
is paired with the repo's ``rss_standard`` preprocessor instead of the
notebook's own RSS encoding.

Note: ``CDAE``'s input corruption is gated on its own ``self.training`` flag
(inherited from this plugin's ``.train()``/``.eval()`` calls), not on the
pretrain/finetune phase. So corruption still applies during phase-2
*training* forward passes (same behaviour as CDAE-MMJ), even though the
CDAE's weights are frozen there -- it only ever affects validation/test
predictions by being switched off during ``model.eval()``.
"""

from .mmj_common import CDAE, DEFAULT_CORRUPTION_PROB, _PretrainedMixerPlugin

MODEL_NAME = "cdae_mmp"


def build_model(input_dim: int):
    return _PretrainedMixerPlugin(CDAE(input_dim, DEFAULT_CORRUPTION_PROB), name="CDAE-MMP")
