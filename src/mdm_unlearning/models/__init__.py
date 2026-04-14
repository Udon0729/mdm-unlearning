"""Model architectures.

Re-exports the three model families used in the experiments:

- :class:`TransEncoder` -- bidirectional masked diffusion model (MDM).
- :class:`GPT` -- causal autoregressive language model (ARM).
- :class:`TransEncoderDecoder` -- encoder-decoder MDM with Q/KV split
  cross-self attention (E2D2-style).

The :class:`Config` class is a unified configuration object shared across
all three architectures.
"""
from mdm_unlearning.models.config import Config
from mdm_unlearning.models.diffmodel import TransEncoder
from mdm_unlearning.models.arm import GPT
from mdm_unlearning.models.enc_dec_diffmodel import (
    TransEncoderDecoder,
    EncoderBlock,
    DecoderBlock,
    forward_process_block,
)

__all__ = [
    "Config",
    "TransEncoder",
    "GPT",
    "TransEncoderDecoder",
    "EncoderBlock",
    "DecoderBlock",
    "forward_process_block",
]
