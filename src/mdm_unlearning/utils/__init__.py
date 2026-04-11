"""Miscellaneous utilities used across training and evaluation.

The contents are largely re-exported from upstream SMDM and lit-gpt
helpers. Submodules:

- :mod:`mdm_unlearning.utils.utils` — default precision selection,
  parameter counting, CSV logger.
- :mod:`mdm_unlearning.utils.speed_monitor` — FLOP estimation /
  Lightning callback.
- :mod:`mdm_unlearning.utils.fused_cross_entropy` — fused CE kernel
  wrapper.

Submodules are intentionally *not* eagerly re-exported here to avoid the
``utils → models → utils`` circular import that would otherwise occur
through ``speed_monitor`` (which imports ``GPT`` / ``Config``). Import the
specific symbol you need from the submodule directly.
"""
__all__: list[str] = []
