"""Unlearning method implementations.

This sub-package collects the loss functions and helpers for the six
unlearning methods evaluated in the paper:

- ``ga``           Naive gradient ascent (UnTrac, Misonuma & Titov 2024).
- ``kl``           KL-constrained gradient ascent (this work).
- ``npo``          Negative Preference Optimization (Zhang et al. 2024).
- ``vdu``          VDU L2-anchor (Heng & Soh 2023, simplified).
- ``fisher_meta``  Fisher-EWC + saliency mask + meta-unlearning hybrid.
- ``exclusive``    Exclusive Unlearning (Sasaki et al. 2026).

Each method exposes a ``compute_loss`` function with a uniform signature so
that the main UnTrac evaluation script can switch between methods via the
``--unlearn-method`` flag without conditionals scattered across the loop.
"""

__all__: list[str] = []
