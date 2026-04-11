"""UnTrac NLL attribution and reconstruction analysis scripts.

Each architecture (MDM, ARM, E2D2) has its own evaluation script because the
forward pass and loss formulation differ:

- ``untrac_mdm`` / ``untrac_ar`` / ``untrac_e2d2``
    Apply each unlearning method to the trained model and measure how the
    test NLL on ToxiGen / WinoBias / TruthfulQA changes. The resulting per-
    subset influence scores are written to JSON for downstream Pearson
    correlation analysis against the leave-one-out ground truth.

- ``reconstruction_mdm`` / ``reconstruction_ar`` / ``reconstruction_e2d2``
    After unlearning a single corpus, measure masked-token (MDM, E2D2) or
    next-token (ARM) reconstruction accuracy on samples from all eight
    training corpora. The selectivity (target_delta - mean_other_delta) is
    the headline metric for actual selective forgetting (as opposed to NLL
    attribution).

Usage::

    python -m mdm_unlearning.evaluate.untrac_mdm \
        --mode untrac --model 113 \
        --ckpt_path workdir/.../iter-040000-ckpt.pth \
        --unlearn_method eu --untrac_corpus bookcorpus \
        --output results/attribution/eu/bookcorpus.json
"""

__all__: list[str] = []
