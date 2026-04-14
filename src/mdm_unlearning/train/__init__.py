"""From-scratch training scripts for the three architectures.

Each module exposes a ``setup()`` function that builds the Lightning Fabric,
configures the model, and starts training. Hyperparameters follow the UnTrac
paper recipe (Adam, lr=5e-5, batch=8, 40K steps).

Modules
-------
train_mdm
    Train the SMDM-style bidirectional masked diffusion model.
train_ar
    Train the GPT-style autoregressive control model.
train_e2d2
    Train the encoder-decoder MDM (E2D2-inspired) with Q/KV split
    cross-self attention and block-causal masking.

These modules are typically invoked through the CLI wrapper
:mod:`mdm_unlearning.cli.train`, but can also be run directly:

    python -m mdm_unlearning.train.train_mdm --model 113 --max_steps 40000
"""

__all__: list[str] = []
