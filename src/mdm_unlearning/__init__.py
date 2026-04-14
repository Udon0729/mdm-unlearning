"""mdm-unlearning: Empirical study of training data attribution and selective
unlearning in masked diffusion language models.

Top-level package re-exports the most commonly used building blocks so that
downstream code can write `from mdm_unlearning import TransEncoder` instead of
the longer fully-qualified path.

Modules
-------
models
    Model architectures: MDM (TransEncoder), ARM (GPT), and the
    encoder-decoder MDM (TransEncoderDecoder).
data
    PackedDataset / CombinedDataset and tokenizer wrappers.
unlearning
    Unlearning method implementations (GA, KL, NPO, VDU, Fisher-Meta, EU).
analysis
    Knowledge localization, neuron suppression, and reconstruction analysis.
utils
    Speed monitor, checkpoint helpers, etc.

Each runnable module exposes its CLI through ``if __name__ == "__main__"``,
so it can be invoked with ``python -m mdm_unlearning.<sub>.<module>`` (e.g.
``python -m mdm_unlearning.evaluate.untrac_mdm --help``).
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mdm-unlearning")
except PackageNotFoundError:  # package is not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
