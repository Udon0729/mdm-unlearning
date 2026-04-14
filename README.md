# mdm-unlearning

> 日本語版の README は [`README.ja.md`](README.ja.md) を参照してください。

Empirical study of **training data attribution** and **selective unlearning** in **masked diffusion language models** (MDMs), benchmarked against autoregressive language models (ARMs) and an encoder-decoder MDM (E2D2-style) baseline.

This repository contains the code, raw result JSONs, and figures used in the technical report at [`docs/untrac_results.md`](docs/untrac_results.md).

## Headline finding

> UnTrac's NLL-based attribution score works on MDMs at parity with the original ARM paper, but token-level reconstruction analysis shows that the underlying mechanism is **differential collapse detection**, not selective forgetting. Whether selective unlearning actually succeeds is governed **not by architecture but by the granularity of the forget set** — fact level (~100 items) succeeds for both ARM and MDM, while corpus level (~40 000 items) fails for both.

See [`docs/untrac_results.md`](docs/untrac_results.md) for the full report and [`docs/experiment_setup.md`](docs/experiment_setup.md) for the controlled comparison setup.

## What this repo contains

- **`src/mdm_unlearning/`** — installable Python package (`src/` layout, PEP 621).
  - `models/` — MDM (`TransEncoder`), ARM (`GPT`), and E2D2-style encoder-decoder MDM (`TransEncoderDecoder`), plus `Config` and `RMSNorm`/`RoPE` building blocks.
  - `data/` — `PackedDataset` / `CombinedDataset` and the TinyLlama-tokenizer wrapper.
  - `train/` — pretraining loops for MDM, ARM, and E2D2 (Lightning Fabric, single-script per architecture).
  - `evaluate/` — UnTrac NLL attribution and masked-token reconstruction analysis for all three architectures.
  - `analysis/` — corpus-specific trigram extraction, knowledge-neuron localization, neuron suppression, and fact-level EU.
  - `utils/` — `SpeedMonitor`, fused cross-entropy, checkpoint helpers.
- **`scripts/`** — shell scripts for end-to-end runs (`train/`, `untrac/`, `reconstruction/`) and a `prepare_data.py` data-prep entry point.
- **`results/`** — raw JSON outputs grouped by experiment family (`ground_truth`, `attribution/{compare,fisher_meta,eu,e2d2,ar,mdm_kl}`, `reconstruction/{mdm_kl,mdm_fm,mdm_eu,e2d2,ar}`, `analysis/`).
- **`docs/`** — technical report (English + Japanese), experiment setup (English + Japanese), and rendered figures.
- **`configs/`** — placeholder for YAML model/training configs.
- **`tests/`** — placeholder for unit tests.

## Architectures and methods

| Architecture | Class | Loss | Where |
| --- | --- | --- | --- |
| MDM (single bidirectional Transformer) | `TransEncoder` | masked-diffusion ELBO | `models/diffmodel.py` |
| ARM (causal LLaMA-style) | `GPT` | NLL | `models/arm.py` |
| Encoder-decoder MDM (E2D2-style) | `TransEncoderDecoder` | block-causal ELBO | `models/enc_dec_diffmodel.py` |

| Unlearning method | Stable? | Collapse-resistant? | Implemented in |
| --- | --- | --- | --- |
| Gradient Ascent (GA) | no | no | `evaluate/untrac_*.py` |
| KL-Constrained GA (KL) | yes | yes (regularized) | `evaluate/untrac_*.py` |
| NPO-style adaptive weight | yes | yes (saturates at step 1) | `evaluate/untrac_*.py` |
| VDU L2-anchor | depends on γ | weak at γ=0.01 | `evaluate/untrac_*.py` |
| Fisher-Meta (Selective Amnesia + SalUn + Meta-Unlearning) | yes | yes | `evaluate/untrac_mdm.py` (`--unlearn_method fisher_meta`) |
| Exclusive Unlearning (EU) | yes | bounded by `log V` | `evaluate/untrac_*.py` (`--unlearn_method eu`) |

## Installation

This project requires **Python 3.11+**, **CUDA 12**, and **flash-attention 2**.

```sh
git clone https://github.com/Udon0729/mdm-unlearning.git
cd mdm-unlearning

# Recommended: uv (https://docs.astral.sh/uv/)
uv venv -p 3.11
source .venv/bin/activate
uv pip install -e ".[viz,dev]"

# flash-attn must be installed separately to match your CUDA toolchain:
uv pip install flash-attn --no-build-isolation
```

`pip install -e .` works as well if you don't use uv. The runtime dependencies are pinned in [`pyproject.toml`](pyproject.toml); the `[viz]` and `[dev]` extras add matplotlib/seaborn and ruff/mypy/pytest/pre-commit respectively.

## Data preparation

The training corpora are 8 splits of public web text (BookCorpus, StackExchange, CCNewsV2, Gutenberg, HackerNews, OpenWebText, Pile-CC, Wikipedia), each tokenized with the TinyLlama tokenizer and packed into 1024-token blocks. See [`docs/experiment_setup.md`](docs/experiment_setup.md) §5 for the exact source of each corpus.

```sh
python scripts/prepare_data.py --output_dir data/untrac
```

The expected output layout is `data/untrac/train_<corpus>_*.bin`. The eight test sets (ToxiGen, WinoBias, TruthfulQA) are downloaded automatically by the evaluation scripts.

## Reproducing the experiments

All commands are run from the repository root. **Checkpoints are not included** (see [`.gitignore`](.gitignore)) — you will need to either pretrain models yourself or point the scripts at your own checkpoints via the environment variables `MDM_CKPT`, `AR_CKPT`, `E2D2_CKPT`.

### 1. Pretraining

```sh
# MDM (113M non-embedding params, 40k steps; 1 GPU example)
python -m mdm_unlearning.train.train_mdm \
    --model 113 --max_steps 40000 --data_dir data/untrac \
    --save_interval 40000 --num_devices 1 \
    --batch_size 8 --micro_batch_size 4 --grad_clip 1.0 --seq_len 1024

# ARM
python -m mdm_unlearning.train.train_ar --model 113 --max_steps 40000 ...

# E2D2 encoder-decoder MDM
python -m mdm_unlearning.train.train_e2d2 --model 113 --max_steps 40000 ...
```

For Leave-One-Out training (8 runs, one per excluded corpus), use:

```sh
bash scripts/train/run_loo.sh
```

### 2. UnTrac NLL attribution

```sh
# Single corpus, KL-constrained GA
MDM_CKPT=workdir/.../iter-040000-ckpt.pth \
python -m mdm_unlearning.evaluate.untrac_mdm \
    --mode untrac --model 113 \
    --ckpt_path "$MDM_CKPT" --data_dir data/untrac \
    --unlearn_method kl --untrac_corpus bookcorpus \
    --output results/attribution/compare/kl_bookcorpus.json

# Sweep all corpora & methods
bash scripts/untrac/run_compare.sh         # GA / KL / NPO / VDU
bash scripts/untrac/run_fisher_meta.sh     # Fisher-Meta
bash scripts/untrac/run_eu.sh              # Exclusive Unlearning
bash scripts/untrac/run_e2d2.sh            # E2D2 KL
bash scripts/untrac/run_ar.sh              # ARM KL/EU
```

### 3. Masked-token reconstruction

```sh
python -m mdm_unlearning.evaluate.reconstruction_mdm \
    --model 113 --ckpt_path "$MDM_CKPT" --data_dir data/untrac \
    --unlearn_corpus bookcorpus --unlearn_steps 10000 \
    --unlearn_method kl --num_samples 100 --mask_ratio 0.15 \
    --output results/reconstruction/mdm_kl/recon_bookcorpus.json

# Sweeps
bash scripts/reconstruction/run_mdm_kl.sh
bash scripts/reconstruction/run_mdm_fm.sh
bash scripts/reconstruction/run_mdm_eu.sh
bash scripts/reconstruction/run_e2d2.sh
bash scripts/reconstruction/run_ar.sh
```

### 4. Knowledge localization and fact-level unlearning

```sh
# Phase 0: extract corpus-specific trigrams + verify MDM can predict them
python -m mdm_unlearning.analysis.extract_corpus_trigrams \
    --ckpt_path "$MDM_CKPT" \
    --output_trigrams results/analysis/corpus_specific_trigrams.json \
    --output_prediction results/analysis/trigram_prediction_results.json

# Phase 1: per-MLP-neuron knowledge importance
python -m mdm_unlearning.analysis.knowledge_localization \
    --ckpt_path "$MDM_CKPT" \
    --output results/analysis/knowledge_localization.json

# Phase 2a: zero out top-k neurons
python -m mdm_unlearning.analysis.neuron_suppression \
    --ckpt_path "$MDM_CKPT" \
    --output results/analysis/neuron_suppression_results.json

# Phase 2b: gradient-based EU at fact level (granularity test)
python -m mdm_unlearning.analysis.fact_level_eu \
    --ckpt_path "$MDM_CKPT" \
    --output results/analysis/fact_level_eu_results.json
```

## Key results at a glance

| Experiment | Result |
| --- | --- |
| SMDM NLL attribution (KL / Fisher-Meta / EU) | r ≈ 0.48-0.49 (parity with ARM paper) |
| SMDM token reconstruction (KL, 10k) | selectivity -0.001 (uniform collapse) |
| SMDM token reconstruction (EU, 10k) | selectivity -0.075 (best of corpus level, retain 18.3%) |
| ARM token reconstruction (EU, 10k) | selectivity -0.073 (≈ MDM EU) |
| E2D2 token reconstruction (KL, 10k) | selectivity +0.001 (cross-attention is not enough) |
| Fact-level gradient-based EU (200 steps) | **selectivity -0.663** (target -95% on gutenberg, others +1.3%) |
| Fact-level neuron suppression (k=200) | selectivity -0.135 |

Full numbers, ablations and discussion: [`docs/untrac_results.md`](docs/untrac_results.md).

## Citation

If you use this code or build on the analysis, please cite as:

```bibtex
@misc{munaoka2026mdmunlearning,
  author       = {Munaoka, Kota},
  title        = {{mdm-unlearning}: empirical study of training data attribution and selective unlearning in masked diffusion language models},
  year         = {2026},
  howpublished = {\url{https://github.com/Udon0729/mdm-unlearning}}
}
```

A machine-readable [`CITATION.cff`](CITATION.cff) is also provided.

## License and attribution

This project is released under the [MIT License](LICENSE). It builds on a number of upstream research codebases — see [`NOTICE.md`](NOTICE.md) for the full attribution chain (SMDM, E2D2, UnTrac, Exclusive Unlearning, NPO).

## Contact

Author: **Kazuki Munaoka** (Shizuoka University) — `munaoka.kazuki.22@shizuoka.ac.jp`
Issues: <https://github.com/Udon0729/mdm-unlearning/issues>
