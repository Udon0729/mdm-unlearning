# NOTICE

This project builds on prior open-source research code. The following
upstream projects are gratefully acknowledged.

## SMDM (Scaling up Masked Diffusion Models on Text)

- Repository: https://github.com/ML-GSAI/SMDM
- Paper: Nie et al., "Scaling up Masked Diffusion Models on Text", arXiv:2410.18514
- License: MIT

The following modules in `src/mdm_unlearning/` are derived from SMDM with
modifications. Original copyright and license notices are preserved at the
top of each file.

| New location | SMDM origin |
| --- | --- |
| `src/mdm_unlearning/models/diffmodel.py` | `lit_gpt/diffmodel.py` |
| `src/mdm_unlearning/models/arm.py` | `lit_gpt/model.py` (renamed) |
| `src/mdm_unlearning/models/config.py` | `lit_gpt/config.py` |
| `src/mdm_unlearning/models/rmsnorm.py` | `lit_gpt/rmsnorm.py` |
| `src/mdm_unlearning/models/rotary.py` | `lit_gpt/fused_rotary_embedding.py` (with Blackwell GPU fix) |
| `src/mdm_unlearning/data/packed_dataset.py` | `lit_gpt/packed_dataset.py` (with epoch-bound fix) |
| `src/mdm_unlearning/data/tokenizer.py` | `lit_gpt/tokenizer.py` |
| `src/mdm_unlearning/utils/speed_monitor.py` | `lit_gpt/speed_monitor.py` |
| `src/mdm_unlearning/utils/utils.py` | `lit_gpt/utils.py` |

### Modifications

- Replaced `fused_rotary_embedding.apply_rotary_emb_func` with
  `flash_attn.layers.rotary.apply_rotary_emb` for compatibility with
  Blackwell-architecture GPUs.
- Added explicit step-bound and `StopIteration` semantics to
  `PackedDataset` (originally an infinite iterator that wraps without
  raising; this caused unbounded unlearning loops in our experiments).

## E2D2 (Encoder-Decoder Diffusion Language Models)

- Repository: https://github.com/kuleshov-group/e2d2
- Paper: Arriola et al., "Encoder-Decoder Diffusion Language Models for
  Efficient Training and Inference", NeurIPS 2025 (arXiv:2510.22852)
- License: see upstream repository

`src/mdm_unlearning/models/enc_dec_diffmodel.py` is an independent
re-implementation of the E2D2 cross-attention encoder-decoder MDM
architecture inside the SMDM codebase. It uses Q/KV-split attention with
block-causal masking via SDPA. No code is copied directly from the E2D2
repository; the implementation is new and adapted to the SMDM module
conventions.

## UnTrac

- Paper: Misonuma & Titov, "Unlearning Traces the Influential Training
  Data of Language Models", ACL 2024
- Repository: https://github.com/misonuma/untrac

The UnTrac evaluation methodology, data preparation pipeline (8 corpora,
test data preprocessing for ToxiGen / WinoBias / TruthfulQA), and
hyperparameters (Adam, lr=5e-5, batch=1, 1 epoch) are all from the
original paper. Our implementation follows their setup as closely as
possible while extending the framework to MDM and ARM architectures.

## Exclusive Unlearning (排他的逆学習)

- Paper: Sasaki, Nakayama, Miyao, Oseki, Isonuma, "排他的逆学習
  (Exclusive Unlearning)", 言語処理学会 第32回年次大会 B2-16, 2026

The Exclusive Unlearning method (Uniform CE forgetting loss + explicit
retain loss) implemented in `src/mdm_unlearning/unlearning/exclusive.py`
is adapted from this paper. Modifications: applied to MDM ELBO loss
instead of autoregressive NLL, and used training corpus data instead of
self-generated text as the forget corpus.

## NPO

- Paper: Zhang et al., "Negative Preference Optimization: From
  Catastrophic Collapse to Effective Unlearning", arXiv:2404.05868

NPO-style adaptive weighting in
`src/mdm_unlearning/unlearning/npo.py` follows the bounded ratio
formulation from this paper.

## Other Acknowledgments

- Test data preprocessing follows the recipe in
  `misonuma/untrac/preprocess_test.py`.
- Knowledge neuron localization analysis follows Dai et al.,
  "Knowledge Neurons in Pretrained Transformers", ACL 2022.
- Tokenizer is `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`
  from Hugging Face Hub.
