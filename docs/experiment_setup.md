# UnTrac on Masked Diffusion Models: Experimental Setup

> A Japanese-language version of this document is available at [`experiment_setup.ja.md`](experiment_setup.ja.md).

## 1. Goal

Apply the unlearning-based training-data attribution methods **UnTrac** and **UnTrac-Inv** (Misonuma et al., 2024) to **masked diffusion language models (MDMs)** as a controlled extension of the original paper's autoregressive (ARM) experiment, and compare the two architectures.

---

## 2. Controlled comparison

### 2.1 Original paper (ARM side)

| Item | Setting |
| --- | --- |
| Model | OPT-125M (decoder-only, 125M params) |
| Training | from scratch |
| Loss | autoregressive cross-entropy (= NLL) |
| Training data | 8 corpora × 40,000 sequences × 1,024 tokens |
| Total tokens | 327,680,000 (~328M tokens) |

### 2.2 This work (MDM side)

| Item | Setting |
| --- | --- |
| Model | Diff_LLaMA_113M (bidirectional Transformer, 162M params) |
| Training | from scratch |
| Loss | ELBO (variational lower bound for masked diffusion) |
| Training data | 8 corpora × 40,000 sequences × 1,024 tokens |
| Total tokens | 327,680,000 (~328M tokens) |

---

## 3. Architecture comparison

| Item | OPT-125M (ARM) | Diff_LLaMA_113M (MDM) |
| --- | --- | --- |
| Architecture | Decoder-only (causal) | Bidirectional Transformer |
| Layers | 12 | 12 |
| Attention heads | 12 | 12 |
| Hidden dim | 768 | 768 |
| FFN dim | 3072 | 3072 |
| Normalisation | LayerNorm | RMSNorm |
| Activation | GeLU | SwiGLU |
| Positional encoding | learned | RoPE |
| Non-embedding params | ~85M | 113M |
| Total params | ~125M | ~162M |
| Vocabulary | 50,272 | 32,000 |
| Tokenizer | GPT-2 BPE (OPT) | LLaMA SentencePiece (TinyLlama) |

The depth/width are matched. Normalisation, activation and positional encoding follow standard practice in each family and are noted as a confounder.

---

## 4. Training hyperparameters

| Item | Original (OPT-125M) | This work (MDM) | Match |
| --- | --- | --- | --- |
| Training steps | 40,000 | 40,000 | ✓ |
| Global batch size | 8 | 8 | ✓ |
| Sequence length | 1,024 | 1,024 | ✓ |
| Optimizer | AdamW (wd=0) | Adam (wd=0) | ✓ (equivalent) |
| Learning rate | 5e-5 | 5e-5 | ✓ |
| LR schedule | constant | constant | ✓ |
| β₁ / β₂ | 0.9 / 0.999 | 0.9 / 0.999 | ✓ |
| Gradient clipping | 1.0 | 1.0 | ✓ |
| Warmup | 0 | 0 | ✓ |
| Weight decay | 0.0 | 0.0 | ✓ |

Source: `arguments.py` and `pretrain_opt.sh` in `misonuma/untrac`.

---

## 5. Training data

### 5.1 Corpora (Equal setting)

| Corpus | Samples | Source | Mapping to original paper |
| --- | --- | --- | --- |
| BookCorpus | 40,000 | bookcorpus/bookcorpus | BookCorpus ✓ |
| StackExchange | 40,000 | The Pile (pile-uncopyrighted) | CC-Stories (substitute) △ |
| CCNewsV2 | 40,000 | vblagoje/cc_news | CCNewsV2 ✓ |
| Gutenberg | 40,000 | deepmind/pg19 | PJ Gutenberg ✓ |
| HackerNews | 40,000 | The Pile (pile-uncopyrighted) | HackerNews ✓ |
| OpenWebText | 40,000 | Skylion007/openwebtext | OpenWebText2 (substitute) △ |
| Pile-CC | 40,000 | The Pile (pile-uncopyrighted) | Pile-CC ✓ |
| Wikipedia | 40,000 | The Pile (pile-uncopyrighted) | Wikipedia ✓ |

Total: 320,000 samples × 1,024 tokens ≈ 328M tokens. CC-Stories (no longer available) is substituted with StackExchange and OpenWebText2 (copyright issues) with OpenWebText.

### 5.2 Preprocessing

Following the original paper: tokenize with the TinyLlama tokenizer, concatenate end-to-end and split into 1,024-token blocks (drop the trailing remainder). See `scripts/prepare_data.py`.

---

## 6. Test data

Preprocessed following the original paper's `preprocess_test.py`.

| Dataset | Subsets | Per-subset size | Total | Filter |
| --- | --- | --- | --- | --- |
| ToxiGen | 13 groups | 256 | 3,328 | prompt_label==1, 8-24 words |
| WinoBias | 4 categories | 256 | 1,024 | type1_pro/anti × gender |
| TruthfulQA | 9 categories | 256 | ~1,852 | wrong answers, ≥128 per category |

---

## 7. Pipeline

### 7.1 Ground truth: Leave-One-Out attribution

For each test subset s and training corpus Z_k:

$$
\text{Influence}_{\text{LOO}}(s, Z_k) = \mathcal{L}_{\text{ELBO}}(s, \theta_{\setminus k}) - \mathcal{L}_{\text{ELBO}}(s, \theta_{\text{full}})
$$

with θ_full trained on all 8 corpora and θ_{\k} retrained without Z_k. See `scripts/train/run_loo.sh`.

### 7.2 UnTrac

Apply gradient ascent on each corpus to a trained model and measure the change in test loss:

$$
I_{\text{UnTrac}}(s, Z_k) = \mathcal{L}_{\text{ELBO}}(s, \theta_T^{(k)}) - \mathcal{L}_{\text{ELBO}}(s, \theta_0)
$$

Hyperparameters: Adam, lr = 5e-5, batch = 1, 1 epoch, no gradient clipping. Implementation: `src/mdm_unlearning/evaluate/untrac_mdm.py` (and `untrac_e2d2.py`, `untrac_ar.py`).

### 7.3 UnTrac-Inv

Run gradient ascent on the test set and measure the change in training-corpus loss (the inverse direction):

$$
I_{\text{UnTrac-Inv}}(s, Z_k) = \mathcal{L}_{\text{ELBO}}(Z_k, \theta_T'^{(s)}) - \mathcal{L}_{\text{ELBO}}(Z_k, \theta_0)
$$

Effective batch size 256, max_steps 50.

### 7.4 MDM-specific considerations

| Aspect | ARM | MDM |
| --- | --- | --- |
| Loss L | NLL (deterministic) | ELBO (stochastic; depends on `t` and mask `m`) |
| ELBO vs NLL | identical | ELBO ≤ NLL (generally not equal) |
| Reproducibility | deterministic | Monte Carlo estimate (variance) |
| Influence accuracy | high | depends on MC sample count `N` (variance ≈ O(1/N)) |

We use N = 128 MC samples to estimate the ELBO.

---

## 8. Evaluation metric

For each test subset, compute the Pearson correlation between the 8-dimensional influence vectors of the method and LOO:

$$
r = \text{Pearson}(\mathbf{I}_{\text{method}}, \mathbf{I}_{\text{LOO}})
$$

with method ∈ {UnTrac, UnTrac-Inv}, averaged over the 26 subsets (13 ToxiGen + 4 WinoBias + 9 TruthfulQA).

---

## 9. Limitations

1. **Architecture differences.** OPT-125M and Diff_LLaMA_113M match in depth/width but differ in normalisation (LayerNorm vs RMSNorm), activation (GeLU vs SwiGLU) and positional encoding (learned vs RoPE). These are confounders for the ARM-vs-MDM comparison.
2. **Corpus substitutions.** CC-Stories (no longer available) → StackExchange and OpenWebText2 (copyright) → OpenWebText. Direct comparison with the ARM paper's numerical values needs caution.
3. **Tokenizer differences.** GPT-2 BPE (50,272 tokens) vs LLaMA SentencePiece (32,000 tokens) tokenize the same text differently.
4. **ELBO stochasticity.** MDM influence estimation has Monte Carlo variance not present in ARM NLL.

---

## 10. Compute environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q (97 GB) |
| GPUs (full training) | 2 |
| GPUs (LOO training) | 1 / run |
| PyTorch | 2.11.0+cu128 |
| Flash-Attention | 2.8.3 |
| Lightning Fabric | 2.6.1 |
| Python | 3.11.13 |

---

## References

- Misonuma et al. (2024). "Unlearning Traces the Influential Training Data of Language Models." arXiv:2401.15241
- Nie et al. (2024). "Scaling up Masked Diffusion Models on Text." arXiv:2410.18514
