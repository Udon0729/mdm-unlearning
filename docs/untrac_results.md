# UnTrac on Masked Diffusion Models: Experimental Results

> A Japanese-language version of this report is available at [`untrac_results.ja.md`](untrac_results.ja.md). The Japanese version is the primary, more detailed write-up; this English summary mirrors its structure but condenses some commentary.

## Overview

We apply the unlearning-based training-data attribution method **UnTrac** (Misonuma & Titov, ACL 2024) to **Masked Diffusion Language Models (MDMs)**, contrast it with autoregressive language models (ARMs), and expose a **gap between "NLL-based attribution" and "actual selective forgetting."**

### Position of this work

UnTrac is a **data attribution** method whose evaluation unit is the leave-one-out (LOO) unit, i.e. **whole training corpora**. In contrast, *practical* unlearning needs (GDPR right-to-be-forgotten, harmful-response removal, PII deletion, ...) operate at a much finer granularity — documents, facts, or patterns.

This work investigates the gap between **attribution granularity** and **practical-unlearning granularity** through four questions:

1. **Attribution.** Does UnTrac's NLL attribution score work on MDMs as well as it does on the original-paper ARMs?
2. **Forgetting mechanism.** When NLL attribution works, is *actual* selective forgetting happening underneath?
3. **Granularity dependence.** Does the success/failure of selective unlearning differ between corpus level and fact level?
4. **Architecture independence.** Does the same behavior hold for both ARM (causal) and MDM (bidirectional)?

### Headline finding (one sentence)

> UnTrac's NLL attribution works on both ARM and MDM at parity with the original paper, but token-level reconstruction analysis shows that what it captures is *differential collapse detection*, not selective forgetting. Whether selective unlearning succeeds is governed **not by architecture but by the granularity of the forget set** — it succeeds at the fact level (~100 items) and fails for both ARM and MDM at the corpus level (~40k items).

---

## 1. Setup

### 1.1 Models and data

| Item | ARM (original paper) | MDM (this work) |
| --- | --- | --- |
| Model | OPT-125M (decoder-only) | Diff_LLaMA_113M (bidirectional) |
| Train steps | 40,000 | 40,000 |
| Batch size | 8 | 8 |
| Sequence length | 1,024 | 1,024 |
| Training data | 8 corpora × 40,000 samples | 8 corpora × 40,000 samples |
| Loss | NLL (deterministic) | ELBO (stochastic) |

### 1.2 Training corpora

BookCorpus, StackExchange, CCNewsV2, Gutenberg, HackerNews, OpenWebText, Pile-CC, Wikipedia (8 corpora × 40,000 sequences × 1,024 tokens ≈ 328M tokens).

### 1.3 Evaluation data

| Dataset | Subsets | Total samples | Target |
| --- | --- | --- | --- |
| ToxiGen | 13 groups | 3,328 | toxic language |
| WinoBias | 4 categories | 1,024 | gender bias |
| TruthfulQA | 9 categories | ~1,852 | factual accuracy |

---

## 2. Methods

### 2.1 UnTrac (original)

UnTrac estimates training-data influence via *unlearning*:

1. Take a trained model θ₀ and "forget" a corpus Z_k by gradient ascent.
2. The change in test loss ΔL = L(θ_after) - L(θ_before) is the influence score.
3. ΔL > 0 ⇒ Z_k is helpful for the test data; ΔL < 0 ⇒ harmful.

**Ground truth (LOO).** Re-train the model excluding each corpus and measure test-loss deltas. Most reliable but expensive.

**Metric.** Pearson correlation between the 8-corpus influence vectors of UnTrac and LOO, averaged over test subsets (ToxiGen 13 + WinoBias 4 + TruthfulQA 9 = 26 subsets).

### 2.2 MDM ELBO loss

```
L_ELBO = E_{t~U[0,1]} E_{mask~Bernoulli(p(t))} [ Σ_{masked positions} CE(logits, x_true) / p_mask ]
```

with `p(t) = (1-ε)t + ε`, `ε = 1e-3`. Unlike ARM's deterministic NLL, the gradient varies between runs because of the random `t` and mask draws.

### 2.3 Implemented unlearning methods

- **GA** — `L = -L_ELBO`. Unbounded; collapses the model.
- **KL-Constrained GA** — `L = -L_ELBO + α · KL(current ‖ frozen ref)`. Stays bounded.
- **NPO-style adaptive weight** — `L = -w(ratio) · L_ELBO`, where the weight goes to zero for already-unlearned samples.
- **VDU L2-anchor** — `L = -L_ELBO + γ · ‖θ - θ₀‖²`.
- **Fisher-Meta** — combines forget GA, Fisher-EWC anchor on retain data, meta-unlearning fine-tune resilience, and SalUn-style saliency masking (~4% of params).
- **Exclusive Unlearning (EU)** — `L = KL(p_θ ‖ uniform) + λ · L_ELBO(retain)`. The forget term is bounded by `log V`.

### 2.4 Analysis methods

- **(a) NLL attribution.** Track per-step NLL deltas on each test subset and correlate with LOO ground truth.
- **(b) Masked-token reconstruction.** Sample 100 sequences per training corpus, mask 15% of tokens with a fixed seed, and compare top-1 / top-5 accuracy before vs. after unlearning. The headline metric is **selectivity** = Δ(target corpus accuracy) − mean Δ(other corpora accuracy). A more negative selectivity means the target corpus dropped more than non-targets.

---

## 3. Results (1): the LOO ground truth has an issue

All 208 LOO influence values are negative (range −1.79 to −0.59, mean −1.15). Cause: `PackedDataset` had `StopIteration` commented out and wraps infinitely, so a 7-corpus LOO model trained for the same 40k steps consumes ~14.3% more samples per unique example than the 8-corpus full model. Every corpus removal therefore *improves* loss.

This adds a constant negative bias to LOO. Pearson correlation is shift-invariant so the relative ranking is still meaningful, but the signal-to-noise ratio is reduced.

---

## 4. Results (2): naive Gradient Ascent collapses the MDM

Applying the original UnTrac recipe (GA, lr=5e-5, batch=1, 1 epoch) to the MDM blows up the loss:

| Metric | Value |
| --- | --- |
| Baseline NLL (toxigen mean) | ~687 |
| After unlearning | ~266,786 (387×) |
| Cross-corpus spread | only 1.29% (noise) |

After step 1 the NLL diverges linearly and the corpus-to-corpus differences disappear.

![NLL trajectory (full + zoom)](figures/untrac_nll_trajectory.png)

![Step-1 influence](figures/untrac_influence_step1.png)

---

## 5. Results (3): four-method comparison (GA / KL / NPO / VDU)

![NLL trajectory comparison (4 methods)](figures/trajectory_comparison.png)

| Method | Final ΔNLL (toxigen) | Behavior |
| --- | --- | --- |
| GA | ~260,000 | linear divergence, total collapse |
| VDU | ~254,000 | almost identical to GA at γ=0.01 |
| KL | ~4,000 | stable convergence |
| NPO | ~5,500 | stable convergence |

![Method comparison: Pearson correlation](figures/method_comparison.png)

| Method | ToxiGen | WinoBias | TruthfulQA | Overall |
| --- | --- | --- | --- | --- |
| ARM UnTrac (original paper) | +0.419 | +0.743 | +0.314 | +0.492 |
| KL-Constrained | +0.543 | +0.508 | +0.421 | +0.491 |
| VDU L2-Anchor | +0.057 | -0.104 | +0.414 | +0.122 |
| Gradient Ascent | -0.323 | -0.169 | +0.413 | -0.026 |
| NPO-Style | -0.324 | -0.170 | +0.411 | -0.028 |

**Key findings.** KL-Constrained GA reaches Overall r = +0.491, on par with ARM (+0.492). NPO collapses to a step-1-equivalent effect because the adaptive weight quickly hits zero. VDU at γ=0.01 is too weak. KL's optimal step is around 10k.

---

## 6. Results (4): Masked-token reconstruction analysis

Goal: ask whether NLL attribution implies actual selective forgetting at the token level. Use KL-Constrained at 10k steps and measure reconstruction accuracy at mask_ratio = 0.15.

![Reconstruction heatmap](figures/reconstruction_analysis.png)

Every cell of the ΔAccuracy matrix is large and negative (−0.33 to −0.50): both target and non-target corpora collapse together.

| Unlearned corpus | Target Δacc | Mean other Δacc | Selectivity |
| --- | --- | --- | --- |
| bookcorpus | -0.502 | -0.367 | -0.135 |
| stackexchange | -0.359 | -0.388 | +0.029 |
| ccnewsv2 | -0.331 | -0.392 | +0.061 |
| gutenberg | -0.418 | -0.374 | -0.044 |
| hackernews | -0.414 | -0.380 | -0.034 |
| openwebtext | -0.361 | -0.388 | +0.027 |
| pilecc | -0.333 | -0.391 | +0.059 |
| wikipedia | -0.358 | -0.388 | +0.030 |

**Mean selectivity: −0.001 (essentially zero).**

Qualitatively, the unlearned model fills every masked position with a meaningless token (`P`, `}`) regardless of the source corpus. KL-Constrained produces a useful NLL attribution signal *not* by selectively forgetting one corpus, but by detecting tiny differential collapse rates inside a globally degraded model.

---

## 7. Discussion

**(7.1) Difficulty of MDM unlearning.** Across five gradient-based methods (GA / KL / NPO / VDU / Fisher-Meta) the corpus-level reconstruction selectivity stays near zero (≈ −0.001 to +0.014) even when the NLL-based attribution score is at parity with the ARM paper.

**(7.2) NLL attribution ≠ knowledge forgetting.**

| Evaluation | Result | Mechanism |
| --- | --- | --- |
| NLL attribution | Effective (r ≈ 0.49) | Detects corpus-to-corpus speed differences inside global degradation |
| Token reconstruction | Fails (selectivity ≈ 0) | All corpora lose 97-99% of reconstruction ability uniformly |

**(7.3) Contrast with image diffusion.** SalUn-style methods on Stable Diffusion erase concepts with FID degradation as small as +0.7. The same family of techniques on an MDM (saliency-masked update of just 4% of params) destroys 97% of reconstruction accuracy. The natural hypothesis is that the U-Net's cross-attention bottleneck localises text-conditioning into ~5-10% of parameters, while a flat bidirectional Transformer has no such bottleneck — but Section 9 below tests and rejects this.

**(7.4) Dynamic attention.** Recent results on attention floating (arXiv:2601.07894, 2025) and attention sinks in diffusion language models (arXiv:2510.15731, 2025) suggest that information routing in MDMs is not anchored to fixed positions. From an unlearning standpoint that is bad: there is no stable surgical target.

**(7.5) Effect of ELBO stochasticity.** ELBO depends on `t` and the mask sample, so gradients are noisy across runs. But KL-Constrained and Fisher-Meta still reach competitive attribution scores, so stochasticity is not the dominant cause of the selective-forgetting failure. The architecture itself is.

**(7.6) Cross-attention MDMs.** GENIE, SeqDiffuSeq, E2D2, and CRoCoDiL do have encoder-decoder + cross-attention structure, providing a candidate "seam." Section 9 tests E2D2 explicitly.

---

## 8. Results (5): Fisher-Meta (Selective Amnesia + SalUn + Meta-Unlearning)

```
L = -L_ELBO(θ; D_forget)                              # forget
  + α · Σ_i F_i^retain · (θ_i - θ_i^0)²              # Fisher-EWC anchor on retain
  + β · ∇_{θ_ft} [-L_ELBO(θ_ft; D_forget)]            # Meta: fine-tune resilience
update mask: saliency(D_forget) > top 30%  AND  Fisher(D_retain) < bottom 70%   # ~4% of params
```

| Method | ToxiGen | WinoBias | TruthfulQA | Overall | Best step |
| --- | --- | --- | --- | --- | --- |
| ARM (paper) | +0.419 | +0.743 | +0.314 | +0.492 | — |
| KL-Constrained | +0.543 | +0.508 | +0.421 | +0.491 | 10k-20k (per dataset) |
| Fisher-Meta | +0.539 | +0.461 | +0.462 | +0.487 | 2k (uniform across datasets) |

Fisher-Meta nearly matches KL on attribution while the optimal step is consistent at 2k for all datasets, removing one tuning knob.

For reconstruction it improves the *retain rate* slightly (3.3% vs 0.2% for KL) but mean selectivity is still +0.014 (essentially zero). Even with a saliency mask that updates only 4% of parameters, the model collapses globally — knowledge is densely entangled across the modified and unmodified parameters.

---

## 9. Results (6): E2D2 Encoder-Decoder MDM with cross-attention

We re-implement an E2D2-style MDM with an 8-layer encoder, a 4-layer decoder with separate Q vs K/V projections (Q from noisy tokens, K/V from `[encoder_output ‖ decoder_hidden]`), and block-causal masking (block size 128). Non-embedding parameter count matches SMDM (113.27M).

![Reconstruction heatmaps (3 conditions)](figures/reconstruction_all_heatmaps.png)
![Selectivity bar chart](figures/selectivity_comparison.png)

| Condition | Base accuracy | Mean ΔAcc | Retain rate | Mean selectivity |
| --- | --- | --- | --- | --- |
| SMDM + KL (10k) | 38.4% | -0.384 | 0.2% | -0.001 |
| SMDM + Fisher-Meta (2k) | 38.4% | -0.372 | 3.3% | +0.014 |
| E2D2 + KL (10k) | 90.3% | -0.896 | 0.8% | +0.001 |

**The cross-attention-bottleneck hypothesis is rejected.** E2D2 has a clear architectural seam yet the corpus-level reconstruction selectivity is essentially zero. The gap to image diffusion is not explained by cross-attention alone — discrete vs continuous representation, distributed semantic encoding, and cross-corpus syntactic overlap likely all contribute.

---

## 10. Results (7): Exclusive Unlearning (EU)

EU (Sasaki, Nakayama, Miyao, Oseki, Isonuma, NLP 2026) reformulates the forget loss to a *bounded* objective:

```
L(θ) = L_forget(θ) + λ · L_retain(θ)

L_forget = KL(p_θ(·|x_forget) ‖ p_uniform)     # bounded above by log V
L_retain = E_{x~D_retain}[ELBO(θ; x)]
```

By construction, `-log p(x)` is unbounded but `KL(· ‖ uniform)` is bounded by `log V`, so no model collapse.

| Method | ToxiGen | WinoBias | TruthfulQA | Overall |
| --- | --- | --- | --- | --- |
| ARM (paper) | +0.419 | +0.743 | +0.314 | +0.492 |
| SMDM KL | +0.543 | +0.508 | +0.421 | +0.491 |
| SMDM EU | +0.505 | +0.368 | +0.572 | +0.481 |

EU's NLL change is only ~120 (vs ~11,594 for KL), so the model is preserved.

![Reconstruction heatmaps (4 conditions)](figures/reconstruction_all_methods.png)
![Selectivity bar chart (all methods)](figures/selectivity_all_methods.png)

| Method | Mean ΔAcc | Retain rate | Mean selectivity |
| --- | --- | --- | --- |
| SMDM KL (10k) | -0.384 | 0.2% | -0.001 |
| SMDM Fisher-Meta (2k) | -0.372 | 3.3% | +0.014 |
| E2D2 KL (10k) | -0.896 | 0.8% | +0.001 |
| **SMDM EU (10k)** | **-0.314** | **18.3%** | **-0.075** |

EU is the first method we tested that produces a meaningfully negative selectivity at the corpus level (-0.075, with bookcorpus reaching -0.347). It also retains 18.3% of base reconstruction accuracy (vs 0.2% for KL). Still, on the absolute scale this is far from "selective forgetting" — non-target corpora drop from ~38% to ~7% as well.

---

## 11. Results (8): Knowledge localization & neuron suppression (fact level)

### 11.1 Motivation

Dai et al. (2022) showed that BERT stores facts in localised MLP neurons. MDMs share BERT's bidirectional masked-LM structure. Does the same localization hold? And if it does, does that resolve the "knowledge is too distributed to remove" hypothesis for the corpus-level failure — or just relocate the failure to a different cause (granularity)?

### 11.2 Corpus-specific trigrams

Sample 500 sequences per training corpus, find token trigrams that appear in *exactly one* corpus, mask the middle token, and ask the trained MDM to fill it in.

| Corpus | Specific trigrams | MDM Top-1 | Examples |
| --- | --- | --- | --- |
| gutenberg | 22,013 | 94.0% | `thou shalt`, `children of Israel` |
| wikipedia | 10,941 | 100.0% | `\nCategory:`, `External links` |
| hackernews | 6,502 | 56.0% | `](https://`, `](http://` |
| bookcorpus | 19,069 | 51.0% | `she didn`, `she was` |

The MDM predicts corpus-specific trigrams with high accuracy, so fact-level analysis is meaningful.

### 11.3 Knowledge neurons

Compute per-neuron L2 gradient norms (of the correct-token logit) at every MLP layer. Note: gradient norms are biased toward shallow layers, which is why Layer 0 always wins below — see the script docstring.

- **Layer 0 dominates** for every corpus (in this gradient-norm proxy).
- **Top-50 neurons (1.6% of 3072)** account for 4-9% of total grad norm — partially localised, not point-localised.
- **Zero overlap** between the top-10 neurons of any two corpora:

```
bookcorpus  top-10: [2410, 2910, 222, 134, 586, 337, 768, 2223, 2270, 2047]
gutenberg   top-10: [688, 1604, 1237, 287, 2760, 1428, 880, 1597, 1006, 599]
hackernews  top-10: [1387, 2045, 2253, 198, 1190, 486, 1097, 1913, 395, 141]
wikipedia   top-10: [3001, 1536, 2622, 2134, 276, 1060, 464, 1941, 2756, 409]

Cross-corpus overlap: 0/10 in every pair
```

### 11.4 Fact-level neuron suppression

Zero out the top-k neurons of the most-important layer (by setting the corresponding rows of `w1` and `w2` to zero), then measure prediction accuracy on each corpus.

| Suppress | k=10 | k=50 | k=100 | k=200 |
| --- | --- | --- | --- | --- |
| bookcorpus selectivity | +0.003 | -0.023 | -0.057 | -0.073 |
| gutenberg selectivity | -0.013 | -0.003 | -0.037 | -0.257 |
| hackernews selectivity | +0.040 | +0.057 | +0.033 | -0.050 |
| wikipedia selectivity | +0.003 | -0.013 | -0.060 | -0.160 |

At k=200 (≈ 6.5% of 3072 neurons):

| Suppress | Target Δ | Other-mean Δ | Selectivity |
| --- | --- | --- | --- |
| bookcorpus | -7.0% | +0.3% | -0.073 |
| gutenberg | -25.0% | +0.7% | -0.257 |
| hackernews | -6.0% | -1.0% | -0.050 |
| wikipedia | -17.0% | -1.0% | -0.160 |

For gutenberg, a 25% drop is concentrated entirely on the target while other corpora are unaffected (+0.7%).

### 11.5 Disentangling granularity from method

Neuron suppression succeeds at fact level; gradient-based optimization fails at corpus level. These differ in *both* granularity and method, so we run a 2×2 cross experiment by applying gradient-based EU at fact level (forget set = 100 corpus-specific trigrams, EU for 200 steps).

| Target | Target Δ | Other-mean Δ | Selectivity |
| --- | --- | --- | --- |
| bookcorpus | -51.0% | -17.0% | -0.340 |
| gutenberg | -95.0% | +1.3% | -0.963 |
| hackernews | -54.0% | -9.3% | -0.447 |
| wikipedia | -100.0% | -10.0% | -0.900 |

### 11.6 The granularity hypothesis is confirmed

| Granularity | Method | Mean selectivity | Effect on other corpora |
| --- | --- | --- | --- |
| fact | gradient-based EU | -0.663 | small or none |
| fact | neuron suppression (k=200) | -0.135 | ~zero |
| corpus | EU (best of corpus level) | -0.075 | large degradation (38% → 7%) |
| corpus | KL | -0.001 | total collapse (38% → 0.1%) |

Gradient-based methods reach -0.663 selectivity at fact level, refuting the alternative "gradient-based methods always collapse." **Granularity, not method, decides whether selective unlearning succeeds.** Knowledge *is* localised per-corpus at the fact level, but a single corpus contributes tens of thousands of facts and removing the union destroys the model.

---

## 12. Conclusions

### 12.1 Big picture

We ran 6 unlearning methods (GA, KL, NPO, VDU, Fisher-Meta, EU) across 2 architectures (single bidirectional Transformer SMDM, encoder-decoder cross-attention MDM E2D2) and an ARM control, plus a fact-level localization experiment.

![Grand summary (with EU)](figures/grand_summary_updated.png)

| Experiment | Result |
| --- | --- |
| SMDM NLL attribution (KL/Fisher-Meta/EU) | r ≈ 0.48-0.49 (parity with ARM paper) |
| E2D2 NLL attribution (KL) | r = 0.26 (uses SMDM LOO; needs E2D2-specific LOO for fair comparison) |
| SMDM token reconstruction (KL, 10k) | selectivity -0.001 (uniform collapse, retain 0.2%) |
| SMDM token reconstruction (Fisher-Meta, 2k) | selectivity +0.014 (retain 3.3%) |
| E2D2 token reconstruction (KL, 10k) | selectivity +0.001 (collapse despite cross-attention, retain 0.8%) |
| SMDM token reconstruction (EU, 10k) | selectivity -0.075 (best so far, retain 18.3%) |
| ARM token reconstruction (EU, 10k) | selectivity -0.073 (≈ MDM EU, retain 19.6%) |
| ARM NLL attribution (KL) | r = 0.191 (uses SMDM LOO, mismatched) |
| ARM NLL attribution (EU) | r = 0.609 (highest of all methods, with SMDM LOO) |
| Fact-level gradient-based EU (200 steps) | selectivity -0.663 (target-only large drop) |
| Fact-level neuron suppression (k=200) | selectivity -0.135 (other corpora barely affected) |
| Image diffusion (SalUn etc.) | FID +0.7 (essentially no degradation) for selective concept removal |

### 12.2 One-sentence claim

> UnTrac's NLL attribution works on both ARM and MDM at parity with the original paper, but token-level reconstruction analysis shows that what it captures is *differential collapse detection*, not selective forgetting. Whether selective unlearning succeeds is governed **not by architecture but by the granularity of the forget set** — fact level (~100) succeeds, corpus level (~40k) fails for ARM and MDM alike.

### 12.3 Attribution granularity vs practical-unlearning granularity

UnTrac is *attribution* — its evaluation unit is fixed at the corpus level because LOO requires "what was excluded" to be well-defined. Real unlearning needs are at finer granularities:

| Use case | Real granularity |
| --- | --- |
| GDPR right-to-be-forgotten | a specific user's data (1-1000s of items) |
| Harmful response prevention (jailbreak defense) | specific harmful patterns (100s-1000s) |
| Copyright / IP | specific books/articles/code (per-document) |
| PII removal | specific names, addresses, ... |
| Bias mitigation | specific bias patterns |

"Forget an entire corpus" is essentially never a real request. Corpus-level unlearning is an attribution-research construct.

The reframed contribution is therefore:

1. **Practical value** — at the granularity practical needs require, MDMs *do* support selective unlearning (selectivity -0.663 with EU).
2. **Theoretical value** — the implicit equation "NLL attribution = selective forgetting" is wrong, as shown by reconstruction analysis.
3. **Methodological value** — attribution granularity and unlearning granularity must not be conflated when designing future studies.

### 12.4 Key findings

1. **NLL-based attribution works on MDMs.** KL-Constrained (r=0.491), Fisher-Meta (r=0.487), EU (r=0.481) all match ARM (r=0.492).
2. **NLL attribution and knowledge forgetting are different mechanisms.** Reconstruction analysis shows that NLL attribution succeeds because of differential collapse rates, not selective knowledge removal.
3. **At practical granularity (facts/patterns), MDMs do support selective unlearning.** Gradient-based EU on 100 trigrams reaches selectivity -0.663 (gutenberg: target -95%, others +1.3%); neuron suppression k=200 reaches -0.135.
4. **At corpus level (the UnTrac evaluation unit), no architecture × method combination succeeds.** Best is EU at -0.075. This is a limit of attribution research devices, not a practical concern.
5. **The "ARM vs MDM architecture" hypothesis is rejected.** ARM with the same data, scale and EU method reaches selectivity -0.073, essentially identical to MDM EU's -0.075.
6. **Cross-attention bottleneck is not sufficient.** E2D2's encoder-decoder + Q/KV split still gets selectivity ≈ +0.001.
7. **Granularity hypothesis confirmed.** 2 methods × 2 granularities cross-comparison: fact level always succeeds, corpus level always fails.
8. **First text-diffusion unlearning study.** 3 architectures × 6 methods × 2 evaluation axes (NLL attribution + token reconstruction) + fact-level localization + ARM control. Not previously done.

### 12.5 Scope and caveats

These conclusions hold for the specific setup studied. Stronger universal claims would require:

- **Scale.** Only 113M parameters tested. Larger models may carry more knowledge redundancy, possibly improving corpus-level selectivity.
- **Training data.** ~328M tokens / 40k steps is small. Longer training may change knowledge localization.
- **Discrete vs continuous.** Continuous-latent diffusion LMs (CDCD, Plaid, ...) may behave more like image diffusion.
- **Method space.** Six methods is broad but not exhaustive.

---

## 13. Future directions

1. **Practical-granularity benchmarks** (top priority): GDPR, jailbreak defense, copyright, TOFU/MUSE.
2. **Bridging attribution and unlearning granularity**: scan 100/500/1k/5k/10k forget-set sizes; quantify the phase transition.
3. **Fix the LOO ground truth**: train LOO models for `40000 × 7/8 = 35000` steps to remove the epoch imbalance, and produce an E2D2-specific LOO baseline.
4. **Inference-time interventions**: activation patching, representation engineering — modify computation rather than parameters.
5. **Low-rank subspace unlearning** (LoRA-style): restrict updates to a low-rank subspace.
6. **Continuous diffusion language models** (CDCD, Plaid): does the discrete-vs-continuous distinction matter?

---

## Appendix

### A. Hardware and software

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Max-Q × 7 (97 GB each) |
| PyTorch | 2.11.0+cu128 |
| Flash-Attention | 2.8.3 |

### B. Implementation fixes (relative to upstream SMDM code)

1. `PackedDataset` infinite-loop bug: `StopIteration` was commented out — added a step cap in evaluation.
2. CPU-only fallback bug in the original training script.
3. Six unlearning methods exposed via `--unlearn_method` (ga, kl, npo, vdu, fisher_meta, eu).
4. Reconstruction analysis scripts (KL / Fisher-Meta / EU).
5. E2D2 architecture (encoder-decoder + Q/KV split + block-causal masking).
6. E2D2 train + attribution + reconstruction scripts.

### C. File index (this repository)

| File | Purpose |
| --- | --- |
| `src/mdm_unlearning/models/diffmodel.py` | SMDM single bidirectional Transformer |
| `src/mdm_unlearning/models/enc_dec_diffmodel.py` | E2D2 encoder-decoder MDM (block-causal) |
| `src/mdm_unlearning/models/arm.py` | Autoregressive (GPT/LLaMA-style) baseline |
| `src/mdm_unlearning/train/train_mdm.py` | SMDM UnTrac training |
| `src/mdm_unlearning/train/train_e2d2.py` | E2D2 UnTrac training |
| `src/mdm_unlearning/train/train_ar.py` | ARM UnTrac training |
| `src/mdm_unlearning/evaluate/untrac_mdm.py` | SMDM UnTrac attribution (GA/KL/NPO/VDU/Fisher-Meta/EU) |
| `src/mdm_unlearning/evaluate/untrac_e2d2.py` | E2D2 UnTrac attribution |
| `src/mdm_unlearning/evaluate/untrac_ar.py` | ARM UnTrac attribution |
| `src/mdm_unlearning/evaluate/reconstruction_mdm.py` | SMDM reconstruction analysis (KL / Fisher-Meta / EU) |
| `src/mdm_unlearning/evaluate/reconstruction_e2d2.py` | E2D2 reconstruction analysis |
| `src/mdm_unlearning/evaluate/reconstruction_ar.py` | ARM reconstruction analysis |
| `src/mdm_unlearning/analysis/extract_corpus_trigrams.py` | corpus-specific trigram extraction + prediction check |
| `src/mdm_unlearning/analysis/knowledge_localization.py` | knowledge-neuron localization (Dai et al. style) |
| `src/mdm_unlearning/analysis/neuron_suppression.py` | neuron-suppression fact-level unlearning |
| `src/mdm_unlearning/analysis/fact_level_eu.py` | gradient-based EU at fact level (granularity test) |
| `results/ground_truth/loo_attribution.json` | LOO ground truth (SMDM) |
| `results/attribution/compare/` | SMDM GA / KL / NPO / VDU four-method comparison (32 JSON) |
| `results/attribution/fisher_meta/` | SMDM Fisher-Meta attribution (8 JSON) |
| `results/attribution/eu/` | SMDM EU attribution (8 JSON) |
| `results/attribution/e2d2/` | E2D2 KL attribution (8 JSON) |
| `results/attribution/ar/{kl,eu}/` | ARM KL / EU attribution (8 JSON each) |
| `results/attribution/mdm_kl/` | initial SMDM KL attribution with per-step trajectories |
| `results/reconstruction/mdm_kl/` | SMDM KL reconstruction analysis (8 JSON) |
| `results/reconstruction/mdm_fm/` | SMDM Fisher-Meta reconstruction (8 JSON) |
| `results/reconstruction/mdm_eu/` | SMDM EU reconstruction (8 JSON) |
| `results/reconstruction/e2d2/` | E2D2 KL reconstruction (8 JSON) |
| `results/reconstruction/ar/` | ARM reconstruction (8 JSON) |
| `results/analysis/corpus_specific_trigrams.json` | extracted corpus-specific trigrams |
| `results/analysis/trigram_prediction_results.json` | prediction accuracy on corpus-specific trigrams |
| `results/analysis/knowledge_localization.json` | per-layer per-neuron knowledge importance |
| `results/analysis/neuron_suppression_results.json` | neuron-suppression sweep results |

### D. References

**UnTrac and attribution**
- Misonuma & Titov (2024). *Unlearning Traces the Influential Training Data of Language Models.* ACL 2024.

**Masked diffusion language models**
- Nie et al. (2024). *Scaling up Masked Diffusion Models on Text.* arXiv:2410.18514.
- Sahoo et al. (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS 2024.
- Lou et al. (2024). *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution.* ICML 2024 Best Paper.
- Nie et al. (2025). *Large Language Diffusion Models.* arXiv:2502.09992.

**Cross-attention MDMs**
- Lin et al. (2023). *Text Generation with Diffusion Language Models: A Pre-training Approach with Continuous Paragraph Denoise (GENIE).* ICML 2023.
- Arriola et al. (2025). *Encoder-Decoder Diffusion Language Models (E2D2).* NeurIPS 2025.
- arXiv:2603.20210. *CRoCoDiL: Continuous and Robust Conditioned Diffusion for Language.* 2026.

**Diffusion model unlearning (image)**
- George et al. (2025). *The Illusion of Unlearning.* CVPR 2025.
- Gao et al. (2025). *Meta-Unlearning on Diffusion Models.* ICCV 2025.
- Heng & Soh (2023). *Selective Amnesia: A Continual Learning Approach to Forgetting in Deep Generative Models.* NeurIPS 2023.
- Fan et al. (2024). *SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency.* ICLR 2024.
- Gandikota et al. (2023). *Erasing Concepts from Diffusion Models (ESD).* ICCV 2023.
- Gandikota et al. (2024). *Unified Concept Editing in Diffusion Models (UCE).* CVPR 2024.

**LLM unlearning**
- Sasaki, Nakayama, Miyao, Oseki, Isonuma (2026). *Exclusive Unlearning.* NLP2026 (言語処理学会 第32回年次大会) B2-16.
- Zhang et al. (2024). *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning.* arXiv:2404.05868.
- Li et al. (2024). *MUSE: Machine Unlearning Six-Way Evaluation.* ICML 2024.

**Knowledge localization and attention structure**
- Dai et al. (2022). *Knowledge Neurons in Pretrained Transformers.* ACL 2022.
- Niu et al. (2024). *What does the Knowledge Neuron Thesis Have to do with Knowledge?* ICLR 2024.
- Meng et al. (2022). *Locating and Editing Factual Associations in GPT (ROME).* NeurIPS 2022.
- Hase et al. (2023). *Does Localization Inform Editing?* NeurIPS 2023.
- arXiv:2601.07894. *Revealing the Attention Floating Mechanism in Masked Diffusion Models.* 2025.
- arXiv:2510.15731. *Attention Sinks in Diffusion Language Models.* 2025.

**Conditioning techniques**
- Peebles & Xie (2023). *Scalable Diffusion Models with Transformers (DiT).* ICCV 2023.
