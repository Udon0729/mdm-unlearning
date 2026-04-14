"""Fact-level Exclusive Unlearning experiment (Phase 2b).

This experiment disambiguates *granularity* from *method* as the cause of
selective unlearning success or failure. The neuron suppression baseline
(``neuron_suppression``) succeeds at fact-level unlearning, but it differs
from the corpus-level UnTrac experiments in *both* granularity (100
trigrams vs 40 000 sequences) and *method* (direct neuron zeroing vs
gradient-based optimization).

Here we apply the same gradient-based EU method used in the corpus-level
experiments, but at fact-level granularity (~100 trigrams). If selective
unlearning succeeds in this controlled setting, then the bottleneck for
the corpus-level case is granularity, not method. The original experiment
(see paper Table) confirmed this hypothesis with mean selectivity of
about -0.66, compared with ~0 at corpus level.

This is the cleaned-up version of the experimental script previously kept
in ``/tmp/phase2b_gradient_fact.py``. The improvements are:

- ``argparse`` instead of hardcoded paths and constants.
- JSON output instead of stdout-only (the original kept the numbers in
  console scrollback).
- Corpus list, EU lambda, learning rate, and step count are configurable.
- Proper docstrings and a clean main entry point.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from mdm_unlearning.analysis.extract_corpus_trigrams import (
    CORPORA,
    MASK_ID,
    TOKENIZER_NAME,
    load_model,
)
from mdm_unlearning.analysis.knowledge_localization import collect_test_cases
from mdm_unlearning.analysis.neuron_suppression import evaluate_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=str, default="data/untrac")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_size", type=int, default=113)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--trigrams_path", type=str,
                        default="results/analysis/corpus_specific_trigrams.json")
    parser.add_argument("--corpora", type=str, nargs="+",
                        default=["bookcorpus", "gutenberg", "hackernews", "wikipedia"])
    parser.add_argument("--max_test_cases", type=int, default=100)
    parser.add_argument("--retain_per_corpus", type=int, default=30,
                        help="Retain sequences sampled per non-target corpus.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--eu_lambda", type=float, default=1.0,
                        help="Weight for the retain loss term.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str,
                        default="results/analysis/fact_level_eu_results.json")
    return parser.parse_args()


def forward_process(batch: torch.Tensor, total_dim: int = 32000, eps: float = 1e-3):
    """SMDM-style mask scheduling for the MDM ELBO."""
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


def run_eu_unlearning(
    model,
    forget_seqs: list[torch.Tensor],
    retain_seqs: list[torch.Tensor],
    n_steps: int,
    lr: float,
    eu_lambda: float,
    device: str,
) -> None:
    """Apply Exclusive Unlearning at fact level for ``n_steps`` steps."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    vocab_size = model.config.padded_vocab_size
    uniform_log_p = -math.log(vocab_size)
    uniform = torch.full(
        (vocab_size,), math.exp(uniform_log_p), device=device
    )

    for step in range(n_steps):
        # Forget pass: drive masked-position predictions toward uniform
        f_seq = forget_seqs[step % len(forget_seqs)].unsqueeze(0).to(device)
        f_noisy, f_mask, _f_pm = forward_process(f_seq)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            f_logits = model(f_noisy)
        forget_loss = F.kl_div(
            F.log_softmax(f_logits[f_mask], dim=-1),
            uniform.unsqueeze(0).expand(f_mask.sum(), -1),
            reduction="batchmean",
        )

        # Retain pass: standard ELBO on a sequence from another corpus
        r_seq = retain_seqs[step % len(retain_seqs)].unsqueeze(0).to(device)
        r_noisy, r_mask, r_pm = forward_process(r_seq)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            r_logits = model(r_noisy)
        retain_ce = F.cross_entropy(
            r_logits[r_mask], r_seq[0][r_mask], reduction="none"
        ) / r_pm[r_mask]
        retain_loss = retain_ce.sum() / r_seq.shape[1]

        total_loss = forget_loss + eu_lambda * retain_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading shared resources...")
    _ = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    with open(args.trigrams_path) as f:
        trigram_data = json.load(f)

    print("Preparing test cases...")
    all_test_cases: dict[str, list[tuple[list[int], int, int]]] = {}
    for corpus in args.corpora:
        good = [(tuple(t), c) for t, c in trigram_data[corpus][:50]]
        target_set = {tri for tri, _ in good}
        cases = collect_test_cases(
            args.data_dir, corpus, target_set, args.seq_len, args.max_test_cases
        )
        all_test_cases[corpus] = cases
        print(f"  {corpus}: {len(cases)} cases")

    print("\n" + "=" * 70)
    print("Experiment B: Gradient-based EU at fact level")
    print("=" * 70)

    results: dict[str, dict] = {}

    for target_corpus in args.corpora:
        print(f"\n--- Target: {target_corpus} ---")

        # Reload fresh model for each target so previous unlearning runs
        # don't bleed into the next experiment.
        model = load_model(
            args.model_size, args.ckpt_path, args.seq_len, args.device
        )

        model.eval()
        baseline = {
            c: evaluate_accuracy(model, all_test_cases[c], args.device)
            for c in args.corpora
        }
        baseline_str = ", ".join(f"{c}={100 * baseline[c]:.0f}%" for c in args.corpora)
        print(f"  Baseline: {baseline_str}")

        # Forget set: sequences containing the target corpus's specific trigrams
        forget_seqs = [
            torch.tensor(seq) for seq, _, _ in all_test_cases[target_corpus]
        ]

        # Retain set: a few sequences from each non-target corpus
        retain_seqs: list[torch.Tensor] = []
        for c in args.corpora:
            if c == target_corpus:
                continue
            for seq, _, _ in all_test_cases[c][: args.retain_per_corpus]:
                retain_seqs.append(torch.tensor(seq))

        run_eu_unlearning(
            model,
            forget_seqs,
            retain_seqs,
            n_steps=args.steps,
            lr=args.lr,
            eu_lambda=args.eu_lambda,
            device=args.device,
        )

        model.eval()
        after = {
            c: evaluate_accuracy(model, all_test_cases[c], args.device)
            for c in args.corpora
        }

        print(f"  After EU ({args.steps} steps):")
        for c in args.corpora:
            delta = after[c] - baseline[c]
            marker = " <<<" if c == target_corpus else ""
            print(
                f"    {c}: {100 * after[c]:.0f}% "
                f"(delta={100 * delta:+.1f}%){marker}"
            )

        target_delta = after[target_corpus] - baseline[target_corpus]
        others_delta = float(
            np.mean([after[c] - baseline[c] for c in args.corpora if c != target_corpus])
        )
        selectivity = target_delta - others_delta
        print(f"    Selectivity: {selectivity:+.4f}")

        results[target_corpus] = {
            "baseline": {c: round(baseline[c], 4) for c in args.corpora},
            "after": {c: round(after[c], 4) for c in args.corpora},
            "delta": {
                c: round(after[c] - baseline[c], 4) for c in args.corpora
            },
            "selectivity": round(selectivity, 4),
            "steps": args.steps,
            "eu_lambda": args.eu_lambda,
        }

        del model
        torch.cuda.empty_cache()

    # Aggregate
    mean_sel = float(np.mean([r["selectivity"] for r in results.values()]))
    summary = {
        "args": vars(args),
        "per_corpus": results,
        "mean_selectivity": round(mean_sel, 4),
    }
    print(f"\nMean selectivity across {len(results)} corpora: {mean_sel:+.4f}")

    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
