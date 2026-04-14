"""Neuron suppression experiment (Phase 2a).

For each target corpus, identify the top-k MLP neurons most important for
predicting corpus-specific trigrams (per Phase 1), zero out the
corresponding rows of ``w1`` and ``w2`` in the SwiGLU layer, and measure
the change in next-token prediction accuracy on the target corpus and on
all other corpora.

This experiment quantifies *fact-level* selective unlearning. The
selectivity metric is::

    selectivity = target_delta_accuracy - mean(other_corpora_delta_accuracy)

A more negative selectivity indicates that the suppression preferentially
hurt the target corpus while leaving other corpora largely intact.

The companion experiment ``fact_level_eu`` repeats the comparison with
gradient-based EU unlearning (instead of direct neuron zeroing) at the
same fact-level granularity, to disentangle "method" from "granularity"
as the cause of selective forgetting success or failure.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from mdm_unlearning.analysis.extract_corpus_trigrams import (
    CORPORA,
    MASK_ID,
    TOKENIZER_NAME,
    load_model,
)
from mdm_unlearning.analysis.knowledge_localization import collect_test_cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=str, default="data/untrac")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_size", type=int, default=113)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--trigrams_path", type=str,
                        default="results/analysis/corpus_specific_trigrams.json")
    parser.add_argument("--localization_path", type=str,
                        default="results/analysis/knowledge_localization.json")
    parser.add_argument("--corpora", type=str, nargs="+",
                        default=["bookcorpus", "gutenberg", "hackernews", "wikipedia"])
    parser.add_argument("--max_test_cases", type=int, default=100)
    parser.add_argument("--suppress_k_values", type=int, nargs="+",
                        default=[10, 50, 100, 200],
                        help="Number of top neurons to suppress per experiment.")
    parser.add_argument("--n_localization_cases", type=int, default=50,
                        help="Number of trigrams used to recompute neuron importance.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str,
                        default="results/analysis/neuron_suppression_results.json")
    return parser.parse_args()


def evaluate_accuracy(
    model,
    cases: list[tuple[list[int], int, int]],
    device: str,
    batch_size: int = 16,
) -> float:
    """Top-1 accuracy on the given (sequence, mask_position, target) cases."""
    if not cases:
        return 0.0
    top1 = 0
    total = len(cases)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_seqs: list[list[int]] = []
        positions: list[int] = []
        targets: list[int] = []
        for seq, pos, tgt in cases[start:end]:
            masked = list(seq)
            masked[pos] = MASK_ID
            batch_seqs.append(masked)
            positions.append(pos)
            targets.append(tgt)
        input_ids = torch.tensor(batch_seqs, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        for k, (pos, tgt) in enumerate(zip(positions, targets, strict=True)):
            if logits[k, pos].argmax().item() == tgt:
                top1 += 1
    return top1 / total


def recompute_neuron_importance(
    model,
    cases: list[tuple[list[int], int, int]],
    layer_idx: int,
    device: str,
    max_cases: int = 50,
) -> np.ndarray:
    """Re-compute per-neuron gradient norms at a single layer for one corpus."""
    n_neurons = model.config.intermediate_size
    importance = np.zeros(n_neurons, dtype=np.float32)
    count = 0
    for seq, pos, tgt in cases[:max_cases]:
        masked = list(seq)
        masked[pos] = MASK_ID
        input_ids = torch.tensor([masked], device=device)
        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        if logits[0, pos].argmax().item() != tgt:
            continue
        logits[0, pos, tgt].backward()
        w3 = model.transformer.h[layer_idx].mlp.swiglu.w3
        if w3.weight.grad is not None:
            importance += w3.weight.grad.float().norm(dim=0).cpu().numpy()
            count += 1
        model.zero_grad()
    if count > 0:
        importance /= count
    return importance


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading model and supporting JSON...")
    _ = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = load_model(args.model_size, args.ckpt_path, args.seq_len, args.device)

    with open(args.trigrams_path) as f:
        trigram_data = json.load(f)
    with open(args.localization_path) as f:
        localization = json.load(f)

    # Pre-compute test cases for all corpora
    print("\nPreparing test cases...")
    test_cases: dict[str, list[tuple[list[int], int, int]]] = {}
    for corpus in args.corpora:
        good = [(tuple(t), c) for t, c in trigram_data[corpus][:50]]
        target_set = {tri for tri, _ in good}
        cases = collect_test_cases(
            args.data_dir, corpus, target_set, args.seq_len, args.max_test_cases
        )
        test_cases[corpus] = cases
        print(f"  {corpus}: {len(cases)} cases")

    # Baseline (no suppression)
    print("\n=== Baseline accuracy (no suppression) ===")
    baseline: dict[str, float] = {}
    for corpus in args.corpora:
        acc = evaluate_accuracy(model, test_cases[corpus], args.device)
        baseline[corpus] = acc
        print(f"  {corpus:<15s}: {100 * acc:.1f}%")

    # Per-k suppression
    print("\n=== Neuron Suppression Experiments ===")
    results: dict[int, dict] = {}

    for k in args.suppress_k_values:
        print(f"\n--- Suppressing top {k} neurons ---")
        results[k] = {}

        for target_corpus in args.corpora:
            best_layer = localization[target_corpus]["best_layer"]
            mlp = model.transformer.h[best_layer].mlp.swiglu

            # Save original weights
            w1_orig = mlp.w1.weight.data.clone()
            w2_orig = mlp.w2.weight.data.clone()

            # Recompute neuron importance for this corpus
            importance = recompute_neuron_importance(
                model,
                test_cases[target_corpus],
                best_layer,
                args.device,
                args.n_localization_cases,
            )
            top_neurons = np.argsort(importance)[::-1][:k].copy()

            # Suppress
            mlp.w1.weight.data[top_neurons] = 0
            mlp.w2.weight.data[top_neurons] = 0

            # Evaluate on all corpora
            row: dict[str, dict[str, float]] = {}
            for eval_corpus in args.corpora:
                acc = evaluate_accuracy(model, test_cases[eval_corpus], args.device)
                row[eval_corpus] = {
                    "acc": round(acc, 4),
                    "delta": round(acc - baseline[eval_corpus], 4),
                }

            # Restore
            mlp.w1.weight.data.copy_(w1_orig)
            mlp.w2.weight.data.copy_(w2_orig)

            target_delta = row[target_corpus]["delta"]
            others_delta = float(
                np.mean(
                    [row[c]["delta"] for c in args.corpora if c != target_corpus]
                )
            )
            selectivity = target_delta - others_delta

            print(f"  Suppress {target_corpus} neurons (layer {best_layer}):")
            for c in args.corpora:
                marker = " <<<" if c == target_corpus else ""
                print(
                    f"    {c}: {100 * row[c]['acc']:.1f}% "
                    f"(delta={100 * row[c]['delta']:+.1f}%){marker}"
                )
            print(f"    Selectivity: {selectivity:+.4f}")

            results[k][target_corpus] = {
                "suppressed_layer": best_layer,
                "n_suppressed": k,
                "accuracies": row,
                "selectivity": round(selectivity, 4),
            }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: selectivity by number of suppressed neurons")
    print("=" * 70)
    print(f"  {'Target':<15s}", end="")
    for k in args.suppress_k_values:
        print(f" {'k=' + str(k):>10s}", end="")
    print()
    for corpus in args.corpora:
        print(f"  {corpus:<15s}", end="")
        for k in args.suppress_k_values:
            sel = results.get(k, {}).get(corpus, {}).get("selectivity", None)
            if sel is None:
                print(f" {'N/A':>10s}", end="")
            else:
                print(f" {sel:>+10.4f}", end="")
        print()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
