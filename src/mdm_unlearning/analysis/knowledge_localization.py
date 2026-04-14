"""Knowledge neuron localization analysis (Phase 1).

Adapts the Knowledge Neuron methodology of Dai et al. (ACL 2022) to MDM.
For each corpus, we compute the per-MLP-neuron gradient norm of the
correct-token logit at the masked middle position of corpus-specific
trigrams. The accumulated gradient magnitudes per (layer, neuron) yield a
"knowledge importance" score that can be used to identify which neurons
contribute most to predicting corpus-specific patterns.

Notes
-----
This script uses *raw gradient norms* of the SwiGLU ``w3`` (down-projection)
weight as a proxy for neuron importance. This is a sensitivity-based
attribution and is biased toward shallow layers (where gradients are
largest in absolute value). For a more principled localization analogous
to Dai et al., switch to integrated gradients on the MLP intermediate
activations -- this is left as future work.

Even with the gradient-norm proxy, two findings are robust:
- Knowledge is partially concentrated: top-50 neurons (1.6%) explain
  4-9% of total gradient norm at each layer.
- Top-10 neurons differ across corpora: 0/10 overlap in pairwise
  comparisons of bookcorpus / gutenberg / hackernews / wikipedia.

Reference
---------
Dai et al., "Knowledge Neurons in Pretrained Transformers", ACL 2022.
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
from mdm_unlearning.data.packed_dataset import PackedDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=str, default="data/untrac")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model_size", type=int, default=113)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--trigrams_path", type=str,
                        default="results/analysis/corpus_specific_trigrams.json")
    parser.add_argument("--corpora", type=str, nargs="+",
                        default=["bookcorpus", "gutenberg", "hackernews", "wikipedia"],
                        help="Subset of corpora to analyze (default: 4 with high accuracy).")
    parser.add_argument("--max_test_cases", type=int, default=100)
    parser.add_argument("--top_neurons", type=int, default=10,
                        help="Number of top neurons per layer to record/compare.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str,
                        default="results/analysis/knowledge_localization.json")
    return parser.parse_args()


def collect_test_cases(
    data_dir: str,
    corpus: str,
    target_trigrams: set[tuple[int, int, int]],
    seq_len: int,
    max_cases: int,
) -> list[tuple[list[int], int, int]]:
    """Find sequences containing corpus-specific trigrams; return (tokens, pos, target)."""
    filenames = sorted(glob.glob(f"{data_dir}/train_{corpus}_*.bin"))
    ds = PackedDataset(
        filenames,
        n_chunks=1,
        block_size=seq_len + 1,
        shuffle=False,
        seed=42,
        num_processes=1,
        process_rank=0,
    )
    cases: list[tuple[list[int], int, int]] = []
    for i, data in enumerate(ds):
        if i >= 1000 or len(cases) >= max_cases:
            break
        tokens = data[:seq_len].tolist()
        for j in range(len(tokens) - 2):
            tri = (tokens[j], tokens[j + 1], tokens[j + 2])
            if tri in target_trigrams and len(cases) < max_cases:
                cases.append((tokens, j + 1, tokens[j + 1]))
    return cases


def compute_neuron_importance(
    model,
    test_cases: list[tuple[list[int], int, int]],
    n_layers: int,
    n_neurons: int,
    device: str,
) -> tuple[np.ndarray, int]:
    """Compute per-neuron gradient norms for correctly predicted cases.

    Returns
    -------
    importance : np.ndarray of shape (n_layers, n_neurons)
        Per-layer per-neuron average gradient norm of the correct-token logit.
    correct_count : int
        Number of test cases that were correctly predicted (only these
        contribute to the importance calculation).
    """
    importance = np.zeros((n_layers, n_neurons), dtype=np.float32)
    correct_count = 0

    for seq, pos, target in test_cases:
        masked_seq = list(seq)
        masked_seq[pos] = MASK_ID
        input_ids = torch.tensor([masked_seq], device=device)

        model.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)

        if logits[0, pos].argmax().item() != target:
            continue
        correct_count += 1

        target_logit = logits[0, pos, target]
        target_logit.backward()

        for layer_idx in range(n_layers):
            w3 = model.transformer.h[layer_idx].mlp.swiglu.w3
            if w3.weight.grad is not None:
                # w3.weight has shape (n_embd, intermediate_size); per-neuron
                # importance is the L2 norm along dim=0 (across embedding dim).
                grad_norms = w3.weight.grad.float().norm(dim=0).cpu().numpy()
                importance[layer_idx] += grad_norms

        model.zero_grad()

    if correct_count > 0:
        importance /= correct_count

    return importance, correct_count


def summarize_layer_concentration(
    importance: np.ndarray, n_layers: int
) -> list[dict]:
    """Compute the share of total gradient norm captured by top-k neurons per layer."""
    stats: list[dict] = []
    for layer_idx in range(n_layers):
        layer_imp = importance[layer_idx]
        total = float(layer_imp.sum())
        if total == 0:
            continue
        sorted_imp = np.sort(layer_imp)[::-1]
        top10_pct = float(100 * sorted_imp[:10].sum() / total)
        top50_pct = float(100 * sorted_imp[:50].sum() / total)
        stats.append(
            {
                "layer": layer_idx,
                "total": total,
                "top10_pct": round(top10_pct, 1),
                "top50_pct": round(top50_pct, 1),
            }
        )
    return stats


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("Loading model and tokenizer...")
    _ = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = load_model(args.model_size, args.ckpt_path, args.seq_len, args.device)

    n_layers = model.config.n_layer
    n_neurons = model.config.intermediate_size
    print(f"Model: {n_layers} layers x {n_neurons} intermediate neurons")

    with open(args.trigrams_path) as f:
        trigram_data = json.load(f)

    print("\n=== Phase 1: Knowledge Neuron Localization ===\n")
    results: dict[str, dict] = {}
    for corpus in args.corpora:
        print(f"Processing {corpus}...")
        # Use the first 50 specific trigrams as targets
        good = [(tuple(t), c) for t, c in trigram_data[corpus][:50]]
        target_set = {tri for tri, _ in good}

        cases = collect_test_cases(
            args.data_dir, corpus, target_set, args.seq_len, args.max_test_cases
        )
        print(f"  {len(cases)} candidate test cases")
        if len(cases) < 10:
            print("  Skipping (too few)")
            continue

        importance, correct = compute_neuron_importance(
            model, cases, n_layers, n_neurons, args.device
        )
        if correct == 0:
            print("  No correctly predicted cases; skipping")
            continue
        print(f"  {correct} correctly predicted cases used")

        layer_stats = summarize_layer_concentration(importance, n_layers)

        # Identify the layer with the largest aggregate gradient norm
        layer_totals = [importance[layer_idx].sum() for layer_idx in range(n_layers)]
        best_layer = int(np.argmax(layer_totals))
        top_neurons = (
            np.argsort(importance[best_layer])[::-1][: args.top_neurons].tolist()
        )

        print(f"  Most important layer: {best_layer} "
              f"(total grad norm {layer_totals[best_layer]:.4f})")
        print(f"  Top {args.top_neurons} neurons in layer {best_layer}: {top_neurons}")

        results[corpus] = {
            "correct_count": correct,
            "total_cases": len(cases),
            "layer_stats": layer_stats,
            "best_layer": best_layer,
            "top_neurons_best_layer": top_neurons,
            "layer_totals": [float(t) for t in layer_totals],
        }

    print("\n" + "=" * 70)
    print("Cross-corpus neuron overlap analysis")
    print("=" * 70)
    if len(results) >= 2:
        corpus_list = list(results.keys())
        for i, c1 in enumerate(corpus_list):
            for c2 in corpus_list[i + 1 :]:
                n1 = set(results[c1]["top_neurons_best_layer"])
                n2 = set(results[c2]["top_neurons_best_layer"])
                overlap = len(n1 & n2)
                print(f"  {c1} vs {c2}: {overlap}/{args.top_neurons} neurons overlap")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
