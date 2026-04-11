"""Extract corpus-specific token trigrams and verify MDM prediction accuracy.

Phase 0 of the knowledge localization analysis. The pipeline has two stages:

1. **Trigram extraction.** Sample N sequences from each of the 8 training
   corpora, decompose them into all overlapping token trigrams, and identify
   trigrams that appear in *exactly one* corpus with a minimum support
   frequency. These are the "corpus-specific" trigrams used as the
   fact-level test set in subsequent phases.

2. **Prediction verification.** For each corpus, take the top-K
   corpus-specific trigrams whose middle token decodes to a meaningful
   alphabetic word, find sequences containing them, mask the middle token,
   and ask the trained MDM to fill in the blank. Compute top-1 / top-5
   accuracy.

The output of stage 1 (a JSON dictionary mapping each corpus to its top
specific trigrams) is consumed by ``localization``, ``suppression``, and
``fact_level_eu`` analysis scripts.

Example
-------
::

    python -m mdm_unlearning.analysis.extract_corpus_trigrams \\
        --data_dir data/untrac \\
        --ckpt_path workdir/.../iter-040000-ckpt.pth \\
        --num_samples 500 \\
        --min_count 3 \\
        --top_k 200 \\
        --output_trigrams results/analysis/corpus_specific_trigrams.json \\
        --output_prediction results/analysis/trigram_prediction_results.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from mdm_unlearning.data.packed_dataset import PackedDataset
from mdm_unlearning.models.config import Config
from mdm_unlearning.models.diffmodel import TransEncoder

CORPORA: list[str] = [
    "bookcorpus",
    "stackexchange",
    "ccnewsv2",
    "gutenberg",
    "hackernews",
    "openwebtext",
    "pilecc",
    "wikipedia",
]
MASK_ID: int = 32000
TOKENIZER_NAME: str = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data_dir", type=str, default="data/untrac",
                        help="Directory containing train_<corpus>_*.bin files.")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the trained MDM checkpoint (.pth).")
    parser.add_argument("--model_size", type=int, default=113,
                        help="Non-embedding parameter count in millions.")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=500,
                        help="Number of sequences to sample per corpus for "
                             "trigram extraction.")
    parser.add_argument("--min_count", type=int, default=3,
                        help="Minimum trigram occurrences within a corpus to "
                             "qualify as 'specific'.")
    parser.add_argument("--top_k", type=int, default=200,
                        help="Number of top-frequency trigrams to keep per "
                             "corpus in the output JSON.")
    parser.add_argument("--prediction_top_k", type=int, default=50,
                        help="Number of trigrams per corpus actually used "
                             "for the prediction-accuracy check.")
    parser.add_argument("--max_test_cases", type=int, default=200,
                        help="Maximum sequences per corpus included in the "
                             "prediction-accuracy check.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_trigrams", type=str,
                        default="results/analysis/corpus_specific_trigrams.json")
    parser.add_argument("--output_prediction", type=str,
                        default="results/analysis/trigram_prediction_results.json")
    return parser.parse_args()


def load_model(model_size: int, ckpt_path: str, seq_len: int, device: str) -> TransEncoder:
    """Instantiate ``TransEncoder`` and load weights from a checkpoint."""
    config = Config.from_name(
        f"Diff_LLaMA_{model_size}M", block_size=seq_len, _norm_class="RMSNorm"
    )
    model = TransEncoder(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in ckpt:
        state_dict = {
            k.replace("_forward_module.", "").replace("_orig_mod.", ""): v
            for k, v in ckpt["model"].items()
        }
        model.load_state_dict(state_dict)
    else:
        from safetensors.torch import load_file

        model.load_state_dict(load_file(ckpt_path))
    model.eval()
    return model


def extract_trigrams(
    data_dir: str, num_samples: int, seq_len: int
) -> dict[str, Counter]:
    """For each corpus, return a Counter of token trigrams from N sampled sequences."""
    corpus_trigrams: dict[str, Counter] = {}
    for corpus in CORPORA:
        filenames = sorted(glob.glob(f"{data_dir}/train_{corpus}_*.bin"))
        if not filenames:
            print(f"  WARNING: no files for corpus {corpus}")
            continue
        ds = PackedDataset(
            filenames,
            n_chunks=1,
            block_size=seq_len + 1,
            shuffle=False,
            seed=42,
            num_processes=1,
            process_rank=0,
        )
        trigrams: Counter = Counter()
        for i, data in enumerate(ds):
            if i >= num_samples:
                break
            tokens = data[:seq_len].tolist()
            for j in range(len(tokens) - 2):
                trigrams[(tokens[j], tokens[j + 1], tokens[j + 2])] += 1
        corpus_trigrams[corpus] = trigrams
        print(f"  {corpus}: {len(trigrams)} unique trigrams from {num_samples} sequences")
    return corpus_trigrams


def find_corpus_specific(
    corpus_trigrams: dict[str, Counter], min_count: int
) -> dict[str, list[tuple[tuple[int, int, int], int]]]:
    """Find trigrams that appear in exactly one corpus with frequency >= min_count."""
    membership: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    for corpus, trigrams in corpus_trigrams.items():
        for tri in trigrams:
            membership[tri].add(corpus)

    specific: dict[str, list[tuple[tuple[int, int, int], int]]] = {c: [] for c in CORPORA}
    for tri, owners in membership.items():
        if len(owners) == 1:
            owner = next(iter(owners))
            count = corpus_trigrams[owner][tri]
            if count >= min_count:
                specific[owner].append((tri, count))

    for corpus in CORPORA:
        specific[corpus].sort(key=lambda x: -x[1])

    return specific


def filter_alphabetic_middle(
    trigrams: list[tuple[tuple[int, int, int], int]],
    tokenizer,
    top_k: int,
) -> list[tuple[tuple[int, int, int], int]]:
    """Keep only trigrams whose middle token decodes to a >=2-letter alphabetic word."""
    out: list[tuple[tuple[int, int, int], int]] = []
    for tri, count in trigrams:
        mid_text = tokenizer.decode([tri[1]]).strip()
        if len(mid_text) >= 2 and mid_text.isalpha():
            out.append((tri, count))
        if len(out) >= top_k:
            break
    return out


def evaluate_prediction_accuracy(
    model: TransEncoder,
    data_dir: str,
    corpus: str,
    target_trigrams: set[tuple[int, int, int]],
    seq_len: int,
    max_test_cases: int,
    device: str,
    batch_size: int = 32,
) -> tuple[int, int, int]:
    """Mask the middle token of test cases and check if MDM predicts it correctly."""
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
    test_cases: list[tuple[list[int], int, int]] = []
    for i, data in enumerate(ds):
        if i >= 1000 or len(test_cases) >= max_test_cases:
            break
        tokens = data[:seq_len].tolist()
        for j in range(len(tokens) - 2):
            tri = (tokens[j], tokens[j + 1], tokens[j + 2])
            if tri in target_trigrams and len(test_cases) < max_test_cases:
                test_cases.append((tokens, j + 1, tokens[j + 1]))

    if not test_cases:
        return 0, 0, 0

    top1_correct = 0
    top5_correct = 0
    total = len(test_cases)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_seqs: list[list[int]] = []
        positions: list[int] = []
        targets: list[int] = []
        for seq, pos, tgt in test_cases[start:end]:
            masked = list(seq)
            masked[pos] = MASK_ID
            batch_seqs.append(masked)
            positions.append(pos)
            targets.append(tgt)

        input_ids = torch.tensor(batch_seqs, device=device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)

        for k, (pos, tgt) in enumerate(zip(positions, targets, strict=True)):
            top1 = logits[k, pos].argmax().item()
            top5 = logits[k, pos].topk(5).indices.tolist()
            if top1 == tgt:
                top1_correct += 1
            if tgt in top5:
                top5_correct += 1

    return total, top1_correct, top5_correct


def main() -> None:
    args = parse_args()
    Path(args.output_trigrams).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_prediction).parent.mkdir(parents=True, exist_ok=True)

    print("=== Phase 0a: extracting trigrams ===")
    corpus_trigrams = extract_trigrams(args.data_dir, args.num_samples, args.seq_len)

    print("\n=== Phase 0b: finding corpus-specific trigrams ===")
    specific = find_corpus_specific(corpus_trigrams, args.min_count)
    for corpus in CORPORA:
        print(f"  {corpus}: {len(specific[corpus])} specific trigrams (count>={args.min_count})")

    # Save trigrams
    out_trigrams = {
        corpus: [[list(tri), count] for tri, count in specific[corpus][: args.top_k]]
        for corpus in CORPORA
    }
    with open(args.output_trigrams, "w") as f:
        json.dump(out_trigrams, f, indent=2)
    print(f"\nSaved corpus-specific trigrams to {args.output_trigrams}")

    print("\n=== Phase 0c: MDM prediction accuracy on corpus-specific trigrams ===")
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = load_model(args.model_size, args.ckpt_path, args.seq_len, args.device)

    print(f"\n{'Corpus':<15s} {'Tested':>7s} {'Top1':>7s} {'Top5':>7s} {'Top1%':>7s} {'Top5%':>7s}")
    print("-" * 55)

    results: dict[str, dict] = {}
    for corpus in CORPORA:
        good = filter_alphabetic_middle(specific[corpus], tokenizer, args.prediction_top_k)
        if not good:
            print(f"  {corpus:<15s} (no suitable trigrams)")
            continue
        target_set = {tuple(t) for t, _ in good}
        total, top1, top5 = evaluate_prediction_accuracy(
            model,
            args.data_dir,
            corpus,
            target_set,
            args.seq_len,
            args.max_test_cases,
            args.device,
        )
        if total == 0:
            print(f"  {corpus:<15s} (no test cases found)")
            continue
        top1_pct = 100 * top1 / total
        top5_pct = 100 * top5 / total
        print(
            f"  {corpus:<15s} {total:>7d} {top1:>7d} {top5:>7d} "
            f"{top1_pct:>6.1f}% {top5_pct:>6.1f}%"
        )
        results[corpus] = {
            "total": total,
            "top1_correct": top1,
            "top5_correct": top5,
            "top1_pct": round(top1_pct, 1),
            "top5_pct": round(top5_pct, 1),
            "examples": [
                (tokenizer.decode(list(t)), c) for t, c in good[:5]
            ],
        }

    with open(args.output_prediction, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved prediction results to {args.output_prediction}")


if __name__ == "__main__":
    main()
