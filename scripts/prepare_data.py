"""Data preparation script for UnTrac data attribution experiments.

Downloads 8 pretraining corpora, tokenizes them with the TinyLlama tokenizer,
and writes them out in ``PackedDataset`` binary format.

Usage::

    python scripts/prepare_data.py --setting equal --out_dir data/untrac
    python scripts/prepare_data.py --setting different --out_dir data/untrac_different
    python scripts/prepare_data.py --corpus hackernews --setting equal --out_dir data/untrac
"""
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

from mdm_unlearning.data.packed_dataset import PackedDatasetBuilder


# Data source types
SOURCE_DIRECT = "direct"       # Load directly from HuggingFace dataset
SOURCE_PILE_FILTER = "pile"    # Filter from monology/pile-uncopyrighted by pile_set_name

CORPUS_CONFIGS = {
    "bookcorpus": {
        "source_type": SOURCE_DIRECT,
        "hf_id": "bookcorpus/bookcorpus",
        "text_field": "text",
        "split": "train",
    },
    "stackexchange": {
        "source_type": SOURCE_PILE_FILTER,
        "pile_set_name": "StackExchange",
        "text_field": "text",
    },
    "ccnewsv2": {
        "source_type": SOURCE_DIRECT,
        "hf_id": "vblagoje/cc_news",
        "text_field": "text",
        "split": "train",
    },
    "gutenberg": {
        "source_type": SOURCE_DIRECT,
        "hf_id": "deepmind/pg19",
        "text_field": "text",
        "split": "train",
    },
    "hackernews": {
        "source_type": SOURCE_PILE_FILTER,
        "pile_set_name": "HackerNews",
        "text_field": "text",
    },
    "openwebtext": {
        "source_type": SOURCE_DIRECT,
        "hf_id": "Skylion007/openwebtext",
        "text_field": "text",
        "split": "train",
    },
    "pilecc": {
        "source_type": SOURCE_PILE_FILTER,
        "pile_set_name": "Pile-CC",
        "text_field": "text",
    },
    "wikipedia": {
        "source_type": SOURCE_PILE_FILTER,
        "pile_set_name": "Wikipedia (en)",
        "text_field": "text",
    },
}

# Sample counts per corpus
EQUAL_SETTING = {name: 40000 for name in CORPUS_CONFIGS}

DIFFERENT_SETTING = {
    "bookcorpus": 32000,
    "stackexchange": 32000,
    "ccnewsv2": 48000,
    "gutenberg": 16000,
    "hackernews": 16000,
    "openwebtext": 64000,
    "pilecc": 96000,
    "wikipedia": 16000,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting', type=str, default='equal', choices=['equal', 'different'],
                        help='data balance setting')
    parser.add_argument('--out_dir', type=str, default='data/untrac', help='output directory')
    parser.add_argument('--seq_len', type=int, default=1024, help='sequence length per sample')
    parser.add_argument('--corpus', type=str, default='all',
                        help='specific corpus to prepare (default: all)')
    parser.add_argument('--chunk_size', type=int, default=1024 * 1024,
                        help='chunk size for PackedDataset files (in tokens)')
    return parser.parse_args()


def load_pile_filtered(pile_set_name):
    """Stream from monology/pile-uncopyrighted, yielding only items matching pile_set_name."""
    ds = load_dataset(
        'monology/pile-uncopyrighted', split='train',
        streaming=True, trust_remote_code=True
    )
    for item in ds:
        if item['meta']['pile_set_name'] == pile_set_name:
            yield item


def prepare_corpus(corpus_name, corpus_config, num_samples, tokenizer, out_dir, seq_len, chunk_size):
    """Download, tokenize, and pack a single corpus."""
    print(f"\n{'='*60}")
    print(f"Preparing {corpus_name}: {num_samples} samples of {seq_len} tokens each")
    print(f"{'='*60}")

    source_type = corpus_config["source_type"]
    text_field = corpus_config["text_field"]

    # Load dataset based on source type
    try:
        if source_type == SOURCE_PILE_FILTER:
            pile_set_name = corpus_config["pile_set_name"]
            print(f"Loading from monology/pile-uncopyrighted (filter: {pile_set_name})")
            dataset = load_pile_filtered(pile_set_name)
        else:
            hf_id = corpus_config["hf_id"]
            split = corpus_config["split"]
            print(f"Loading from {hf_id}")
            dataset = load_dataset(hf_id, split=split, streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Failed to load {corpus_name}: {e}")
        print(f"Skipping {corpus_name}. You may need to download it manually.")
        return False

    # Initialize PackedDatasetBuilder
    prefix = f"train_{corpus_name}"
    builder = PackedDatasetBuilder(
        outdir=out_dir,
        prefix=prefix,
        chunk_size=chunk_size,
        sep_token=tokenizer.eos_token_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    # Tokenize and pack
    total_tokens_needed = num_samples * seq_len
    total_tokens = 0
    doc_count = 0

    for doc in dataset:
        if total_tokens >= total_tokens_needed:
            break

        text = doc[text_field]
        if not text or len(text.strip()) == 0:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 10:
            continue

        token_array = np.array(tokens, dtype=np.uint16)
        builder.add_array(token_array)
        total_tokens += len(tokens)
        doc_count += 1

        if doc_count % 10000 == 0:
            print(f"  {corpus_name}: {doc_count} docs, {total_tokens}/{total_tokens_needed} tokens "
                  f"({100*total_tokens/total_tokens_needed:.1f}%)")

    builder.write_reminder()

    actual_samples = total_tokens // seq_len
    print(f"  Done: {corpus_name} - {doc_count} docs, {total_tokens} tokens, ~{actual_samples} samples")

    if total_tokens < total_tokens_needed:
        print(f"  WARNING: Only got {total_tokens} tokens, needed {total_tokens_needed}")

    return True


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
        use_fast=True
    )

    sample_counts = EQUAL_SETTING if args.setting == 'equal' else DIFFERENT_SETTING

    if args.corpus == 'all':
        corpora_to_prepare = list(CORPUS_CONFIGS.keys())
    else:
        corpora_to_prepare = [args.corpus]
        assert args.corpus in CORPUS_CONFIGS, f"Unknown corpus: {args.corpus}. Choose from: {list(CORPUS_CONFIGS.keys())}"

    print(f"Setting: {args.setting}")
    print(f"Output directory: {out_dir}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Corpora to prepare: {corpora_to_prepare}")
    print(f"Sample counts: {[(c, sample_counts[c]) for c in corpora_to_prepare]}")

    results = {}
    for corpus_name in corpora_to_prepare:
        success = prepare_corpus(
            corpus_name=corpus_name,
            corpus_config=CORPUS_CONFIGS[corpus_name],
            num_samples=sample_counts[corpus_name],
            tokenizer=tokenizer,
            out_dir=out_dir,
            seq_len=args.seq_len,
            chunk_size=args.chunk_size,
        )
        results[corpus_name] = "OK" if success else "FAILED"

        # Report disk usage after each corpus
        import shutil
        total, used, free = shutil.disk_usage(out_dir)
        data_size = sum(f.stat().st_size for f in Path(out_dir).glob("*"))
        print(f"  Disk free: {free/1e9:.1f} GB | Data dir: {data_size/1e6:.1f} MB")

    print(f"\n{'='*60}")
    print("Summary:")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
