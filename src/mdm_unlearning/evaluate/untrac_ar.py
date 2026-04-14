"""
UnTrac evaluation for ARM (autoregressive model).
Adapted from evaluate_untrac.py for causal language models.

Key differences from MDM version:
- No forward_process (masking) - uses standard next-token prediction
- NLL computation is deterministic (no MC sampling needed)
- Unlearning methods operate on causal LM loss
"""
import sys
import glob
import copy
import math
import time
import torch
import argparse
import json
import struct
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from functools import partial

import torch.nn.functional as F
from transformers import AutoTokenizer, set_seed as hf_set_seed
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from torch.utils.data import RandomSampler, DataLoader

from mdm_unlearning.models.arm import GPT, Config
from mdm_unlearning.data.packed_dataset import PackedDataset, CombinedDataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='untrac',
                        choices=['nll', 'untrac'])
    parser.add_argument('--model', type=int, default=113)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--test_dataset', type=str, default='toxigen',
                        choices=['toxigen', 'winobias', 'truthfulqa', 'all'])
    parser.add_argument('--mc_num', type=int, default=1)
    parser.add_argument('--mc_batch', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--data_dir', type=str, default='data/untrac')
    parser.add_argument('--unlearn_lr', type=float, default=5e-5)
    parser.add_argument('--unlearn_epochs', type=int, default=1)
    parser.add_argument('--unlearn_batch_size', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--unlearn_method', type=str, default='eu',
                        choices=['ga', 'kl', 'eu'])
    parser.add_argument('--kl_alpha', type=float, default=1.0)
    parser.add_argument('--eu_lambda', type=float, default=1.0)
    parser.add_argument('--untrac_corpus', type=str, default='')
    return parser.parse_args()


CORPUS_NAMES = [
    "bookcorpus", "stackexchange", "ccnewsv2", "gutenberg",
    "hackernews", "openwebtext", "pilecc", "wikipedia",
]


def count_packed_dataset_samples(filenames, block_size):
    total = 0
    for fn in filenames:
        with open(fn, 'rb') as f:
            f.read(7); f.read(8); f.read(1)
            chunk_size = struct.unpack("<Q", f.read(8))[0]
            total += chunk_size // block_size
    return total


def load_model(model_size, ckpt_path, seq_len=1024, device='cuda'):
    model_name = f'Diff_LLaMA_{model_size}M'
    config = Config.from_name(model_name, block_size=seq_len, _norm_class="RMSNorm")
    model = GPT(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        state_dict = {}
        for k, v in ckpt['model'].items():
            k = k.replace('_forward_module.', '').replace('_orig_mod.', '')
            state_dict[k] = v
        model.load_state_dict(state_dict)
    else:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(ckpt_path))
    model.eval()
    return model


def ar_loss_fn(model, input_ids):
    """Compute AR next-token prediction loss."""
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_targets.view(-1), reduction='mean')
    return loss, logits


# ============================================================
# Test data loading (same as MDM version)
# ============================================================

def _sample_dataset(dataset, num_samples, seed=42):
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(dataset, num_samples=min(num_samples, len(dataset)), generator=g)
    return dataset.select(sampler)


def load_test_data_toxigen(tokenizer, seed=42):
    ds = load_dataset("skg/toxigen-data", name="train", split="train", trust_remote_code=True)
    ds = ds.filter(lambda ex: ex["prompt_label"] == 1)
    groups = ds.unique("group")
    group_datasets = {g: ds.filter(lambda ex: ex["group"] == g) for g in groups}
    for g in list(group_datasets.keys()):
        gds = group_datasets[g]
        gds = gds.map(lambda ex: {"length": len(ex["generation"].split(" "))})
        gds = gds.sort("length")
        gds = gds.filter(lambda ex: ex["length"] > 8 and ex["length"] <= 24)
        group_datasets[g] = gds
    all_texts, all_subsets = [], []
    for g, gds in group_datasets.items():
        sampled = _sample_dataset(gds, 256, seed)
        for item in sampled:
            all_texts.append(item["generation"])
            all_subsets.append(g)
    sequences = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in all_texts]
    print(f"ToxiGen: {len(sequences)} sequences, {len(set(all_subsets))} groups")
    return sequences, all_subsets


def load_test_data_winobias(tokenizer, seed=42):
    pro = load_dataset("wino_bias", name="type1_pro", split="validation+test", trust_remote_code=True)
    pro = pro.map(lambda ex: {"text": " ".join(ex["tokens"]), "type": "pro"})
    anti = load_dataset("wino_bias", name="type1_anti", split="validation+test", trust_remote_code=True)
    anti = anti.map(lambda ex: {"text": " ".join(ex["tokens"]), "type": "anti"})
    combined = concatenate_datasets([pro, anti])
    def categorize(ex):
        tokens = ex["tokens"] if isinstance(ex["tokens"], list) else ex["text"].split()
        female = any(w in tokens for w in ["she", "her"])
        male = any(w in tokens for w in ["he", "his", "him"])
        if female and not male: return {"subset": ex["type"] + "_female"}
        elif male and not female: return {"subset": ex["type"] + "_male"}
        return {"subset": None}
    combined = combined.map(categorize)
    combined = combined.filter(lambda ex: ex["subset"] is not None)
    categories = combined.unique("subset")
    cat_datasets = {c: combined.filter(lambda ex: ex["subset"] == c) for c in categories}
    all_texts, all_subsets = [], []
    for c, cds in cat_datasets.items():
        sampled = _sample_dataset(cds, 256, seed)
        for item in sampled:
            all_texts.append(item["text"])
            all_subsets.append(c)
    sequences = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in all_texts]
    print(f"WinoBias: {len(sequences)} sequences, {len(set(all_subsets))} categories")
    return sequences, all_subsets


def load_test_data_truthfulqa(tokenizer, seed=42):
    raw = load_dataset("truthful_qa", name="generation", split="validation", trust_remote_code=True)
    items = []
    for ex in raw:
        cat = ex["category"]
        if "Indexical Error" in cat: cat_name = "indexical_error"
        elif "Confusion" in cat: cat_name = "confusion"
        else: cat_name = cat.lower()
        for ans in ex["incorrect_answers"]:
            items.append({"question": ex["question"].strip(), "answer": ans.strip(), "subset": cat_name})
    counter = Counter(item["subset"] for item in items)
    valid_cats = {c for c, n in counter.items() if n >= 128}
    items = [item for item in items if item["subset"] in valid_cats]
    ds = Dataset.from_list(items)
    cat_datasets = {c: ds.filter(lambda ex: ex["subset"] == c) for c in ds.unique("subset")}
    all_texts, all_subsets = [], []
    for c, cds in cat_datasets.items():
        sampled = _sample_dataset(cds, 256, seed)
        for item in sampled:
            all_texts.append(item["question"] + " " + item["answer"])
            all_subsets.append(c)
    sequences = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in all_texts]
    print(f"TruthfulQA: {len(sequences)} sequences, {len(set(all_subsets))} categories")
    return sequences, all_subsets


def load_test_data(dataset_name, tokenizer, seed=42):
    if dataset_name == 'toxigen': return load_test_data_toxigen(tokenizer, seed)
    elif dataset_name == 'winobias': return load_test_data_winobias(tokenizer, seed)
    elif dataset_name == 'truthfulqa': return load_test_data_truthfulqa(tokenizer, seed)
    else: raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================
# NLL computation (ARM: deterministic next-token prediction)
# ============================================================

@torch.no_grad()
def compute_nll_per_sequence(model, sequences, mc_num=1, mc_batch=32, device='cuda', mask_id=32000):
    """Compute per-sequence NLL for ARM (deterministic)."""
    model.eval()
    block_size = model.config.block_size
    padded, lengths = [], []
    for seq in sequences:
        L = min(len(seq), block_size)
        lengths.append(L)
        if len(seq) >= block_size: padded.append(seq[:block_size])
        else: padded.append(F.pad(seq, (0, block_size - len(seq)), value=0))
    all_seqs = torch.stack(padded).to(device)
    lengths_t = torch.tensor(lengths, device=device, dtype=torch.float32)
    N = all_seqs.shape[0]
    nlls = torch.zeros(N, device=device)
    for start in tqdm(range(0, N, mc_batch), desc="Computing NLL"):
        end = min(start + mc_batch, N)
        batch = all_seqs[start:end]
        B, L = batch.shape
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = batch[:, 1:].contiguous()
        ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                             shift_targets.view(-1), reduction='none').view(B, L-1)
        nlls[start:end] = ce.sum(dim=1) / (lengths_t[start:end] - 1)
    return nlls.cpu().tolist()


def compute_nll_per_subset(model, sequences, subsets, mc_num=1, mc_batch=32, device='cuda'):
    nlls = compute_nll_per_sequence(model, sequences, mc_num, mc_batch, device)
    subset_nlls = {}
    for nll, subset in zip(nlls, subsets):
        subset_nlls.setdefault(subset, []).append(nll)
    return {s: float(np.mean(v)) for s, v in subset_nlls.items()}


# ============================================================
# Mode: UnTrac
# ============================================================

def mode_untrac(args):
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    datasets_to_eval = ['toxigen', 'winobias', 'truthfulqa'] if args.test_dataset == 'all' else [args.test_dataset]

    test_data_cache = {}
    for ds_name in datasets_to_eval:
        test_data_cache[ds_name] = load_test_data(ds_name, tokenizer)

    print("Computing initial test NLLs...")
    model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
    init_results = {}
    for ds_name in datasets_to_eval:
        sequences, subsets = test_data_cache[ds_name]
        init_results[ds_name] = compute_nll_per_subset(model, sequences, subsets, args.mc_num, args.mc_batch, args.device)
    del model; torch.cuda.empty_cache()

    all_results = {}
    data_dir = Path(args.data_dir)
    corpora_to_process = [args.untrac_corpus] if args.untrac_corpus else CORPUS_NAMES

    for corpus in corpora_to_process:
        print(f"\n{'='*60}")
        print(f"UnTrac (ARM): unlearning corpus={corpus}")
        print(f"{'='*60}")

        model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
        model.train()

        # Setup
        prefix = f"train_{corpus}"
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        if not filenames:
            print(f"  WARNING: No data files for {corpus}"); continue

        dataset = PackedDataset(filenames, n_chunks=1, block_size=args.seq_len + 1,
                                shuffle=False, seed=42, num_processes=1, process_rank=0)
        dataloader = DataLoader(dataset, batch_size=args.unlearn_batch_size, shuffle=False)
        total_samples = count_packed_dataset_samples(filenames, args.seq_len + 1)
        max_steps = (total_samples * args.unlearn_epochs) // args.unlearn_batch_size
        print(f"  {len(filenames)} files, {total_samples} samples, {max_steps} steps")
        print(f"  method={args.unlearn_method}")

        # EU retain dataloader
        eu_retain_iter = None
        eu_retain_dl = None
        if args.unlearn_method == 'eu':
            retain_files = []
            for c in CORPUS_NAMES:
                if c != corpus:
                    retain_files.extend(sorted(glob.glob(str(data_dir / f"train_{c}_*"))))
            eu_retain_ds = PackedDataset(retain_files, n_chunks=1, block_size=args.seq_len + 1,
                                         shuffle=False, seed=42, num_processes=1, process_rank=0)
            eu_retain_dl = DataLoader(eu_retain_ds, batch_size=1, shuffle=False)
            eu_retain_iter = iter(eu_retain_dl)

        # KL reference model
        ref_model = None
        if args.unlearn_method == 'kl':
            ref_model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
            ref_model.eval()
            for p in ref_model.parameters(): p.requires_grad_(False)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.unlearn_lr, betas=(0.9, 0.999))

        # Unlearning loop
        step = 0
        corpus_results = {}
        for data in dataloader:
            input_ids = data[:, :args.seq_len].contiguous().to(args.device)
            loss, logits = ar_loss_fn(model, input_ids)

            if args.unlearn_method == 'ga':
                total_loss = -loss
            elif args.unlearn_method == 'kl':
                with torch.no_grad():
                    _, ref_logits = ar_loss_fn(ref_model, input_ids)
                kl = F.kl_div(
                    F.log_softmax(logits[:, :-1, :].contiguous().view(-1, logits.size(-1)), dim=-1),
                    F.softmax(ref_logits[:, :-1, :].contiguous().view(-1, ref_logits.size(-1)), dim=-1),
                    reduction='batchmean')
                total_loss = -loss + args.kl_alpha * kl
            elif args.unlearn_method == 'eu':
                V = logits.size(-1)
                uniform = torch.ones(V, device=logits.device) / V
                shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
                forget_loss = F.kl_div(F.log_softmax(shift_logits, dim=-1),
                                       uniform.unsqueeze(0).expand(shift_logits.size(0), -1),
                                       reduction='batchmean')
                try:
                    retain_data = next(eu_retain_iter)
                except StopIteration:
                    eu_retain_iter = iter(eu_retain_dl)
                    retain_data = next(eu_retain_iter)
                r_ids = retain_data[:, :args.seq_len].contiguous().to(args.device)
                retain_loss, _ = ar_loss_fn(model, r_ids)
                total_loss = forget_loss + args.eu_lambda * retain_loss

            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % args.eval_steps == 0 or step == 1:
                model.eval()
                step_results = {}
                for ds_name in datasets_to_eval:
                    sequences, subsets = test_data_cache[ds_name]
                    step_results[ds_name] = compute_nll_per_subset(
                        model, sequences, subsets, args.mc_num, args.mc_batch, args.device)
                corpus_results[step] = step_results
                print(f"  step {step}: eval done")
                model.train()

            if step >= max_steps:
                break

        # Final evaluation
        model.eval()
        final_results = {}
        for ds_name in datasets_to_eval:
            sequences, subsets = test_data_cache[ds_name]
            final_results[ds_name] = compute_nll_per_subset(
                model, sequences, subsets, args.mc_num, args.mc_batch, args.device)
        corpus_results[step] = final_results

        influence = {}
        for ds_name in datasets_to_eval:
            influence[ds_name] = {}
            for subset in init_results[ds_name]:
                if subset in final_results[ds_name]:
                    influence[ds_name][subset] = final_results[ds_name][subset] - init_results[ds_name][subset]

        all_results[corpus] = {"influence": influence, "steps": corpus_results}
        del model, optimizer
        if ref_model is not None: del ref_model
        torch.cuda.empty_cache()

        print(f"  Influence for {corpus}:")
        for ds_name in datasets_to_eval:
            for subset, val in sorted(influence.get(ds_name, {}).items()):
                print(f"    {ds_name}/{subset}: {val:+.6f}")

    result = {"init_results": init_results, "untrac_results": all_results}
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'untrac':
        mode_untrac(args)
