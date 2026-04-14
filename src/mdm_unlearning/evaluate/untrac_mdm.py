"""
UnTrac evaluation and attribution for MDM.

Modes:
  nll          - Compute per-subset NLL on test data for a single checkpoint
  attribution  - Leave-one-out attribution (ground truth baseline)
  untrac       - UnTrac: unlearn each training corpus, measure test loss change
  untrac_inv   - UnTrac-Inv: unlearn test data, measure training corpus loss change

Test data filtering matches the original UnTrac codebase (misonuma/untrac):
  ToxiGen:    prompt_label==1, word length 8-24, 256 samples/group (13 groups)
  WinoBias:   type1_pro + type1_anti, val+test, gender-categorized, 256/category (4 categories)
  TruthfulQA: incorrect answers, categories with >=128 samples, 256/category (9 categories)
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

from mdm_unlearning.models.diffmodel import TransEncoder, Config
from mdm_unlearning.data.packed_dataset import PackedDataset, CombinedDataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    hf_set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='nll',
                        choices=['nll', 'attribution', 'untrac', 'untrac_inv'])
    parser.add_argument('--model', type=int, default=113)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--full_ckpt', type=str, default='')
    parser.add_argument('--loo_dir', type=str, default='workdir/untrac')
    parser.add_argument('--test_dataset', type=str, default='toxigen',
                        choices=['toxigen', 'winobias', 'truthfulqa', 'all'])
    parser.add_argument('--mc_num', type=int, default=1)
    parser.add_argument('--mc_batch', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='')
    # UnTrac-specific
    parser.add_argument('--data_dir', type=str, default='data/untrac')
    parser.add_argument('--unlearn_lr', type=float, default=5e-5)
    parser.add_argument('--unlearn_epochs', type=int, default=1)
    parser.add_argument('--unlearn_batch_size', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5000)
    # UnTrac-Inv specific
    parser.add_argument('--inv_batch_size', type=int, default=256)
    parser.add_argument('--inv_max_steps', type=int, default=50)
    parser.add_argument('--inv_eval_steps', type=int, default=5)
    # Unlearning method
    parser.add_argument('--unlearn_method', type=str, default='ga',
                        choices=['ga', 'kl', 'npo', 'vdu', 'fisher_meta', 'eu'],
                        help='ga/kl/npo/vdu/fisher_meta/eu')
    parser.add_argument('--kl_alpha', type=float, default=1.0)
    parser.add_argument('--npo_beta', type=float, default=0.1)
    parser.add_argument('--vdu_gamma', type=float, default=0.01)
    # Fisher-Meta specific
    parser.add_argument('--ewc_alpha', type=float, default=1.0,
                        help='EWC regularization weight')
    parser.add_argument('--saliency_top_pct', type=float, default=30,
                        help='Top-k%% saliency for param mask')
    parser.add_argument('--fisher_bottom_pct', type=float, default=70,
                        help='Bottom-k%% Fisher for param mask')
    parser.add_argument('--fisher_samples', type=int, default=500,
                        help='Batches for Fisher/saliency computation')
    parser.add_argument('--meta_k', type=int, default=10,
                        help='Inner fine-tuning steps for meta-unlearning')
    parser.add_argument('--meta_every', type=int, default=500,
                        help='Meta robustness check frequency')
    parser.add_argument('--meta_beta', type=float, default=0.01,
                        help='Meta gradient step size')
    # Exclusive Unlearning (EU) specific
    parser.add_argument('--eu_lambda', type=float, default=1.0,
                        help='Retain loss weight for EU method')
    # Single-corpus mode for parallel execution
    parser.add_argument('--untrac_corpus', type=str, default='',
                        help='Process only this corpus (for parallel execution)')
    return parser.parse_args()


CORPUS_NAMES = [
    "bookcorpus", "stackexchange", "ccnewsv2", "gutenberg",
    "hackernews", "openwebtext", "pilecc", "wikipedia",
]


def count_packed_dataset_samples(filenames, block_size):
    """Count total samples across packed dataset files by reading headers."""
    total = 0
    for fn in filenames:
        with open(fn, 'rb') as f:
            f.read(7)   # HDR_MAGIC
            f.read(8)   # version
            f.read(1)   # dtype_code
            chunk_size = struct.unpack("<Q", f.read(8))[0]
            total += chunk_size // block_size
    return total


# ============================================================
# Fisher / Saliency / Parameter Mask (for fisher_meta method)
# ============================================================

def compute_fisher_diagonal(model, dataloader, seq_len, device, max_batches=500):
    """Compute diagonal Fisher Information Matrix on data (retain set)."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    count = 0
    for data in dataloader:
        if count >= max_batches:
            break
        input_ids = data[:, :seq_len].contiguous().to(device)
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        model.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(noisy_input)
        loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices],
                               reduction='none') / p_mask[mask_indices]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)
        count += 1
    for n in fisher:
        fisher[n] /= max(count, 1)
    model.zero_grad()
    return fisher


def compute_saliency(model, dataloader, seq_len, device, max_batches=500):
    """Compute gradient saliency on data (forget set)."""
    model.eval()
    saliency = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    count = 0
    for data in dataloader:
        if count >= max_batches:
            break
        input_ids = data[:, :seq_len].contiguous().to(device)
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        model.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(noisy_input)
        loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices],
                               reduction='none') / p_mask[mask_indices]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                saliency[n] += p.grad.data.abs()
        count += 1
    for n in saliency:
        saliency[n] /= max(count, 1)
    model.zero_grad()
    return saliency


def create_param_mask(fisher, saliency, saliency_top_pct=30, fisher_bottom_pct=70):
    """Create binary mask: high saliency (forget-specific) AND low Fisher (retain-safe)."""
    # Use numpy percentile to avoid torch.quantile memory limit
    all_sal = torch.cat([s.flatten() for s in saliency.values()]).cpu().numpy()
    all_fis = torch.cat([f.flatten() for f in fisher.values()]).cpu().numpy()
    sal_thresh = float(np.percentile(all_sal, 100 - saliency_top_pct))
    fis_thresh = float(np.percentile(all_fis, fisher_bottom_pct))
    del all_sal, all_fis
    mask = {}
    for n in fisher:
        mask[n] = ((saliency[n] >= sal_thresh) & (fisher[n] <= fis_thresh)).float()
    total = sum(m.numel() for m in mask.values())
    active = sum(m.sum().item() for m in mask.values())
    print(f"  Param mask: {active:.0f}/{total} ({100*active/total:.1f}%) parameters selected")
    print(f"  Saliency threshold: {sal_thresh:.2e}, Fisher threshold: {fis_thresh:.2e}")
    return mask


# ============================================================
# Model loading
# ============================================================

def load_model(model_size, ckpt_path, seq_len=1024, device='cuda'):
    model_name = f'Diff_LLaMA_{model_size}M'
    config = Config.from_name(model_name, block_size=seq_len, _norm_class="RMSNorm")
    model = TransEncoder(config).to(device)

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


# ============================================================
# Test data loading (matches misonuma/untrac preprocess_test.py)
# ============================================================

def _sample_dataset(dataset, num_samples, seed=42):
    """Random sample from HF dataset, matching original RandomSampler."""
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(dataset, num_samples=min(num_samples, len(dataset)), generator=g)
    return dataset.select(sampler)


def load_test_data_toxigen(tokenizer, seed=42):
    """ToxiGen: prompt_label==1, word length 8-24, 256 samples per group."""
    ds = load_dataset("skg/toxigen-data", name="train", split="train", trust_remote_code=True)
    ds = ds.filter(lambda ex: ex["prompt_label"] == 1)

    groups = ds.unique("group")
    group_datasets = {g: ds.filter(lambda ex: ex["group"] == g) for g in groups}

    # Filter by word length and sort
    for g in list(group_datasets.keys()):
        gds = group_datasets[g]
        gds = gds.map(lambda ex: {"length": len(ex["generation"].split(" "))})
        gds = gds.sort("length")
        gds = gds.filter(lambda ex: ex["length"] > 8 and ex["length"] <= 24)
        group_datasets[g] = gds

    # Sample 256 per group, collect with subset labels
    all_texts, all_subsets = [], []
    for g, gds in group_datasets.items():
        sampled = _sample_dataset(gds, 256, seed)
        for item in sampled:
            all_texts.append(item["generation"])
            all_subsets.append(g)

    # Tokenize
    sequences = [torch.tensor(tokenizer.encode(t), dtype=torch.long) for t in all_texts]
    print(f"ToxiGen: {len(sequences)} sequences, {len(set(all_subsets))} groups")
    return sequences, all_subsets


def load_test_data_winobias(tokenizer, seed=42):
    """WinoBias: type1_pro + type1_anti, val+test, gender-categorized, 256 per category."""
    pro = load_dataset("wino_bias", name="type1_pro", split="validation+test", trust_remote_code=True)
    pro = pro.map(lambda ex: {"text": " ".join(ex["tokens"]), "type": "pro"})
    anti = load_dataset("wino_bias", name="type1_anti", split="validation+test", trust_remote_code=True)
    anti = anti.map(lambda ex: {"text": " ".join(ex["tokens"]), "type": "anti"})
    combined = concatenate_datasets([pro, anti])

    def categorize(ex):
        tokens = ex["tokens"] if isinstance(ex["tokens"], list) else ex["text"].split()
        female = any(w in tokens for w in ["she", "her"])
        male = any(w in tokens for w in ["he", "his", "him"])
        if female and not male:
            return {"subset": ex["type"] + "_female"}
        elif male and not female:
            return {"subset": ex["type"] + "_male"}
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
    """TruthfulQA: incorrect answers, categories with >=128 samples, 256 per category."""
    raw = load_dataset("truthful_qa", name="generation", split="validation", trust_remote_code=True)

    items = []
    for ex in raw:
        cat = ex["category"]
        if "Indexical Error" in cat:
            cat_name = "indexical_error"
        elif "Confusion" in cat:
            cat_name = "confusion"
        else:
            cat_name = cat.lower()
        for ans in ex["incorrect_answers"]:
            items.append({
                "question": ex["question"].strip(),
                "answer": ans.strip(),
                "subset": cat_name,
            })

    # Filter categories with >= 128 samples
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
    """Load test data with subset labels. Returns (sequences, subset_labels)."""
    if dataset_name == 'toxigen':
        return load_test_data_toxigen(tokenizer, seed)
    elif dataset_name == 'winobias':
        return load_test_data_winobias(tokenizer, seed)
    elif dataset_name == 'truthfulqa':
        return load_test_data_truthfulqa(tokenizer, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


# ============================================================
# NLL computation (MDM-specific Monte Carlo estimation)
# ============================================================

@torch.no_grad()
def compute_nll_per_sequence(model, sequences, mc_num=128, mc_batch=32, device='cuda', mask_id=32000):
    """Compute per-sequence NLL via Monte Carlo. Processes multiple sequences per forward pass."""
    model.eval()
    block_size = model.config.block_size

    # Pad all sequences to block_size
    padded = []
    lengths = []
    for seq in sequences:
        L = min(len(seq), block_size)
        lengths.append(L)
        if len(seq) >= block_size:
            padded.append(seq[:block_size])
        else:
            padded.append(F.pad(seq, (0, block_size - len(seq)), value=0))
    all_seqs = torch.stack(padded).to(device)  # (N, block_size)
    lengths_t = torch.tensor(lengths, device=device, dtype=torch.float32)
    N = all_seqs.shape[0]

    nlls = torch.zeros(N, device=device)
    for start in tqdm(range(0, N, mc_batch), desc="Computing NLL"):
        end = min(start + mc_batch, N)
        batch = all_seqs[start:end]  # (B, L)
        B, L = batch.shape
        batch_lengths = lengths_t[start:end]
        loss_acc = torch.zeros(B, device=device)

        for _ in range(mc_num):
            t = torch.rand((B,), device=device)
            p_mask = (1 - 1e-3) * t + 1e-3
            p_mask_2d = p_mask[:, None].expand(B, L)
            mask_idx = torch.rand((B, L), device=device) < p_mask_2d
            noisy = torch.where(mask_idx, mask_id, batch)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(noisy)  # (B, L, V)

            # Vectorized per-sequence loss
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1), reduction='none').view(B, L)
            weighted = ce / p_mask_2d
            masked_loss = (weighted * mask_idx.float()).sum(dim=1) / batch_lengths
            loss_acc += masked_loss

        nlls[start:end] = loss_acc / mc_num

    return nlls.cpu().tolist()


def compute_nll_per_subset(model, sequences, subsets, mc_num=128, mc_batch=32, device='cuda'):
    """Compute NLL grouped by subset. Returns dict {subset: mean_nll}."""
    nlls = compute_nll_per_sequence(model, sequences, mc_num, mc_batch, device)
    subset_nlls = {}
    for nll, subset in zip(nlls, subsets):
        subset_nlls.setdefault(subset, []).append(nll)
    return {s: float(np.mean(v)) for s, v in subset_nlls.items()}


# ============================================================
# MDM forward process (for UnTrac training)
# ============================================================

def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


# ============================================================
# Mode: NLL (per-subset)
# ============================================================

def mode_nll(args):
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')
    model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)

    datasets_to_eval = ['toxigen', 'winobias', 'truthfulqa'] if args.test_dataset == 'all' else [args.test_dataset]
    result = {}
    for ds_name in datasets_to_eval:
        sequences, subsets = load_test_data(ds_name, tokenizer)
        subset_nlls = compute_nll_per_subset(model, sequences, subsets, args.mc_num, args.mc_batch, args.device)
        result[ds_name] = subset_nlls
        print(f"\n{ds_name} per-subset NLL:")
        for s, v in sorted(subset_nlls.items()):
            print(f"  {s:30s}: {v:.4f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    return result


# ============================================================
# Mode: Leave-one-out attribution
# ============================================================

def mode_attribution(args):
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

    datasets_to_eval = ['toxigen', 'winobias', 'truthfulqa'] if args.test_dataset == 'all' else [args.test_dataset]

    # Load full model and compute baseline NLLs
    print("Computing full model NLLs...")
    model = load_model(args.model, args.full_ckpt, args.seq_len, args.device)
    full_results = {}
    test_data_cache = {}
    for ds_name in datasets_to_eval:
        sequences, subsets = load_test_data(ds_name, tokenizer)
        test_data_cache[ds_name] = (sequences, subsets)
        full_results[ds_name] = compute_nll_per_subset(model, sequences, subsets, args.mc_num, args.mc_batch, args.device)
    del model; torch.cuda.empty_cache()

    # Load each LOO model
    loo_dir = Path(args.loo_dir)
    loo_results = {ds: {} for ds in datasets_to_eval}
    for corpus in CORPUS_NAMES:
        pattern = f"mdm-untrac-{args.model}M-*-excl-{corpus}"
        matches = list(loo_dir.glob(pattern))
        if not matches:
            print(f"WARNING: No LOO checkpoint for {corpus}")
            continue
        ckpts = sorted(matches[0].glob("iter-*-ckpt.pth"))
        if not ckpts:
            continue
        print(f"\nLOO: exclude={corpus}, ckpt={ckpts[-1].name}")
        model = load_model(args.model, str(ckpts[-1]), args.seq_len, args.device)
        for ds_name in datasets_to_eval:
            sequences, subsets = test_data_cache[ds_name]
            loo_results[ds_name][corpus] = compute_nll_per_subset(
                model, sequences, subsets, args.mc_num, args.mc_batch, args.device)
        del model; torch.cuda.empty_cache()

    # Compute attribution scores per subset
    result = {}
    for ds_name in datasets_to_eval:
        result[ds_name] = {}
        subset_names = sorted(full_results[ds_name].keys())
        for subset in subset_names:
            full_nll = full_results[ds_name][subset]
            scores = {}
            for corpus in CORPUS_NAMES:
                if corpus in loo_results[ds_name] and subset in loo_results[ds_name][corpus]:
                    scores[corpus] = loo_results[ds_name][corpus][subset] - full_nll
            result[ds_name][subset] = scores

        print(f"\n{'='*60}")
        print(f"Attribution scores for {ds_name}:")
        for subset in subset_names:
            print(f"\n  Subset: {subset}")
            for corpus, score in sorted(result[ds_name][subset].items(), key=lambda x: -x[1]):
                print(f"    {corpus:20s}: {score:+.6f}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
    return result


# ============================================================
# Mode: UnTrac (gradient ascent unlearning on training data)
# ============================================================

def mode_untrac(args):
    """
    UnTrac: For each training corpus, unlearn it from the trained model via gradient ascent.
    Measure influence as: eval_loss_after_unlearning - eval_loss_before_unlearning.
    Matches misonuma/untrac: loss = -loss for gradient ascent, 1 epoch, batch=1, lr=5e-5.
    """
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

    datasets_to_eval = ['toxigen', 'winobias', 'truthfulqa'] if args.test_dataset == 'all' else [args.test_dataset]

    # Load test data
    test_data_cache = {}
    for ds_name in datasets_to_eval:
        test_data_cache[ds_name] = load_test_data(ds_name, tokenizer)

    # Compute initial (pre-unlearning) test NLLs
    print("Computing initial test NLLs...")
    model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
    init_results = {}
    for ds_name in datasets_to_eval:
        sequences, subsets = test_data_cache[ds_name]
        init_results[ds_name] = compute_nll_per_subset(
            model, sequences, subsets, args.mc_num, args.mc_batch, args.device)
    del model; torch.cuda.empty_cache()

    # For each training corpus, unlearn and measure influence
    all_results = {}
    data_dir = Path(args.data_dir)

    corpora_to_process = [args.untrac_corpus] if args.untrac_corpus else CORPUS_NAMES
    for corpus in corpora_to_process:
        print(f"\n{'='*60}")
        print(f"UnTrac: unlearning corpus={corpus}")
        print(f"{'='*60}")

        # Load model fresh from checkpoint
        model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
        model.train()

        # Load training data for this corpus (forget set)
        prefix = f"train_{corpus}"
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        if not filenames:
            print(f"  WARNING: No data files for {corpus}")
            continue

        dataset = PackedDataset(
            filenames, n_chunks=1, block_size=args.seq_len + 1,
            shuffle=False, seed=42, num_processes=1, process_rank=0,
        )
        dataloader = DataLoader(dataset, batch_size=args.unlearn_batch_size, shuffle=False)

        # PackedDataset never raises StopIteration (wraps forever), so we
        # compute the epoch length from file headers and break manually.
        total_samples = count_packed_dataset_samples(filenames, args.seq_len + 1)
        max_steps = (total_samples * args.unlearn_epochs) // args.unlearn_batch_size
        print(f"  {len(filenames)} files, {total_samples} samples, {max_steps} steps ({args.unlearn_epochs} epoch(s))")
        print(f"  method={args.unlearn_method}")

        # ---- fisher_meta: pre-computation phase ----
        param_mask = None
        original_params = None
        retain_dl = None
        fisher_diag = None
        if args.unlearn_method == 'fisher_meta':
            # 1. Build retain dataloader (all corpora except forget)
            retain_files = []
            for c in CORPUS_NAMES:
                if c != corpus:
                    retain_files.extend(sorted(glob.glob(str(data_dir / f"train_{c}_*"))))
            retain_ds = PackedDataset(
                retain_files, n_chunks=1, block_size=args.seq_len + 1,
                shuffle=False, seed=42, num_processes=1, process_rank=0,
            )
            retain_dl = DataLoader(retain_ds, batch_size=1, shuffle=False)

            # 2. Compute Fisher diagonal on retain data
            print(f"  Computing Fisher diagonal ({args.fisher_samples} batches)...")
            fisher_diag = compute_fisher_diagonal(
                model, retain_dl, args.seq_len, args.device, args.fisher_samples)

            # 3. Compute saliency on forget data
            print(f"  Computing saliency ({args.fisher_samples} batches)...")
            forget_sal_ds = PackedDataset(
                filenames, n_chunks=1, block_size=args.seq_len + 1,
                shuffle=False, seed=42, num_processes=1, process_rank=0,
            )
            forget_sal_dl = DataLoader(forget_sal_ds, batch_size=1, shuffle=False)
            saliency = compute_saliency(
                model, forget_sal_dl, args.seq_len, args.device, args.fisher_samples)

            # 4. Create parameter mask
            param_mask = create_param_mask(
                fisher_diag, saliency, args.saliency_top_pct, args.fisher_bottom_pct)
            del saliency

            # 5. Store original parameters for EWC
            original_params = {n: p.clone().detach() for n, p in model.named_parameters()}

            # Refresh retain dataloader for meta inner loop
            retain_ds_meta = PackedDataset(
                retain_files, n_chunks=1, block_size=args.seq_len + 1,
                shuffle=False, seed=42, num_processes=1, process_rank=0,
            )
            retain_dl = DataLoader(retain_ds_meta, batch_size=1, shuffle=False)
            retain_iter = iter(retain_dl)

        # ---- EU: build retain dataloader ----
        eu_retain_iter = None
        if args.unlearn_method == 'eu':
            retain_files = []
            for c in CORPUS_NAMES:
                if c != corpus:
                    retain_files.extend(sorted(glob.glob(str(data_dir / f"train_{c}_*"))))
            eu_retain_ds = PackedDataset(
                retain_files, n_chunks=1, block_size=args.seq_len + 1,
                shuffle=False, seed=42, num_processes=1, process_rank=0,
            )
            eu_retain_dl = DataLoader(eu_retain_ds, batch_size=1, shuffle=False)
            eu_retain_iter = iter(eu_retain_dl)
            print(f"  EU: retain data from {len(retain_files)} files, lambda={args.eu_lambda}")

        # ---- Setup for other methods ----
        ref_model = None
        ref_params = None
        if args.unlearn_method in ('kl', 'npo'):
            ref_model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
        elif args.unlearn_method == 'vdu':
            ref_params = {n: p.clone().detach() for n, p in model.named_parameters()}

        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.unlearn_lr, betas=(0.9, 0.999))
        model.train()

        # ---- Unlearning loop ----
        step = 0
        corpus_results = {}
        for data in dataloader:
            input_ids = data[:, :args.seq_len].contiguous().to(args.device)
            noisy_input, mask_indices, p_mask = forward_process(input_ids)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(noisy_input)
            loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            if args.unlearn_method == 'ga':
                total_loss = -loss

            elif args.unlearn_method == 'kl':
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        ref_logits = ref_model(noisy_input)
                kl = F.kl_div(
                    F.log_softmax(logits[mask_indices], dim=-1),
                    F.softmax(ref_logits[mask_indices], dim=-1),
                    reduction='batchmean')
                total_loss = -loss + args.kl_alpha * kl

            elif args.unlearn_method == 'npo':
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        ref_logits = ref_model(noisy_input)
                    ref_loss = F.cross_entropy(ref_logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
                    ref_loss = ref_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
                    ratio = torch.exp(ref_loss - loss.detach())
                    weight = (2 * ratio.pow(args.npo_beta) / (ratio.pow(args.npo_beta) + 1))
                total_loss = -(weight * loss)

            elif args.unlearn_method == 'vdu':
                l2_penalty = sum(((p - ref_params[n]).pow(2)).sum()
                                 for n, p in model.named_parameters())
                total_loss = -loss + args.vdu_gamma * l2_penalty

            elif args.unlearn_method == 'fisher_meta':
                # ① Forget + ② EWC regularization
                ewc_loss = sum((fisher_diag[n] * (p - original_params[n]).pow(2)).sum()
                               for n, p in model.named_parameters())
                total_loss = -loss + args.ewc_alpha * ewc_loss

            elif args.unlearn_method == 'eu':
                # Exclusive Unlearning: uniform CE on forget + NLL on retain
                # Forget: drive masked position predictions toward uniform distribution
                V = logits.size(-1)
                log_uniform = torch.full((V,), -math.log(V), device=logits.device)
                forget_loss = F.kl_div(
                    F.log_softmax(logits[mask_indices], dim=-1),
                    log_uniform.unsqueeze(0).expand(mask_indices.sum(), -1).exp(),
                    reduction='batchmean')

                # Retain: standard ELBO on a batch from other corpora
                try:
                    retain_data = next(eu_retain_iter)
                except StopIteration:
                    eu_retain_iter = iter(eu_retain_dl)
                    retain_data = next(eu_retain_iter)
                r_ids = retain_data[:, :args.seq_len].contiguous().to(args.device)
                r_noisy, r_mask, r_pm = forward_process(r_ids)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    r_logits = model(r_noisy)
                retain_loss = F.cross_entropy(
                    r_logits[r_mask], r_ids[r_mask], reduction='none') / r_pm[r_mask]
                retain_loss = retain_loss.sum() / (r_ids.shape[0] * r_ids.shape[1])

                total_loss = forget_loss + args.eu_lambda * retain_loss

            total_loss.backward()

            # ④ Apply saliency mask (fisher_meta only)
            if param_mask is not None:
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad.data.mul_(param_mask[n])

            optimizer.step()
            optimizer.zero_grad()
            step += 1

            # ③ Meta robustness check (fisher_meta only)
            if (args.unlearn_method == 'fisher_meta' and
                    step % args.meta_every == 0 and step > 0):
                model.eval()
                # Save current state
                saved_state = {n: p.data.clone() for n, p in model.named_parameters()}

                # Inner loop: simulate fine-tuning on retain data
                model.train()
                meta_opt = torch.optim.SGD(model.parameters(), lr=args.unlearn_lr)
                for _ in range(args.meta_k):
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(retain_dl)
                        retain_batch = next(retain_iter)
                    r_ids = retain_batch[:, :args.seq_len].contiguous().to(args.device)
                    r_noisy, r_mask, r_pm = forward_process(r_ids)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        r_logits = model(r_noisy)
                    r_loss = F.cross_entropy(r_logits[r_mask], r_ids[r_mask],
                                             reduction='none') / r_pm[r_mask]
                    r_loss = r_loss.sum() / (r_ids.shape[0] * r_ids.shape[1])
                    r_loss.backward()
                    meta_opt.step()
                    meta_opt.zero_grad()

                # Evaluate: is forget data still forgotten at θ_ft?
                f_ids = input_ids  # reuse current forget batch
                f_noisy, f_mask, f_pm = forward_process(f_ids)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    f_logits = model(f_noisy)
                f_loss = F.cross_entropy(f_logits[f_mask], f_ids[f_mask],
                                         reduction='none') / f_pm[f_mask]
                f_loss = f_loss.sum() / (f_ids.shape[0] * f_ids.shape[1])

                # We want f_loss HIGH → gradient ascent on f_loss at θ_ft
                model.zero_grad()
                (-f_loss).backward()
                meta_grads = {n: p.grad.data.clone() if p.grad is not None
                              else torch.zeros_like(p)
                              for n, p in model.named_parameters()}

                # Restore original state and apply meta gradient with mask
                for n, p in model.named_parameters():
                    p.data.copy_(saved_state[n])
                    if n in meta_grads:
                        p.data.sub_(args.meta_beta * meta_grads[n] * param_mask[n])

                del saved_state, meta_opt, meta_grads
                torch.cuda.empty_cache()
                model.train()
                print(f"  step {step}: meta check done (forget_loss_at_ft={f_loss.item():.2f})")

            # Periodic evaluation
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

        # Compute influence: final - initial
        influence = {}
        for ds_name in datasets_to_eval:
            influence[ds_name] = {}
            for subset in init_results[ds_name]:
                if subset in final_results[ds_name]:
                    influence[ds_name][subset] = final_results[ds_name][subset] - init_results[ds_name][subset]

        all_results[corpus] = {"influence": influence, "steps": corpus_results}
        del model, optimizer
        if ref_model is not None:
            del ref_model; ref_model = None
        if ref_params is not None:
            del ref_params; ref_params = None
        if param_mask is not None:
            del param_mask, original_params, fisher_diag; param_mask = None
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


# ============================================================
# Mode: UnTrac-Inv (unlearn test data, evaluate on training corpora)
# ============================================================

def mode_untrac_inv(args):
    """
    UnTrac-Inv: Unlearn TEST data from the model, then measure loss change on each
    training corpus. Matches paper: batch=256 (effective), max_steps=50, eval every 5 steps.
    """
    set_seed()
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

    datasets_to_eval = ['toxigen', 'winobias', 'truthfulqa'] if args.test_dataset == 'all' else [args.test_dataset]
    data_dir = Path(args.data_dir)

    # Helper: compute per-corpus training loss
    def compute_train_corpus_losses(model, data_dir, seq_len, device, num_samples=256):
        model.eval()
        corpus_losses = {}
        for corpus in CORPUS_NAMES:
            prefix = f"train_{corpus}"
            filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
            if not filenames:
                continue
            dataset = PackedDataset(
                filenames, n_chunks=1, block_size=seq_len + 1,
                shuffle=False, seed=42, num_processes=1, process_rank=0,
            )
            dl = DataLoader(dataset, batch_size=8, shuffle=False)
            losses = []
            count = 0
            for data in dl:
                input_ids = data[:, :seq_len].contiguous().to(device)
                noisy_input, mask_indices, p_mask = forward_process(input_ids)
                with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(noisy_input)
                loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
                loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
                losses.append(loss.cpu().item())
                count += input_ids.shape[0]
                if count >= num_samples:
                    break
            corpus_losses[corpus] = float(np.mean(losses))
        return corpus_losses

    for ds_name in datasets_to_eval:
        print(f"\n{'='*60}")
        print(f"UnTrac-Inv: unlearning test data ({ds_name})")
        print(f"{'='*60}")

        # Load model
        model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)

        # Initial training corpus losses
        init_corpus_losses = compute_train_corpus_losses(model, data_dir, args.seq_len, args.device)
        print(f"Initial corpus losses: { {k: f'{v:.4f}' for k,v in init_corpus_losses.items()} }")

        # Prepare test data as unlearning source
        sequences, subsets = load_test_data(ds_name, tokenizer)
        # Pad/truncate sequences to seq_len for batching
        padded = []
        for seq in sequences:
            if len(seq) >= args.seq_len:
                padded.append(seq[:args.seq_len])
            else:
                padded.append(F.pad(seq, (0, args.seq_len - len(seq)), value=tokenizer.eos_token_id))
        test_tensor = torch.stack(padded)  # (N, seq_len)

        # Unlearn with effective batch_size=256 via gradient accumulation
        micro_batch = min(8, len(test_tensor))
        grad_accum = max(1, args.inv_batch_size // micro_batch)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.unlearn_lr, betas=(0.9, 0.999))

        step_results = {}
        global_step = 0
        for epoch in range(100):  # max epochs, will break on max_steps
            indices = torch.randperm(len(test_tensor))
            for i in range(0, len(test_tensor), micro_batch):
                if global_step >= args.inv_max_steps:
                    break
                batch_idx = indices[i:i+micro_batch]
                input_ids = test_tensor[batch_idx].to(args.device)
                noisy_input, mask_indices, p_mask = forward_process(input_ids)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    logits = model(noisy_input)
                loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
                loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

                neg_loss = -loss / grad_accum
                neg_loss.backward()

                if (i // micro_batch + 1) % grad_accum == 0 or (i + micro_batch) >= len(test_tensor):
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % args.inv_eval_steps == 0:
                        model.eval()
                        corpus_losses = compute_train_corpus_losses(model, data_dir, args.seq_len, args.device)
                        step_results[global_step] = corpus_losses
                        print(f"  step {global_step}: { {k: f'{v:.4f}' for k,v in corpus_losses.items()} }")
                        model.train()

                    if global_step >= args.inv_max_steps:
                        break
            if global_step >= args.inv_max_steps:
                break

        # Final evaluation
        model.eval()
        final_corpus_losses = compute_train_corpus_losses(model, data_dir, args.seq_len, args.device)

        # Influence: final - initial (positive = test data helped this corpus)
        influence = {c: final_corpus_losses.get(c, 0) - init_corpus_losses.get(c, 0) for c in CORPUS_NAMES}

        print(f"\nUnTrac-Inv influence for {ds_name}:")
        for c, v in sorted(influence.items(), key=lambda x: -x[1]):
            print(f"  {c:20s}: {v:+.6f}")

        del model, optimizer; torch.cuda.empty_cache()

    result = {"init_corpus_losses": init_corpus_losses, "influence": influence, "step_results": step_results}
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'nll':
        mode_nll(args)
    elif args.mode == 'attribution':
        mode_attribution(args)
    elif args.mode == 'untrac':
        mode_untrac(args)
    elif args.mode == 'untrac_inv':
        mode_untrac_inv(args)
