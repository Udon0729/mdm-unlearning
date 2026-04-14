"""
Masked Token Reconstruction Analysis for UnTrac on MDM.

For each unlearned corpus, compares token reconstruction accuracy
before vs after KL-Constrained unlearning across all 8 training corpora.

Expected: accuracy drops on the unlearned corpus, stays similar on others.
"""
import sys
import glob
import copy
import json
import struct
import torch
import argparse
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm

import torch.nn.functional as F
from transformers import AutoTokenizer

from mdm_unlearning.models.diffmodel import Config
from mdm_unlearning.models.enc_dec_diffmodel import TransEncoderDecoder, forward_process_block
from mdm_unlearning.data.packed_dataset import PackedDataset
from torch.utils.data import DataLoader


CORPUS_NAMES = [
    "bookcorpus", "stackexchange", "ccnewsv2", "gutenberg",
    "hackernews", "openwebtext", "pilecc", "wikipedia",
]
MASK_ID = 32000


def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_model(model_size, ckpt_path, seq_len=1024, device='cuda'):
    config = Config.from_name(f'Diff_LLaMA_{model_size}M',
                              block_size=seq_len, _norm_class="RMSNorm")
    config.n_encoder_layers = 8
    config.n_decoder_layers = 4
    config.diffusion_block_size = 128
    model = TransEncoderDecoder(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        sd = {k.replace('_forward_module.', '').replace('_orig_mod.', ''): v
              for k, v in ckpt['model'].items()}
        model.load_state_dict(sd)
    else:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(ckpt_path))
    model.eval()
    return model


def sample_corpus(data_dir, corpus, seq_len, num_samples=100):
    """Sample sequences from a training corpus."""
    filenames = sorted(glob.glob(str(Path(data_dir) / f"train_{corpus}_*")))
    if not filenames:
        return None
    dataset = PackedDataset(
        filenames, n_chunks=1, block_size=seq_len + 1,
        shuffle=False, seed=42, num_processes=1, process_rank=0,
    )
    samples = []
    for i, data in enumerate(dataset):
        if i >= num_samples:
            break
        samples.append(data[:seq_len])
    return torch.stack(samples)


@torch.no_grad()
def evaluate_reconstruction(model, sequences, mask_ratio=0.15, batch_size=32, device='cuda'):
    """Evaluate masked token reconstruction accuracy with fixed mask."""
    model.eval()
    N, L = sequences.shape
    # Fixed mask for fair comparison across models
    gen = torch.Generator().manual_seed(12345)
    mask = torch.rand(N, L, generator=gen) < mask_ratio

    all_preds = []
    all_logits_at_mask = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = sequences[start:end].to(device)
        batch_mask = mask[start:end].to(device)
        masked_input = torch.where(batch_mask, MASK_ID, batch)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch, masked_input)  # E2D2: (clean, noisy)

        preds = logits.argmax(dim=-1).cpu()
        all_preds.append(preds)
        # Top-5 at masked positions
        for i in range(preds.shape[0]):
            m = batch_mask[i]
            if m.any():
                top5 = logits[i][m].topk(5, dim=-1).indices.cpu()
                all_logits_at_mask.append((start + i, top5))

    all_preds = torch.cat(all_preds, dim=0)

    correct = (all_preds == sequences) & mask
    acc = correct.float().sum() / mask.float().sum()

    # Top-5
    top5_hits = 0
    top5_total = 0
    for idx, top5 in all_logits_at_mask:
        m = mask[idx]
        targets = sequences[idx][m]
        hits = (top5 == targets.unsqueeze(-1)).any(dim=-1)
        top5_hits += hits.sum().item()
        top5_total += hits.shape[0]

    top5_acc = top5_hits / top5_total if top5_total > 0 else 0.0

    return {
        'accuracy': acc.item(),
        'top5_accuracy': top5_acc,
        'preds': all_preds,
        'mask': mask,
    }


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)
    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


def count_packed_dataset_samples(filenames, block_size):
    total = 0
    for fn in filenames:
        with open(fn, 'rb') as f:
            f.read(7); f.read(8); f.read(1)
            chunk_size = struct.unpack("<Q", f.read(8))[0]
            total += chunk_size // block_size
    return total


def run_kl_unlearning(model, ref_model, data_dir, corpus, seq_len,
                      max_steps=10000, lr=5e-5, alpha=1.0, device='cuda'):
    """Run KL-constrained gradient ascent unlearning."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    filenames = sorted(glob.glob(str(Path(data_dir) / f"train_{corpus}_*")))
    dataset = PackedDataset(
        filenames, n_chunks=1, block_size=seq_len + 1,
        shuffle=False, seed=42, num_processes=1, process_rank=0,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    step = 0
    for data in tqdm(dataloader, total=max_steps, desc=f"Unlearning {corpus}"):
        input_ids = data[:, :seq_len].contiguous().to(device)
        noisy_input, mask_indices, p_mask = forward_process_block(input_ids, block_size=128)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(input_ids, noisy_input)
        loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices],
                               reduction='none') / p_mask[mask_indices]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                ref_logits = ref_model(input_ids, noisy_input)
        kl = F.kl_div(
            F.log_softmax(logits[mask_indices], dim=-1),
            F.softmax(ref_logits[mask_indices], dim=-1),
            reduction='batchmean')

        total_loss = -loss + alpha * kl
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if step >= max_steps:
            break

    model.eval()
    del optimizer
    torch.cuda.empty_cache()
    return model


def compute_fisher_diagonal(model, dataloader, seq_len, device, max_batches=500):
    """Compute diagonal Fisher Information Matrix on data."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    count = 0
    for data in dataloader:
        if count >= max_batches:
            break
        input_ids = data[:, :seq_len].contiguous().to(device)
        noisy_input, mask_indices, p_mask = forward_process_block(input_ids, block_size=128)
        model.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids, noisy_input)
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
    """Compute gradient saliency on data."""
    model.eval()
    saliency = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    count = 0
    for data in dataloader:
        if count >= max_batches:
            break
        input_ids = data[:, :seq_len].contiguous().to(device)
        noisy_input, mask_indices, p_mask = forward_process_block(input_ids, block_size=128)
        model.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids, noisy_input)
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


def run_fisher_meta_unlearning(model, data_dir, corpus, seq_len,
                               max_steps=2000, lr=5e-5, ewc_alpha=1.0,
                               saliency_top_pct=30, fisher_bottom_pct=70,
                               fisher_samples=500, meta_k=10, meta_every=500,
                               meta_beta=0.01, device='cuda'):
    """Run Fisher-Meta unlearning: EWC + saliency mask + meta robustness."""
    # 1. Build retain dataloader
    retain_files = []
    for c in CORPUS_NAMES:
        if c != corpus:
            retain_files.extend(sorted(glob.glob(str(Path(data_dir) / f"train_{c}_*"))))
    retain_ds = PackedDataset(retain_files, n_chunks=1, block_size=seq_len + 1,
                              shuffle=False, seed=42, num_processes=1, process_rank=0)
    retain_dl = DataLoader(retain_ds, batch_size=1, shuffle=False)

    # 2. Compute Fisher on retain data
    print(f"  Computing Fisher ({fisher_samples} batches)...")
    fisher = compute_fisher_diagonal(model, retain_dl, seq_len, device, fisher_samples)

    # 3. Compute saliency on forget data
    print(f"  Computing saliency ({fisher_samples} batches)...")
    forget_files = sorted(glob.glob(str(Path(data_dir) / f"train_{corpus}_*")))
    sal_ds = PackedDataset(forget_files, n_chunks=1, block_size=seq_len + 1,
                           shuffle=False, seed=42, num_processes=1, process_rank=0)
    sal_dl = DataLoader(sal_ds, batch_size=1, shuffle=False)
    saliency = compute_saliency(model, sal_dl, seq_len, device, fisher_samples)

    # 4. Create parameter mask
    all_sal = torch.cat([s.flatten() for s in saliency.values()]).cpu().numpy()
    all_fis = torch.cat([f.flatten() for f in fisher.values()]).cpu().numpy()
    sal_thresh = float(np.percentile(all_sal, 100 - saliency_top_pct))
    fis_thresh = float(np.percentile(all_fis, fisher_bottom_pct))
    del all_sal, all_fis
    param_mask = {}
    for n in fisher:
        param_mask[n] = ((saliency[n] >= sal_thresh) & (fisher[n] <= fis_thresh)).float()
    total = sum(m.numel() for m in param_mask.values())
    active = sum(m.sum().item() for m in param_mask.values())
    print(f"  Param mask: {active:.0f}/{total} ({100*active/total:.1f}%) selected")
    del saliency

    # 5. Store original params for EWC
    original_params = {n: p.clone().detach() for n, p in model.named_parameters()}

    # 6. Prepare dataloaders
    forget_ds = PackedDataset(forget_files, n_chunks=1, block_size=seq_len + 1,
                              shuffle=False, seed=42, num_processes=1, process_rank=0)
    forget_dl = DataLoader(forget_ds, batch_size=1, shuffle=False)
    retain_ds2 = PackedDataset(retain_files, n_chunks=1, block_size=seq_len + 1,
                               shuffle=False, seed=42, num_processes=1, process_rank=0)
    retain_dl2 = DataLoader(retain_ds2, batch_size=1, shuffle=False)
    retain_iter = iter(retain_dl2)

    # 7. Unlearning loop
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    step = 0
    for data in tqdm(forget_dl, total=max_steps, desc=f"Unlearning {corpus}"):
        input_ids = data[:, :seq_len].contiguous().to(device)
        noisy_input, mask_indices, p_mask = forward_process_block(input_ids, block_size=128)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids, noisy_input)
        loss = F.cross_entropy(logits[mask_indices], input_ids[mask_indices],
                               reduction='none') / p_mask[mask_indices]
        loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

        ewc_loss = sum((fisher[n] * (p - original_params[n]).pow(2)).sum()
                       for n, p in model.named_parameters())
        total_loss = -loss + ewc_alpha * ewc_loss
        total_loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                p.grad.data.mul_(param_mask[n])

        optimizer.step()
        optimizer.zero_grad()
        step += 1

        # Meta robustness check
        if step % meta_every == 0:
            saved = {n: p.data.clone() for n, p in model.named_parameters()}
            model.train()
            meta_opt = torch.optim.SGD(model.parameters(), lr=lr)
            for _ in range(meta_k):
                try:
                    rb = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_dl2)
                    rb = next(retain_iter)
                r_ids = rb[:, :seq_len].contiguous().to(device)
                r_n, r_m, r_p = forward_process_block(r_ids, block_size=128)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    r_log = model(r_ids, r_n)
                r_loss = F.cross_entropy(r_log[r_m], r_ids[r_m], reduction='none') / r_p[r_m]
                r_loss = r_loss.sum() / (r_ids.shape[0] * r_ids.shape[1])
                r_loss.backward()
                meta_opt.step()
                meta_opt.zero_grad()

            f_n, f_m, f_p = forward_process_block(input_ids, block_size=128)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                f_log = model(input_ids, f_n)
            f_loss = F.cross_entropy(f_log[f_m], input_ids[f_m], reduction='none') / f_p[f_m]
            f_loss = f_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            model.zero_grad()
            (-f_loss).backward()
            meta_grads = {n: p.grad.data.clone() if p.grad is not None
                          else torch.zeros_like(p) for n, p in model.named_parameters()}
            for n, p in model.named_parameters():
                p.data.copy_(saved[n])
                p.data.sub_(meta_beta * meta_grads[n] * param_mask[n])
            del saved, meta_opt, meta_grads
            torch.cuda.empty_cache()
            model.train()

        if step >= max_steps:
            break

    model.eval()
    del optimizer, fisher, original_params, param_mask
    torch.cuda.empty_cache()
    return model


def get_text_examples(sequences, preds_before, preds_after, mask, tokenizer,
                      num_examples=3, max_tokens=64):
    """Generate text examples showing reconstruction differences."""
    examples = []
    for idx in range(min(num_examples, sequences.shape[0])):
        m = mask[idx]
        seq = sequences[idx][:max_tokens]
        m_short = m[:max_tokens]
        pb = preds_before[idx][:max_tokens]
        pa = preds_after[idx][:max_tokens]

        # Build annotated text
        original_tokens = []
        before_tokens = []
        after_tokens = []
        for pos in range(len(seq)):
            orig_tok = tokenizer.decode([seq[pos].item()])
            if m_short[pos]:
                b_tok = tokenizer.decode([pb[pos].item()])
                a_tok = tokenizer.decode([pa[pos].item()])
                b_mark = b_tok if pb[pos] == seq[pos] else f"[{b_tok}]"
                a_mark = a_tok if pa[pos] == seq[pos] else f"[{a_tok}]"
                original_tokens.append(f"_{orig_tok}_")
                before_tokens.append(b_mark)
                after_tokens.append(a_mark)
            else:
                original_tokens.append(orig_tok)
                before_tokens.append(orig_tok)
                after_tokens.append(orig_tok)

        # Count differences
        masked_positions = m_short.sum().item()
        correct_before = ((pb == seq) & m_short)[:max_tokens].sum().item()
        correct_after = ((pa == seq) & m_short)[:max_tokens].sum().item()

        examples.append({
            'original': ''.join(original_tokens),
            'before': ''.join(before_tokens),
            'after': ''.join(after_tokens),
            'masked_positions': int(masked_positions),
            'correct_before': int(correct_before),
            'correct_after': int(correct_after),
        })
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=113)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/untrac')
    parser.add_argument('--unlearn_corpus', type=str, required=True)
    parser.add_argument('--unlearn_steps', type=int, default=10000)
    parser.add_argument('--unlearn_method', type=str, default='kl',
                        choices=['kl', 'fisher_meta'])
    parser.add_argument('--kl_alpha', type=float, default=1.0)
    parser.add_argument('--ewc_alpha', type=float, default=1.0)
    parser.add_argument('--saliency_top_pct', type=float, default=30)
    parser.add_argument('--fisher_bottom_pct', type=float, default=70)
    parser.add_argument('--fisher_samples', type=int, default=500)
    parser.add_argument('--meta_k', type=int, default=10)
    parser.add_argument('--meta_every', type=int, default=500)
    parser.add_argument('--meta_beta', type=float, default=0.01)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(
        'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

    # Sample data from all corpora
    print("Sampling training data...")
    corpus_data = {}
    for corpus in CORPUS_NAMES:
        data = sample_corpus(args.data_dir, corpus, args.seq_len, args.num_samples)
        if data is not None:
            corpus_data[corpus] = data
            print(f"  {corpus}: {data.shape[0]} sequences")

    # Evaluate BASE model reconstruction
    print("\nEvaluating base model reconstruction...")
    base_model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
    base_results = {}
    base_preds = {}
    for corpus in CORPUS_NAMES:
        if corpus not in corpus_data:
            continue
        r = evaluate_reconstruction(base_model, corpus_data[corpus],
                                    args.mask_ratio, device=args.device)
        base_results[corpus] = {'accuracy': r['accuracy'],
                                'top5_accuracy': r['top5_accuracy']}
        base_preds[corpus] = r['preds']
        print(f"  {corpus:15s}: acc={r['accuracy']:.4f}  top5={r['top5_accuracy']:.4f}")

    # Run unlearning
    print(f"\nRunning {args.unlearn_method} unlearning on '{args.unlearn_corpus}' "
          f"for {args.unlearn_steps} steps...")

    model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)

    if args.unlearn_method == 'kl':
        ref_model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
        model = run_kl_unlearning(model, ref_model, args.data_dir, args.unlearn_corpus,
                                  args.seq_len, max_steps=args.unlearn_steps,
                                  alpha=args.kl_alpha, device=args.device)
        del ref_model; torch.cuda.empty_cache()
    elif args.unlearn_method == 'fisher_meta':
        model = run_fisher_meta_unlearning(
            model, args.data_dir, args.unlearn_corpus, args.seq_len,
            max_steps=args.unlearn_steps, lr=5e-5,
            ewc_alpha=args.ewc_alpha,
            saliency_top_pct=args.saliency_top_pct,
            fisher_bottom_pct=args.fisher_bottom_pct,
            fisher_samples=args.fisher_samples,
            meta_k=args.meta_k, meta_every=args.meta_every,
            meta_beta=args.meta_beta, device=args.device)

    # Evaluate UNLEARNED model reconstruction
    print("\nEvaluating unlearned model reconstruction...")
    unlearned_results = {}
    unlearned_preds = {}
    for corpus in CORPUS_NAMES:
        if corpus not in corpus_data:
            continue
        r = evaluate_reconstruction(model, corpus_data[corpus],
                                    args.mask_ratio, device=args.device)
        unlearned_results[corpus] = {'accuracy': r['accuracy'],
                                     'top5_accuracy': r['top5_accuracy']}
        unlearned_preds[corpus] = r['preds']
        delta_acc = r['accuracy'] - base_results[corpus]['accuracy']
        marker = " <<<" if corpus == args.unlearn_corpus else ""
        print(f"  {corpus:15s}: acc={r['accuracy']:.4f} "
              f"(delta={delta_acc:+.4f}){marker}")

    # Generate text examples for the unlearned corpus
    print(f"\nText examples for '{args.unlearn_corpus}':")
    mask_gen = torch.Generator().manual_seed(12345)
    ex_mask = torch.rand(corpus_data[args.unlearn_corpus].shape, generator=mask_gen) < args.mask_ratio

    examples = get_text_examples(
        corpus_data[args.unlearn_corpus],
        base_preds[args.unlearn_corpus],
        unlearned_preds[args.unlearn_corpus],
        ex_mask, tokenizer, num_examples=3, max_tokens=80)

    for i, ex in enumerate(examples):
        print(f"\n  --- Example {i+1} (masked={ex['masked_positions']}, "
              f"before={ex['correct_before']}, after={ex['correct_after']}) ---")
        print(f"  Original:  {ex['original'][:200]}")
        print(f"  Before:    {ex['before'][:200]}")
        print(f"  After:     {ex['after'][:200]}")

    # Also generate examples for a RETAINED corpus (for contrast)
    retained_corpus = [c for c in CORPUS_NAMES if c != args.unlearn_corpus][0]
    print(f"\nText examples for RETAINED '{retained_corpus}':")
    ret_mask = torch.rand(corpus_data[retained_corpus].shape, generator=torch.Generator().manual_seed(12345)) < args.mask_ratio
    ret_examples = get_text_examples(
        corpus_data[retained_corpus],
        base_preds[retained_corpus],
        unlearned_preds[retained_corpus],
        ret_mask, tokenizer, num_examples=2, max_tokens=80)

    for i, ex in enumerate(ret_examples):
        print(f"\n  --- Example {i+1} (masked={ex['masked_positions']}, "
              f"before={ex['correct_before']}, after={ex['correct_after']}) ---")
        print(f"  Original:  {ex['original'][:200]}")
        print(f"  Before:    {ex['before'][:200]}")
        print(f"  After:     {ex['after'][:200]}")

    # Save results
    output = {
        'unlearn_corpus': args.unlearn_corpus,
        'unlearn_steps': args.unlearn_steps,
        'mask_ratio': args.mask_ratio,
        'base_results': base_results,
        'unlearned_results': unlearned_results,
        'delta': {c: {
            'accuracy': unlearned_results[c]['accuracy'] - base_results[c]['accuracy'],
            'top5_accuracy': unlearned_results[c]['top5_accuracy'] - base_results[c]['top5_accuracy'],
        } for c in CORPUS_NAMES if c in base_results},
        'examples_unlearned': examples,
        'examples_retained': ret_examples,
    }

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")

    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
