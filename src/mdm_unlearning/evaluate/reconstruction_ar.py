"""
Next-token prediction accuracy analysis for ARM UnTrac.
Equivalent to MDM's masked token reconstruction analysis.

For each unlearned corpus, compares next-token prediction accuracy
before vs after EU unlearning across all 8 training corpora.
"""
import sys, glob, json, struct, torch, argparse, numpy as np, random, math
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer

from mdm_unlearning.models.arm import GPT, Config
from mdm_unlearning.data.packed_dataset import PackedDataset
from torch.utils.data import DataLoader

CORPUS_NAMES = [
    "bookcorpus", "stackexchange", "ccnewsv2", "gutenberg",
    "hackernews", "openwebtext", "pilecc", "wikipedia",
]


def set_seed(seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)


def load_model(model_size, ckpt_path, seq_len=1024, device='cuda'):
    config = Config.from_name(f'Diff_LLaMA_{model_size}M', block_size=seq_len, _norm_class="RMSNorm")
    model = GPT(config).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'model' in ckpt:
        sd = {k.replace('_forward_module.', '').replace('_orig_mod.', ''): v
              for k, v in ckpt['model'].items()}
        model.load_state_dict(sd)
    model.eval()
    return model


def sample_corpus(data_dir, corpus, seq_len, num_samples=100):
    filenames = sorted(glob.glob(str(Path(data_dir) / f"train_{corpus}_*")))
    if not filenames: return None
    ds = PackedDataset(filenames, n_chunks=1, block_size=seq_len + 1,
                       shuffle=False, seed=42, num_processes=1, process_rank=0)
    samples = []
    for i, data in enumerate(ds):
        if i >= num_samples: break
        samples.append(data[:seq_len])
    return torch.stack(samples)


@torch.no_grad()
def evaluate_next_token_accuracy(model, sequences, batch_size=16, device='cuda'):
    """Evaluate next-token prediction accuracy (top-1 and top-5)."""
    model.eval()
    N, L = sequences.shape
    total_correct = 0
    total_top5 = 0
    total_tokens = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = sequences[start:end].to(device)
        B = batch.shape[0]
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(batch)
        # Next-token: predict position t+1 from position t
        preds = logits[:, :-1, :].argmax(dim=-1)  # (B, L-1)
        targets = batch[:, 1:]  # (B, L-1)
        correct = (preds == targets).sum().item()
        top5 = logits[:, :-1, :].topk(5, dim=-1).indices  # (B, L-1, 5)
        top5_correct = (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
        total_correct += correct
        total_top5 += top5_correct
        total_tokens += B * (L - 1)
    return {
        'accuracy': total_correct / total_tokens,
        'top5_accuracy': total_top5 / total_tokens,
    }


def ar_loss_fn(model, input_ids):
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = input_ids[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_targets.view(-1), reduction='mean')
    return loss, logits


def forward_process(batch, total_dim=32000, eps=1e-3):
    """Dummy - not used for ARM but kept for compatibility."""
    pass


def run_eu_unlearning_ar(model, data_dir, corpus, seq_len,
                         max_steps=10000, lr=5e-5, eu_lambda=1.0, device='cuda'):
    """EU unlearning for ARM."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    forget_files = sorted(glob.glob(str(Path(data_dir) / f"train_{corpus}_*")))
    forget_ds = PackedDataset(forget_files, n_chunks=1, block_size=seq_len + 1,
                              shuffle=False, seed=42, num_processes=1, process_rank=0)
    forget_dl = DataLoader(forget_ds, batch_size=1, shuffle=False)

    retain_files = []
    for c in CORPUS_NAMES:
        if c != corpus:
            retain_files.extend(sorted(glob.glob(str(Path(data_dir) / f"train_{c}_*"))))
    retain_ds = PackedDataset(retain_files, n_chunks=1, block_size=seq_len + 1,
                              shuffle=False, seed=42, num_processes=1, process_rank=0)
    retain_dl = DataLoader(retain_ds, batch_size=1, shuffle=False)
    retain_iter = iter(retain_dl)

    step = 0
    V = 32000
    uniform = torch.ones(V, device=device) / V
    for data in tqdm(forget_dl, total=max_steps, desc=f"EU-ARM {corpus}"):
        input_ids = data[:, :seq_len].contiguous().to(device)
        _, logits = ar_loss_fn(model, input_ids)

        shift_logits = logits[:, :-1, :].contiguous().view(-1, V)
        forget_loss = F.kl_div(F.log_softmax(shift_logits, dim=-1),
                               uniform.unsqueeze(0).expand(shift_logits.size(0), -1),
                               reduction='batchmean')

        try:
            retain_data = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_dl)
            retain_data = next(retain_iter)
        r_ids = retain_data[:, :seq_len].contiguous().to(device)
        retain_loss, _ = ar_loss_fn(model, r_ids)

        total_loss = forget_loss + eu_lambda * retain_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        step += 1
        if step >= max_steps: break

    model.eval()
    del optimizer; torch.cuda.empty_cache()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=int, default=113)
    parser.add_argument('--seq_len', type=int, default=1024)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='data/untrac')
    parser.add_argument('--unlearn_corpus', type=str, required=True)
    parser.add_argument('--unlearn_steps', type=int, default=10000)
    parser.add_argument('--eu_lambda', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output', type=str, default='')
    args = parser.parse_args()

    set_seed()
    tokenizer = AutoTokenizer.from_pretrained('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T')

    print("Sampling training data...")
    corpus_data = {}
    for corpus in CORPUS_NAMES:
        data = sample_corpus(args.data_dir, corpus, args.seq_len, args.num_samples)
        if data is not None:
            corpus_data[corpus] = data
            print(f"  {corpus}: {data.shape[0]} sequences")

    print("\nEvaluating base model...")
    base_model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
    base_results = {}
    for corpus in CORPUS_NAMES:
        if corpus not in corpus_data: continue
        r = evaluate_next_token_accuracy(base_model, corpus_data[corpus], device=args.device)
        base_results[corpus] = r
        print(f"  {corpus:15s}: acc={r['accuracy']:.4f}  top5={r['top5_accuracy']:.4f}")
    del base_model; torch.cuda.empty_cache()

    print(f"\nRunning EU unlearning on '{args.unlearn_corpus}' for {args.unlearn_steps} steps...")
    model = load_model(args.model, args.ckpt_path, args.seq_len, args.device)
    model = run_eu_unlearning_ar(model, args.data_dir, args.unlearn_corpus,
                                 args.seq_len, max_steps=args.unlearn_steps,
                                 eu_lambda=args.eu_lambda, device=args.device)

    print("\nEvaluating unlearned model...")
    unlearned_results = {}
    for corpus in CORPUS_NAMES:
        if corpus not in corpus_data: continue
        r = evaluate_next_token_accuracy(model, corpus_data[corpus], device=args.device)
        unlearned_results[corpus] = r
        delta = r['accuracy'] - base_results[corpus]['accuracy']
        marker = " <<<" if corpus == args.unlearn_corpus else ""
        print(f"  {corpus:15s}: acc={r['accuracy']:.4f} (delta={delta:+.4f}){marker}")

    output = {
        'unlearn_corpus': args.unlearn_corpus,
        'unlearn_steps': args.unlearn_steps,
        'base_results': base_results,
        'unlearned_results': unlearned_results,
        'delta': {c: {
            'accuracy': unlearned_results[c]['accuracy'] - base_results[c]['accuracy'],
            'top5_accuracy': unlearned_results[c]['top5_accuracy'] - base_results[c]['top5_accuracy'],
        } for c in CORPUS_NAMES if c in base_results},
    }
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved to {args.output}")
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
