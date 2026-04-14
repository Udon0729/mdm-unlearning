"""
MDM training script for UnTrac data attribution experiments.
Based on train_mdm.py, modified for:
- 4 GPU training with small batch size (global batch=8)
- Step-based training (40K steps) instead of FLOPs-based
- Constant learning rate (Adam, lr=5e-5) to match UnTrac paper
- 8-corpus data configuration for attribution tracking
- Configurable sequence length (default 1024)
"""
import glob
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from functools import partial


from mdm_unlearning.models.diffmodel import TransEncoder, Block, Config
from mdm_unlearning.data.packed_dataset import CombinedDataset, PackedDataset
from mdm_unlearning.utils.speed_monitor import SpeedMonitorFabric as Monitor
from mdm_unlearning.utils.speed_monitor import estimate_flops
from mdm_unlearning.utils.utils import get_default_supported_precision, num_parameters, step_csv_logger
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import random
import argparse


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, default=113, help='model non-embedding params in millions')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of nodes')
    parse.add_argument('--num_devices', type=int, default=4, help='number of GPUs')
    parse.add_argument('--batch_size', type=int, default=8, help='global batch size')
    parse.add_argument('--micro_batch_size', type=int, default=2, help='micro batch size per GPU')
    parse.add_argument('--max_steps', type=int, default=40000, help='max training steps')
    parse.add_argument('--seq_len', type=int, default=1024, help='sequence length')
    parse.add_argument('--lr', type=float, default=5e-5, help='learning rate')
    parse.add_argument('--wd', type=float, default=0.0, help='weight decay')
    parse.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parse.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parse.add_argument('--grad_clip', type=float, default=0.0, help='gradient clipping (0=off)')
    parse.add_argument('--decay_lr', action='store_true', help='use cosine LR decay (default: constant)')
    parse.add_argument('--save_interval', type=int, default=2000, help='checkpoint save interval (steps)')
    parse.add_argument('--log_interval', type=int, default=10, help='log interval (steps)')
    parse.add_argument('--data_dir', type=str, default='data/untrac', help='training data directory')
    parse.add_argument('--val_data_dir', type=str, default='', help='validation data directory (empty=none)')
    parse.add_argument('--data_setting', type=str, default='equal', choices=['equal', 'different'],
                        help='data balance setting')
    parse.add_argument('--exclude_corpus', type=str, default='',
                        help='corpus name to exclude (for leave-one-out experiments)')
    args = parse.parse_args()
    return args


args = parse_args()
model_name = f'Diff_LLaMA_{args.model}M'
out_dir = Path('workdir')

model_para_config = {
    '6': 6.294784, '19': 18.880896, '34': 33.563136, '48': 47.786688, '66': 65.54944,
    '85': 85.21408, '75': 75.38752, '113': 113.265408, '142': 141.581568, '170': 169.897728,
    '180': 179.856768, '206': 205.550464, '231': 231.24416, '268': 268.469248, '302': 302.027776,
    '336': 335.586304, '472': 471.90656, '551': 550.55744, '571': 571.001728, '629': 629.20832,
    '666': 666.168448, '717': 717.285888, '761': 761.335168, '831': 830.541312, '944': 943.796736,
    '1028': 1027.677952, '1233': 1233.213184, '1476': 1476.487168, '1678': 1677.826048, '2121': 2121.39328
}

# Hyperparameters
num_of_devices = args.num_devices
global_batch_size = int(args.batch_size / args.nodes_num)
learning_rate = args.lr
micro_batch_size = args.micro_batch_size
max_step = args.max_steps
warmup_steps = 0
log_step_interval = args.log_interval
eval_iters = int(100 * 1024 / global_batch_size)
save_step_interval = args.save_interval
eval_step_interval = 999999999999  # inf

weight_decay = args.wd
beta1 = args.beta1
beta2 = args.beta2
grad_clip = args.grad_clip
decay_lr = args.decay_lr
min_lr = learning_rate / 10

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0, (
    f"batch_size per device ({batch_size}) must be >= micro_batch_size ({micro_batch_size})"
)
warmup_iters = warmup_steps * gradient_accumulation_steps

max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


# 8-corpus data configuration for UnTrac experiment
CORPUS_NAMES = [
    "train_bookcorpus",
    "train_stackexchange",
    "train_ccnewsv2",
    "train_gutenberg",
    "train_hackernews",
    "train_openwebtext",
    "train_pilecc",
    "train_wikipedia",
]

# Build train_data_config, excluding specified corpus for leave-one-out
excluded = args.exclude_corpus.strip()
train_data_config = []
for name in CORPUS_NAMES:
    short_name = name.replace("train_", "")
    if excluded and short_name == excluded:
        continue
    train_data_config.append((name, 1.0))

val_data_config = [
    ("validation", 1.0),
]

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)


def forward_process(batch, total_dim=32000, eps=1e-3):
    b, l = batch.shape
    t = torch.rand((b,), device=batch.device)

    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask


def setup(
    devices: int = None,
    train_data_dir: Path = None,
    val_data_dir: Path = None,
    precision: Optional[str] = None,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    if devices is None:
        devices = num_of_devices
    if train_data_dir is None:
        train_data_dir = Path(args.data_dir)
    if val_data_dir is None:
        val_data_dir = Path(args.val_data_dir) if args.val_data_dir else None

    exclude_tag = f'-excl-{excluded}' if excluded else ''
    hp_name = f'mdm-untrac-{args.model}M-{args.max_steps}steps{exclude_tag}'
    out_dir = Path('workdir/untrac') / hp_name
    wandb_logger = WandbLogger(name=hp_name, save_dir=out_dir, project='untrac')

    precision = precision or get_default_supported_precision(training=True)

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy=None,
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    main(fabric, train_data_dir, val_data_dir, resume)


def main(fabric, train_data_dir, val_data_dir, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name, block_size=args.seq_len, _norm_class="RMSNorm")

    train_dataloader, val_dataloader = create_dataloaders(
        batch_size=micro_batch_size,
        block_size=config.block_size,
        fabric=fabric,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        seed=3407,
    )
    if val_dataloader is None:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    else:
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.seed_everything(3407)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights, n_layer=config.n_layer))

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)

    # Use Adam (not AdamW) with no weight decay to match UnTrac paper
    if weight_decay > 0:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay,
            betas=(beta1, beta2), foreach=False
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate,
            betas=(beta1, beta2), foreach=False
        )
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        import re
        def extract_number(filename):
            match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
            return int(match.group(1)) if match else 0
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            resume = False
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric, state, train_dataloader, val_dataloader, monitor, resume):
    model = state["model"]
    optimizer = state["optimizer"]

    with torch.device("meta"):
        meta_model = TransEncoder(model.config)
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
        del meta_model, x

    total_lengths = 0
    total_t0 = time.perf_counter()

    initial_iter = state["iter_num"]
    curr_iter = 0

    loss_func = CrossEntropyLoss(reduction='none')
    for train_data in train_dataloader:
        # Resume loader state
        if resume:
            if curr_iter < initial_iter:
                curr_iter += 1
                continue
            else:
                resume = False
                curr_iter = -1
                fabric.barrier()
                fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
        if state["iter_num"] >= max_iters:
            break

        # Set learning rate
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_t0 = time.perf_counter()

        input_ids = train_data[:, 0 : model.config.block_size].contiguous()
        noisy_input, mask_indices, p_mask = forward_process(input_ids)
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(noisy_input)
            loss = loss_func(logits[mask_indices], input_ids[mask_indices]) / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])

            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            if grad_clip > 0:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        state["iter_num"] += 1
        total_lengths += input_ids.size(1)
        t1 = time.perf_counter()
        fabric.print(
            f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
            f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours."
        )

        monitor.on_train_batch_end(
            state["iter_num"] * micro_batch_size,
            t1 - total_t0,
            fabric.world_size,
            state["step_count"],
            flops_per_batch=estimated_flops,
            lengths=total_lengths,
            train_loss=loss.item()
        )

        if val_dataloader is not None and not is_accumulating and (state["step_count"] % eval_step_interval == 0 or state["step_count"] == max_step):
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss:.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.log_dict({"metric/val_loss": val_loss.item()}, state["step_count"])
            fabric.barrier()

        if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k, val_data in enumerate(val_dataloader):
        if k >= eval_iters:
            break

        mc_loss = torch.zeros(128, device=fabric.device)
        for i in range(128):
            input_ids = val_data[:, 0 : model.config.block_size].contiguous()
            noisy_input, mask_indices, p_mask = forward_process(input_ids)
            logits = model(noisy_input)
            loss = torch.nn.functional.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
            mc_loss[i] = loss

        losses[k] = mc_loss.mean().item()

    losses = fabric.all_reduce(losses, reduce_op="mean")
    out = losses.mean()

    model.train()
    return out


def create_dataloader(
    batch_size: int, block_size: int, data_dir: Path, fabric, shuffle: bool = True, seed: int = 12345, split="train"
) -> DataLoader:
    datasets = []
    data_config = train_data_config if split == "train" else val_data_config
    for prefix, _ in data_config:
        filenames = sorted(glob.glob(str(data_dir / f"{prefix}*")))
        if not filenames:
            fabric.print(f"Warning: no files found for prefix '{prefix}' in {data_dir}")
            continue
        random.seed(seed)
        random.shuffle(filenames)

        dataset = PackedDataset(
            filenames,
            n_chunks=8 if split == "train" else 1,
            block_size=block_size,
            shuffle=shuffle,
            seed=seed+fabric.global_rank,
            num_processes=fabric.world_size,
            process_rank=fabric.global_rank,
        )
        datasets.append(dataset)

    if not datasets:
        raise RuntimeError(
            f"No data found at {data_dir}. Make sure you ran scripts/prepare_untrac_data.py to create the dataset."
        )

    weights = [weight for _, weight in data_config if any(
        glob.glob(str(data_dir / f"{prefix}*")) for prefix, _ in [(_, weight)]
    )]
    # Recalculate weights for found datasets only
    weights = [1.0] * len(datasets)
    sum_weights = sum(weights)
    weights = [el / sum_weights for el in weights]

    combined_dataset = CombinedDataset(datasets=datasets, seed=seed, weights=weights)

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def create_dataloaders(
    batch_size: int,
    block_size: int,
    fabric,
    train_data_dir: Path = Path("data/untrac"),
    val_data_dir: Optional[Path] = None,
    seed: int = 12345,
) -> Tuple[DataLoader, DataLoader]:
    effective_block_size = block_size + 1
    train_dataloader = create_dataloader(
        batch_size=batch_size,
        block_size=effective_block_size,
        fabric=fabric,
        data_dir=train_data_dir,
        shuffle=True,
        seed=seed,
        split="train"
    )
    val_dataloader = (
        create_dataloader(
            batch_size=batch_size,
            block_size=effective_block_size,
            fabric=fabric,
            data_dir=val_data_dir,
            shuffle=False,
            seed=seed,
            split="validation"
        )
        if val_data_dir
        else None
    )
    return train_dataloader, val_dataloader


# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup()
