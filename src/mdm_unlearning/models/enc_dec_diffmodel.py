"""Encoder-Decoder Masked Diffusion Model with block-level cross-attention.

Based on E2D2 (Arriola et al., NeurIPS 2025). Key features:
- Block-causal encoder: block i attends to blocks 0..i only
- Offset block-causal decoder: cross-attends to encoder blocks 0..i-1,
  self-attends within decoder block i only
- No information leakage: decoder never sees the clean version of tokens
  in its own block through the encoder

Uses SDPA with custom masks (not flash_attn) for block-level masking.
"""
import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_func
from xformers.ops import SwiGLU

from mdm_unlearning.models.config import Config
from mdm_unlearning.models.diffmodel import (
    LLaMAMLP, SelfAttention,
    build_rope_cache,
    RoPECache, KVCache,
)

FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


# ============================================================
# Block mask utilities
# ============================================================

def make_block_causal_mask(seq_len: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Block-causal mask: block i attends to blocks 0..i.
    Returns: (seq_len, seq_len) bool tensor, True = attend.
    """
    blocks = torch.arange(seq_len, device=device) // block_size
    return blocks.unsqueeze(0) >= blocks.unsqueeze(1)


def make_decoder_cross_mask(
    T_dec: int, T_enc: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """Decoder cross-self-attention mask.
    - Cross-attn to encoder: block i attends to encoder blocks 0..i-1 (offset)
    - Self-attn in decoder: block i attends to decoder block i only (diagonal)
    Returns: (T_dec, T_enc + T_dec) bool tensor, True = attend.
    """
    q_blocks = torch.arange(T_dec, device=device) // block_size
    enc_blocks = torch.arange(T_enc, device=device) // block_size
    dec_blocks = torch.arange(T_dec, device=device) // block_size

    # Cross-attention: offset block-causal (strictly previous blocks)
    enc_part = q_blocks.unsqueeze(1) > enc_blocks.unsqueeze(0)   # (T_dec, T_enc)
    # Self-attention: same block only
    dec_part = q_blocks.unsqueeze(1) == dec_blocks.unsqueeze(0)  # (T_dec, T_dec)

    return torch.cat([enc_part, dec_part], dim=1)  # (T_dec, T_enc + T_dec)


# ============================================================
# Attention with custom mask (SDPA-based)
# ============================================================

class MaskedSelfAttention(nn.Module):
    """Self-attention using SDPA with custom boolean mask."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        self.attn = nn.Linear(config.n_embd, shape, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.config = config

    def forward(self, x: torch.Tensor, rope: RoPECache,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.size()
        hs = self.config.head_size
        n_h = self.config.n_head
        n_kv = self.config.n_query_groups
        q_per_kv = n_h // n_kv

        qkv = self.attn(x)
        qkv = qkv.view(B, T, n_kv, q_per_kv + 2, hs)
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)
        q = q.reshape(B, T, n_h, hs)
        k = k.reshape(B, T, n_kv, hs)
        v = v.reshape(B, T, n_kv, hs)

        cos, sin = rope
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        # SDPA expects (B, nheads, T, hs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if n_h != n_kv:
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        scale = 1.0 / math.sqrt(hs)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scale
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class CrossSelfAttention(nn.Module):
    """Q/KV split attention with custom mask via SDPA."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.q_proj = nn.Linear(config.n_embd, config.n_head * config.head_size, bias=False)
        kv_size = 2 * config.n_query_groups * config.head_size
        self.kv_proj = nn.Linear(config.n_embd, kv_size, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.config = config

    def forward(
        self,
        x_full: torch.Tensor,       # (B, T_enc + T_dec, C)
        rope_q: RoPECache,
        rope_kv: RoPECache,
        q_start_idx: int,
        attn_mask: Optional[torch.Tensor] = None,  # (T_dec, T_enc + T_dec)
    ) -> torch.Tensor:
        B, T_full, C = x_full.size()
        T_dec = T_full - q_start_idx
        hs = self.config.head_size
        n_h = self.config.n_head
        n_kv = self.config.n_query_groups
        q_per_kv = n_h // n_kv

        x_q = x_full[:, q_start_idx:]
        q = self.q_proj(x_q).view(B, T_dec, n_h, hs)

        kv = self.kv_proj(x_full).view(B, T_full, n_kv, 2, hs)
        k, v = kv.unbind(dim=3)

        cos_q, sin_q = rope_q
        cos_kv, sin_kv = rope_kv
        q = apply_rotary_emb_func(q, cos_q, sin_q, False, True)
        k = apply_rotary_emb_func(k, cos_kv, sin_kv, False, True)

        # SDPA: (B, nheads, L, hs)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if n_h != n_kv:
            k = k.repeat_interleave(q_per_kv, dim=1)
            v = v.repeat_interleave(q_per_kv, dim=1)

        scale = 1.0 / math.sqrt(hs)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scale
        )
        y = y.transpose(1, 2).reshape(B, T_dec, C)
        return self.proj(y)


# ============================================================
# Blocks
# ============================================================

class EncoderBlock(nn.Module):
    """Encoder block with block-causal masked self-attention."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = MaskedSelfAttention(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

    def forward(self, x: torch.Tensor, rope: RoPECache,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.attn(self.norm_1(x), rope, attn_mask=attn_mask)
        x = x + h
        x = x + self.mlp(self.norm_2(x))
        return x


class DecoderBlock(nn.Module):
    """Decoder block with block-masked cross-self-attention."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CrossSelfAttention(config)
        self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)

    def forward(
        self,
        enc_hidden: torch.Tensor,
        dec_x: torch.Tensor,
        rope_q: RoPECache,
        rope_kv: RoPECache,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T_enc = enc_hidden.size(1)
        full_x = torch.cat([enc_hidden, dec_x], dim=1)
        normed = self.norm_1(full_x)
        h = self.attn(normed, rope_q, rope_kv, q_start_idx=T_enc, attn_mask=attn_mask)
        dec_x = dec_x + h
        dec_x = dec_x + self.mlp(self.norm_2(dec_x))
        return dec_x


# ============================================================
# Main model
# ============================================================

class TransEncoderDecoder(nn.Module):
    """Encoder-Decoder MDM with block-level masking.

    block_size controls the granularity of the block-level attention masks.
    Encoder uses block-causal attention. Decoder uses offset-block-causal
    cross-attention + block-diagonal self-attention.
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.wte = nn.Embedding(config.padded_vocab_size + 1, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)

        n_enc = getattr(config, 'n_encoder_layers', 8)
        n_dec = getattr(config, 'n_decoder_layers', 4)
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(n_enc)])
        self.encoder_norm = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(n_dec)])
        self.decoder_norm = config.norm_class(config.n_embd, eps=config.norm_eps)

        self.rope_cache: Optional[RoPECache] = None
        self.n_encoder_layers = n_enc
        self.n_decoder_layers = n_dec

        # Block-level masks (precomputed, registered as buffers)
        self._enc_mask: Optional[torch.Tensor] = None
        self._dec_mask: Optional[torch.Tensor] = None
        self.diffusion_block_size = getattr(config, 'diffusion_block_size', 128)

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, (LLaMAMLP, CrossSelfAttention,
                                                               MaskedSelfAttention, SelfAttention))) or \
               (name == "w3.weight" and isinstance(module, SwiGLU)):
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd) / n_layer)

    def _get_masks(self, T: int, device: torch.device):
        """Lazily create and cache block-level attention masks."""
        if self._enc_mask is None or self._enc_mask.size(0) != T:
            bs = self.diffusion_block_size
            self._enc_mask = make_block_causal_mask(T, bs, device)
            self._dec_mask = make_decoder_cross_mask(T, T, bs, device)
        return self._enc_mask, self._dec_mask

    def forward(self, clean_idx: torch.Tensor, noisy_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clean_idx: (B, T) clean token ids
            noisy_idx: (B, T) noisy token ids (with mask tokens, per-block t)
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = noisy_idx.size()

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(noisy_idx)
        cos, sin = self.rope_cache

        enc_mask, dec_mask = self._get_masks(T, noisy_idx.device)

        # Encoder: block-causal attention on clean tokens
        enc_x = self.wte(clean_idx)
        rope_enc = (cos[:T], sin[:T])
        for block in self.encoder:
            enc_x = block(enc_x, rope_enc, attn_mask=enc_mask)
        enc_x = self.encoder_norm(enc_x)

        # Decoder: cross-self-attention with block masks
        dec_x = self.wte(noisy_idx)
        rope_q = (cos[T:2*T], sin[T:2*T])
        rope_kv = (cos[:2*T], sin[:2*T])
        for block in self.decoder:
            dec_x = block(enc_x, dec_x, rope_q, rope_kv, attn_mask=dec_mask)
        dec_x = self.decoder_norm(dec_x)

        return self.lm_head(dec_x)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size * 2,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )


# ============================================================
# Block-level forward process for training
# ============================================================

def forward_process_block(batch, block_size=128, total_dim=32000, eps=1e-3):
    """Forward process with per-block timestep sampling.

    Each block of block_size tokens gets its own t ~ U[0,1].
    Returns noisy batch, mask indices, and per-token mask probabilities.
    """
    b, l = batch.shape
    n_blocks = l // block_size
    assert l % block_size == 0, f"seq_len {l} must be divisible by block_size {block_size}"

    # Sample t per block
    t = torch.rand((b, n_blocks), device=batch.device)
    p_mask = (1 - eps) * t + eps  # (b, n_blocks)

    # Expand to per-token
    p_mask = p_mask.unsqueeze(-1).expand(b, n_blocks, block_size).reshape(b, l)

    mask_indices = torch.rand((b, l), device=batch.device) < p_mask
    noisy_batch = torch.where(mask_indices, total_dim, batch)
    return noisy_batch, mask_indices, p_mask
