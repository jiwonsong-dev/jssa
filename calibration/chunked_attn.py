"""`chunked_math` attention implementation, for calibrating large-head_dim models
at full context.

Motivation: Gemma4-26B-A4B's full_attention layers use `global_head_dim=512`
(the config's top-level `head_dim=256` describes only its 25 sliding layers).
The fused SDPA backends we used do not accept head_dim 512 (flash caps at
256, mem-efficient at 128), so PyTorch falls back to the math kernel, which materialises the full [B, H, Lq, Lk] score matrix:
16 heads x 131072^2 x 4 B = **1.1 TB** at 128k. On top of that, the eager/sdpa
mask path builds a dense `[1, 1, L, L]` bf16 hybrid mask (32 GiB at 128k).

This module removes both:
  * `chunked_math_attention` streams over the QUERY axis in blocks of
    `JSSA_ATTN_CHUNK` (default 1024), so peak score memory is
    H x chunk x slab x 4 B regardless of Lq.
  * `no_mask` is registered as the mask builder for this implementation, so
    transformers never materialises the dense mask. Causality and the
    sliding window are applied per chunk from position indices instead.

Each chunk materialises only the key SLAB it can legally attend to, rather than
the full Lk width followed by a mask. That is ~2x on causal layers and
Lk/(chunk + W) on sliding ones: at 128k with window=1024 it is a 64x cut on
Gemma4's 25 sliding layers, which otherwise dominate calibration wall-clock.

Numerics match the eager path (same fp32 softmax, same scaling) to ~1 ulp of
fp32 — the slab reduces a different number of (exactly-zero) terms, so the
summation tree differs; the attended-key set is identical. Only used for
calibration (`jssa_calibrate.py --attn_impl chunked_math`), never at deployment.
"""
from __future__ import annotations

import os

import torch

from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
from transformers.modeling_utils import AttentionInterface

_CHUNK = int(os.environ.get("JSSA_ATTN_CHUNK", "1024"))


def chunked_math_attention(
    module,
    query: torch.Tensor,          # [B, Hq, Lq, D]
    key: torch.Tensor,            # [B, Hkv, Lk, D]
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    **kwargs,
):
    B, Hq, Lq, D = query.shape
    Hkv, Lk = key.shape[1], key.shape[2]
    grp = Hq // Hkv
    if scaling is None:
        scaling = D ** -0.5

    # Broadcast the KV heads instead of repeat_kv: at 128k, materialising K and V
    # at Hq heads costs 2 x Hq x Lk x D x 2 B (4.3 GiB for Gemma4), and the point
    # here is to stay inside one GPU next to a 52 GB model.
    qg = query.view(B, Hkv, grp, Lq, D)
    k5 = key.unsqueeze(2)                     # [B, Hkv, 1, Lk, D]
    v5 = value.unsqueeze(2)
    og = torch.empty_like(qg)

    q_off = Lk - Lq                           # aligns query t with key position

    for i in range(0, Lq, _CHUNK):
        j = min(i + _CHUNK, Lq)
        # Materialise ONLY the key slab this chunk can legally attend to. Causality caps
        # it at the chunk's last query position; a sliding window also floors it. Building
        # the full [chunk, Lk] block and masking afterwards costs 2x on causal layers and
        # Lk/(chunk+W) on sliding ones -- at 128k with window=1024 that is ~64x wasted
        # compute and HBM traffic on Gemma4's 25 sliding layers. The dropped entries were filled with finfo.min, whose
        # exp underflows to exactly 0, so no attended key is lost -- but the softmax then
        # reduces a different NUMBER of terms, which reshuffles the pairwise-summation
        # tree. Agreement with the full-width block is therefore ~1 ulp of fp32, not bit-
        # exact (measured worst case 2.7 ulp over 450 configs).
        k_hi = min(Lk, j + q_off)
        k_lo = max(0, i + q_off - sliding_window + 1) if sliding_window is not None else 0
        ks = slice(k_lo, k_hi)
        w = torch.matmul(qg[:, :, :, i:j], k5[:, :, :, ks].transpose(-1, -2)) * scaling
        qpos = torch.arange(i, j, device=query.device) + q_off
        kpos = torch.arange(k_lo, k_hi, device=query.device)
        bad = kpos[None, :] > qpos[:, None]                       # causal
        if sliding_window is not None:
            bad |= kpos[None, :] <= qpos[:, None] - sliding_window
        w = w.masked_fill(bad[None, None, None], torch.finfo(w.dtype).min)
        if attention_mask is not None:                            # padding, if any
            w = w + attention_mask[:, :, i:j, ks].view(B, -1, 1, j - i, k_hi - k_lo)
        w = torch.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        og[:, :, :, i:j] = torch.matmul(w, v5[:, :, :, ks])
        del w

    out = og.view(B, Hq, Lq, D)
    return out.transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):
    """chunked_math_attention applies causal/sliding masking itself, so skip the
    dense [1, 1, L, L] mask entirely (32 GiB at 128k)."""
    return None


AttentionInterface.register("chunked_math", chunked_math_attention)
ALL_MASK_ATTENTION_FUNCTIONS.register("chunked_math", _no_mask)
