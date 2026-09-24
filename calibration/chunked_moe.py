"""Token-chunked grouped-MoE experts forward, for calibrating MoE models at full context.

At 131k tokens, `transformers/integrations/moe.py:grouped_mm_experts_forward` materializes
several (num_tokens * top_k, hidden) buffers at once (~11 GiB for Gemma4-26B-A4B), which is
what keeps a 128k calibration of that model from fitting on one GPU. Experts act on each
token independently (routing, grouping and un-permutation are computed per call from the
call's own rows), so running the original function over slices of the token axis and
concatenating is exactly equivalent, with peak memory scaling as chunk / num_tokens.

Enable with `JSSA_MOE_CHUNK=<tokens>` (0 or unset disables). Calibration only.
"""
from __future__ import annotations

import os

import torch

from transformers.integrations.moe import (
    ALL_EXPERTS_FUNCTIONS,
    grouped_mm_experts_forward as _orig_grouped_mm,
)

_installed = False


def chunked_grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights,
                                       *args, **kwargs):
    """`grouped_mm_experts_forward` applied over slices of the token axis."""
    chunk = int(os.environ.get("JSSA_MOE_CHUNK", "0"))
    n = hidden_states.size(0)
    if chunk <= 0 or n <= chunk:
        return _orig_grouped_mm(self, hidden_states, top_k_index, top_k_weights,
                                *args, **kwargs)
    out = None
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        part = _orig_grouped_mm(self, hidden_states[i:j], top_k_index[i:j],
                                top_k_weights[i:j], *args, **kwargs)
        if out is None:                      # allocate once, in the part's dtype/device
            out = torch.empty((n,) + tuple(part.shape[1:]), dtype=part.dtype,
                              device=part.device)
        out[i:j] = part
        del part
    return out


def install() -> bool:
    """Route `grouped_mm` through the chunked wrapper. No-op unless JSSA_MOE_CHUNK > 0.

    Overrides the registry entry rather than plumbing a new `_experts_implementation`
    through the config, because transformers resolves that name at module-construction
    time and models pick it themselves; overriding the name the model already asks for is
    the only interception point that does not depend on load order.
    """
    global _installed
    if _installed:
        return True
    chunk = int(os.environ.get("JSSA_MOE_CHUNK", "0"))
    if chunk <= 0:
        return False
    ALL_EXPERTS_FUNCTIONS.register("grouped_mm", chunked_grouped_mm_experts_forward)
    _installed = True
    print(f"[chunked_moe] grouped_mm experts run over {chunk}-token chunks")
    return True
