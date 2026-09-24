"""Attention-stage decode latency of one layer on a synthetic paged KV cache
(no model): Full-KV vs JSSA gather-then-attend vs JSSA gather-free.

Two timings are reported for JSSA:
  attend   attention over the B selected tokens only (top-B indices given):
           gather K/V + FlashAttention, or the gather-free indexed kernel
  total    FP8 projected score over all cached tokens + radix top-B + attend
Full-KV is FlashAttention paged decode over all ``--context`` tokens.

Default dimensions: Llama-3.1-8B (32 query heads, 8 KV heads, d=128), r=32, B=2048.

  python attention_microbench.py --context 131072 --batch 1 4 16
"""
import argparse

import torch

from vllm.v1.attention.ops.triton_jssa_gather import jssa_gather
from vllm.v1.attention.ops.triton_jssa_indexed_attend import jssa_indexed_attend
from vllm.v1.attention.ops.triton_jssa_score_topk import jssa_score_topk
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


def _time_ms(fn, reps, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def run(batch, a):
    dev, dt = "cuda", torch.bfloat16
    d, Hq, Hkv, r, B, bs = a.head_size, a.num_q_heads, a.num_kv_heads, a.rank, a.budget, 16
    blocks_per_req = (a.context + bs - 1) // bs
    num_blocks = batch * blocks_per_req
    kv = torch.randn(num_blocks, bs, Hkv, 2 * d, device=dev, dtype=dt)  # [K | V] slot
    projk = torch.randn(num_blocks, bs, Hkv, r, device=dev).to(torch.float8_e4m3fn)
    bt = torch.randperm(num_blocks, device=dev, dtype=torch.int32).view(batch, blocks_per_req)
    seq_lens = torch.full((batch,), a.context, device=dev, dtype=torch.int32)
    q = torch.randn(batch, Hq, d, device=dev, dtype=dt)
    proj_q = torch.randn(batch, Hkv, r, device=dev, dtype=dt)
    scale = d**-0.5
    out = torch.empty_like(q)
    scores = torch.empty(batch, Hkv, blocks_per_req * bs, device=dev)
    top_idx = torch.empty(batch, Hkv, B, device=dev, dtype=torch.int32)
    top_sc = torch.empty(batch, Hkv, B, device=dev)
    K_sel = torch.empty(batch * B, Hkv, d, device=dev, dtype=dt)
    V_sel = torch.empty_like(K_sel)
    cu_q = torch.arange(batch + 1, device=dev, dtype=torch.int32)
    cu_k = cu_q * B
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    scratch = {}

    def _scratch(name, shape, dtype, device):
        key = (name, tuple(shape), dtype)
        if key not in scratch:
            scratch[key] = torch.empty(shape, dtype=dtype, device=device)
        return scratch[key]

    def select():
        jssa_score_topk(proj_q, kv, bt, seq_lens, B, d, r, scores_out=scores,
                        top_idx_out=top_idx, top_scores_out=top_sc, use_radix=True,
                        projk_cache=projk)

    def fkv():
        flash_attn_varlen_func(q=q, k=kv[..., :d], v=kv[..., d:], out=out,
                               cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=seq_lens,
                               max_seqlen_k=a.context, softmax_scale=scale, causal=True,
                               block_table=bt)

    def gather_attend():
        jssa_gather(kv, bt, top_idx, head_size=d, dtype=dt, K_sel_out=K_sel, V_sel_out=V_sel)
        flash_attn_varlen_func(q=q, k=K_sel, v=V_sel, out=out, cu_seqlens_q=cu_q,
                               max_seqlen_q=1, cu_seqlens_k=cu_k, max_seqlen_k=B,
                               softmax_scale=scale, causal=False)

    def gather_free():
        jssa_indexed_attend(q, kv, bt, top_idx, head_size=d, scale=scale, out=out,
                            scratch=_scratch, sm_count=sm)

    select()  # top_idx for the attend-only timings
    t_fkv = _time_ms(fkv, a.reps)
    t_sel = _time_ms(select, a.reps)
    t_gat = _time_ms(gather_attend, a.reps)
    t_free = _time_ms(gather_free, a.reps)
    print(f"batch={batch:3d} context={a.context}  fkv={t_fkv:.3f}ms  select={t_sel:.3f}ms")
    print(f"   attend: gather={t_gat:.3f}ms ({t_fkv / t_gat:.2f}x)  "
          f"gather-free={t_free:.3f}ms ({t_fkv / t_free:.2f}x)")
    print(f"   total : gather={t_sel + t_gat:.3f}ms ({t_fkv / (t_sel + t_gat):.2f}x)  "
          f"gather-free={t_sel + t_free:.3f}ms ({t_fkv / (t_sel + t_free):.2f}x)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--head_size", type=int, default=128)
    ap.add_argument("--num_q_heads", type=int, default=32)
    ap.add_argument("--num_kv_heads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()
    torch.manual_seed(0)
    for b in a.batch:
        run(b, a)


if __name__ == "__main__":
    main()
