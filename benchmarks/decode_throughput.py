"""End-to-end decoding throughput: Full-KV vs JSSA in the same vLLM stack.

Backends (one per invocation):
  fkv       dense attention (vLLM default backend), the Full-KV baseline
  jssa      JSSA on a GQA model (Qwen3, Llama, Ministral, Gemma 4)
  jssa_mla  JSSA on an MLA model (GLM-4.7-Flash, GLM-5.2, DeepSeek-V3.x)
  dsa       the model's native DeepSeek Sparse Attention (GLM-5.2, DeepSeek-V3.2)

Protocol (paper, Appendix A.7): CUDA graphs on, batch of B random prompts of
``--context`` tokens, 128 new tokens with ignore_eos, prefix caching off. The
prefill time is removed with a two-point measurement: T1 = generate 1 token,
T = generate 1 + D tokens, decode time = T - T1. One warm-up pair, then
``--reps`` measured pairs; reported numbers are means over the pairs.
  per-stream tok/s = D / (T - T1),  aggregate tok/s = B * D / (T - T1)

Examples:
  python decode_throughput.py --backend fkv  --model Qwen/Qwen3-8B --context 131072 --batch 8
  python decode_throughput.py --backend jssa --model Qwen/Qwen3-8B --context 131072 --batch 8 \\
      --basis bases/qwen3_8b.pt --rank 32 --budget 2048
  python decode_throughput.py --backend jssa --model google/gemma-4-26b-a4b-it --rank 64 ...
  python decode_throughput.py --backend jssa_mla --model zai-org/GLM-4.7-Flash --rank 64 ...
"""
import argparse
import json
import os
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def yarn_overrides(model: str) -> dict:
    """Qwen3: YaRN factor 4 (32k -> 128k), as used for serving and calibration."""
    if "qwen3" not in model.lower():
        return {}
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model)
    rp = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    if rp.get("rope_type", "default") != "default":
        return {}
    theta = rp.get("rope_theta", getattr(cfg, "rope_theta", None) or 1000000)
    return {"rope_parameters": {"rope_theta": theta, "rope_type": "yarn", "factor": 4.0,
                                "original_max_position_embeddings": 32768}}


def build_llm(a) -> LLM:
    kw = dict(
        model=a.model,
        max_model_len=a.context + a.decode_len + 16,
        gpu_memory_utilization=a.gpu_mem,
        enforce_eager=a.eager,
        tensor_parallel_size=a.tp,
        enable_prefix_caching=False,
        max_num_seqs=a.max_num_seqs or a.batch,
        # keep prefill and decode in separate steps (steady-state decode)
        enable_chunked_prefill=a.chunked_prefill,
        trust_remote_code=True,
    )
    ov = yarn_overrides(a.model)
    if ov:
        kw["hf_overrides"] = ov
    ac: dict = {}
    if a.backend == "jssa":
        assert a.basis, "--basis is required for jssa"
        kw["kv_cache_dtype"] = "jssa"
        ac = {"backend": "JSSA", "jssa_basis_path": a.basis, "jssa_budget": a.budget,
              "jssa_rank": a.rank, "jssa_fp8_projk": a.fp8_projk,
              "jssa_indexed_attend": a.indexed_attend}
        # Hybrid models (Gemma 4): sliding-window layers keep the standard
        # backend and a window-bounded cache (also applied automatically).
        kw["kv_cache_dtype_skip_layers"] = ["sliding_window"]
    elif a.backend == "jssa_mla":
        assert a.basis, "--basis is required for jssa_mla"
        ac = {"jssa_mla": True, "jssa_basis_path": a.basis, "jssa_budget": a.budget,
              "jssa_rank": a.rank}
        kw["kv_cache_dtype"] = a.mla_kv_dtype
    elif a.backend == "dsa":
        # Native DSA indexer (built automatically for v3.2-style checkpoints).
        kw["kv_cache_dtype"] = a.mla_kv_dtype
    if a.mla_backend:
        ac["backend"] = a.mla_backend
    if ac:
        kw["attention_config"] = ac
    if a.cudagraph_mode and not a.eager:
        kw["compilation_config"] = {"cudagraph_mode": a.cudagraph_mode}
    return LLM(**kw)


def timed_generate(llm, prompts, max_tokens) -> float:
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=["fkv", "jssa", "jssa_mla", "dsa"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--basis", default=None, help="JSSA basis .pt (jssa_calibrate.py)")
    ap.add_argument("--rank", type=int, default=32, help="JSSA rank r (64 for Gemma 4 / MLA)")
    ap.add_argument("--budget", type=int, default=2048, help="token budget B")
    ap.add_argument("--fp8_projk", action=argparse.BooleanOptionalAction, default=True,
                    help="GQA: FP8 projected-key cache (default on, as in the paper)")
    ap.add_argument("--indexed_attend", action=argparse.BooleanOptionalAction, default=True,
                    help="GQA: gather-free indexed attention; --no-indexed_attend "
                         "uses gather-then-attend (ablation)")
    ap.add_argument("--context", type=int, default=32768, help="prompt tokens per request")
    ap.add_argument("--decode_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--max_num_seqs", type=int, default=0, help="0 -> batch")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--eager", action="store_true", help="disable CUDA graphs")
    ap.add_argument("--chunked_prefill", action="store_true")
    ap.add_argument("--mla_kv_dtype", default="auto",
                    help="MLA latent KV cache dtype (auto = bf16; fp8_ds_mla for DSA-style fp8)")
    ap.add_argument("--mla_backend", default=None, help="force an MLA attention backend")
    ap.add_argument("--cudagraph_mode", default=None,
                    help="e.g. PIECEWISE (native DSA on GLM-5.2 needs it; use the same "
                         "mode for every backend of that comparison)")
    ap.add_argument("--json_out", default=None, help="append one JSON line with the result")
    a = ap.parse_args()

    torch.manual_seed(0)
    prompts = [TokensPrompt(prompt_token_ids=torch.randint(1000, 30000, (a.context,)).tolist())
               for _ in range(a.batch)]
    llm = build_llm(a)

    for _ in range(a.warmup):
        timed_generate(llm, prompts, 1)
        timed_generate(llm, prompts, 1 + a.decode_len)
    per_stream, agg = [], []
    for _ in range(a.reps):
        t1 = timed_generate(llm, prompts, 1)
        t = timed_generate(llm, prompts, 1 + a.decode_len)
        dec = t - t1
        per_stream.append(a.decode_len / dec)
        agg.append(a.batch * a.decode_len / dec)
    mean_ps = sum(per_stream) / len(per_stream)
    mean_agg = sum(agg) / len(agg)
    print(f"[throughput] model={a.model} backend={a.backend} context={a.context} "
          f"batch={a.batch} budget={a.budget} rank={a.rank}")
    print(f"  DECODE_MS_PER_TOK={1000.0 / mean_ps:.3f}  PER_STREAM_TOK_S={mean_ps:.1f}  "
          f"AGG_TOK_S={mean_agg:.1f}  (reps={a.reps})")
    if a.json_out:
        with open(a.json_out, "a") as f:
            f.write(json.dumps({
                "model": a.model, "backend": a.backend, "context": a.context,
                "batch": a.batch, "budget": a.budget, "rank": a.rank,
                "fp8_projk": a.fp8_projk, "indexed_attend": a.indexed_attend,
                "agg_tok_s": mean_agg, "per_stream_tok_s": mean_ps,
                "agg_tok_s_reps": agg}) + "\n")


if __name__ == "__main__":
    main()
