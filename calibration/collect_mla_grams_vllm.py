"""JSSA-MLA calibration inside vLLM, for MLA models that need tensor parallelism
(GLM-5.2, DeepSeek-V3.x), which transformers cannot load on a single node.

The model runs with DENSE MLA attention (``jssa_mla_collect_grams``; any native
DSA indexer is disabled) and eager execution. A collector on every MLA layer
accumulates, over the prefill tokens of the calibration sequences,

    G_Qm = sum_t q_bar_t q_bar_t^T,  q_bar_t = mean over all heads of W_UK^T q_nope
    G_K  = sum_t c_t c_t^T           (normed latent c_KV)

(per-rank head sums are all-reduced, so every rank holds the full-head G_Qm).
The basis is M = TopEig_r(G_Qm/tr + G_K/tr), saved as {'M': {layer: [1, r, kv_lora]}}
for ``attention_config.jssa_basis_path`` with ``jssa_mla=True``. This is the same
objective and data as jssa_calibrate.py (the HuggingFace path used for
GLM-4.7-Flash).

  python collect_mla_grams_vllm.py --model zai-org/GLM-5.2-FP8 --tp 8 \\
      --output bases/glm_5_2.pt
"""
import argparse
import os
import sys
import time

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
# collective_rpc ships the two helper functions below to the workers.
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from jssa_calibrate import (  # noqa: E402
    compute_basis,
    load_longalign_packed,
    tokenize_truncate,
)


def _collectors(worker):
    from vllm.model_executor.layers.jssa_mla_indexer import JSSAMLAGramCollector

    model = worker.model_runner.model
    return [m for m in model.modules() if isinstance(m, JSSAMLAGramCollector)]


def _reset(worker):
    cs = _collectors(worker)
    for c in cs:
        c.reset()
    return len(cs)


def _dump(worker, path):
    """Save the grams from TP rank 0 (identical on every rank) to ``path``;
    results are written to disk rather than returned through the RPC."""
    from vllm.distributed import get_tensor_model_parallel_rank

    grams = {c.layer_idx: (c.g_q.cpu(), c.g_k.cpu(), c.num_tokens)
             for c in _collectors(worker) if c.g_q is not None}
    if get_tensor_model_parallel_rank() == 0:
        torch.save(grams, path)
    return len(grams)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=131072)
    ap.add_argument("--data", default=os.path.join(_HERE, "data", "longalign_calib.jsonl"))
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_num_batched_tokens", type=int, default=16384)
    ap.add_argument("--save_grams", default=None)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(
        model=a.model,
        tensor_parallel_size=a.tp,
        max_model_len=a.max_len + 16,
        gpu_memory_utilization=a.gpu_mem,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_num_batched_tokens=a.max_num_batched_tokens,
        max_num_seqs=1,
        kv_cache_dtype="auto",
        attention_config={"jssa_mla_collect_grams": True},
        trust_remote_code=True,
    )
    # Drop anything accumulated by profiling / warm-up forwards.
    n_layers = llm.collective_rpc(_reset)[0]
    print(f"[calib] {n_layers} MLA layers with collectors")

    sp = SamplingParams(temperature=0.0, max_tokens=1)
    t0 = time.time()
    for idx in range(a.num_samples):
        prompt = load_longalign_packed(a.data, idx, int(a.max_len * 1.15))
        ids = tokenize_truncate(tokenizer, prompt, a.max_len)[0].tolist()
        print(f"[calib] sample {idx}: {len(ids)} tokens")
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False)
    print(f"[calib] {a.num_samples} prefills in {time.time() - t0:.1f}s")

    dump_path = os.path.abspath(a.output) + ".grams_tmp.pt"
    llm.collective_rpc(_dump, args=(dump_path,))
    grams = torch.load(dump_path, weights_only=False)
    os.remove(dump_path)
    G_Qm = {ly: g[0].unsqueeze(0) for ly, g in grams.items()}  # [1, kv_lora, kv_lora]
    G_K = {ly: g[1].unsqueeze(0) for ly, g in grams.items()}
    ntok = {ly: g[2] for ly, g in grams.items()}
    print(f"[calib] grams for {len(G_K)} layers, tokens/layer = {sorted(set(ntok.values()))}")
    if a.save_grams:
        torch.save({"G_Qm": G_Qm, "G_K": G_K, "model": a.model}, a.save_grams)
    M = compute_basis(G_Qm, G_K, a.rank)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    torch.save({"M": M, "rank": a.rank, "model": a.model,
                "objective": "TopEig(G_Qm/tr + G_K/tr)"}, a.output)
    k0 = sorted(M)[0]
    print(f"[calib] wrote {a.output}: {len(M)} layers, M[{k0}] {tuple(M[k0].shape)}")


if __name__ == "__main__":
    main()
