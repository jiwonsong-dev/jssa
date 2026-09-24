# JSSA: Joint Subspace Sparse Attention — supplementary code

This directory contains the implementation used for the paper's system and
calibration experiments:

| directory | contents |
|---|---|
| `vllm_patch/` | JSSA integrated into vLLM, as a patch against a pinned upstream vLLM commit, plus two small patches for the MLA path |
| `calibration/` | basis calibration (closed-form joint-subspace basis from second moments of group-mean queries and keys) |
| `benchmarks/` | end-to-end decoding-throughput benchmark (Figure 4), the gather-free ablation, and an attention-stage microbenchmark |

Paper-to-code map:

| paper | code |
|---|---|
| joint subspace basis, `M = TopEig_r(G_Qm/tr(G_Qm) + G_K/tr(G_K))` | `calibration/jssa_calibrate.py: compute_basis` |
| per-KV-head selection with the group-mean projected query | `vllm/v1/attention/backends/jssa_attn.py: _sparse_decode_forward` |
| FP8 projected-key scoring cache | `vllm/v1/attention/ops/triton_jssa_store.py`, `triton_jssa_score_topk.py` |
| gather-free indexed attention | `vllm/v1/attention/ops/triton_jssa_indexed_attend.py` |
| JSSA on MLA (64 latent + 64 RoPE scoring dims, FlashMLA sparse attention) | `vllm/model_executor/layers/jssa_mla_indexer.py` |

(`vllm/...` paths refer to the patched vLLM tree.)

## 1. Install

Requires Python 3.12 and an NVIDIA driver supporting CUDA 12.9 or 13.0 (the
precompiled vLLM binaries of the pinned commit). The paper's throughput numbers
were measured on a single NVIDIA B200.

```bash
cd vllm_patch
bash setup_vllm.sh vllm-jssa             # GQA models: Qwen3, Llama, Ministral, Gemma 4
JSSA_MLA=1 bash setup_vllm.sh vllm-jssa  # additionally MLA models: GLM, DeepSeek
```

`setup_vllm.sh` clones vLLM at commit `39910f2b25aacc09f5e7f166cdf0030b19f8b9e8`,
applies `jssa_vllm.patch`, and installs it with the precompiled binaries of that
commit (`VLLM_USE_PRECOMPILED=1`, CUDA 13.0 build by default,
`VLLM_PRECOMPILED_WHEEL_VARIANT=cu129` for CUDA 12.9). The patch is
Python/Triton only, so no CUDA build is needed.

The two additional patches only matter for MLA models:

* `deepgemm_mqa_logits_no_relu.patch` (applied by `JSSA_MLA=1`). JSSA-MLA reuses
  DeepGEMM's FP8 paged MQA-logits kernel of DeepSeek Sparse Attention (DSA), whose
  epilogue applies a per-head ReLU that DSA's trained indexer expects. JSSA's
  score is the plain head sum, so the ReLU is compiled out. The kernels are
  JIT-compiled at run time, so the script clears DeepGEMM's JIT cache
  (`$VLLM_CACHE_ROOT/deep_gemm`, or `$DG_JIT_CACHE_DIR` if set). **Revert it
  (`patch -R`) before running a model's native DSA indexer.**
* `flashmla_long_prefill_int64_strides.patch` (optional, not applied by the
  script). vLLM's sparse MLA prefill sends the whole sequence through FlashMLA's
  sparse decode interface; beyond ~32K prompt tokens some strides exceed int32
  and the combine kernel's grid exceeds the CUDA y-limit. Needed for the 64K/128K
  MLA measurements. It applies to FlashMLA commit `a6ec2ba7` (the one vLLM pins)
  and requires building vLLM from source:
  ```bash
  git clone https://github.com/vllm-project/FlashMLA && cd FlashMLA
  git checkout a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b
  git apply ../flashmla_long_prefill_int64_strides.patch
  cd ../vllm-jssa && FLASH_MLA_SRC_DIR=$PWD/../FlashMLA pip install -e . --no-build-isolation
  ```

The JSSA GQA backend runs on any GPU supported by vLLM's Triton backends
(compute capability >= 8.0). The MLA path needs FlashMLA sparse attention and
DeepGEMM (Hopper / Blackwell data-center GPUs).

Kernel unit tests (in the patched tree):
`python -m pytest --noconftest tests/v1/attention/test_jssa_*.py`.

## 2. Calibration

Calibration runs one prefill per calibration sequence with HuggingFace
transformers and records post-RoPE queries and keys; the basis is a single
eigendecomposition per (layer, KV head). No training or iterative optimization.

```bash
cd calibration
pip install "transformers==5.8.0" datasets accelerate
python prepare_longalign.py             # data/longalign_calib.jsonl (64 LongAlign-10k samples)

# eight 131,072-token sequences packed from LongAlign-10k (the paper's recipe)
python jssa_calibrate.py --model meta-llama/Llama-3.1-8B-Instruct   --rank 32 --output ../bases/llama_3_1_8b_instruct.pt
python jssa_calibrate.py --model Qwen/Qwen3-8B                      --rank 32 --output ../bases/qwen3_8b.pt
python jssa_calibrate.py --model Qwen/Qwen3-14B --yarn              --rank 32 --output ../bases/qwen3_14b.pt
python jssa_calibrate.py --model mistralai/Ministral-3-8B-Instruct-2512 --rank 32 --output ../bases/ministral_3_8b_instruct_2512.pt
JSSA_MOE_CHUNK=16384 python jssa_calibrate.py --model google/gemma-4-26b-a4b-it --rank 64 \
    --attn_impl chunked_math --output ../bases/gemma_4_26b_a4b_it.pt
python jssa_calibrate.py --model zai-org/GLM-4.7-Flash --rank 64 --attn_impl sdpa --output ../bases/glm_4_7_flash.pt
```

* The output is `{'M': {layer: Tensor[H_kv, r, d]}}` (MLA: `[1, r, kv_lora_rank]`),
  the file passed to vLLM as `jssa_basis_path`. `--save_grams` also stores the
  second moments.
* `--yarn` (Qwen3-14B) calibrates with the YaRN RoPE (factor 4) used for serving;
  calibration and serving RoPE must match.
* Gemma 4: only the global-attention layers are recorded; the sliding-window
  layers keep dense attention at serving time. `chunked_math` streams attention
  over query blocks (head_dim 512 is not supported by the fused SDPA kernels)
  and `JSSA_MOE_CHUNK` bounds the MoE buffers, so the 128K calibration fits on
  one GPU.
* Large MLA models that need tensor parallelism (GLM-5.2, DeepSeek-V3.x) are
  calibrated inside vLLM with dense attention; the objective and data are the
  same:
  ```bash
  python collect_mla_grams_vllm.py --model <model> --tp 8 --rank 64 --output ../bases/<name>.pt
  ```

## 3. Throughput

```bash
cd benchmarks
# one cell: Qwen3-8B, 128K context, batch 8
python decode_throughput.py --backend fkv  --model Qwen/Qwen3-8B --context 131072 --batch 8
python decode_throughput.py --backend jssa --model Qwen/Qwen3-8B --context 131072 --batch 8 \
    --basis ../bases/qwen3_8b.pt --rank 32 --budget 2048

# Figure 4 grids (32K and 128K, all batch sizes, FKV and JSSA)
bash run_throughput_sweep.sh qwen3  ../bases
bash run_throughput_sweep.sh gemma4 ../bases
bash run_throughput_sweep.sh glm    ../bases

# gather-then-attend vs gather-free (end to end)
EXTRA="--no-indexed_attend" OUT=throughput_gemma4_gather.jsonl bash run_throughput_sweep.sh gemma4 ../bases

# attention stage only (Llama-3.1-8B dimensions, 128K context)
python attention_microbench.py --context 131072 --batch 1 4 16
```

`attention_microbench.py` times one layer on a synthetic paged cache and reports,
against Full-KV FlashAttention decode, (i) the attention over the B selected tokens
alone (gather-then-attend vs. gather-free) and (ii) the total including the FP8
projected score over all cached tokens and the top-B selection.

Measurement protocol (`decode_throughput.py`): CUDA graphs enabled, B random
prompts of the given length, 128 new tokens with `ignore_eos`, prefix caching
off, `max_num_seqs = B`. Prefill time is removed by a two-point measurement
(generate 1 token vs. 1 + 128 tokens); one warm-up pair is followed by three
measured pairs and the mean aggregate throughput `B * 128 / (T_{129} - T_1)` is
reported. The KV cache is BF16; JSSA's projected scoring cache is FP8.

## 4. vLLM configuration reference

GQA models use `kv_cache_dtype="jssa"` and the `JSSA` attention backend:

```python
from vllm import LLM
llm = LLM(model="Qwen/Qwen3-8B", kv_cache_dtype="jssa",
          attention_config={"backend": "JSSA", "jssa_basis_path": "bases/qwen3_8b.pt",
                            "jssa_budget": 2048, "jssa_rank": 32})
```

MLA models keep their latent KV cache and enable the JSSA indexer:

```python
llm = LLM(model="zai-org/GLM-4.7-Flash",
          attention_config={"jssa_mla": True, "jssa_basis_path": "bases/glm_4_7_flash.pt",
                            "jssa_budget": 2048, "jssa_rank": 64})
```

| `attention_config` key | default | meaning |
|---|---|---|
| `jssa_basis_path` | — | calibrated basis file |
| `jssa_budget` | 2048 | token budget B per decode step and KV head |
| `jssa_rank` | head_size // 4 (MLA: 64) | subspace rank r; must match the basis |
| `jssa_fp8_projk` | True | GQA: FP8 projected-key cache (False: BF16 proj_K in the KV slot) |
| `jssa_indexed_attend` | True | GQA: gather-free indexed attention (False: gather-then-attend) |
| `jssa_mla` | False | enable JSSA on an MLA model |
| `jssa_mla_collect_grams` | False | calibration mode used by `collect_mla_grams_vllm.py` |

Notes:
* Prefill and mixed prefill/decode steps use dense attention; selection applies
  to decode steps. Layers absent from the basis file, and sliding-window layers,
  use dense attention.
* The GQA FP8 projected-key cache (r bytes per token and KV head, 6.25% of the
  BF16 KV cache at r = 32, d = 128) is allocated outside vLLM's KV-cache memory
  budget; leave that much headroom in `gpu_memory_utilization`.

## License

The vLLM patch is a derivative of vLLM (Apache-2.0). The remaining code is
released under the Apache-2.0 license.
