#!/bin/bash
# Install vLLM with JSSA: upstream vLLM at a pinned commit + jssa_vllm.patch.
# The patch is Python-only, so vLLM's precompiled binaries for that commit are used
# (no CUDA build).
#
#   bash setup_vllm.sh [target_dir]          # GQA models (Qwen3, Llama, Ministral, Gemma 4)
#   JSSA_MLA=1 bash setup_vllm.sh [dir]      # + MLA models (GLM, DeepSeek): DeepGEMM header patch
#
# Run inside a fresh Python 3.12 environment (the wheel pulls a matching PyTorch).
set -euo pipefail
VLLM_COMMIT=39910f2b25aacc09f5e7f166cdf0030b19f8b9e8
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="${1:-vllm-jssa}"

git clone https://github.com/vllm-project/vllm.git "$DIR"
cd "$DIR"
git checkout "$VLLM_COMMIT"
git apply "$HERE/jssa_vllm.patch"
# Precompiled binaries of the same upstream commit (published for CUDA 13.0 and 12.9;
# set VLLM_PRECOMPILED_WHEEL_VARIANT=cu129 for the latter).
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$VLLM_COMMIT" pip install -e .

if [ "${JSSA_MLA:-0}" = "1" ]; then
  # JSSA-MLA scores with the plain head sum: compile the per-head ReLU out of the
  # DeepGEMM MQA-logits kernels (JIT-compiled at runtime), then drop the JIT cache.
  # Revert (patch -R) before running a model's native DSA indexer.
  patch -p1 -d vllm/third_party/deep_gemm < "$HERE/deepgemm_mqa_logits_no_relu.patch"
  rm -rf "${DG_JIT_CACHE_DIR:-${VLLM_CACHE_ROOT:-$HOME/.cache/vllm}/deep_gemm}"
fi
echo "vLLM with JSSA installed in $(pwd)"
