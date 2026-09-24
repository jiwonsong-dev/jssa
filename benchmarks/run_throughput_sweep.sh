#!/bin/bash
# Decoding-throughput sweep of Figure 4 (single GPU, B = 2048, 32K and 128K context).
# Each (model, context, batch, backend) cell runs in its own process; results are
# appended to $OUT as JSON lines.
#
#   bash run_throughput_sweep.sh qwen3   BASES_DIR   # Qwen3-8B          (GQA)
#   bash run_throughput_sweep.sh gemma4  BASES_DIR   # Gemma4-26B-A4B-It (hybrid)
#   bash run_throughput_sweep.sh glm     BASES_DIR   # GLM-4.7-Flash     (MLA)
#
# BASES_DIR holds the calibrated bases (qwen3_8b.pt, gemma_4_26b_a4b_it.pt,
# glm_4_7_flash.pt; see ../calibration). Ablation: EXTRA="--no-indexed_attend"
# measures gather-then-attend instead of the gather-free kernel.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KEY=${1:?model key: qwen3 | gemma4 | glm}
BASES=${2:?directory with calibrated bases}
OUT=${OUT:-throughput_${KEY}.jsonl}
PY=${PYTHON:-python}
BUDGET=${BUDGET:-2048}
EXTRA=${EXTRA:-}

case $KEY in
  qwen3)  MODEL=Qwen/Qwen3-8B;             SPARSE=jssa;     RANK=32
          BASIS=$BASES/qwen3_8b.pt;        B32="1 2 4 8 16 32";    B128="1 2 4 8" ;;
  gemma4) MODEL=google/gemma-4-26b-a4b-it; SPARSE=jssa;     RANK=64
          BASIS=$BASES/gemma_4_26b_a4b_it.pt; B32="1 2 4 8 16 32 64"; B128="1 2 4 8 16" ;;
  glm)    MODEL=zai-org/GLM-4.7-Flash;     SPARSE=jssa_mla; RANK=64
          BASIS=$BASES/glm_4_7_flash.pt;   B32="1 2 4 8 16 32 64"; B128="1 2 4 8 16" ;;
  *) echo "unknown model key $KEY"; exit 1 ;;
esac

for CTX in 32768 131072; do
  if [ "$CTX" = 32768 ]; then BATCHES=$B32; else BATCHES=$B128; fi
  for BS in $BATCHES; do
    for BE in fkv $SPARSE; do
      echo "=== $MODEL ctx=$CTX batch=$BS backend=$BE"
      $PY "$HERE/decode_throughput.py" --backend $BE --model $MODEL --basis "$BASIS" \
        --rank $RANK --budget $BUDGET --context $CTX --batch $BS --json_out "$OUT" $EXTRA \
        || echo "FAILED: ctx=$CTX batch=$BS backend=$BE (e.g. out of memory)"
    done
  done
done
echo "results in $OUT"
