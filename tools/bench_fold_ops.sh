#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
export PEARL_TRIAL_LIBRARY=/tmp/libpearl-fold-ops.so
export PEARL_RESULT_PREFIX="${PEARL_RESULT_PREFIX:-/tmp/pearl-fold-ops}"
export PEARL_TRIAL_NAMES='["production baseline","clone sanity","native XOR","native XOR no init/load","native XOR direct output","native XOR register transcript","shuffle register transcript","native XOR coalesced epilogue","shuffle coalesced epilogue","native XOR no init/load coalesced","register transcript s4","register transcript BK64","no init/load coalesced s4","no init/load coalesced BK64","warp4x2 native shared","warp4x2 register s3","warp4x2 register s4","warp4x2 register BK64 s3","warp4x2 coalesced BK64","warp4x2 register BK64 s4","warp2x4 register BK64 s3","warp2x4 register s4","warp1x8 register BK64 s3"]'
if [ "${1:-}" = grid ]; then
  shift
  .venv/bin/python tests/bench_cute_grid.py "$@"
else
  .venv/bin/python tests/bench_cute_variants.py
fi
