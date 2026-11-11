#!/usr/bin/env bash
# Run after stopping other GPU 0 workloads. Does not alter the miner or binary.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-v3-profile-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$PROFILE_OUT"
export CUDA_VISIBLE_DEVICES=0
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  -o "$PROFILE_OUT/grid" .venv/bin/python tools/benchmark_v3_grid.py
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum \
  "$PROFILE_OUT/grid.nsys-rep" > "$PROFILE_OUT/systems-summary.txt"
ncu --kernel-name 'regex:.*cute_pearl_fold.*' --launch-skip 10 --launch-count 3 \
  --section SpeedOfLight --section ComputeWorkloadAnalysis \
  --section MemoryWorkloadAnalysis --section Occupancy \
  --section SchedulerStats --section WarpStateStats \
  -o "$PROFILE_OUT/gemm" .venv/bin/python tests/bench_capi.py \
  > "$PROFILE_OUT/compute-summary.txt" 2>&1
printf 'Profiles saved to %s\n' "$PROFILE_OUT"
