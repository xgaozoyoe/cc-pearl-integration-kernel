#!/usr/bin/env bash
# Restricted hardware counters on this machine require sudo. GPU 0 only.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-mapping-reuse-profile}"
mkdir -p "$PROFILE_OUT"
export CUDA_VISIBLE_DEVICES=0
export PEARL_TRIAL_LIBRARY="${PEARL_TRIAL_LIBRARY:-/tmp/libpearl-pure-gemm.so}"
read -r -a SELECTED_VARIANTS <<< "${PROFILE_VARIANTS:-67 68 71}"
METRICS='gpu__time_duration.sum,dram__bytes.sum,lts__t_bytes.sum,sm__sass_data_bytes_mem_shared_op_ldsm.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active'
for VARIANT in "${SELECTED_VARIANTS[@]}"; do
  [[ "$VARIANT" =~ ^[0-9]+$ ]] || { printf 'Invalid variant: %s\n' "$VARIANT" >&2; exit 1; }
  REPORT="$PROFILE_OUT/warm-v$VARIANT"
  /usr/local/cuda/bin/ncu --devices 0 --clock-control base \
    --replay-mode application --cache-control none \
    --kernel-name 'regex:.*cute_pearl_fold.*' \
    --launch-skip 32 --launch-count 3 --metrics "$METRICS" \
    -o "$REPORT" .venv/bin/python tests/profile_fold_traffic.py "$VARIANT" \
    > "$REPORT.log" 2>&1
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page raw --csv > "$REPORT.csv"
done
# Each unpadded 4096^3 region executes 16,777,216 warp-level m16n8k32
# MMA instructions (4096^3 / 4096 MACs). Divide by measured layer bytes
# to obtain MMA/byte. Treat L2/DRAM totals as all traffic, not input only.
# 256x192 variants 82-91 execute 17,301,504 due to N padding to 4224;
# report both issued-MMA/byte and useful-MMA/byte for padded candidates.
printf 'Reports: %s\n' "$PROFILE_OUT"
