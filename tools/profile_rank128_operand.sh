#!/usr/bin/env bash
# Offline GPU0 only: R128 baseline vs row-paired MMA candidate.
# Requires counter access. Validated for Nsight Compute2024.3.2 CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-rank128-operand-profile}"
mkdir -p "$PROFILE_OUT"
METRICS='gpu__time_duration.sum,dram__bytes.sum,lts__t_bytes.sum,sm__sass_data_bytes_mem_shared_op_ldsm.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active'
for VARIANT in 0 1; do
  REPORT="$PROFILE_OUT/v$VARIANT"
  if [[ ! -s "$REPORT.ncu-rep" ]]; then
  /usr/local/cuda/bin/ncu --devices 0 --clock-control none \
    --replay-mode application --cache-control none --kernel-name-base demangled \
    --kernel-name 'regex:.*rank128_operand.*::kernel.*' \
    --launch-skip 32 --launch-count 3 --metrics "$METRICS" \
    --section SourceCounters --section SchedulerStats --section WarpStateStats \
    --warp-sampling-interval auto --warp-sampling-max-passes 5 --import-source yes \
    -o "$REPORT" /home/xgao/pearl-official/.venv/bin/python tests/bench_rank128_operand.py profile --variant "$VARIANT" \
    > "$REPORT.log" 2>&1
  else
    printf 'Reusing existing report: %s\n' "$REPORT.ncu-rep"
  fi
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page raw --print-units base --csv > "$REPORT.raw.csv"
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page source --print-source sass --csv > "$REPORT.sass.csv"
done
/home/xgao/pearl-official/.venv/bin/python tools/analyze_instruction_stalls.py "$PROFILE_OUT"/v*.sass.csv --output "$PROFILE_OUT/pc-stalls.json"
/home/xgao/pearl-official/.venv/bin/python tools/analyze_ada_ws.py "$PROFILE_OUT" --variants 0 1 --useful-mma 67108864
printf 'Reports: %s\n' "$PROFILE_OUT"
