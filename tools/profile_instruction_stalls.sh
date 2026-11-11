#!/usr/bin/env bash
# GPU0 only. Requires sudo for restricted hardware counter access.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-instruction-stalls}"
mkdir -p "$PROFILE_OUT"
export CUDA_VISIBLE_DEVICES=0
export PEARL_TRIAL_LIBRARY="${PEARL_TRIAL_LIBRARY:-/tmp/libpearl-stall-attribution.so}"
[[ -f "$PEARL_TRIAL_LIBRARY" ]] || { printf 'Build the lineinfo library first.\n' >&2; exit 1; }
NCU=/usr/local/cuda/bin/ncu
KERNEL_REGEX="${KERNEL_REGEX:-.*cute_pearl_fold.*}"
CLOCK_CONTROL="${CLOCK_CONTROL:-base}"
AVAILABLE_SECTIONS="$($NCU --list-sections)"
for SECTION in SourceCounters SchedulerStats WarpStateStats; do
  if ! awk -v section="$SECTION" '$1 == section { found=1 } END { exit !found }' <<< "$AVAILABLE_SECTIONS"; then
    printf 'Missing Nsight section: %s\n' "$SECTION" >&2
    exit 1
  fi
done
read -r -a CACHE_MODES <<< "${CACHE_MODES:-warm}"
read -r -a PROFILE_VARIANTS <<< "${PROFILE_VARIANTS:-88 89}"
PC_METRICS='smsp__pcsamp_sample_count,smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_long_scoreboard,smsp__pcsamp_warps_issue_stalled_wait,smsp__pcsamp_warps_issue_stalled_mio_throttle,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle,smsp__pcsamp_warps_issue_stalled_barrier_not_issued,smsp__pcsamp_warps_issue_stalled_short_scoreboard_not_issued,smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued,smsp__pcsamp_warps_issue_stalled_wait_not_issued,smsp__pcsamp_warps_issue_stalled_mio_throttle_not_issued,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle_not_issued'
for CACHE in "${CACHE_MODES[@]}"; do
  case "$CACHE" in warm) POLICY=none;; cold) POLICY=all;; *) exit 1;; esac
  for VARIANT in "${PROFILE_VARIANTS[@]}"; do
    REPORT="$PROFILE_OUT/$CACHE-v$VARIANT"
    "$NCU" --devices 0 --clock-control "$CLOCK_CONTROL" \
      --replay-mode application --cache-control "$POLICY" \
      --kernel-name-base demangled --kernel-name "regex:$KERNEL_REGEX" \
      --launch-skip 32 --launch-count 3 \
      --section SourceCounters --section SchedulerStats --section WarpStateStats \
      --warp-sampling-interval auto --warp-sampling-max-passes 5 \
      --import-source yes -o "$REPORT" \
      .venv/bin/python tests/profile_fold_traffic.py "$VARIANT" > "$REPORT.log" 2>&1
    /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page source \
      --print-source sass --metrics "$PC_METRICS" --csv > "$REPORT-sass.csv"
    /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page source \
      --print-source cuda,sass --csv > "$REPORT-cuda-sass.csv"
    /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page raw \
      --csv > "$REPORT-raw.csv"
  done
done
printf 'Reports: %s\n' "$PROFILE_OUT"
