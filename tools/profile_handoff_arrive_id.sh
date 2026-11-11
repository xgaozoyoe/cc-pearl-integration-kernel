#!/usr/bin/env bash
# Offline GPU0 only: correct-arithmetic early BAR.ARV controls.
# Requires counter access. Validated for Nsight Compute2024.3.2 CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $EUID -ne 0 ]]; then
 echo 'Hardware counters require root: sudo bash tools/profile_handoff_arrive_id.sh' >&2
 exit 1
fi
.venv/bin/python tools/check_gpu0_idle.py --allow-supervisor-pid "${1:-0}"
export CUDA_VISIBLE_DEVICES=0

PROFILE_OUT="${PROFILE_OUT:-$(mktemp -d /tmp/pearl-handoff-arrive-id.XXXXXX)}"
printf '%s\n' "$PROFILE_OUT" > /tmp/pearl-handoff-arrive-id-last.txt
chmod 644 /tmp/pearl-handoff-arrive-id-last.txt
mkdir -p "$PROFILE_OUT"
# Preserve readable captures even when a later report generator fails.
trap 'chmod -R a+rX "$PROFILE_OUT"' EXIT
if compgen -G "$PROFILE_OUT/v*.ncu-rep" > /dev/null; then
 echo 'Use a fresh PROFILE_OUT directory; existing reports must not mix with changed binaries.' >&2
 exit 1
fi
sha256sum artifacts/handoff-arrive-id/manifest.json artifacts/handoff-arrive-id/*.cubin > "$PROFILE_OUT/inputs.sha256"
nvidia-smi -i 0 --query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv > "$PROFILE_OUT/gpu-before.csv"
/usr/local/cuda/bin/ncu --version > "$PROFILE_OUT/ncu-version.txt"
METRICS='sm__cycles_elapsed.avg.per_second,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_tensor.sum,smsp__warps_eligible.avg.per_cycle_active,sm__cycles_elapsed.sum,smsp__cycles_elapsed.sum,smsp__inst_executed_pipe_tensor_v2.sum,smsp__inst_executed_pipe_tensor_v2.avg.per_cycle_elapsed,gpu__time_duration.sum,dram__bytes.sum,lts__t_bytes.sum,sm__sass_data_bytes_mem_shared_op_ldsm.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active'
for VARIANT in 0 1 2 3; do
  REPORT="$PROFILE_OUT/v$VARIANT"
  if [[ ! -s "$REPORT.ncu-rep" ]]; then
  /usr/local/cuda/bin/ncu --devices 0 --clock-control base \
    --replay-mode application --cache-control none --kernel-name-base demangled \
    --kernel-name 'regex:.*handoff_lean_.*::kernel.*' \
    --launch-skip 32 --launch-count 3 --metrics "$METRICS" \
    --section SourceCounters --section SchedulerStats --section WarpStateStats \
    --warp-sampling-interval auto --warp-sampling-max-passes 5 --import-source yes \
    -o "$REPORT" "$PWD/.venv/bin/python" tests/bench_handoff_arrive_id.py profile --variant "$VARIANT" \
    > "$REPORT.log" 2>&1
  else
    printf 'Reusing existing report: %s\n' "$REPORT.ncu-rep"
  fi
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page raw --print-units base --csv > "$REPORT.raw.csv"
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page source --print-source sass --csv > "$REPORT.sass.csv"
done
"$PWD/.venv/bin/python" tools/analyze_instruction_stalls.py "$PROFILE_OUT"/v*.sass.csv --output "$PROFILE_OUT/pc-stalls.json"
"$PWD/.venv/bin/python" tools/analyze_ada_ws.py "$PROFILE_OUT" --variants 0 1 2 3 --useful-mma 67108864
nvidia-smi -i 0 --query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv > "$PROFILE_OUT/gpu-after.csv"
"$PWD/.venv/bin/python" tools/analyze_imma_issue.py "$PROFILE_OUT" --variants 0 1 2 3

"$PWD/.venv/bin/python" tools/report_handoff_arrive_id.py --capture "$PROFILE_OUT"
chmod -R a+rX "$PROFILE_OUT"
printf 'Reports: %s\n' "$PROFILE_OUT"
