#!/usr/bin/env bash
# Offline GPU0 only: Wrong-arithmetic baseline SASS operand probes; diagnostic only.
# Requires counter access. Validated for Nsight Compute2024.3.2 CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $EUID -ne 0 ]]; then
 echo 'Hardware counters require root: sudo bash tools/profile_dependency_ablation.sh' >&2
 exit 1
fi
GPU_UUID=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ "$APPS" == *"$GPU_UUID"* ]]; then echo 'GPU0 already has a compute process' >&2; exit 1; fi
export CUDA_VISIBLE_DEVICES=0

PROFILE_OUT="${PROFILE_OUT:-$(mktemp -d /tmp/pearl-dependency-ablation.XXXXXX)}"
printf '%s\n' "$PROFILE_OUT" > /tmp/pearl-dependency-ablation-last.txt
chmod 644 /tmp/pearl-dependency-ablation-last.txt
mkdir -p "$PROFILE_OUT"
if compgen -G "$PROFILE_OUT/v*.ncu-rep" > /dev/null; then
 echo 'Use a fresh PROFILE_OUT directory; existing reports must not mix with changed binaries.' >&2
 exit 1
fi
sha256sum artifacts/dependency-ablation/manifest.json artifacts/dependency-ablation/baseline.cubin artifacts/dependency-ablation/c_zero.cubin artifacts/dependency-ablation/a_zero.cubin artifacts/dependency-ablation/b_zero.cubin artifacts/dependency-ablation/ab_zero.cubin > "$PROFILE_OUT/inputs.sha256"
nvidia-smi -i 0 --query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv > "$PROFILE_OUT/gpu-before.csv"
/usr/local/cuda/bin/ncu --version > "$PROFILE_OUT/ncu-version.txt"
METRICS='sm__cycles_elapsed.sum,smsp__cycles_elapsed.sum,smsp__inst_executed_pipe_tensor_v2.sum,smsp__inst_executed_pipe_tensor_v2.avg.per_cycle_elapsed,gpu__time_duration.sum,dram__bytes.sum,lts__t_bytes.sum,sm__sass_data_bytes_mem_shared_op_ldsm.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active'
for VARIANT in 0 1 2 3 4; do
  REPORT="$PROFILE_OUT/v$VARIANT"
  if [[ ! -s "$REPORT.ncu-rep" ]]; then
  /usr/local/cuda/bin/ncu --devices 0 --clock-control none \
    --replay-mode application --cache-control none --kernel-name-base demangled \
    --kernel-name 'regex:.*handoff_control_baseline::kernel.*' \
    --launch-skip 32 --launch-count 3 --metrics "$METRICS" \
    --section SourceCounters --section SchedulerStats --section WarpStateStats \
    --warp-sampling-interval auto --warp-sampling-max-passes 5 --import-source yes \
    -o "$REPORT" "$PWD/.venv/bin/python" tests/bench_dependency_ablation.py profile --variant "$VARIANT" \
    > "$REPORT.log" 2>&1
  else
    printf 'Reusing existing report: %s\n' "$REPORT.ncu-rep"
  fi
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page raw --print-units base --csv > "$REPORT.raw.csv"
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page source --print-source sass --csv > "$REPORT.sass.csv"
done
"$PWD/.venv/bin/python" tools/analyze_instruction_stalls.py "$PROFILE_OUT"/v*.sass.csv --output "$PROFILE_OUT/pc-stalls.json"
"$PWD/.venv/bin/python" tools/analyze_ada_ws.py "$PROFILE_OUT" --variants 0 1 2 3 4 --useful-mma 67108864
nvidia-smi -i 0 --query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv > "$PROFILE_OUT/gpu-after.csv"
"$PWD/.venv/bin/python" tools/analyze_imma_issue.py "$PROFILE_OUT" --variants 0 1 2 3 4
chmod -R a+rX "$PROFILE_OUT"
printf 'Reports: %s\n' "$PROFILE_OUT"
