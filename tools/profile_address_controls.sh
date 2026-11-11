#!/usr/bin/env bash
# Exact frozen fastest binary, GPU0 only; no build, mining or production mutation.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $EUID -ne 0 ]]; then
 echo 'Counter access requires: sudo bash tools/profile_address_controls.sh' >&2
 exit 1
fi
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python tools/check_gpu0_idle.py --allow-supervisor-pid "${1:-0}"
.venv/bin/python - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('artifacts/fold-preamble');v=json.loads((p/'manifest.json').read_text())['variants']['both_compact']
expected='104c0c1b265260dea5c9ab0bb174ebbbe7ee9cceb0cf0acebf2a9d1dfa4d50aa'
assert v['sha256']==expected and hashlib.sha256((p/'both_compact.cubin').read_bytes()).hexdigest()==expected
print('Verified frozen fastest binary:',expected)
PY
PROFILE_OUT="$(mktemp -d /tmp/pearl-address-controls.XXXXXX)"
printf '%s\n' "$PROFILE_OUT" > /tmp/pearl-address-controls-last.txt
chmod 644 /tmp/pearl-address-controls-last.txt
trap 'chmod -R a+rX "$PROFILE_OUT"' EXIT
sha256sum artifacts/fold-preamble/both_compact.cubin artifacts/fold-preamble/manifest.json tests/bench_address_controls.py artifacts/address-controls/manifest.json artifacts/address-controls/*.cubin tools/profile_address_controls.sh tools/report_address_controls.py > "$PROFILE_OUT/inputs.sha256"
nvidia-smi -i 0 --query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv > "$PROFILE_OUT/gpu-before.csv"
NCU=/usr/local/cuda/bin/ncu
"$NCU" --version > "$PROFILE_OUT/ncu-version.txt"
METRICS='sm__cycles_elapsed.avg.per_second,sm__cycles_elapsed.sum,smsp__cycles_elapsed.sum,smsp__inst_executed_pipe_alu.sum,smsp__inst_executed_pipe_tensor.sum,smsp__inst_executed_pipe_tensor_v2.sum,smsp__inst_executed_pipe_tensor_v2.avg.per_cycle_elapsed,gpu__time_duration.sum,dram__bytes.sum,lts__t_bytes.sum,sm__sass_data_bytes_mem_shared_op_ldsm.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_eligible.avg.per_cycle_active,smsp__warp_issue_stalled_barrier_per_warp_active.pct,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct,sm__pipe_tensor_op_imma_cycles_active_v2.avg.pct_of_peak_sustained_active'
# Six controlled variants in manifest order; three warmed launches each.
# NCU clock-control base is temporary during profiling, GPU0 only. No persistent nvidia-smi changes.
for RUN in 0 1 2 3 4 5; do
 REPORT="$PROFILE_OUT/v$RUN"
 "$NCU" --devices 0 --clock-control base --replay-mode application --cache-control none \
  --kernel-name-base demangled --kernel-name 'regex:.*handoff_lean_toggle::kernel.*' \
  --launch-skip 32 --launch-count 3 --metrics "$METRICS" \
  --section SourceCounters --section SchedulerStats --section WarpStateStats \
  --warp-sampling-interval auto --warp-sampling-max-passes 5 --import-source yes \
  -o "$REPORT" "$PWD/.venv/bin/python" tests/bench_address_controls.py profile --variant "$RUN" \
  > "$REPORT.log" 2>&1 || { cat "$REPORT.log" >&2; exit 1; }
 "$NCU" --import "$REPORT.ncu-rep" --page raw --print-units base --csv > "$REPORT.raw.csv"
 "$NCU" --import "$REPORT.ncu-rep" --page source --print-source sass --csv > "$REPORT.sass.csv"
done
.venv/bin/python tools/analyze_instruction_stalls.py "$PROFILE_OUT"/v*.sass.csv --output "$PROFILE_OUT/pc-stalls.json"
.venv/bin/python tools/analyze_ada_ws.py "$PROFILE_OUT" --variants 0 1 2 3 4 5 --useful-mma 67108864
.venv/bin/python tools/analyze_imma_issue.py "$PROFILE_OUT" --variants 0 1 2 3 4 5
.venv/bin/python tools/report_address_controls.py "$PROFILE_OUT"
nvidia-smi -i 0 --query-gpu=uuid,name,driver_version,clocks.sm,clocks.mem,power.limit,temperature.gpu --format=csv > "$PROFILE_OUT/gpu-after.csv"
printf 'Reports: %s\n' "$PROFILE_OUT"
