#!/usr/bin/env bash
# Short CUDA timeline capture. Does not stop existing miners or change clocks.
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep device numbering identical to the verified --devices 1 mining command.
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
PEAK_BIN=/home/xgao/peakminer/peakminer
[[ -x "$PEAK_BIN" ]] || { echo "Missing $PEAK_BIN" >&2; exit 1; }
GPU_UUID=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader)
if printf '%s\n' "$APPS" | grep -F "$GPU_UUID"; then
  echo 'GPU 1 is busy. Stop only its PeakMiner with Ctrl+C, then rerun. GPU 2 stays running.' >&2
  exit 2
fi
PROFILE_OUT=$(mktemp -d /tmp/pearl-peakminer-profile.XXXXXX)
chmod 755 "$PROFILE_OUT"
printf 'Reports: %s\n' "$PROFILE_OUT"
printf '%s\n' "$PROFILE_OUT" > /tmp/pearl-peakminer-last-profile.txt
chmod 644 /tmp/pearl-peakminer-last-profile.txt
sha256sum "$PEAK_BIN" > "$PROFILE_OUT/binary.sha256"
nvidia-smi --query-gpu=index,uuid,pci.bus_id,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem --format=csv > "$PROFILE_OUT/gpus-before.csv"
# Target process tree only; CPU/system-wide sampling disabled. 20s warmup +20s trace.
# Use a distinct pool worker and disable HTTP API to avoid port conflicts.
set +e
timeout --signal=TERM --kill-after=10s 90s /usr/local/cuda/bin/nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --delay="${PROFILE_DELAY:-20}" --duration=20 --kill=sigterm --wait=all \
  --output="$PROFILE_OUT/peakminer" \
  "$PEAK_BIN" --coin pearl --devices 1 \
  -o pearl-eu2.luckypool.io:3360 \
  -u prl1pp2fh58veqcg8x6g3msx0hshznkml6l2q9ev5efj5tsjzjcc2fm6q58xfw9 \
  -w peakminer-gpu1-profile --api-port 0 --log-file "$PROFILE_OUT/miner.log" \
  > "$PROFILE_OUT/nsys.log" 2>&1
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$PROFILE_OUT/exit-status.txt"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv > "$PROFILE_OUT/processes-after.csv"
if [[ ! -s "$PROFILE_OUT/peakminer.nsys-rep" ]]; then
  tail -60 "$PROFILE_OUT/nsys.log"
  echo "No trace generated. Keep logs for diagnosis: $PROFILE_OUT" >&2
  exit 3
fi
/usr/local/cuda/bin/nsys export --type sqlite --output "$PROFILE_OUT/peakminer.sqlite" "$PROFILE_OUT/peakminer.nsys-rep" > "$PROFILE_OUT/export.log" 2>&1
/usr/local/cuda/bin/nsys stats --report cuda_gpu_kern_sum,cuda_gpu_kern_gb_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
  --format csv "$PROFILE_OUT/peakminer.sqlite" > "$PROFILE_OUT/summary.csv" 2> "$PROFILE_OUT/stats.log"
python3 tools/analyze_peakminer_trace.py "$PROFILE_OUT/peakminer.sqlite" > "$PROFILE_OUT/kernels.json"
chmod -R a+rX "$PROFILE_OUT"
printf 'Capture complete: %s\nResume your normal GPU1 command after checking processes-after.csv.\n' "$PROFILE_OUT"
