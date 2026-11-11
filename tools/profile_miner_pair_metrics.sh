#!/usr/bin/env bash
# External same-tool metrics for physical GPU1 and GPU2; no miner restarts.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ${EUID} -ne 0 ]]; then
  echo 'GPU hardware counters require root on this host (RmProfilingAdminOnly=1).' >&2
  echo 'Run: sudo bash tools/profile_miner_pair_metrics.sh' >&2
  exit 1
fi
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
OUT=$(mktemp -d /tmp/pearl-miner-pair-metrics.XXXXXX)
chmod 755 "$OUT"
printf 'Report directory: %s\n' "$OUT"
printf '%s\n' "$OUT" > /tmp/pearl-miner-pair-last-metrics.txt
chmod 644 /tmp/pearl-miner-pair-last-metrics.txt
/usr/local/cuda/bin/nsys profile --gpu-metrics-devices=help > "$OUT/device-list.txt" 2>&1
METRIC_IDS=()
for PHYSICAL_GPU in 1 2; do
  PCI_BUS=$(nvidia-smi -i "$PHYSICAL_GPU" --query-gpu=pci.bus_id --format=csv,noheader)
  ID=$(python3 tools/select_nsys_metric_device.py "$OUT/device-list.txt" "$PCI_BUS")
  METRIC_IDS+=("$ID")
  printf 'Physical GPU %s PCI %s -> Nsight metrics device %s\n' "$PHYSICAL_GPU" "$PCI_BUS" "$ID" | tee -a "$OUT/device-selection.txt"
done
METRIC_DEVICE="${METRIC_IDS[0]},${METRIC_IDS[1]}"
curl -fsS --max-time 5 http://127.0.0.1:4068/summary > "$OUT/miner-before.json"
nvidia-smi -i 1,2 --query-gpu=index,uuid,temperature.gpu,power.draw,clocks.sm,clocks.mem --format=csv > "$OUT/gpu-before.csv"
# No CUDA injection into the miner: /bin/sleep is the only launched target.
# All selected GPU metrics are device-wide; cannot attribute them to kernel PCs.
/usr/local/cuda/bin/nsys profile --trace=none --sample=none --cpuctxsw=none \
  --gpu-metrics-devices="$METRIC_DEVICE" --gpu-metrics-frequency=1000 \
  --duration=30 --kill=none --wait=primary --output="$OUT/pair" /bin/sleep 32 > "$OUT/nsys.log" 2>&1
/usr/local/cuda/bin/nsys export --type sqlite --output "$OUT/pair.sqlite" "$OUT/pair.nsys-rep" > "$OUT/export.log" 2>&1
curl -fsS --max-time 5 http://127.0.0.1:4068/summary > "$OUT/miner-after.json"
chmod -R a+rX "$OUT"
printf 'Complete: %s\n' "$OUT"
