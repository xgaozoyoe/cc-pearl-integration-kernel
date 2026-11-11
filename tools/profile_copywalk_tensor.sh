#!/usr/bin/env bash
# External same-tool Tensor Active comparison; GPU0 workload, GPU1 monitoring only.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $EUID -ne 0 ]]; then
  echo 'Hardware counters require root (RmProfilingAdminOnly=1). Run sudo bash tools/profile_copywalk_tensor.sh' >&2
  exit 1
fi
unset CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER
CANDIDATE=${1:-artifacts/libpearl-rank128-copywalk-fulla-candidate.so}
[[ -f "$CANDIDATE" ]]
GPU_UUID=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
APPS=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)
if [[ "$APPS" == *"$GPU_UUID"* ]]; then
  echo 'GPU0 has an existing compute process. Stop the GPU0 experiment before profiling.' >&2
  exit 1
fi
OUT=$(mktemp -d /tmp/pearl-copywalk-tensor.XXXXXX)
chmod 755 "$OUT"
echo "$OUT" | tee /tmp/pearl-copywalk-tensor-last.txt
chmod 644 /tmp/pearl-copywalk-tensor-last.txt
/usr/local/cuda/bin/nsys profile --gpu-metrics-devices=help > "$OUT/devices.txt" 2>&1
IDS=()
for PHYSICAL_GPU in 0 1; do
 PCI=$(nvidia-smi -i "$PHYSICAL_GPU" --query-gpu=pci.bus_id --format=csv,noheader)
 ID=$(python3 tools/select_nsys_metric_device.py "$OUT/devices.txt" "$PCI")
 IDS+=("$ID")
 printf 'Physical GPU %s PCI %s -> Nsight %s\n' "$PHYSICAL_GPU" "$PCI" "$ID" | tee -a "$OUT/selection.txt"
done
for VARIANT in baseline candidate; do
 LIB=artifacts/libpearl-rank.so
 if [[ "$VARIANT" == candidate ]]; then LIB=$CANDIDATE; fi
 sha256sum "$LIB" > "$OUT/$VARIANT.sha256"
 nvidia-smi -i 0,1 --query-gpu=index,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv > "$OUT/$VARIANT-before.csv"
 /usr/local/cuda/bin/nsys profile --trace=none --sample=none --cpuctxsw=none \
  --gpu-metrics-devices="${IDS[0]},${IDS[1]}" --gpu-metrics-frequency=1000 \
  --duration=35 --kill=none --wait=primary --output="$OUT/$VARIANT" \
  "$PWD/.venv/bin/python" tools/run_copywalk_metrics.py --library "$LIB" > "$OUT/$VARIANT.log" 2>&1
 /usr/local/cuda/bin/nsys export --type sqlite --output "$OUT/$VARIANT.sqlite" "$OUT/$VARIANT.nsys-rep" > "$OUT/$VARIANT-export.log" 2>&1
 nvidia-smi -i 0,1 --query-gpu=index,uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu --format=csv > "$OUT/$VARIANT-after.csv"
done
python3 tools/analyze_copywalk_tensor.py "$OUT" | tee "$OUT/report.txt"
chmod -R a+rX "$OUT"
echo "Complete: $OUT (analyze common window 10-30s; physical GPU0 trial, GPU1 PeakMiner)"
