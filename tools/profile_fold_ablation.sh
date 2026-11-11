#!/usr/bin/env bash
# Run with sudo for restricted hardware counters. GPU 0 only, no live mining.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-fold-ablation-profile}"
mkdir -p "$PROFILE_OUT"
export CUDA_VISIBLE_DEVICES=0
export PEARL_TRIAL_LIBRARY=/tmp/libpearl-fold-ablation.so
METRICS=$(.venv/bin/python -c 'import re,pathlib; print(re.search("METRICS="+chr(39)+"([^"+chr(39)+"]+)",pathlib.Path("tools/profile_fold_traffic.sh").read_text())[1])')
for VARIANT in 20 24 27; do
  REPORT="$PROFILE_OUT/warm-v$VARIANT"
  /usr/local/cuda/bin/ncu --devices 0 --clock-control base \
    --replay-mode application --cache-control none \
    --kernel-name 'regex:.*cute_pearl_fold.*' \
    --launch-skip 32 --launch-count 3 --metrics "$METRICS" \
    -o "$REPORT" .venv/bin/python tests/profile_fold_traffic.py "$VARIANT" \
    > "$REPORT.log" 2>&1
  /usr/local/cuda/bin/ncu --import "$REPORT.ncu-rep" --page raw --csv > "$REPORT.csv"
done
printf 'Reports: %s\n' "$PROFILE_OUT"
