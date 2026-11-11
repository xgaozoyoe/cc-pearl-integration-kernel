#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=artifacts/fold-ready-window
names=(native after_wait before_wait)
timeout -k 5 150 artifacts/sanitizer-12.9.79/cuda_sanitizer_api-linux-x86_64-12.9.79-archive/bin/compute-sanitizer --tool racecheck --error-exitcode 99 .venv/bin/python tests/bench_fold_ready_window.py validate --names "${names[@]}" > "$out/racecheck.log" 2>&1
timeout -k 5 30 .venv/bin/python tests/bench_fold_ready_window_grids.py validate --names "${names[@]}" > "$out/pow-validation.log" 2>&1
for run in 1 2; do
 .venv/bin/python tests/bench_fold_ready_window.py bench --names "${names[@]}" > "$out/bench-run${run}.log" 2>&1
 cp "$out/events.json" "$out/events-run${run}.json"
done
