#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out=artifacts/transcript-tasks
# Initial full racecheck reached its240s bound after46successful comparisons.
# Finish the last2 cases without repeating previously completed coverage.
timeout -k 5 90 artifacts/sanitizer-12.9.79/cuda_sanitizer_api-linux-x86_64-12.9.79-archive/bin/compute-sanitizer --tool racecheck --error-exitcode 99 .venv/bin/python tests/bench_transcript_tasks.py validate --race-tail --names consumer_late consumer_fine > "$out/racecheck-tail.log" 2>&1
timeout -k 5 45 .venv/bin/python tests/bench_transcript_tasks_grids.py validate > "$out/pow-validation.log" 2>&1
for run in 1 2; do
 .venv/bin/python tests/bench_transcript_tasks.py bench > "$out/bench-run${run}.log" 2>&1
 cp "$out/events.json" "$out/events-run${run}.json"
done
