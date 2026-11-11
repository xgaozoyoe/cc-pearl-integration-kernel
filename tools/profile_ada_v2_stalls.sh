#!/usr/bin/env bash
# GPU 0 only: compare packed-B baseline with the wider warp mapping.
# Run with sudo because this driver restricts GPU performance counters.
set -euo pipefail
cd "$(dirname "$0")/.."
export PEARL_TRIAL_LIBRARY="${PEARL_TRIAL_LIBRARY:-/tmp/libpearl-ada-wide96.so}"
export PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-ada-v2-stalls}"
export PROFILE_VARIANTS="${PROFILE_VARIANTS:-110 115}"
export KERNEL_REGEX='.*pearl_ada_v2.*'
export CLOCK_CONTROL="${CLOCK_CONTROL:-none}"
export CACHE_MODES="${CACHE_MODES:-warm}"
exec bash tools/profile_instruction_stalls.sh
