#!/usr/bin/env bash
# GPU 0 only. Counter access requires sudo on the current driver.
set -euo pipefail
cd "$(dirname "$0")/.."
export PEARL_TRIAL_LIBRARY="${PEARL_TRIAL_LIBRARY:-/tmp/libpearl-ada-linear-copy.so}"
export PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-ada-swizzle-stalls}"
export PROFILE_VARIANTS="${PROFILE_VARIANTS:-110 117 121}"
exec bash tools/profile_ada_v2_stalls.sh
