#!/usr/bin/env bash
# GPU 0 only; restricted hardware counters require sudo on this host.
set -euo pipefail
cd "$(dirname "$0")/.."
export PROFILE_OUT=/tmp/pearl-b-prefetch-profile
export PEARL_TRIAL_LIBRARY=/tmp/libpearl-prefetch.so
export PROFILE_VARIANTS='88 89'
bash tools/profile_mapping_reuse.sh
