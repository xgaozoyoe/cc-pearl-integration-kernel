#!/usr/bin/env bash
# GPU0 only. Variants99/101/102 have identical instruction encodings;
# profile99 once against89 to explain the changed loading cadence.
set -euo pipefail
cd "$(dirname "$0")/.."
export PEARL_TRIAL_LIBRARY=/tmp/libpearl-operand-ring.so
export PROFILE_VARIANTS="89 99"
export PROFILE_OUT="${PROFILE_OUT:-/tmp/pearl-operand-ring-stalls}"
bash tools/profile_instruction_stalls.sh
