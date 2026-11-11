#!/usr/bin/env bash
# GPU 0 only. Hardware counter access requires sudo on this host.
set -euo pipefail
cd "$(dirname "$0")/.."
export PROFILE_OUT=/tmp/pearl-operand-stream-profile
export PEARL_TRIAL_LIBRARY=/tmp/libpearl-operand-stream.so
export PROFILE_VARIANTS='67 86 88'
bash tools/profile_mapping_reuse.sh
