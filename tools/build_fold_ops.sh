#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRA_DEFINES=()
if [ "${FOCUS_SQUARE:-0}" = 1 ]; then EXTRA_DEFINES=(-DPEARL_FOCUS_SQUARE); fi
"${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" -shared -o "${OUTPUT_LIBRARY:-/tmp/libpearl-fold-ops.so}" \
  tests/fold_ops.cu csrc/gemm/pearl_gemm_only_sm61.cu csrc/gemm/pearl_blake3_sm61.cu \
  -I csrc -I csrc/gemm -I "${CUTLASS_DIR:-/home/xgao/cutlass/include}" \
  -Xcompiler -fPIC -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda \
  --use_fast_math -gencode arch=compute_89,code=sm_89 -O3 -DNDEBUG -allow-unsupported-compiler "${EXTRA_DEFINES[@]}"
