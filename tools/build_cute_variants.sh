#!/usr/bin/env bash
# Build isolated experiments; never overwrite libp40cuda.so.
set -euo pipefail
cd "$(dirname "$0")/.."
"${NVCC:-/usr/local/cuda-12.6/bin/nvcc}" -shared -o /tmp/libpearl-cute-variants.so \
  tests/cute_fold_variants.cu csrc/gemm/pearl_gemm_only_sm61.cu \
  csrc/gemm/pearl_blake3_sm61.cu \
  -I csrc -I csrc/gemm -I "${CUTLASS_DIR:-/home/xgao/cutlass/include}" \
  -Xcompiler -fPIC -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda \
  --use_fast_math -gencode arch=compute_89,code=sm_89 -O3 -DNDEBUG \
  -allow-unsupported-compiler
