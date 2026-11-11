#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
EXTRA_NVCC_FLAGS=()
EXTRA_CUDA_SOURCES=()
if [[ "${ADA_V2:-0}" == 1 ]]; then
  EXTRA_NVCC_FLAGS+=(-DPEARL_ADA_V2)
  EXTRA_CUDA_SOURCES+=(csrc/gemm/pearl-gemm-ada-v2.cu)
fi
if [[ "${REUSE_FOCUS:-0}" == 1 ]]; then EXTRA_NVCC_FLAGS+=(-DPEARL_REUSE_FOCUS); fi
if [[ "${LINEINFO:-0}" == 1 ]]; then EXTRA_NVCC_FLAGS+=(-lineinfo); fi
if [[ "${REUSE_FOCUS:-0}" == 1 ]]; then
  OUTPUT_LIBRARY="${OUTPUT_LIBRARY:-/tmp/libpearl-operand-stream.so}"
else
  OUTPUT_LIBRARY="${OUTPUT_LIBRARY:-/tmp/libpearl-pure-gemm.so}"
fi
/usr/local/cuda-12.6/bin/nvcc -shared -o "$OUTPUT_LIBRARY" \
  tests/pure_gemm_measure.cu csrc/gemm/pearl_gemm_only_sm61.cu csrc/gemm/pearl_blake3_sm61.cu "${EXTRA_CUDA_SOURCES[@]}" \
  -I csrc -I csrc/gemm -I /home/xgao/cutlass/include \
  -Xcompiler -fPIC -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda \
  --use_fast_math -gencode arch=compute_89,code=sm_89 -O3 -DNDEBUG -allow-unsupported-compiler "${EXTRA_NVCC_FLAGS[@]}"
