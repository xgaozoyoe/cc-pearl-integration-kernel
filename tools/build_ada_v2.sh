#!/usr/bin/env bash
# Standalone experimental library; never overwrite the live miner library.
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR="${ADA_V2_BUILD_DIR:-/tmp/pearl-ada-v2-build}"
OUTPUT_LIBRARY="${OUTPUT_LIBRARY:-/tmp/libpearl-ada-v2.so}"
mkdir -p "$BUILD_DIR"
NVCC=/usr/local/cuda-12.6/bin/nvcc
FLAGS=(-I csrc -I csrc/gemm -I /home/xgao/cutlass/include -Xcompiler -fPIC
  -std=c++20 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math
  -gencode arch=compute_89,code=sm_89 -O3 -DNDEBUG -allow-unsupported-compiler
  -lineinfo -DPEARL_REUSE_FOCUS -DPEARL_ADA_V2)
DEPENDENCIES="$(find tests csrc -type f \( -name '*.cuh' -o -name '*.hpp' -o -name '*.h' -o -path 'tests/fold_ops.cu' \) -print)"
for SRC in tests/pure_gemm_measure.cu csrc/gemm/pearl_gemm_only_sm61.cu csrc/gemm/pearl_blake3_sm61.cu csrc/gemm/pearl-gemm-ada-v2.cu; do
  OBJ="$BUILD_DIR/$(basename "$SRC").o"
  REBUILD=0
  [[ -f "$OBJ" && "$SRC" -ot "$OBJ" ]] || REBUILD=1
  # Include both experiment and production headers in the cache invalidation.
  while IFS= read -r DEP; do
    if [[ "$DEP" -nt "$OBJ" ]]; then REBUILD=1; fi
  done <<< "$DEPENDENCIES"
  if [[ "$REBUILD" == 1 ]]; then "$NVCC" "${FLAGS[@]}" -c "$SRC" -o "$OBJ"; fi
done
"$NVCC" -shared "$BUILD_DIR/pure_gemm_measure.cu.o" "$BUILD_DIR/pearl_gemm_only_sm61.cu.o" \
  "$BUILD_DIR/pearl_blake3_sm61.cu.o" "$BUILD_DIR/pearl-gemm-ada-v2.cu.o" -o "$OUTPUT_LIBRARY"
printf 'Built %s\n' "$OUTPUT_LIBRARY"
