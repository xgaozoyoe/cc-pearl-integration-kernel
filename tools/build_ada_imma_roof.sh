#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
/usr/local/cuda-12.6/bin/nvcc -std=c++20 -O3 -lineinfo   -gencode arch=compute_89,code=sm_89 -Xcompiler -fPIC   --ptxas-options=-v -shared tests/ada_imma_roof.cu   -o /tmp/libpearl-imma-roof.so
