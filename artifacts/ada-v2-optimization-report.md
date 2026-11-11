# Ada v2 optimization results

GPU 0 only; RTX 4090, SM89, CUDA 12.6. Two matched-input A/B runs, with 40 measured stable grids per variant after warmup. Clock/power policy was unchanged; clocks were not locked. The baseline was the frozen original v104 library, not the refactored v104 alias in the candidate library.

| Full-grid measurement | Original hand-written v104 | Optimized v110 |
|---|---:|---:|
| Median wall time | 366.85 ms | 352.70 ms |
| Mean wall time | 367.04 ms | 352.685 ms |
| p95 wall time | 369.915 ms | 354.335 ms |
| Grids/sec from median | 2.72591 | 2.83527 |
| Equivalent end-to-end throughput | 191.82 TH/s | 199.51 TH/s |
| Reported search throughput, median | 200.335 TH/s | 208.655 TH/s |

Median throughput improved 4.01%; median full-grid wall time fell 14.15 ms (3.86%). Reported search throughput excludes setup and is not a CUDA-event kernel measurement. Regional CUDA-event median for final v110 was 0.301827 ms; this is one region, not one full grid.

## Changes

1. Refactored shared-memory address expressions; raw-layout v109 gained about 1.34% end-to-end over the frozen v104 baseline.
2. Changed B's LDSM lane addresses so its output registers form adjacent pairs in the order consumed by MMA. This removes repeated register-pair rearrangements while preserving MMA count and loaded bytes.
3. Specialized the packed-B branch to the already-required K=4096, removing dynamic-K overhead and the temporary local-memory spills introduced by initial packing.

The final kernel retains a 256x192 CTA tile, 64x64 warp tile, 12 warps/CTA, K64, two shared-memory stages, and INT8 m16n8k32 MMA. It uses 168 registers/thread and 69,632 bytes shared memory/CTA, allowing one CTA/SM. Disassembly shows zero LDL/STL and zero local allocation. Per warp/K64 there remain 64 IMMA and 16 LDSM.x4 instructions. Static instruction count fell from 875 in the original to 748 in the final candidate.

Planar K64 and K96 shared-memory layout experiments were slower in regional measurements and were not selected.

## Correctness and dispatch

All five full transcript comparisons against DP4A passed: 256x512, 512x256, 4096x4096, 256x768, and 768x512, with K4096/R256.

Nsight Systems recorded 2,048 invocations across two full grids of `pearl_ada_v2::kernel<2,64,false,true>`, confirming the packed-B handwritten path actually ran. The trace is a dispatch check, not the source of the stable performance statistics. Remaining CuTe kernels in that trace perform noise GEMM.

V3 seed derivation, rank-penalized target logic, and proof construction were not changed. The production library was not replaced. A live accepted share for this new candidate remains unverified; these results do not claim pool acceptance for v110.

## Artifacts and rollback

- Frozen original: `/tmp/libpearl-ada-v2-retest.so`, SHA256 `0bd2620677bafa6b141a70a243d5eecd781e439c4ca184d118407b62b7e2395b`.
- Final candidate: `/tmp/libpearl-ada-bpack-fixedk.so`, SHA256 `b056dea25bfbe622ea82510cab49e18b7756c01e23965d5209d059d06f89fabd`.
- Production `libp40cuda.so` remains unchanged, SHA256 `dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6`.
- Combined measurements: `ada-final-combined-results.json`; individual runs: `ada-final-v110-run1.json`, `ada-final-v110-run2.json`.
- Kernel/benchmark round diff: `ada-v2-optimization-round.diff`; fixed-K kernel diff: `ada-bpack-fixedk-kernel.diff`.
- Final resource/disassembly records: `ada-bpack-fixedk-sass.json`.
- Dispatch trace summary: `ada-packed-launch-kernels.csv`.

No production promotion or commit was performed. The next useful investigation is whether further operand-address and register-pair overhead can be removed without changing transcript ownership or increasing register pressure.
