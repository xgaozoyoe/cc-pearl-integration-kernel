# GEMM main-loop experiments

GPU 0 only; experimental /tmp/libpearl-pure-gemm.so. Production libp40cuda.so unchanged (SHA256 dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6). No changes to V3 seeds, target or proof construction; no candidates promoted.

All seven new full-transcript candidates passed DP4A reference comparisons at 256x512, 512x256 and 4096x4096, K4096/R256. Normal PoW is retained for full-grid benchmarks. Inputs are deterministic and matched between variants, measured in forward/reverse blocks with two warmup grids per block and 20 stable grids per variant. Clocks were not administratively locked, so sub-percent changes should not be considered established gains.

| Variant | Region CUDA median (ms) | Registers/thread | Shared bytes/CTA |
|---|---:|---:|---:|
| K64 s3 baseline 20 | 0.340403 | 228 | 73728 |
| Prefetch before LDSM 49 | 0.346235 | 230 | 73728 |
| Serpentine MMA traversal 50 | 0.340425 | 236 | 73728 |
| Linear row-major traversal 51 | 0.340412 | 236 | 73728 |
| Consume K32 fragments separately 52 | 0.344535 | 230 | 73728 |
| B LDSM x4 53 | 0.338077 | 236 | 73728 |
| K128 s2 streaming 54 | 0.343332 | 216 | 98304 |
| K128 s2 full-fragment load 55 | 0.356320 | 208 | 98304 |

All have 256 threads, eight warps, one resident CTA/SM and zero local allocation reported by CUDA kernel attributes. Region runs use five alternating-order passes; these timings are not full-grid times.

Full-grid B LDSM x4 trial: baseline 20 median 391.15 ms, candidate 53 median 391.45 ms: no reproducible benefit despite the region improvement.

| Latest full-grid comparison | Median ms | Mean ms | p95 ms | Grids/s | Effective mining TH/s |
|---|---:|---:|---:|---:|---:|
| Production 0 | 427.55 | 427.205 | 430.29 | 2.33891 | 164.586 |
| Best K64 s3 20 | 391.60 | 391.255 | 392.62 | 2.55363 | 179.695 |
| K128 s2 streaming 54 | 397.00 | 396.555 | 397.965 | 2.51889 | 177.251 |

K128 reduces K-loop iterations and barriers from 64 to 32, but increases shared storage to 96 KiB and loses ~1.38% in full-grid latency. Lower register counts do not increase occupancy. Keep variant 20; none of these new changes improves full-grid performance.

Next useful direction: quantify operand delivery with Nsight Compute for baseline 20 and K128 54 before making another mapping change. Prior cached-region profiling showed high L2 throughput; that suggests investigating cross-CTA A/B reuse and bytes per MMA, rather than further instruction-order adjustments. It does not establish the full-grid bottleneck by itself.

Raw measurements: mainloop-region-results.json and mainloop-grid-v54-results.json. No commits or live mining tests were performed for these candidates.
