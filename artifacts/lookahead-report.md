# Fine-grained shared-load/MMA pipeline experiment

Candidates 56/57 preserve CTA128x256, warp2x4, K64, three shared stages, INT8 MMA and full transcript/PoW. They load K32(0), compute its first accumulator row group, then look ahead to K32(1). Candidate 56 loads A/B together; 57 staggers B to the next row group.

Both passed full transcript comparisons against DP4A for 256x512, 512x256 and 4096x4096 at K4096/R256. Registers decreased from 228 to 222; shared remains 73728 bytes/CTA, eight warps, one CTA/SM. SASS shows zero LDL/STL, 64 IMMA and 24 LDSM instructions in all three kernels. Both new candidates produced the same extracted LDSM/IMMA sequence. Crucially, baseline SASS already interleaves LDSM and IMMA despite the bulk-load source structure. This experiment therefore adjusts an existing compiler-scheduled overlap rather than introducing overlap from scratch.

Region CUDA medians: baseline 20 0.340057 ms; joint 56 0.344911 ms; staggered 57 0.345033 ms. Five alternating-order passes, 20 warmups and 150 measured launches per pass.

GPU0 full-grid test, deterministic matched inputs, two forward/reverse blocks per variant, two warmup grids discarded per block, 20 stable grids each, normal BLAKE3 retained:

| Variant | Median ms/grid | Mean | p95 | Grids/s | Effective TH/s |
|---|---:|---:|---:|---:|---:|
| Best baseline 20 | 390.55 | 390.39 | 392.44 | 2.56049 | 180.179 |
| Joint lookahead 56 | 394.80 | 394.51 | 396.25 | 2.53293 | 178.239 |

Candidate 56 increases latency 1.09%. No promotion. This does not rule out cross-K64 iteration register pipelining, but that requires more careful stage-lifetime/barrier handling than this intra-stage experiment. Actual clocks were not administratively locked; no sub-percent benefit is claimed.

Production libp40cuda.so SHA256 remains dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6. No seed/target/proof edits or live mining tests. Experimental changes remain uncommitted in tests/fold_ops.cuh, tests/fold_ops.cu and benchmark scripts. Raw data: lookahead-region-results.json, lookahead-grid-results.json, lookahead-sass-summary.json.
