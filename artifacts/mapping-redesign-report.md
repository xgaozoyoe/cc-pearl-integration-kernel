# Operand-reuse mapping experiment

No new winner; preserve variant67 N8. Production library, V3 seeds, target and proof construction unchanged. GPU0 only, no live shares or commits. Experimental sources remain in tests/fold_ops.cu/cuh and independent /tmp/libpearl-pure-gemm.so.

## Reuse model

For warp-level m16n8k32 INT8 MMA (4096 MAC/instruction), ideal input-only requested bytes/MMA = 4096*(1/BM+1/BN). Ideal shared operand bytes/MMA = 4096*(1/warpM+1/warpN). These are analytic payload values, not measured transactions. Actual L2 bytes include cache transaction effects and other traffic.

| Mapping | Global requested bytes/useful MMA | Ideal shared bytes/MMA | Notes |
|---|---:|---:|---|
| 128x256, warp64x64 | 48 | 128 | Current baseline |
| 256x128, warp64x64 | 48 | 128 | Changes input symmetry, no arithmetic intensity gain |
| 128x128, warp64x64 | 64 | 128 | +33.3% input payload; two CTAs/SM |
| 128x320, warp64x64 | 45 | 128 | -6.25% requested input over full 4096 region; +1.5625% padded MMA |

The baseline has 1/48 MMA per requested input byte and 1/128 MMA per ideal shared operand byte. The wide candidate improves only the first. For 256x256, a complete INT32 accumulator alone needs 65536 registers, exhausting a SM's register capacity before operands and transcript. Increasing tile area requires changing live-state storage or reuse, not simply changing dimensions.

## Launch/resource and correctness results

Six launchable new mappings passed complete DP4A transcript reference at 256x512,512x256,4096x4096,256x768,768x512 (K4096/R256). A first test attempt used a matrix smaller than a control CTA; the invalid launch was corrected by changing the extra test dimensions. Four wide candidates excluded by occupancy query before execution:

| Variant | Registers/thread | Threads | Shared bytes/CTA | Resident CTA/SM |
|---|---:|---:|---:|---:|
| 67 baseline | 228 | 256 | 73728 | 1 |
| 68 256x128 warp4x2 | 232 | 256 | 73728 | 1 |
| 69 256x128 warp2x4 | 250 | 256 | 73728 | 1 |
| 70 256x128 warp8x1 | 216 | 256 | 73728 | 1 |
| 71 128x128 warp2x2 | 226 | 128 | 49152 | 2 |
| 72 128x128 warp1x4 | 254 | 128 | 49152 | 2 |
| 73 128x128 warp4x1 | 220 | 128 | 49152 | 2 |
| 74 wide register transcript s4 | 201 | 320 | 57344 | 0 |
| 75 wide direct transcript s4 | 233 | 320 | 57344 | 0 |
| 76 wide shared transcript s3 | 190 | 320 | 53248 | 0 |
| 77 wide shared transcript s4 | 190 | 320 | 67584 | 0 |

All report zero local allocation. Register resource allocation is granular; total registers/thread*threads alone cannot predict residency. Excluded variants have no correctness/performance claim. Small two-CTA kernels are ~33% slower in region tests; transposed CTA variants are ~2% slower. See mapping-final-results.json.

## Complete grids

20 stable grids each, deterministic matched inputs, two warmup grids discarded/block, forward/reverse order, normal setup/noise/BLAKE3 retained, offline zero target. Clocks not administratively locked.

| Kernel | Median ms/grid | Mean | p95 | Effective TH/s |
|---|---:|---:|---:|---:|
| 67 current N8 | 377.35 | 377.49 | 378.80 | 186.481 |
| 68 best transposed control | 380.00 | 379.685 | 381.73 | 185.181 |

The control loses 0.70% latency. Current baseline's different absolute speed versus prior rounds is run-to-run variation, not a new optimization. Keep variant67.

## Next evidence

tools/profile_mapping_reuse.sh prepares Nsight comparisons 67/68/71. Hardware counters on this host require sudo; the script was syntax-checked, not run. It profiles warmed 4096 regions, not entire grids. Each measured unpadded region executes 16777216 warp-level MMA instructions; divide by measured DRAM/L2/shared bytes separately. Cached-region counters must not be presented as full-grid traffic. Run: sudo bash tools/profile_mapping_reuse.sh.

To pursue a larger tile, first reduce operand and temporary register lifetimes enough to allow residency without spills, then repeat intensity and wall-clock measurements. This experiment does not establish L2 as the causal bottleneck.
