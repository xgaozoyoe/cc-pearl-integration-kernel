# Grouped CTA scheduling, GPU 0

Candidate 61 groups 16 M-axis CTAs and sweeps N within each group. Only block-to-tile coordinates change; CTA128x256, eight warps (2x4), BK64, three stages, MMA, transcript and PoW stay unchanged. Alternative groups 1/4/8 also tested. Mapping is bijective for full and partial groups.

All four candidates passed complete DP4A transcript comparisons at 256x512,512x256,4096x4096,128x256,384x768, K4096/R256. All report 228 registers/thread, 73728 shared bytes/CTA, one CTA/SM and zero local allocation.

Region medians (five forward/reverse passes, 20 warmups and 150 measured launches each): baseline 344.943 us; M1 333.626; M4 326.492; M8 324.123; M16 322.219. Unlocked clocks and thermal drift limit exact region comparisons.

Full-grid matched-input forward/reverse comparison, GPU0, 20 stable grids per variant, two warmup grids discarded per block. Normal noise/setup and BLAKE3 retained, offline zero target, no live shares:

| Kernel | Median ms | Mean ms | p95 ms | Grids/s | Effective TH/s |
|---|---:|---:|---:|---:|---:|
| Production 0 | 427.05 | 427.35 | 430.265 | 2.34165 | 164.779 |
| Previous best 20 | 391.35 | 390.95 | 392.585 | 2.55526 | 179.810 |
| Group M16 61 | 381.80 | 382.21 | 384.235 | 2.61917 | 184.308 |

Median latency decreases 9.55 ms (2.44%); effective throughput increases 2.50%. Mean latency decreases 8.74 ms (2.24%). Clocks are not administratively locked. The observed benefit does not prove decreased L2 traffic; collect counters to establish the mechanism. No mining promotion or accepted-share claim for this candidate.

Production SHA256 unchanged: dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6. Isolated library saved as /tmp/libpearl-grouped-candidates.so, variant61; it contains other experimental variants and is not a drop-in production library. No seed/target/proof edits or commits.

Review kernel delta in grouped-kernel.diff. Sources: tests/fold_ops.cuh and tests/fold_ops.cu. Measurements: grouped-region-results.json and grouped-grid-results.json. Reproduce with bash tools/build_pure_gemm.sh, then PEARL_TRIAL_LIBRARY=/tmp/libpearl-pure-gemm.so .venv/bin/python tests/bench_fold_ablation.py reuse and .venv/bin/python tests/bench_mainloop_grids.py 61.
