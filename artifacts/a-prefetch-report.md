# Further prefetch controls

Instruction-level reports were not present at /tmp/pearl-instruction-stalls when this experiment started. These candidates are hypothesis tests, not optimizations selected from PC-level attribution.

89 remains the best candidate. Added92: same B prefetch with three shared stages;93: two stages with next K32 A row loaded immediately after its last current-slice consumer;94: same A prefetch with three stages. CTA256x192, warp4x3, BK64, complete transcript, N8 scheduling and requested operand payload unchanged. Independent /tmp/libpearl-prefetch2.so; previous /tmp/libpearl-prefetch.so retained. No production, V3, target or proof edits, no shares, GPU0 only.

All new candidates passed five complete DP4A transcript comparisons at256x512,512x256,4096x4096,256x768,768x512, K4096/R256. Two-stage variants89/93 use164 registers/thread and69632 total shared bytes; three-stage92/94 use167 registers and98304 shared bytes. All have384 threads, one CTA/SM, zero local allocation,64 IMMA and24 LDSM instructions per warp per K64 body.

SASS instruction encodings for93 are identical to89. Explicit A dead-row prefetch reproduces scheduling already performed by the compiler, so no additional full-grid test for93 is justified. Evidence: a-prefetch-sass.json.

Region medians:89 315.651 us;92 320.846;93 315.625;94 324.765. Five alternating-order passes with20 warmups and150 measured launches per pass; clocks not locked. Fastest new executable candidate92 is slower, verified in the paired full-grid run recorded in a-prefetch-grid-results.json. Each variant gets20 stable grids, deterministic matched inputs and forward/reverse blocks; discard two warmup grids/block. Normal setup/noise/BLAKE3 retained, offline zero target.

Full-grid medians: baseline89 370.80 ms (189.775 TH/s); candidate92 376.05 ms (187.126 TH/s),1.42% worse latency. Means370.765 vs375.855 ms; p95372.325 vs376.95 ms. This run's absolute baseline throughput differs from earlier runs; it is not a code regression.

No candidate promoted. More source-level preload rearrangement or extra stages has not produced another improvement. Obtain the prepared instruction-level PC reports before choosing a more disruptive wait/barrier change; preserving producer/consumer stage lifetime is essential. Run sudo bash tools/profile_instruction_stalls.sh; sudo is required for this host's restricted hardware counters. That script profiles unchanged88/89 with an independent lineinfo binary whose machine code was verified identical.

Build: REUSE_FOCUS=1 OUTPUT_LIBRARY=/tmp/libpearl-prefetch2.so bash tools/build_pure_gemm.sh. Verify: PEARL_TRIAL_LIBRARY=/tmp/libpearl-prefetch2.so .venv/bin/python tests/bench_fold_ablation.py prefetch2. Grid: PEARL_TRIAL_LIBRARY=/tmp/libpearl-prefetch2.so .venv/bin/python tests/bench_mainloop_grids.py 92 89.

Raw region: a-prefetch-region-results.json. Kernel delta: a-prefetch-kernel.diff. No claim about measured PC-level stalls in these new candidates.
