# Grouped scheduling refinement

Six new candidates: M2/M12/M24 and N2/N4/N8. N-group scheduling visits a band of N coordinates before moving along M. CTA128x256, warp2x4, BK64, three stages, MMA and full transcript remain unchanged. Only GPU0 used. All six pass five full-transcript DP4A reference cases, including partial groups. All use 228 registers/thread, 73728 shared bytes/CTA, one resident CTA/SM and zero local allocation.

Region medians in us: original20 344.658, M16 325.064, M2 328.416, M12 327.708, M24 331.162, N2 336.111, N4 334.828, N8 321.058. Five alternating-order passes, 20 warmups plus 150 measured launches per pass.

Two independent complete-grid comparisons: each uses forward/reverse blocks, deterministic matched inputs, two discarded warmup grids/block and 20 stable grids/variant. Normal setup/noise and BLAKE3 retained; offline zero target, no live shares. Clocks not administratively locked.

| Run | M16 median ms | N8 median ms | M16 effective TH/s | N8 effective TH/s |
|---|---:|---:|---:|---:|
| First | 382.00 | 379.20 | 184.211 | 185.572 |
| Repeat | 383.40 | 379.80 | 183.539 | 185.278 |

Candidate67 N8 improves effective throughput 0.74% and 0.95% respectively. This is a modest reproducible advantage in these runs, not a large architectural gain. Actual L2 traffic mechanism remains unverified. No candidate promoted to production or live-share tested. Production library is untouched; independent experimental library saved at /tmp/libpearl-grouped2-candidates.so, variant67 (not a production drop-in).

Diff: grouped2-kernel.diff. Raw: grouped2-region-results.json, grouped2-grid-results.json, grouped2-grid-repeat-results.json and grouped2-combined-results.json. Reproduce: bash tools/build_pure_gemm.sh; PEARL_TRIAL_LIBRARY=/tmp/libpearl-pure-gemm.so .venv/bin/python tests/bench_fold_ablation.py reuse; .venv/bin/python tests/bench_mainloop_grids.py 67 61.

Further small scheduling sweeps are unlikely to explain the public performance gap. Next structural experiment should consider more output reuse per CTA while controlling accumulator registers, or changing warp ownership to improve shared operand reuse. Current evidence does not establish which is better; full-grid hardware counters would narrow the choice.
