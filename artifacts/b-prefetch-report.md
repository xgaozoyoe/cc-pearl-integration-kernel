# One-group-ahead B prefetch

Candidates89/90/91 preserve variant88's CTA256x192, warp4x3, BK64, two shared stages, grouped-N8 scheduling, MMA and complete transcript/PoW. They load the next B copy group before current MMAs, after half0 finishes, or midway through half0. Each B group is loaded once; analytic operand traffic is unchanged.

All three pass complete DP4A transcript comparisons at256x512,512x256,4096x4096,256x768,768x512, K4096/R256. All remain resident, use164 registers/thread (baseline166),384 threads,69632 total shared bytes, one CTA/SM and zero local allocation. SASS encodings for89/90/91 are identical; benchmark the simplest89. Per-warp K64 body remains64 IMMA,8 LDSM.x4,16 LDSM.x2, zero LDL/STL. Disassembled instruction count falls720->712. Encodings/hash evidence: b-prefetch-sass.json.

Region tests have timing drift: baseline88 median325.298 us versus candidates~315.6-315.8 us, but per-pass differences are smaller and variable. Do not interpret the median gap as a stable3% improvement.

Two full-grid forward/reverse comparisons, deterministic matched inputs, normal setup/noise/BLAKE3, two warmup grids discarded/block,20 stable grids per variant per run; GPU0 only, offline zero target. Clocks are not administratively locked.

| Run | Baseline88 median ms | Prefetch89 median ms | Baseline TH/s | Prefetch TH/s |
|---|---:|---:|---:|---:|
| First | 370.65 | 368.80 | 189.852 | 190.805 |
| Repeat | 369.55 | 367.60 | 190.417 | 191.427 |
| Combined40 grids | 370.40 | 368.10 | 189.980 | 191.167 |

Combined means370.3225->368.0425 ms. Each independent run improves~0.50-0.53% throughput; combined-median improvement~0.62%. Modest reproducible benefit in these conditions, not evidence of a large performance jump. Actual short-scoreboard reduction has not yet been measured.

Independent library /tmp/libpearl-prefetch.so, variant89. Production libp40cuda.so unchanged, SHA256 dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6. No seed/target/proof changes, live shares or commits; prior variant88 library retained at /tmp/libpearl-operand-stream.so.

Build: REUSE_FOCUS=1 OUTPUT_LIBRARY=/tmp/libpearl-prefetch.so bash tools/build_pure_gemm.sh. Verify: PEARL_TRIAL_LIBRARY=/tmp/libpearl-prefetch.so .venv/bin/python tests/bench_fold_ablation.py prefetch. Grid: PEARL_TRIAL_LIBRARY=/tmp/libpearl-prefetch.so .venv/bin/python tests/bench_mainloop_grids.py 89 88.

Counter script prepared and syntax-checked: sudo bash tools/profile_b_prefetch.sh; requires privileged hardware counter access on this host, not run during this experiment. Compares88/89. Both execute17301504 issued MMA instructions and16777216 useful MMA equivalents per4096 region. Use it to check actual short-scoreboard changes and verify unchanged traffic. No mining promotion before live-share validation.

Raw: b-prefetch-region-results.json, b-prefetch-grid-results.json, b-prefetch-grid-repeat-results.json, b-prefetch-grid-combined.json. Kernel delta: b-prefetch-kernel.diff.
