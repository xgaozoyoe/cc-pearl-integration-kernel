# Operand reuse with reduced register lifetime

Best new offline candidate: variant88, CTA256x192, warp4x3 (12 warps), BK64, two shared stages, full transcript stored in shared memory and coalesced at the end. One K32 A slice remains live while each B copy group (two 8-column MMA halves) is consumed by all four row groups. Normal PoW BLAKE3 preserved. No production or seed/target/proof changes, no live shares, GPU0 only.

## Correctness and implementation

Initial streaming-copy indexing incorrectly treated one B copy group as one MMA half and caused an illegal access in an experimental process. Fixed after inspecting actual CuTe copy/fragment layouts: one group covers two halves. Static assertions check group counts. All launchable final candidates passed complete transcript comparisons against DP4A at 256x512,512x256,4096x4096,256x768,768x512, K4096/R256, including padded edges. Nonresident variants80/84 excluded before launch.

Shared transcript plus shorter operand lifetime lowers the larger CTA register count from 190 to 168 (s3) or 166 (s2), making it launchable without local allocation. Full register-transcript variants remain nonresident. SASS for 67/86/88 shows identical per-warp K64 body counts: 64 IMMA.16832.S8.S8, eight LDSM.x4, sixteen LDSM.x2, zero LDL/STL. Mode9 adds transcript shared stores and one epilogue barrier.

| Resource | Previous best67 | Candidate88 |
|---|---:|---:|
| Registers/thread | 228 | 166 |
| Threads/CTA | 256 | 384 |
| Warps/CTA | 8 | 12 |
| Total shared bytes/CTA | 73728 | 69632 |
| Resident CTA/SM | 1 | 1 |
| Theoretical resident warps / 48 | 8/48 | 12/48 |
| K iterations/CTA | 64 | 64 |

## Reuse accounting

Input-only requested bytes/useful MMA decrease from 48 to 38 over a complete 4096 region (-20.83%). Interior value is 37.333; edge rounding increases it to 38 and adds 3.125% MMA work. This is an analytic requested-payload calculation, not measured L2 transactions. Ideal LDSM bytes/issued MMA remain128; bytes/useful MMA increase to132 due to padding. Transcript shared traffic is additional.

K32 larger-CTA candidates were slower (~363-380 us versus baseline~322 us). K64 restores fewer barriers; best region candidate88 is 322.758 us versus67 322.280 us, essentially tied. The full-grid advantage below must not be inferred from warmed-region performance alone.

## End-to-end results

Two independent forward/reverse comparisons with deterministic matched inputs and normal setup/noise/BLAKE3, two warmup grids discarded/block, 20 stable grids per variant per run. Offline zero target, no shares submitted. Clocks not administratively locked.

| Run | Baseline67 median ms | Candidate88 median ms | Baseline effective TH/s | Candidate effective TH/s |
|---|---:|---:|---:|---:|
| First | 378.50 | 370.60 | 185.915 | 189.878 |
| Repeat | 378.40 | 371.40 | 185.964 | 189.469 |
| Combined 40 stable grids | 378.40 | 370.95 | 185.964 | 189.699 |

Combined means: baseline378.335 ms, candidate370.9625 ms. Median latency improves1.97%; effective throughput improves2.01%. Keep candidate88 for further profiling; do not replace the known-working mining library before live-share validation. These measurements do not establish that measured L2 traffic actually falls20.83% or explain why the cached-region result differs.

## Artifacts and reproduction

Isolated library: /tmp/libpearl-operand-stream.so. Kernel delta: operand-stream-kernel.diff. Raw resources/timings: operand-stream-results.json; full grids: operand-stream-grid-results.json, operand-stream-grid-repeat-results.json, operand-stream-grid-combined.json; SASS counts: operand-stream-sass-counts.json. Production libp40cuda.so SHA256 remains dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6.

Build: REUSE_FOCUS=1 bash tools/build_pure_gemm.sh. Correctness/region test: PEARL_TRIAL_LIBRARY=/tmp/libpearl-operand-stream.so .venv/bin/python tests/bench_fold_ablation.py operand. Grid: PEARL_TRIAL_LIBRARY=/tmp/libpearl-operand-stream.so .venv/bin/python tests/bench_mainloop_grids.py 88 67.

Counter script prepared and syntax-checked, not executed: sudo bash tools/profile_operand_stream.sh. Compares67/86/88 on warmed regions. Hardware counter access requires sudo on this machine. Padded candidates execute17301504 warp-level MMA instructions/region; compare both issued-MMA/byte and useful-MMA/byte (16777216 useful instructions). Next analysis should separate actual L2 reduction from shared, barrier and issue-efficiency costs; warmed-region counters do not replace full-grid measurements.
