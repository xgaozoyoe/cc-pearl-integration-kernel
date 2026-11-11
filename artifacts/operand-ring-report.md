# Explicit rolling-operand experiment

GPU0 only; isolated /tmp/libpearl-operand-ring.so. Normal production binary/seed/target/proof paths unchanged. No commit or promotion.

## Source change

Variants101/102 reuse A fragment slice0 across both K32 slices. Each A row is overwritten after its last previous-slice consumer. B fragments use a three-slot/two-slot source-level ring, with two-group/one-group lookahead respectively. Same CTA256×192,warp4×3,warp tile64×64,K64,2sharedstages,N8 scheduling. Original shared-stage synchronization and transcript boundaries remain unchanged.

## Verification and measurements

Both variants match the complete DP4A transcript for256×512,512×256,4096²,256×768,768×512; K4096/R256. Five forward/reverse batches,20warmups/150timed launches each.

| Variant | Region CUDA event median ms | Registers/thread | Shared bytes/CTA | Resident CTA/SM | Local bytes |
|---|---:|---:|---:|---:|---:|
|89 baseline|0.315465|164|69632|1|0|
|101 three slots|0.319256|166|69632|1|0|
|102 two slots|0.318975|166|69632|1|0|

Crucially,101/102/99 have exactly identical SASS instruction encodings (both encoding words compared), including register assignment: SHA256 e645a875f4038759c66eacc877c523617df4aa972484fe6ecdf109b3f5ddd5c8. All have712static instructions,64IMMA,24LDSM,zeroLDL/STL. The compiler optimized source-level ring reuse into the same schedule as the previous flattened pipeline. Logical slots in C++ do not pin physical registers or constrain ptxas scheduling.

No duplicate full-grid measurement is justified. Previous matched-input20stable-grid A/B measurement of this same99 machine code:89 370.9ms/grid,189.724TH/s;99 375.7ms/grid,187.300TH/s (~1.3% slower). These numbers are prior measurements, not new end-to-end measurements of101/102. Timing differences among identical101/102 code are noise.

## Decision and next evidence

Keep89. Ring indexing alone neither reduces registers nor realizes a fixed cadence. Do not infer that all fixed-cadence software pipelines are ineffective.

Before adding artificial runtime dependencies, compare PC stalls of89 and99 to determine whether the changed cadence worsens operand dependencies, math-pipe contention or main-loop synchronization. Prepared command: sudo bash tools/profile_operand_ring.sh. It uses GPU0,32warmups/3launches,base clock control, warm-cache application replay, SourceCounters/SchedulerStats/WarpStateStats and writes a separate /tmp/pearl-operand-ring-stalls directory. Existing instruction reports are preserved. Section identifiers and shell syntax validated. Collection has not been performed; restricted counter access requires the user’s sudo invocation on this host.

Once collected, use tools/analyze_instruction_stalls.py on warm-v89-sass.csv and warm-v99-sass.csv. PC offsets for99 must come from its own SASS; do not reuse89 hot-PC offsets blindly. Samples are warp-state fractions, not recoverable wall-clock percentages.

Evidence: operand-ring-kernel.diff,operand-ring-sass.json,operand-ring-region-results.json; prior k64-rhythm-grid-results.json.
