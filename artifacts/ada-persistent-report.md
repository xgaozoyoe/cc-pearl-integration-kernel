# Ada persistent tile scheduling experiment

No demonstrated throughput gain. Keep v121 with region8192. Both persistent schedules pass full transcript/digest validation and retain zero accumulator/local-memory spill. This failure is different from the earlier producer/consumer experiment.

## Measured full-grid performance

GPU0 only; mainnet M=N=131072,K4096,R256; region8192. Two runs with20 stable grids per variant after4 warm-up grids per block; second run reverses variant order. Seed sequences match across variants. Generation, V3, noise, GEMM, transcript, PoW and jackpot checks are included. Offline zero target and pool stub: no live shares/proofs/network.

| Variant | Median ms/grid | Mean ms | p95 ms | Effective TH/s | Grids/s | Change vs baseline |
|---|---:|---:|---:|---:|---:|---:|
|v121|318.30|318.408|320.910|221.08|3.1417|+0.00%|
|fixed8192 ordinary|317.85|318.118|320.025|221.39|3.1461|+0.14%|
|persistent N8|319.50|319.800|322.435|220.25|3.1299|-0.38%|
|persistent M16 N8|320.80|320.562|322.210|219.35|3.1172|-0.78%|

The fixed-size ordinary control changes sign relative to baseline between runs; do not promote its aggregate0.14% difference as a reliable gain. Persistent N8 and M16N8 were slower in both run orders. Differences are small, clocks were not locked, and telemetry was sampled after buffer release; no claim of precise cycle-level regression is made. Power policy stayed450W.

## What was changed

- v0: ordinary v121, identical SASS to frozen /tmp/libpearl-ada-linear-copy.so.
- v3: ordinary CTA grid with compile-time8192 dimensions, isolating specialization from persistence.
- v1:128 persistent CTAs process the original N8 task index order, each taking work=blockIdx.x+j*128.
- v2: same persistent assignment, but logical task order groups16 M tiles and8 N tiles, sweeps N for that M window, then advances M.

All have168 registers/thread,384 threads/12 warps,69,632 shared bytes/CTA, one resident CTA/SM, and zero local bytes/LDL/STL. All retain64 IMMA per K64, the same warp64x64/native accumulator ownership, and full transcript.128 accumulators/thread stay in registers.

The two persistent candidates are specialized to8192; their API rejects other dimensions. This is an offline-only generated derivative, not a production dispatch change.

## Spill avoidance without discarding accumulators

The initial persistent form hoisted four per-thread epilogue-index values across the long task loop and spilled16 bytes/thread outside the K loop. SASS showed no accumulator spill, but these saves were eliminated anyway. A fresh PTX tid.x read at the output epilogue prevents that loop-invariant hoisting in persistent variants; the epilogue indices are recomputed after MMA rather than kept live through every tile. Final compiler and SASS reports confirm zero spill in all four variants.

## Data reuse and synchronization scope

For8192, there are32 M tiles x43 N tiles =1376 output CTAs per conventional launch. Each persistent CTA processes10 or11 tasks. The M16N8 window holds a logical16MiB A set and6MiB B set for full windows, targeting L2 locality. This is cache-oriented ordering, not guaranteed cache residency. No extra accumulator bank or producer warp is introduced.

Each output tile starts with zero accumulators, traverses all K4096, writes its complete transcript and advances. Shared operand stages are refilled for each tile. This prototype does not pipeline the next output tile into the current tile or keep a full operand slab in shared memory.

Task-loop safety: the prior tile final CTA barrier completes all operand reads. Output reads only the transcript partition, while next-tile cp.async writes the disjoint operand partition. The next tile first K-loop CTA barrier ensures every old transcript read is complete before any new transcript fold writes it. No grid-wide barrier, atomic queue or new CTA barrier is needed.

This is persistence within each8192 region:256 GEMM launches per complete grid remain in every variant. It reduces CTA creation per launch, not host launch count. The result does not rule out a different design that merges regions or changes the data-production pipeline. Those were not implemented.

## Correctness and evidence

- Every transcript word and BLAKE3 digest matched frozen v121 for two independent random full-range INT8 cases, each8192 squared. Output buffers were poisoned before each candidate launch to expose missing tasks.
- All task indices are computed without atomics; full output comparison exercises final partial N groups and the M-window boundary.
- Racecheck for both persistent variants reports0 hazards/errors/warnings on a complete8192 workload, including repeated shared-stage/task reuse.
- Synccheck for both persistent variants reports0 errors on the complete workload.
- Production libp40cuda.so hash is unchanged; V3, penalized target and proof source were not modified. No GPU1–3 kernels were launched.

## Interpretation and next evidence

The completed counters show no measured data-traffic advantage from the tested schedules. L2 bytes/useful MMA and shared LDSM traffic are essentially identical. Persistent N8 leaves DRAM traffic unchanged; M16N8 increases it. Barrier stall percentage rises, while short-scoreboard and math-throttle percentages fall, without a throughput improvement. These are changes in normalized stall distributions, not additive changes in elapsed time.

Keep v121 with region8192. The evidence does not support further tuning of these two within-region schedules. It does not rule out cross-region fusion or a different operand pipeline; neither was tested.

Profiling reproduction (driver requires privileged counters):

    sudo bash tools/profile_ada_persistent.sh

Reports go to /tmp/pearl-ada-persistent-profile. The script reuses existing ncu-rep files: these captures are dated October7 at14:38–14:39, while the inspected CSV/JSON exports were refreshed at17:47. The later export timestamp is not a new measurement. Use a fresh PROFILE_OUT after rebuilding or when requesting a new capture.

## Reproduce / review

    python3 tools/build_ada_persistent.py
    .venv/bin/python tests/bench_ada_persistent.py validate
    .venv/bin/python tests/bench_ada_persistent.py bench

Build generates /tmp/pearl-persistent-generated.cu from the unchanged Ada source and produces /tmp/libpearl-persistent.so. A copy of the exact generated source, SASS, build/validation/benchmark/sanitizer logs is in ada-persistent-evidence/.

Raw results: ada-persistent-results.json. Correctness: ada-persistent-correctness.json. Source review: ada-persistent.patch. No commit and no production promotion.

## Completed Nsight measurements

Each row averages three8192x8192x4096 regional launches. Kernel timing here is instrumented regional timing, not the normal full-grid wall time above. Replay uses warm caches and unchanged clock policy. No L2 hit-rate counter was collected; do not infer hit rate from throughput utilization.

| Variant | Profile kernel ms | Tensor active | L2 B/useful MMA | L2 utilization | DRAM MB/region | Barrier | Long SB | Short SB | Math throttle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|v121|1.069643|74.88%|37.75840|44.54%|49.178|13.46%|2.79%|5.73%|43.68%|
|fixed8192 ordinary|1.069739|74.92%|37.75915|44.60%|50.466|13.47%|2.80%|5.73%|43.68%|
|persistent N8|1.076789|74.60%|37.75878|44.28%|49.228|21.00%|2.84%|4.05%|38.61%|
|persistent M16 N8|1.071211|75.21%|37.75837|44.63%|54.981|17.09%|2.94%|4.41%|40.88%|

All four variants transfer8,657,043,456 shared LDSM bytes per region. L2 traffic is approximately2.5339GB in every variant. M16N8 DRAM traffic is11.80% above baseline, but only about5.8MB extra per region; that alone does not establish DRAM bandwidth as the limiting factor.

Eligible warps per scheduler/cycle:0.34 /0.34 /0.32 /0.34 (v0/v3/v1/v2). Issue-active cycle fractions:0.22 /0.22 /0.22 /0.2233. Active warps per scheduler are3.00 in all variants. Tensor active is normalized to peak sustained active cycles, not the fraction of theoretical useful MAC throughput delivered.

### Instruction-level attribution

Not-issued barrier samples concentrate at these PCs:

- v0:98.86% at0xe00, following the K-loop BAR.SYNC.DEFER_BLOCKING at0xde0.
- v3:98.83% at0xe00, same K-loop barrier location.
- v1:98.98% at0xd50, following the K-loop barrier at0xd30.
- v2:99.06% at0xf60, following the K-loop barrier at0xf40.

Each preceding sequence is DEPBAR.LE (cp.async wait), loop predicate work, BAR.SYNC.DEFER_BLOCKING, then the sampled branch. These are barrier stalls recorded at deferred blocking points, not branch-resolution stalls. Only about1% of barrier samples occur at the tile epilogue's deferred point. The experiment therefore did not remove the dominant repeated K-loop synchronization.

Short-scoreboard samples remain concentrated at IMMA consumers: the hottest PCs include0x1490/0x1520/0x15b0 in v1 and0x19a0/0x1800/0x16e0 in v2. Math-throttle samples also remain on IMMA sequences. Their reduced aggregate percentages do not demonstrate a speedup: Tensor utilization and eligible-warp supply are effectively unchanged, and normal grid throughput did not improve. There is no accumulator spill in these kernels.

This supports a narrower conclusion than blaming one stall category for all lost time: keeping CTAs resident and reordering their tasks did not reduce traffic or resolve the inner-loop supply/synchronization issue. It is not evidence that L2 reuse is universally unhelpful or that persistent kernels cannot work on Ada.

Archived evidence: ada-persistent-evidence/profile/ (raw metrics, SASS exports, logs, summary and compact PC aggregation). Production code/library remain unchanged; no new GPU experiment was run for this analysis.
