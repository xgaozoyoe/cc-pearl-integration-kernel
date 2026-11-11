# Ada warp-specialization experiment

**Qualified C: reject these two implementations.** WS1/WS2 are approximately five times slower. All12 consumer warps were retained; the slowdown is associated with register spilling after adding producer resources. This is not evidence that removing consumer warps caused the slowdown, nor proof that every Ada warp-specialized mapping fails.

Production library, live-mining dispatcher, V3 seeds, target logic and proof code were not changed. No pool connection or GPU1–3 execution.

## Controlled measurements

GPU0; M=N=131072, K4096, R256,1024 regions of4096x4096 per grid. Two runs of40 measured grids per variant after warm-up. Rotating variant order; identical production A/B generation, V3/noise setup and seed per matched sample. Full transcript enabled. PoW/jackpot/pool omitted equally.

Kernel time sums1024 CUDA event intervals. TH/s counts each MAC once, matching the miner's MNK/time convention. Offline wall time includes generation, noise, event/dispatch overhead and synchronization. These are not live-miner end-to-end results.

| Variant | Kernel median ms | TH/s equiv, median | Tensor util | L2 B/useful MMA | Barrier | Math throttle |
|---|---:|---:|---|---|---|---|
| Baseline v121 |307.308|228.98|74.69%|38.27|13.42%|43.55%|
| WS1 |1521.418|46.25|15.12%|365.98|13.48%|1.94%|
| WS2 |1510.815|46.58|15.16%|365.59|14.71%|1.85%|

All three privileged profiles are complete, with three regional launches per variant. Tensor percentage is IMMA active cycles relative to peak sustained active; the normal full-grid timings in this table are separate from instrumented regional replay timings. Baseline was captured earlier and reused; these profiles are not an interleaved, locked-clock timing comparison. No historical counters were substituted.

| Variant | Kernel mean / p95 ms | Kernel grids/s, mean | Offline wall median / mean / p95 ms | Offline grids/s | Offline TH/s |
|---|---:|---:|---:|---:|---:|
| Baseline |307.267 /307.871|3.2545|332.684 /332.602 /333.335|3.0066|211.57|
| WS1 |1520.939 /1522.420|0.6575|1547.031 /1546.521 /1547.958|0.6466|45.50|
| WS2 |1510.366 /1512.929|0.6621|1536.631 /1536.104 /1538.792|0.6510|45.81|

| Variant | Run1 kernel median ms | Run2 kernel median ms |
|---|---:|---:|
| Baseline |307.158|307.489|
| WS1 |1521.189|1521.666|
| WS2 |1510.303|1510.914|

Same450W power limit and driver settings; memory clock10501MHz. Sampled core boost2685–2730MHz: these were not locked-clock tests. Alternation controls drift but not workload-dependent DVFS. This small clock variation cannot plausibly account for the approximately5x difference. Full conditions are in the grid log.

The warm-cache4096-cubed regional test also used2x40 samples, each averaging32 launches: median0.29354 /1.47893 /1.46813ms. Full-grid measurements above are the primary results.

## Resources, correctness and instruction evidence

| Variant | Producers / consumers | Threads | Registers/thread | Shared/CTA | Local/thread | CTAs/SM | Resident warps/SM | Static LDL / STL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline |0 /12|384|168|69,632 B|0|1|12 (25%)|0 /0|
| WS1 |1 /12|416|128|69,632 B|296 B|1|13 (27.08%)|87 /84|
| WS2 |2 /12|448|128|69,632 B|296 B|1|14 (29.17%)|87 /84|

Producer-only threads still consume the kernel's uniform per-thread register allocation. More resident warps do not imply more useful MMA capacity. ptxas reports444 bytes spill stores and420 bytes spill loads for each WS kernel; these are static compiler reports, not dynamic traffic. The baseline has no spills.

All candidates retain CTA256x192x64, warp64x64, two stages,64 K iterations, INT8 m16n8k32 MMA and192 transcript tiles/CTA. Same useful MMA count and input-copy volume, but actual L2 traffic per useful MMA increased approximately9.6x. The measured arithmetic-intensity success criterion therefore fails despite preserving the geometric operand reuse.

- All three variants pass the existing five-shape full-transcript validation against DP4A, including partial CTA edges.
- Compute Sanitizer synccheck:0 errors. Racecheck:0 hazards/errors/warnings on a64-iteration workload.
- Experimental baseline SASS is instruction-identical to frozen v121.
- Every kernel retains64 static IMMA and16 LDSM instructions in the K64 loop.
- WS SASS has separate consumer LDSM/IMMA and producer LDGSTS/LDGDEPBAR/DEPBAR paths, joined with BAR.ARV/BAR.SYNC handoffs. No SM90 mechanisms.

## Mapping and synchronization

    CTA256x192:        wn0      wn1      wn2
    wm0               warp0    warp4    warp8
    wm1               warp1    warp5    warp9
    wm2               warp2    warp6    warp10
    wm3               warp3    warp7    warp11

WS1 adds warp12; WS2 adds warps12–13. Preserving consumers avoids changing output coverage and accumulator ownership, but increases resource demand. This is not a constant-total-thread comparison.

    baseline: wait copyK -> CTA sync -> enqueue copyK+1 -> LDSM/IMMA K -> fold
    producer: copy0/wait -> ready0 -> copy1/wait -> ready1 -> wait empty0 -> copy2
    consumer: wait ready0 -> LDSM/IMMA/fold0 -> empty0 -> wait ready1 -> ...

Baseline already overlaps asynchronous DMA with compute. WS changes who issues and waits. Named ready/empty barriers include all CTA threads but use nonblocking arrival for the publishing/releasing role. Producers wait for their own copies before publishing; consumers release only after completing stage reads. One final full-CTA barrier protects transcript output. Exact lane coordinates, safety reasoning and PCs: [design](ada-ws-design.md).

## Decision and remaining work

Dedicated data-movement warps did not outperform the homogeneous CTA in this experiment. WS2 improves only slightly on WS1. Counters and SASS strongly support spill traffic and dependent accumulator reloads as the dominant failure mechanism, not fewer consumer warps. A spill-free counterfactual was not measured, so the exact fraction of the slowdown caused by spilling versus handoff overhead is not isolated.

**Qualified C: abandon WS1/WS2 as implemented.** A design that redistributes accumulator ownership within a fixed thread budget is a different experiment. It was not started. Do not promote either candidate to production.

Build: bash tools/build_ada_ws.sh

Offline checks: .venv/bin/python tests/bench_ada_ws.py validate

Regional benchmark: .venv/bin/python tests/bench_ada_ws.py bench

Full-grid benchmark: .venv/bin/python tests/bench_ada_ws_grids.py

Completed privileged profiling: sudo bash tools/profile_ada_ws.sh (existing reports are reused).

Run profiling with GPU0 idle. It writes raw metrics, SASS PC samples and summaries to /tmp/pearl-ada-ws-profile, keeping driver clock settings. Instrumented replay timings must be distinguished from normal benchmark timing.

Review patch: [ada-ws.patch](ada-ws.patch). Raw timing: [full grids](ada-ws-grid-results.json), [regions](ada-ws-results.json). Evidence: ada-ws-evidence/. No commit.

Production SHA256 remains dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6. Frozen baseline: /tmp/libpearl-ada-linear-copy.so. Experiment: /tmp/libpearl-ada-ws.so. Binary/source hashes: ada-ws-binaries.json.

## Completed instruction-level profile, October7

| Metric | Baseline | WS1 | WS2 |
|---|---:|---:|---:|
| Profiled regional kernel mean, us |315.904|1459.936|1446.272|
| L2 throughput utilization |40.98%|89.70%|90.45%|
| L2 bytes/region |642,086,763|6,140,086,400|6,133,631,467|
| Long scoreboard |2.91%|48.36%|44.28%|
| Short scoreboard |5.77%|4.47%|4.27%|
| Eligible warps/scheduler/cycle |0.34|0.09|0.09|
| Issue-active cycles, fraction |0.22|0.07|0.07|
| Active warps/scheduler |3.00|3.25|3.50|
| LDSM bytes/region |2,214,592,512|2,214,592,512|2,214,592,512|

The active-warps and eligible-warps values are scheduler-normalized, not whole-SM counts. Stall percentages are normalized warp-state metrics, not additive fractions of elapsed grid time. PC percentages below are distributions within one stall sample column, not percentages of kernel time.

### Accumulator reloads block IMMA

For WS1/WS2,70.40%/68.10% of not-issued long-scoreboard PC samples land on IMMA. The hottest single PC is0x1a90 (13.34%/13.32% of those samples). SASS shows:

    0x1a70 LDL.LU.64 R10, [R1+0xd8]
    0x1a80 LDL.LU.64 R8,  [R1+0xd0]
    0x1a90 IMMA ... R8, R24.ROW, R104.COL, R8

The R8–R11 accumulator is reloaded immediately before consumption. Another hot IMMA at0x1b40 consumes R112–R115 following local reloads of R113–R115. These are memory dependencies observed at the consuming instruction; a sampled IMMA does not imply Tensor-pipe saturation. Baseline, in contrast, places98.10% of its not-issued math-throttle samples on IMMA.

Equal LDSM byte totals support unchanged shared operand consumption. Extra L2 traffic is approximately5.5GB per region, consistent with the compiler/SASS spill evidence. The collected counters do not directly isolate local-memory bytes from every other L2 source, so attributing every additional byte to spill would overstate the measurement. DRAM traffic is approximately0/32.93/31.93MB per warm-cache region: this is principally an L2 traffic/dependency problem, not a multi-gigabyte DRAM transfer.

### Handoff waiting was not eliminated

Baseline:98.82% of not-issued barrier samples occur at0xe00, after the main-loop BAR.SYNC.DEFER_BLOCKING at0xde0.

WS1/WS2:94.43%/78.64% occur at0xca0, after the consumer ready barrier at0xc30;3.72%/18.97% occur at0x2840, after the producer empty barrier at0x2820. The remainder is at the final epilogue barrier's deferred blocking point. The sampled STL at0xca0 must not be interpreted as a spill-memory stall: its recorded reason is barrier, inherited from the earlier deferred barrier. Likewise the producer BRA location is not branch-resolution evidence.

WS2 shifts the barrier-sample distribution toward producers waiting for empty stages but improves normal full-grid kernel time by only approximately0.7%. More producers do not rescue this resource-constrained design. Source and SASS permit producer/consumer overlap; these samples do not quantify the fraction of time that overlap was actually achieved.

### Conclusion scope

C for these implementations: retain the homogeneous baseline. Extra producer threads reduced the compiled register budget from168 to128/thread, introduced accumulator spills, increased L2 B/MMA approximately9.6x and collapsed Tensor utilization. All12 consumer warps were retained, so this experiment does not demonstrate a loss caused by removing MMA warps. It also does not cleanly measure the value of specialization independently of register pressure. A fixed-total-thread, spill-free alternative remains untested; no further redesign or tuning was started.

Raw metrics, SASS exports, logs and compact PC attribution are archived in ada-ws-evidence/completed/. All variants passed the earlier offline correctness and sanitizer checks; production remains unchanged.
