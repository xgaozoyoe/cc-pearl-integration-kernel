# Current handwritten Ada kernel: instruction-level profile

## Conclusion

Keep v110. The v115 wide-warp mapping genuinely reduces shared-load dependency samples, but increases fixed-latency wait samples and reduces eligible warps. Increasing operand reuse alone did not improve the kernel. The next experiments should preserve useful warp concurrency while improving issue scheduling and the async-copy producer layout.

The collected reports contain three profiled regional launches per variant, application replay, warm caches, GPU 0 only, unchanged clocks. These are not full-grid wall times or mining throughput measurements. The selected kernel is `pearl_ada_v2::kernel<2,64,false,true,false,false,2,Wide>`, with Wide=false for v110 and true for v115. The v110 instruction sequence was previously checked identical to the frozen approximately 200 TH/s baseline.

## Comparison

| Measurement | v110, warp64x64 | v115, warp64x96 |
|---|---:|---:|
| Mean profiled regional duration | 0.317291 ms | 0.334944 ms |
| Registers/thread | 168 | 247 |
| Resident warps/CTA; one CTA/SM | 12 | 8 |
| Active warps per scheduler | 2.9991 | 2.0000 |
| Eligible warps per scheduler | 0.33955 | 0.27494 |
| Issue-active percentage | 22.58% | 21.60% |
| GPC frequency inferred from cycles/duration | ~2.522 GHz | ~2.517 GHz |
| Dropped sampling bytes | 0 | 0 |

The wide mapping is 5.56% slower in this profile; eligible warps fall 19.03%. Frequencies differ by only about 0.17%, so the timing difference is not explained by a large clock change. The issue-active percentage describes scheduler instruction issue, not Tensor pipe utilization. This script did not collect current Tensor pipe or DRAM-byte metrics.

## Warp samples

Percentages below divide each reason's all-sample count by the total warp PC sample count, summed over three launches. They are not percentages of wall-clock duration or independent recoverable time.

| Reason | v110 | v115 |
|---|---:|---:|
| Math pipe throttle | 41.55% | 38.40% |
| Fixed-latency wait | 19.63% | 25.43% |
| CTA barrier | 15.27% | 14.21% |
| Short scoreboard | 6.03% | 3.02% |
| Long scoreboard | 3.08% | 1.72% |
| MIO throttle | 1.22% | 1.28% |

Absolute not-issued samples clarify the comparison:

- Math throttle: 107,004 to 106,058, essentially unchanged. Its lower percentage in v115 is not evidence of proportional time saved.
- Fixed-latency wait: 41,478 to 61,828, up 49.1%.
- Short scoreboard: 13,432 to 7,493, down 44.2%.
- CTA barrier: 36,619 to 36,607, essentially unchanged.

98.48% and 98.50% of not-issued math-throttle samples land on IMMA instructions in v110 and v115 respectively. The strongest math-throttle PC is v110 offset0x1780 (`IMMA ... R44, R148.ROW, R146.COL, R44`), accounting for 28,252 samples; v115 offset0x1990 (`IMMA ... R24, R152.ROW, R218.COL, R24`) accounts for 46,025. These are instruction hotspots, not isolated instruction runtimes.

In v110, the largest short-scoreboard consumers are IMMA PCs0x16f0,0x1570,0x1830,0x1620,0x14c0; 71.96% of short-scoreboard not-issued samples occur on IMMA. The producer can be a preceding shared load; the consumer PC does not by itself prove an accumulator dependency.

Barrier samples land mostly on the branch immediately following the deferred BAR.SYNC, not the BAR.SYNC opcode itself: v1100xe00 and v1150xdb0. Inspecting neighboring SASS confirms this is the main loop's cp.async wait / CTA synchronization boundary; classifying these as ordinary branch stalls would be incorrect.

## Shared-memory producer finding

The source-counter export reports approximately 4.98 million excessive shared wavefronts per regional launch in both variants. All exported excessive-wavefront counts are attributed to LDGSTS/cp.async stores into shared memory; LDSM shows zero excessive wavefronts in this collection.

Baseline hot producer PCs0x1010 (A) and0x1230 (B) account for nearly all these counts; their corresponding v115 PCs are0xfc0 and0x11b0. Baseline A's producer has excessive wavefronts equal to its ideal-wavefront count. This is evidence to investigate the producer mapping, not proof that eliminating these counts will produce a matching speedup.

Ideal shared wavefronts fall from23,461,888 to20,578,304 per launch, whereas measured shared wavefronts fall from28,443,207 to25,559,187. The unchanged excess limits the benefit from fewer LDSM instructions. N-Way source counters include instruction width effects and should not be read as a direct slowdown factor.

The CUDA+SASS export repeats some instructions under inlined source mappings. Shared-memory analysis deduplicates addresses per launch and checks totals against raw counters. The pure SASS PC-sample export already matches raw sample totals.

## Ranked follow-up experiments

1. Preserve the12-warp baseline and regularize the MMA/LDSM/address schedule around hot consumer PCs. Evaluate eligible warps, fixed-latency wait, and stable full-grid timing together. High math throttle can reflect useful execution-pipe saturation, so reducing that counter alone is not the objective.
2. Change only the cp.async shared-write mapping/swizzle while preserving native MMA ownership. Target the producer excessive-wavefront counts, with bit-exact transcript checks and confirmation that LDSM remains conflict-free.
3. Test a smaller register footprint with multiple resident CTAs, measuring the operand-reuse tradeoff. Avoid widening warp tiles further without a plan to preserve ready warps.

The data supports a scheduling/concurrency problem in addition to reuse. It does not yet identify a specific patch or prove that300 TH/s is achievable under the same conditions.

## Sources and artifacts

NVIDIA defines math-pipe throttle as waiting for the execution pipe, wait as a fixed-latency execution dependency, and short scoreboard as a dependency on an MIO operation such as shared memory: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html .

- Raw reports: `/tmp/pearl-ada-v2-stalls/warm-v110.ncu-rep`, `warm-v115.ncu-rep`, plus their source/raw CSV exports.
- Parsed PC records: `ada-v2-current-stall-analysis.json`.
- Selected raw metrics: `ada-v2-current-stall-metrics.json`.
- Compact comparison: `ada-v2-current-profile-summary.json`.
- Shared-memory instruction counters: `ada-v2-current-shared-conflicts.json`.

No GPU execution, kernel modification or production-library replacement was performed while analyzing these reports.
