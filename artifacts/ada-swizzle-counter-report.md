# Ada swizzle counter follow-up

The producer-wavefront hypothesis was not supported: changing swizzle or reordering producer lanes did not materially reduce excessive shared wavefronts. Keep v121 as the small measured performance improvement, but do not attribute its gain to removing these wavefronts.

| Measurement, mean of 3 regional launches | v110 baseline | v117 xor3 | v121 direct producer |
|---|---:|---:|---:|
| Profiled kernel duration |317.323 us|315.317 us|315.723 us|
| Excessive shared wavefronts |4,981,313|4,981,408|4,981,121|
| LDSM excessive wavefronts |0|0|0|
| Eligible warps/scheduler |0.3396|0.3612|0.3359|
| Issue-active percentage |22.585%|22.851%|21.789%|
| Inferred GPC frequency |2.5208 GHz|2.5206 GHz|2.5197 GHz|
| Dropped sample bytes |0|0|0|

The v121 excessive-wavefront count differs from baseline by only about0.004%, whereas regional duration improves about0.50%. This agrees in direction and size with the earlier40-grid end-to-end comparison (198.98→199.97 TH/s), although regional profile timings are not mining throughput.

Deduplicating repeated source mappings confirms that all excessive shared wavefronts still belong to LDGSTS/cp.async in every variant. LDSM has no excessive wavefronts in this collection. Shared ideal traffic and LDSM traffic are identical across variants. These counters do not demonstrate that the measured producer excess can be removed by a lane permutation, nor establish a recoverable wall-time fraction.

## Stall distribution

Percentages are shares of all PC samples, not percentages of wall-clock duration.

| Reason |v110|v117|v121|
|---|---:|---:|---:|
|Math pipe throttle|41.55%|47.77%|43.55%|
|Barrier|15.22%|8.87%|13.39%|
|Fixed-latency wait|19.64%|20.09%|20.70%|
|Short scoreboard|6.06%|5.42%|5.76%|
|Long scoreboard|3.14%|3.02%|2.90%|
|MIO throttle|1.20%|0.86%|0.49%|

For v117, not-issued barrier samples fall from36,370 to20,968, but not-issued math-throttle samples rise from106,859 to123,061. Over98% of not-issued math-throttle samples occur on IMMA in every variant. Lower barrier samples alone therefore do not predict a large speedup.

For v121, simpler address generation (740 static SASS instructions versus748), changed scheduling, and lower MIO/barrier samples are consistent with a small improvement. This is an interpretation; the profile does not isolate a single causal mechanism. Issue-active percentage is a scheduler metric, not Tensor Core utilization; this capture has no direct Tensor pipe utilization measurement.

## Decision

Stop broadening the swizzle search for now. The experiment removed redundant address work, but did not affect the targeted wavefront count. The next kernel experiment should retain12 resident warps and examine how MMA, LDSM, address arithmetic and transcript work are scheduled around the main-loop boundary. Judge candidates by matched full-grid timings, correct transcripts, register pressure and eligible warps together. Do not interpret high math throttle as entirely wasted time: it can accompany useful execution-pipe saturation.

No kernel changes or GPU workloads were performed during this analysis. Production remains unchanged.

Artifacts:
- `ada-swizzle-counter-summary.json`: raw metric means and sample fractions.
- `ada-swizzle-stall-analysis.json`: per-PC samples and opcode attribution.
- `ada-swizzle-shared-attribution.json`: deduplicated shared-wavefront attribution.
- Original reports: `/tmp/pearl-ada-swizzle-stalls/`.
