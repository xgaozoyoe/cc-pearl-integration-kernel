# B-prefetch counters

User-run profile_b_prefetch.sh: three warmed4096 regions per variant, application replay, cache-control none, requested base clocks. Each variant executes17301504 issued MMA instructions,16777216 useful equivalents. No new kernel changes or full-grid tests during this analysis.

| Metric (median) | 88 baseline | 89 prefetch |
|---|---:|---:|
| Profiled kernel us | 383.936 | 382.112 |
| L2 bytes/useful MMA | 38.262707 | 38.262465 |
| LDSM bytes/useful MMA | 132.000029 | 132.000029 |
| LDSM bytes/issued MMA | 128.000028 | 128.000028 |
| Tensor active metric % | 69.3607 | 69.6341 |
| Short scoreboard stall % | 6.0746 | 3.4789 |
| Long scoreboard stall % | 1.0966 | 1.0151 |
| Barrier stall % | 5.5445 | 12.7871 |

Short-scoreboard decreases42.73% relative (2.596 percentage points), confirming that the targeted waiting metric improved. Global/LDSM traffic is unchanged; kernel latency improves0.475%, consistent with previous~0.5% per-run full-grid gains. DRAM medians zero: warmed cache, not full-grid traffic.

Barrier sampling rises7.243 percentage points. Do not subtract/add stall percentages as wall-clock costs. Prefetch may change warp arrival timing or redistribute where stalls are sampled; exact barrier cause needs PC/source attribution. It would be unsupported to claim this12.8% can be eliminated as a12.8% speed gain.

Conclusion: retain89 as the best isolated offline candidate; prefetch helps, but only modestly. Next useful investigation is stall-by-PC around cp.async.wait_group, BAR.SYNC and first LDSM/MMA consumers, together with eligible-warps/issue metrics, rather than increasing prefetch depth blindly. Residency is tight and deeper prefetch may increase live registers. Raw data: b-prefetch-counters.json. Original reports: /tmp/pearl-b-prefetch-profile/warm-v{88,89}.{csv,ncu-rep}.
