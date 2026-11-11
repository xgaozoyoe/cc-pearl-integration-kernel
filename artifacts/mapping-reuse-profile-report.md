# Mapping operand-reuse counters

User-run tools/profile_mapping_reuse.sh completed successfully: three measured 4096^3 regions per variant, 32 warmups, application replay, cache-control none, requested base clock control. Each unpadded region executes 16777216 warp-level m16n8k32 instructions. Values below are medians, normalized to MMA instructions, not per-thread operations. L2 totals include output/other traffic, whereas shared column counts LDSM reads only.

| Metric | 67 best N8 128x256 | 68 transposed 256x128 | 71 small 128x128 |
|---|---:|---:|---:|
| Profiled kernel us | 372.448 | 377.728 | 499.008 |
| L2 bytes/MMA | 48.263 | 48.263 | 64.264 |
| MMA/L2 byte | 0.02072 | 0.02072 | 0.01556 |
| Shared LDSM bytes/MMA | 128.000 | 128.000 | 128.000 |
| L2 throughput utilization % | 95.072 | 93.758 | 94.524 |
| Tensor metric % | 65.931 | 65.307 | 48.643 |
| Barrier stall % | 18.343 | 20.601 | 21.418 |
| Long scoreboard stall % | 6.368 | 3.898 | 6.028 |
| Short scoreboard stall % | 0.845 | 0.722 | 1.938 |
| DRAM byte median | 0 | 0 | 0 |

Small CTA increases measured L2 bytes/MMA by 33.16% and profiled time by 33.98%, with unchanged shared LDSM bytes/MMA. Two resident CTAs do not compensate. In this warmed-region experiment the strong traffic/time scaling supports L2 operand delivery as the main limiting factor; it is not an isolated causal proof because CTA geometry/concurrency also change.

Baseline requested input model predicts 48 B/MMA; measured L2 total is 48.263. Small predicts 64, measured 64.264. This close agreement supports the repeated-CTA-read interpretation. Logical reuse being possible in L2 does not eliminate per-CTA L2-to-SM transfer requests. Measured effective L2 traffic rate is about 2.17 TB/s for baseline and 2.16 TB/s for small CTA.

Transposed CTA has nearly identical traffic and is 1.42% slower under profiling. Long scoreboard decreases while barrier increases, so optimizing any single stall percentage is not sufficient. Stall percentages are scheduler sampling metrics, not additive wall-clock percentages. The Tensor metric is pct_of_peak_sustained_active, not a direct fraction of end-to-end useful work.

DRAM medians of zero mean these repeated inputs fit the warmed cache; do not infer that real complete grids perform zero DRAM transfers. NCU warns caches are uncontrolled. Three samples and application replay limit precision; no full-grid hardware counters were captured.

Next priority: raise MMA/L2 byte while keeping kernels launchable. For example the attempted wide CTA promises input-only 45 B/useful MMA versus 48 (-6.25%) but does not currently fit register residency constraints. Improving this modest amount alone would not bridge the public-miner gap. Shared reuse remains a distinct metric; changing CTA scheduling alone did not improve its 128 B/MMA.

Raw normalized results: mapping-reuse-counters.json. Original CSV/report files: /tmp/pearl-mapping-reuse-profile/warm-v{67,68,71}.{csv,ncu-rep}. No kernels or production libraries modified during this analysis.
