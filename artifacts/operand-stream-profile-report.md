# Operand-stream hardware counters

User-run profile_operand_stream.sh completed: three measured warmed 4096 regions per variant, application replay, cache-control none, requested base clocks. NCU warns uncontrolled caches. Profiles are not full-grid timings. Baseline67 executes16777216 warp-level MMA instructions; padded variants86/88 execute17301504, including3.125% extra N work.

| Metric (median) | Baseline67 | Large s3 86 | Large s2 88 |
|---|---:|---:|---:|
| Kernel us | 372.672 | 395.392 | 383.872 |
| L2 bytes/useful MMA | 48.266 | 38.264 | 38.263 |
| L2 bytes/issued MMA | 48.266 | 37.104 | 37.104 |
| Useful MMA/L2 byte | 0.02072 | 0.02613 | 0.02613 |
| LDSM bytes/useful MMA | 128.000 | 132.000 | 132.000 |
| LDSM bytes/issued MMA | 128.000 | 128.000 | 128.000 |
| L2 throughput metric % | 95.254 | 35.711 | 36.717 |
| Tensor metric % | 66.148 | 67.601 | 69.347 |
| Barrier stall % | 18.346 | 12.373 | 5.555 |
| Long scoreboard stall % | 6.306 | 0.643 | 1.102 |
| Short scoreboard stall % | 0.848 | 9.165 | 6.072 |

DRAM byte medians are zero in all cases: this is a warmed-cache experiment. Measured L2 bytes/useful MMA fall20.72%, close to the model's20.83% requested-input reduction. Useful MMA per L2 byte improves26.14%. Operand-reuse improvement is confirmed at this layer, not just theoretical.

The reduction does not translate proportionally to performance. Candidate88 is3.01% slower in this profiled warmed region; previous two unprofiled full-grid comparisons showed~2% end-to-end improvement. These are different workloads/clock/instrumentation conditions and should not be substituted for one another. No new full-grid measurements in this analysis.

The L2-throughput percentage decreases much more than the byte/time rate: effective byte rates are~2.173 TB/s baseline and~1.672 TB/s candidate88. Do not treat36.7% as literal occupied fraction of an aggregate byte-bandwidth ceiling, or infer2.7x attainable performance. Composite metric behavior and traffic mix need further attribution.

Barrier and long-scoreboard stalls decrease, while short-scoreboard increases~7.16x. This is consistent with reduced global-delivery pressure and increased short-latency operand dependencies after streaming B immediately before MMA. It does not identify the exact responsible instruction or prove bank conflicts: shared throughput, bank-conflict metrics and stall-by-PC were not collected. Tensor metric is pct_of_peak_sustained_active, not wall-clock utilization.

Next controlled experiment: prefetch the next B copy group before consuming the current group, keeping one extra group live. This targets LDSM-to-MMA latency while preserving the improved global-reuse geometry. Check register residency first: candidate88 has166 registers/thread and a tight resource budget, so more overlap may make it nonresident. Compare short-scoreboard, actual shared bytes, L2 bytes and full-grid wall time; do not promote on utilization alone.

Raw normalized data: operand-stream-counters.json. Original reports: /tmp/pearl-operand-stream-profile/warm-v{67,86,88}.{csv,ncu-rep}. Production and experiment kernels were unchanged during this analysis.
