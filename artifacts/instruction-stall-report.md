# Instruction-level stall attribution: variants 88 and 89

GPU 0 only; SM89; Nsight Compute 2024.3.2; base clock control; application replay; uncontrolled/warm caches. Three sampled 4096³, R256 launches per variant, after 32 warmups. No dropped PC-sampling bytes. These are sampled region kernels, not end-to-end grids. No production library changes.

| Metric | 88: streamed B | 89: B prefetch |
|---|---:|---:|
| Median sampled kernel, µs | 384.032 | 382.176 |
| Math-pipe throttle / all PC samples | 49.735% | 43.853% |
| Fixed-latency wait / all PC samples | 18.907% | 20.976% |
| Barrier / all PC samples | 5.513% | 12.675% |
| Short scoreboard / all PC samples | 6.113% | 3.450% |
| Long scoreboard / all PC samples | 1.049% | 1.017% |
| MIO throttle / all PC samples | 1.079% | 1.284% |
| Issue-active cycles | ~26.56% | ~26.46% |
| Eligible warps / scheduler / cycle | ~0.457 | ~0.434 |
| Active warps / scheduler / cycle | ~3.0 | ~3.0 |
| Registers / thread | 166 | 164 |
| Shared bytes / CTA | 69632 | 69632 |

Percentages are warp-state sample fractions, not fractions of kernel time. Not-issued samples are subsets and must not be added to their parent columns. Math throttle at IMMA indicates unavailable target-pipeline issue capacity; it is not a direct measurement of Tensor Core utilization or recoverable speedup. Fixed-latency wait is distinct from async-copy wait.

## PC attribution

Offsets refer to each kernel entry, so ASLR addresses do not matter.

- Async wait: DEPBAR at 0x0fa0; no long-scoreboard samples attributed to this PC. This does not establish zero copy latency: deferred dependencies can be attributed to later instructions.
- Main-loop barrier: BAR.SYNC.DEFER_BLOCKING at 0x0fb0; 68.43% (88) / 77.47% (89) of long-scoreboard samples are here.
- Instruction immediately following the main-loop barrier: branch at 0x0fc0; 15922/16119 = 98.78% (88), 36165/36914 = 97.97% (89) of barrier samples are here. The rest are at the instruction after the transcript epilogue barrier: 0x29b0 / 0x2980.
- IMMA accounts for 93.75% / 93.45% of math-pipe-throttle samples.
- IMMA accounts for 71.64% / 43.41% of short-scoreboard samples. Absolute IMMA short-scoreboard counts fall 12806 -> 4361 across three launches. Total short-scoreboard fractions fall 43.6% relatively.
- 89 math-throttle concentration: IMMA at 0x1990, 0x1a40, 0x1730. Together 71785/127718 = 56.2% of math-throttle samples. Inspect these instruction bundles and their neighboring LDSM before changing scheduling.
- No local spill instructions appear in the source export; both variants report local allocation zero.

## Interpretation and next experiments

Prefetch genuinely reduces shared-operand dependency stalls, but the measured kernel improvement is only 0.48%. Waiting shifts toward main-loop synchronization and fixed-latency scheduling; transcript epilogue synchronization is a minor contributor. More prefetch distance alone is not supported as a large performance opportunity.

Ranked next experiments (not performed in this report):
1. Redistribute LDSM/address work between MMA bursts at the identified hot PCs, preserving the MMA count and operand reuse; inspect resulting SASS and benchmark full grids.
2. Investigate per-warp work balance and async-copy ownership around the main-loop barrier. A barrier sample can reflect arrival skew; do not remove required synchronization or equate the sampled percentage with recoverable time.
3. Reduce register/CTA resource pressure enough to test two resident CTAs with comparable operand reuse. Prior smaller-tile experiments lost performance, so occupancy alone is not a justification.

Current best previously measured end-to-end throughput remains roughly 190–191 TH/s. This profile does not establish a new end-to-end throughput figure.

Raw files: /tmp/pearl-instruction-stalls/warm-v{88,89}.{ncu-rep}, accompanying source/raw CSVs. Machine-readable attribution: artifacts/instruction-stall-analysis.json.
