# Main operator diagnostic results, 2026-10-06

GPU 0 only; isolated `/tmp/libpearl-fold-ablation.so`. Production library SHA256
remains `dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6`.

Each measurement uses M=N=K=4096, R=256; five forward/reverse passes,
20 warmup and 150 timed launches per configuration per pass, CUDA events.
Clocks were not locked. These are region-kernel timings, not wall-clock mining
throughput or live accepted-share measurements.

## Fold ablation

| Layout | Full transcript | Local XOR only | Final-only GEMM checksum |
|---|---:|---:|---:|
| 8x1 | 352.788 us | 345.566 us | 347.450 us |
| 2x4 | 340.342 us | 338.493 us | 338.384 us |

Removing intermediate sampling improves the median 2x4 kernel time by only
0.575%, smaller than the across-pass spread. It does not reveal a large hidden
fold cost. The compiler chooses 228 registers/thread for full 2x4 and 234 for
final-only, so this is a compiled-configuration comparison, not an exact
instruction-level attribution of fold time. Final-only still performs one
complete final XOR and writes one checksum word per tile; it does not write
the full C matrix. All C elements contribute to the checksum, preventing
dead-code elimination of the matrix product.

Full variants matched all transcript words of DP4A; final-only variants matched
word 15 and left other words untouched. Three random cases: 256x512, 512x256,
4096x4096, K4096/R256. Diagnostic IDs 23-26 are rejected by trial_pow.

## Occupancy experiment (separate run)

| CTA / warp / BK | Registers | Shared bytes | Resident CTAs/SM | Median |
|---|---:|---:|---:|---:|
| 128x256 / 2x4 / 64 | 228 | 73728 | 1 | 342.891 us |
| 128x256 / 2x4 / 32 | 208 | 36864 | 1 | 343.641 us |
| 128x128 / 2x4 / 64 | 128 | 49152 | 2 | 431.964 us |
| 128x128 / 2x4 / 32 | 126 | 24576 | 2 | 436.274 us |

All four variants passed complete transcript checks. A 64x256/1x8 trial failed
all three transcript checks and was excluded; it is not a performance candidate.
Two resident CTAs alone are not sufficient: halving BN increases ideal global
operand traffic per update from 768 to 1024 bytes (+33.3%). This is a layout
calculation, not a new NCU traffic measurement. Runtime becomes 26-27% worse.

## Instruction inspection

Cuobjdump SASS for BK64/full transcript contains 64 IMMA instructions in the
main-loop body for both layouts. The 8x1 body has 2 LDSM.x4 and 64 LDSM.x2;
2x4 has 8 LDSM.x4 and 16 LDSM.x2. LDSM is already interleaved with IMMA by
ptxas. Both have a dependency wait and a CTA barrier in the main loop. These
are static instruction counts, not measured stalls or executed transaction
counts. There is no local-memory allocation reported for the tested kernels.

## Conclusion and remaining measurement

Large speedups must address the GEMM/loading main loop while preserving reuse.
Neither removing intermediate folds nor increasing CTA occupancy in isolation
approaches the reported 300 TH/s target. The final-only 2x4 configuration reaches
about 406 arithmetic TOPS (counting multiply and add separately), equivalent to
203 trillion dot-product multiply-adds/s for this isolated region. That number
is diagnostic throughput and is not a valid mining hashrate.

Fresh hardware counters for 2x4 full, final-only, and small CTA are still needed.
Run `sudo bash tools/profile_fold_ablation.sh`. It compares IDs 20,24,27 in warm
application-replay mode, base clocks, using the existing 22 traffic/stall metrics.
The 32 MiB region-input working set fits cache; this does not reproduce full
mainnet allocation residency. Reports go to `/tmp/pearl-fold-ablation-profile`.
The current session has no passwordless sudo, so this capture was not executed.

## Hardware counters supplied after manual sudo capture

The user subsequently completed the capture. Three measured kernels per variant,
medians, base clocks, application replay and warm cache:

| Metric | 20: full 2x4 | 24: final-only | 27: small CTA |
|---|---:|---:|---:|
| Duration (us) | 389.536 | 387.552 | 505.920 |
| L2 bytes / equivalent update | 772.204 | 770.222 | 1028.227 |
| L2 throughput (% peak sustained elapsed) | 90.893 | 91.355 | 93.270 |
| Shared LDSM bytes / equivalent update | 2048.000 | 2048.000 | 3072.000 |
| Tensor IMMA metric (%) | 63.520 | 63.542 | 48.295 |
| Barrier stall (%) | 20.950 | 23.768 | 27.898 |
| Long scoreboard stall (%) | 6.019 | 6.094 | 30.350 |
| Short scoreboard stall (%) | 0.769 | 0.880 | 2.916 |
| Local load/store sectors | 0 / 0 | 0 / 0 | 0 / 0 |
| DRAM bytes | 0 | 0 | 0 |

Normalization remains 1,048,576 logical R-boundary updates for identical GEMM
work. Final-only performs no intermediate transcript updates, so its denominator
is an equivalent-work normalization rather than its number of actual samples.
The cached-region test does not prove zero DRAM traffic in live full-grid mining.
Stall percentages and tensor utilization use their own counter denominators;
they are not shares of grid wall time and must not be added together.

Removing folds yields only 0.512% improvement under NCU, agreeing with the
unprofiled diagnostic. Small CTA increases L2 traffic 33.15% and LDSM traffic
50%, while duration worsens 29.88%. The strongest explanation is an operand-
delivery limit dominated by L2 traffic in this cached-region experiment; the
near-peak L2 throughput is evidence, not an independent causal proof of every
stall. High barrier stalls can include waiting for slower warps/async loads.

The measured 2x4 shared traffic (2048 B/update) also confirms the earlier layout
estimate versus 8x1's measured 4352 B/update. Cutting that traffic by 53% bought
only a modest runtime gain, while L2 bytes stayed near 772 B/update. The next
design should target L2 operand reuse and synchronization while respecting the
accumulator register budget; maximizing occupancy or eliminating folds alone
does not address the observed limit. No new production optimization is claimed.
