# Operand-reuse experiments, GPU 0, 2026-10-06

No new performance winner. Production `libp40cuda.so` remains unchanged.
The current best isolated kernel is variant 20 (128x256 CTA, warp2x4, BK64,
three stages, register transcript). Its complete-grid offline throughput remains
about 180 TH/s wall-clock / 187 TH/s reported search speed.

## L1 caching and CTA wave pairing

Built `/tmp/libpearl-fold-l1.so`. Tested eight new candidates plus two controls:
small 128x128 CTAs with cp.async.ca; normal/pair132/pair128 ordering; BK32/BK64;
three/four stages; and cp.async.cg paired-order control. Large 128x256 CA copies
were also compared. All candidates passed complete transcript comparisons on
three random matrix cases against DP4A.

Five interleaved forward/reverse timing passes, 20 warmups and 150 launches per
candidate per pass, M=N=K4096/R256, unlocked clocks:

| Configuration | Median region time |
|---|---:|
| Current best 2x4 CG BK64 | 340.173 us |
| Large CA BK64 | 343.217 us |
| Large CA BK32 | 348.740 us |
| Small CA BK64 pair132 | 396.629 us |
| Small CA BK32 normal | 435.514 us |
| Small CA BK32 pair132 | 435.930 us |
| Small CA BK32 pair128 | 437.248 us |
| Small CG normal | 439.383 us |
| Small CG pair132 | 440.156 us |
| Small CA BK32 s4 pair132 | 437.499 us |

CTA pairing is a scheduling heuristic, not a guarantee of two CTAs landing on
the same SM. The index mapping was checked as a bijection on partial waves.
No new cache counters were collected; actual L1 hit rate / L2 traffic reduction
cannot be inferred from this timing experiment alone.

## Square 192x192 CTAs

Added exact-extent 16-byte cp.async copies, zero-filled global edge rows and
guarded transcript stores. Both register and coalesced shared transcript modes
were tested. Reference checks cover random 256x512,512x256,4096x4096,128x256 and
384x768 cases at K4096/R256, including complete and partial square tiles.

A nine-warp prototype could not launch according to CUDA occupancy queries and
was never timed. Twelve-warp BK64 also cannot launch (194 registers, 384 threads).
All launchable candidates below passed all five complete transcript checks and
report zero local-memory allocation.

| Square variant | Registers/thread | Threads | Median region time |
|---|---:|---:|---:|
| 12warp 3x4 BK32 s3 | 161 | 384 | 389.657 us |
| 12warp 3x4 BK32 s4 | 161 | 384 | 384.811 us |
| 12warp 4x3 BK32 s3 | 162 | 384 | 386.115 us |
| 8warp 4x2 BK32 s3 | 226 | 256 | 368.032 us |
| 8warp 4x2 BK32 s4 | 226 | 256 | 368.114 us |
| 8warp 4x2 BK64 shared transcript | 246 | 256 | 407.893 us |
| 8warp 4x2 BK32 shared transcript | 210 | 256 | 406.431 us |
| 8warp 2x4 BK32 s3 | 226 | 256 | 364.769 us |
| Same-run current best 128x256 | 228 | 256 | 343.715 us |

Ideal interior input bytes/update improve 768 -> 682.667. Accounting for the
4096-region boundary gives 704 B/update (-8.33%); this is calculated requested
input payload, not measured L2 transaction traffic. Padded MMA work increases
6.35% because 4096 is rounded to 4224 for the square tile. Manual-copy loop and
edge addressing add instructions. These costs plausibly offset traffic savings;
their exact attribution needs counters and is not established here.

## Complete mainnet grids

Six grids per configuration, zero target/no HIT; median excludes first grid.
Original baseline brackets the experiment; the previous best runs on both sides
of the square candidate to reveal drift.

| Run order | Wall seconds/grid | Effective TH/s | Reported search TH/s |
|---|---:|---:|---:|
| Production baseline | 0.4282 | 164.34 | 170.50 |
| Previous best 2x4 | 0.3914 | 179.79 | 187.13 |
| Best square candidate | 0.4206 | 167.31 | 173.70 |
| Previous best 2x4 again | 0.3906 | 180.16 | 187.59 |
| Production baseline again | 0.4259 | 165.22 | 171.65 |

No live shares submitted; only GPU 0 used. Square candidate is about 7% slower
in effective complete-grid throughput than the previous best, so it is not
promoted. Sources, isolated libraries and JSON measurements are retained for
review. Mainnet difficulty-bound and V3 setup code were not changed.

This rejects these particular cache/pairing and square-CTA designs. It does not
establish that reducing L2 traffic is useless, or that 180 TH/s is a hardware
limit. A next experiment should address instruction scheduling / register
operand reuse in the main loop while retaining the current grid-aligned CTA,
or introduce a reuse scheme without extra padded MMA work.
