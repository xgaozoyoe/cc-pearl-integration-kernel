# Controlled Ada pure-GEMM comparison

**Conclusion: redesign Ada GEMM mapping.** Removing transcript integration
does not reveal a large speedup. No redesign experiments were started for this
task. This conclusion selects the next area to investigate; it does not prove
which new mapping will be faster or that 300 TH/s is achievable.

## Method

GPU 0, RTX 4090 SM89, CUDA 12.6. Compared the fastest previously measured
experimental Ada configuration (variant 20: 128x256 CTA, warp2x4, K64, three
stages, lane-register transcript) with an otherwise identical specialization
using `DisableTranscript=true` (variant 48). This is the existing fused kernel,
not cuBLAS or a different CUTLASS GEMM. The production live-mining dispatcher
and library were not replaced.

Mainnet M=N=131072,K4096,R256, traversed as 1024 regions of 4096x4096. Original
GPU setup_job, V3 seeds, random generation and noise application are unchanged.
Both modes use identical deterministic setup seeds, job header and key. Both
omit PoW BLAKE3/jackpot checks, so their difference isolates the transcript
implementation rather than adding a separate BLAKE3 cost to the full variant.

Each timing phase uses full/pure/pure/full ordering, four warmup grids plus ten
measured grids in each block: **20 stable grids per mode per phase**. Main wall
measurements use original kernel launches, without events or graphs. A separate
CUDA-event phase sums 1024 region-kernel brackets per grid. Each bracket is a
prebuilt graph containing only start event, unchanged kernel, end event, avoiding
CPU event-enqueue gaps. Graph/event instrumentation changes dispatch cadence;
its absolute times should not replace the uninstrumented wall measurements.

Power limit remained 450 W; memory clock 10501 MHz. Stable event-phase telemetry
samples show 2715 MHz core for both variants. No privileged clock locking was
performed. In the wall phase, full ranged 2685-2715 MHz and pure sampled 2715 MHz
(both medians 2715 MHz), so the subpercent wall difference is not an exact
intrinsic transcript-cost attribution. GPU telemetry is retained as CSV.

## Measurements

| Timing | Variant | Median ms/grid | Mean ms/grid | p95 ms/grid | grids/sec | Equivalent TH/s |
|---|---|---:|---:|---:|---:|---:|
| Kernel CUDA events | Full transcript | 360.368 | 360.685 | 363.763 | 2.775 | 195.269 |
| Kernel CUDA events | Pure GEMM | 361.372 | 361.948 | 365.300 | 2.767 | 194.727 |
| Uninstrumented full-grid wall | Full transcript | 382.900 | 382.875 | 383.830 | 2.612 | 183.778 |
| Uninstrumented full-grid wall | Pure GEMM | 382.150 | 382.025 | 382.725 | 2.617 | 184.139 |

Equivalent TH/s = 70.368744177664 trillion multiply-adds per mainnet grid /
seconds. It is a work-equivalent rate, **not accepted-share mining hashrate**.
Both wall measurements omit BLAKE3/checks, explaining some difference from prior
normal-mining grid measurements.

Using median kernel times as requested:

- Transcript overhead: **-1.004 ms**, or **-0.278% of full time**.
- Transcript slowdown ratio: **0.997223x**.
- Uninstrumented wall difference: **+0.750 ms**, **0.196% of full time**,
  slowdown ratio **1.001963x**.

The tiny opposite-sign differences do not identify a useful positive fold
overhead. They include compiler scheduling/resource changes and measurement
variation. They do not establish a zero-cost fold, or prove folding itself
makes GEMM faster. Pure GEMM remains around 360 ms under event instrumentation,
consistent with Case B rather than the proposed 180-220 ms Case A.

## Resources and identical launch/work configuration

| Resource | Full transcript | Pure GEMM |
|---|---:|---:|
| Registers/thread | 228 | 180 |
| Shared memory/CTA | 73728 B (72 KiB) | 73728 B (72 KiB) |
| Threads / warps per CTA | 256 / 8 | 256 / 8 |
| Resident CTAs/SM (CUDA occupancy query) | 1 | 1 |
| Theoretical warp occupancy | 16.67% | 16.67% |
| Local allocation/thread | 0 | 0 |

Occupancy is theoretical, not newly sampled achieved occupancy. Prior NCU
measurements for the full variant also reported zero local load/store spills.
Current SASS inspection finds no local spill loads/stores in either variant.

- CTA tile: 128x256; logical warp ownership area: 64x64, interleaved rather than
  a single contiguous rectangle.
- MMA: signed INT8 x signed INT8 -> INT32, TN,
  `SM80_16x8x32_S32S8S8S32_TN`, SASS `IMMA.16832.S8.S8`.
- K64 contains two K32 MMA substeps; 64 K64 iterations per CTA.
- Region launch: 32x16 CTAs, 256 threads each.
- Grid: 1024 region launches, 524288 CTAs, **33554432 CTA-level K iterations**.
- Each main-loop body retains 64 warp IMMA instructions, 8 LDSM.x4 and 16
  LDSM.x2 in both compiled variants. Dynamic MMA work is unchanged.

## Lane / warp / transcript ownership

Within a CTA, warp w has `wm=w%2`, `wn=w//2`. Its hash-tile row indices are
`wm+2*tm` for tm=0..3; column indices `wn+4*tc` for tc=0..3.

```text
CTA hash-tile grid: 8 rows x 16 columns (each tile 16x16)

                  column classes
                 0,4,8,12  1,5,9,13  2,6,10,14  3,7,11,15
rows 0,2,4,6:       warp 0    warp 2     warp 4     warp 6
rows 1,3,5,7:       warp 1    warp 3     warp 5     warp 7

Each warp owns 16 complete 16x16 hash tiles.
Each lane owns 8 accumulator elements per hash tile = 128 INT32 elements.
```

For lane l, let g=l//4, q=l%4. Within each 16x16 hash tile, lane l owns:

```text
rows:    g and g+8
columns: 2q, 2q+1, 2q+8, 2q+9
Cartesian product -> 8 accumulator elements per lane
```

This formula was checked against CuTe's actual partition_C coordinates for all
32768 accumulator elements in the CTA; the raw coordinate array is retained.

At every R256 boundary, each lane XORs its eight elements for each owned hash
tile. `__reduce_xor_sync` performs a native 32-lane XOR reduction (`REDUX.XOR`),
not a five-instruction shuffle tree. For boundary rb=0..15, lane rb retains the
tile's result in rT. Each warp maintains 16 transcripts x 16 words = 256 logical
words; each CTA 128 transcripts x 16 words = 2048 logical words. Lanes 0..15
hold 16 words each, one boundary word for each tile; lanes 16..31 do not retain
transcript words. Physical register allocation is uniform across all lanes.

In this current fastest path, transcript ownership already follows whole-tile
MMA ownership. There is no cross-warp transcript exchange, shared-memory
transcript rearrangement, or fold-specific CTA barrier. Shared memory holds A/B
pipeline stages only. `cp_async_wait<1>()` and `__syncthreads()` occur once per
K64 iteration, 64 times per CTA, in both variants. There is no final transcript
barrier for the lane-register mode.

The fold appears after preceding MMA and before the next iteration: dependent
XOR/reduction instructions interrupt that warp's MMA issue sequence at 16 R
boundaries. Other warps / the Tensor pipe can still execute outstanding work.
The measured net compiled-configuration effect is small; instruction ordering
alone does not establish a large Tensor pipe stall caused by transcript work.

## Observable sink and safety

Pure mode has no rT initialization, R-boundary folds, transcript output or
transcript shared/global stores. Every final accumulator element feeds a
thread-local checksum; one native reduction and one lane-0 store per warp occur
only after GEMM. Output is eight uint32 checksums per CTA (16 KiB/region) rather
than a 4 MiB/region transcript. No atomic, CTA barrier or extra kernel is used.
Checksum pointers cannot alias A/B (`restrict`).

Per-warp checksums were compared against XORs of the 16 relevant final-word
values from the DP4A transcript on three random cases, including 4096x4096.
The normal full transcript matched DP4A bit-for-bit. Sentinel checks prove pure
mode writes only the checksum region. SASS retains every required IMMA. Calling
trial_pow with pure variant 48 is rejected before any kernel launch.

Production V3 seeds, penalized target, proof construction and library are
unchanged. SHA256 remains
`dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6`.
No commit, deployment or live benchmark-mode mining was performed.

## Top three next experiments (proposals only)

1. **Improve operand delivery/reuse while respecting the accumulator register
   budget.** Revisit CTA/warp ownership and L2 -> shared -> register traffic.
   Existing NCU puts L2 throughput near 91%; traffic-reduction designs need
   to avoid the extra padding/work and reuse losses seen in prior attempts.
2. **Change MMA fragment traversal / operand register reuse and instruction
   scheduling.** Keep arithmetic identical; evaluate accumulator dependency
   spacing, register-bank pressure and loads per IMMA before broad tiling sweeps.
3. **Restructure K-stage consumption and synchronization.** Test finer overlap
   of LDSM with IMMA and async-stage retirement, using counter evidence to
   separate barrier waiting from shared/L2 latency. Preserve R-boundary samples.

Ranking is a hypothesis based on the current evidence, not a promised speedup.
None of these experiments has been started in this task.

Reproduce: `bash tools/build_pure_gemm.sh`, then
`.venv/bin/python tests/benchmark_pure_gemm.py` with GPU 0 otherwise idle.
Source review: `tests/fold_ops.cuh`, `tests/fold_ops.cu`,
`tests/pure_gemm_measure.cu`, `tests/benchmark_pure_gemm.py`.
