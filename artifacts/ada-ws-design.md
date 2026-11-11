# Ada WS experiment: pre-edit inspection and design

## Current v121 CTA

- 384 threads,12 warps; CTA256x192x64, warp64x64.
- MMA: mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.
- Two shared stages;69,632 bytes including12,288-byte transcript.
-168 registers/thread, zero local allocation; one CTA/SM,12 resident warps/SM (25% of48).
- All12 warps issue cp.async, LDSM and IMMA. Copies use the v121 producer-lane permutation and direct shared destinations.
- K4096 means64 K64 iterations. Each iteration executes one cp.async.wait_group0 and one CTA BAR.SYNC, then prefetches the next stage and computes the current stage. One final CTA barrier precedes transcript output. Transcript is reduced every fourth iteration.

Steady state:

```
all warps: wait(copy K) -> CTA sync -> enqueue copy K+1 -> LDSM/IMMA K -> fold if R boundary
async copy:                                            [copy K+1 overlaps compute K]
```

The baseline already overlaps asynchronous transfer with compute. WS tests whether moving issue/address/wait work to dedicated warps improves that overlap; it does not introduce overlap from zero.

## Variants

Keep12 consumer warps and the exact256x192 output tile. WS1 adds1 producer warp (416 threads/13 warps); WS2 adds2 (448 threads/14 warps). Both use double buffering initially. This isolates the handoff design from reductions in consumer tile coverage. Extra producer threads still consume statically allocated registers on Ada and may force spills:168 registers times416/448 threads exceeds the65,536-register SM budget. This cost is part of the measurement, not hidden by claiming unchanged occupancy. Do not use SM90 dynamic register allocation.

Use named CTA barriers, supported on SM89, with explicit participant counts. Each stage has ready and empty barriers; IDs1..4, leaving barrier0 for the final epilogue. No TMA, WGMMA or transaction-count barriers.

For stage s:

1. Producer skips empty wait on the initial fill; on reuse it executes bar.sync(empty_s,total_threads).
2. Producers cooperatively issue all cp.async for the stage, commit and wait_group0. Each producer waits for its own outstanding copies.
3. Producers execute nonblocking bar.arrive(ready_s,total_threads), then advance to the next stage.
4. Consumers execute bar.sync(ready_s,total_threads), then LDSM/IMMA and the unchanged transcript fold.
5. Consumers execute nonblocking bar.arrive(empty_s,total_threads), then wait for the next ready stage.

Ready completion proves every producer's copies are complete. Empty completion proves all consumers have finished reading the stage. Separate ready/empty barriers prevent overwrite races and premature barrier reuse. All participant counts are warp multiples and role branches are warp-uniform. No spin polling or global-memory flags. The first stages need no empty arrival because no consumer has used them yet; final empty barriers need not be reused after the last consumption. One final full-CTA barrier joins the producer/consumer branches for the unchanged transcript epilogue.

```
producer: copy0/wait -> publish0 -> copy1/wait -> publish1 -> wait empty0 -> copy2 ...
consumer: wait ready0 -> compute0 -> release0 -> wait ready1 -> compute1 ...
```

The handoff barriers include all CTA threads but only the dependent role blocks; they are not full-CTA rendezvous before every load/compute phase. SASS must confirm this. Overlap is permitted by this ordering; source/SASS alone cannot quantify achieved hardware overlap.

## Controls and acceptance

Start with the full transcript since preserving the existing consumer computation is straightforward. Validate against frozen v121 and DP4A, including edge tiles. Benchmark identical seeded INT8 inputs at4096x4096x4096,40 or more stable independent event measurements per variant in two runs. Report regional kernel-equivalent TH/s separately from mining end-to-end throughput.

Preserve global operand volume and CTA scheduling. Theoretical global bytes/CTA/K64=28,672 and useful MMA count per full4096-cubed region=16,777,216; actual executed MMA includes the partial N-edge padding. Measure L2 bytes/useful MMA with Nsight; do not substitute theoretical bytes for L2 counters. Existing approximately38.26 B/MMA is historical, not a new measurement.

References: NVIDIA PTX bar.arrive/bar.sync producer/consumer pattern and cp.async.wait_group semantics: https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html .

## Implementation and resource observations

The compile uses SM89 only. The isolated library exports ws_run/ws_bench and is
not registered in the production mining dispatcher. The included homogeneous
baseline has instruction-identical SASS to frozen v121.

Measured CUDA function attributes:
baseline: 168 registers, 384 threads, 69,632 shared bytes, 0 local bytes/thread.
WS1: 128 registers, 416 threads, 69,632 shared bytes, 296 local bytes/thread.
WS2: 128 registers, 448 threads, 69,632 shared bytes, 296 local bytes/thread.
All have one resident CTA/SM. Total resident warps are 12/13/14, but consumer
warps remain 12 for every variant. Producer-only warps still pay the uniform
per-thread register allocation. Dynamic register redistribution is not used.

The WS compiler output contains 444 bytes spill stores and 420 bytes spill loads
in the ptxas static report. These are compiler reports, NOT measured runtime
traffic. SASS contains 87 static LDL-family and 84 STL-family instructions in
each WS kernel, versus zero in the baseline. All variants contain 64 static
IMMA instructions in the K64 loop and 16 LDSM instructions. Runtime counts
include loops, predication and edge padding; static instruction counts do not
measure cycle attribution.

The initial regional benchmark is supplemented by 2 x 40 full mainnet grids
per variant: M=N=131072, K=4096, dispatched as 1024 regions of 4096 squared.
The standalone grid runner uses production setup_job (including V3 seeds),
noise_gen, transpose and noise_gemm, with the same deterministic seed for all
three variants at a given sample. It omits jackpot/PoW/pool for all variants.
Kernel events bracket each region; their elapsed times are summed per grid.
Grid wall time includes generation, noise, event/launch overhead, and final
synchronization. Event extraction is outside the wall interval. Therefore this
is an offline pipeline benchmark, not actual end-to-end mining throughput.

TH/s follows the existing miner convention: M*N*K/time/1e12, counting each MAC
once. INT8 TOPS counting multiplication and addition separately would be twice
this number, and must not be compared to reported Pearl TH/s.

Five existing transcript-validation shapes (including partial CTA output tiles)
pass against DP4A for all three variants. Compute Sanitizer synccheck and
racecheck pass on a full CTA workload (64 K iterations, repeated stage reuse).
These checks do not establish speed or pool compatibility of a new mining path;
no experimental path was connected to a pool.

## Lane mapping preserved

    CTA output 256 rows x 192 columns:
                     wn=0        wn=1        wn=2
    wm=0          warp 0      warp 4      warp 8
    wm=1          warp 1      warp 5      warp 9
    wm=2          warp 2      warp 6      warp10
    wm=3          warp 3      warp 7      warp11

Each consumer warp owns 64x64, split into 4x4 Pearl16x16 tiles.
For lane l, tile(row,group), half h, accumulator index v:
    output row = wm*64 + row*16 + floor(l/4) + 8*floor(v/2)
    output col = wn*64 + group*16 + 8*h + 2*(l%4) + (v%2)
There are128 accumulator scalars/thread. Each lane XORs8 values per16x16 tile;
REDUX.XOR across32 lanes yields its tile reduction. Lane0 writes one sample
per tile perR256 boundary,16 samples acrossK4096. Thus16 transcript tiles/warp,
192 tiles/CTA and3072 transcript words/CTA. The final CTA barrier protects
the coalesced output read from shared transcript storage.

The specialization moves only copies/addressing/handoff into warps12 (WS1)
or12-13 (WS2). Native accumulator ownership, rolling operand prefetch, MMA
sequence, fold and output layout are retained. LDSM and IMMA stay exclusively
in warps0-11.

Observed SASS WS1 PCs:
consumer ready BAR.SYNC at0xc30 -> first LDSM0xd10 -> first IMMA0xd70;
consumer empty BAR.ARV0x2710;
producer empty BAR.SYNC0x2820 -> LDGSTS0x29c0/0x2ba0 ->
LDGDEPBAR0x2bf0 -> DEPBAR.LE0x2c30 -> ready BAR.ARV0x2c50;
epilogue BAR.SYNC0x2c90.
These role-separated paths permit asynchronous overlap; no PC-sampled or
cycle-level overlap claim is made without counters. Named barrier IDs are
register operands; ptxas reserves16 barriers (versus1 for baseline). Shared
memory already limits all candidates to one CTA/SM, so this does not change
the observed active-CTA count.
