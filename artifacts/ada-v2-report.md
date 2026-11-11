# Hand-written Pearl Ada v2: first verified benchmark

Implementation: csrc/gemm/pearl-gemm-ada-v2.cu. Independent SM89 library /tmp/libpearl-ada-v2.so; normal production dispatcher/library unchanged. No commit, live-share test or promotion. GPU0 only.

## Build and reproduce

```bash
bash tools/build_ada_v2.sh
PEARL_TRIAL_LIBRARY=/tmp/libpearl-ada-v2.so .venv/bin/python tests/bench_fold_ablation.py ada-v2
PEARL_TRIAL_LIBRARY=/tmp/libpearl-ada-v2.so .venv/bin/python tests/bench_mainloop_grids.py 104 89
```

The cached builder uses CUDA12.6,SM89,-O3,-lineinfo and builds isolated objects under /tmp/pearl-ada-v2-build. The existing CuTe library is linked only for baseline/reference comparison and unchanged BLAKE3. The new GEMM contains no CuTe mapping.

## Mapping and instruction ownership

```text
CTA:256×192,384threads,12warps,2 shared stages of K64
warp w: wm=w%4, wn=w/4; contiguous64×64 output
  4 row fragments × 4 B groups × 2 MMA halves
lane l: MMA row=l/4; columns=2*(l%4),+1
  C registers0,1 own that row;2,3 own row+8
  half0 covers columns0..7;half1 covers8..15
16×16 transcript tile = 8 C words/lane × 32 lanes
  local XOR -> native warp REDUX XOR -> lane0 shared store
  one transcript word sampled everyR256;16 samples forK4096
CTA epilogue: barrier -> coalesced global transcript stores
```

Native PTX m16n8k32 signed INT8/S32 MMA, manual cp.async.cg,16-byte-aligned swizzled shared operands,ldmatrix.x4 for bothA and a full16-columnBgroup. Stage copy bounds zero-fill padded rows. N8 CTA ordering is preserved; warp N ownership is now contiguous rather than the prior CuTe permutation.

Winner104 uses two-group lookahead across the eight B groups of K64, explicit fragment arrays, and next-slice A row loads after final old-row consumers. It is still ptxas-scheduled PTX, not manually scheduled SASS; exact issue cadence is not guaranteed by source order.

## Correctness and scope

All four final variants103..106 match full DP4A transcripts at256×512,512×256,4096²,256×768,768×512, K4096/R256. Unsupported K/R shapes are rejected before launch. This prototype is explicitly specialized to K4096/R256 and M/N divisible by16. V3 salted seed setup, official rank-penalized target and proof construction were not modified. Complete-grid offline zero-target runs exercise normal A/B generation,noise,transcript and unchanged PoW BLAKE3; no shares are submitted.

## Final resource and event comparison

|Variant|Registers/thread|Shared bytes/CTA|Resident CTA/SM|Local allocation bytes/thread|Region CUDA event median ms|
|---|---:|---:|---:|---:|---:|
|89|164|69632|1|0|0.325589|
|103|168|69632|1|16|0.316791|
|104|168|69632|1|0|0.315508|
|105|168|69632|1|56|0.325571|
|106|168|69632|1|16|0.320805|

Winner104:168registers/thread,384threads,oneCTA/SM,12/48residentwarps=25% theoretical warp occupancy. Zero LDL/STL in SASS. K64 body has64IMMA,16LDSM.x4, compared with89’s64IMMA,8LDSM.x4+16LDSM.x2. Requested shared operand bytes per MMA remain128; fewer load instructions do not mean fewer bytes. Full K4096 has64 K64 iterations perCTA,128 K32 slices,4096MMA perwarp. Total winner instruction body875static instructions versus baseline712; direct PTX is not intrinsically a smaller kernel.

## End-to-end performance

Two matched-input A/B comparisons,20 stable grids pervariant perrun,40combined. Per-block12grids with first2 discarded; production/89/104/104/89/production order. Clocks/power policy unchanged within comparisons but clocks were not locked. Wall-log resolution0.1ms.

|Metric,40stable grids|89 baseline|104 Ada v2|
|---|---:|---:|
|Median ms/grid|370.4500|366.4500|
|Mean ms/grid|370.4200|366.4400|
|p95 ms/grid|372.1100|368.9100|
|grids/s|2.6994|2.7289|
|Effective TH/s|189.9548|192.0282|
|Miner-reported search TH/s|198.2300|200.5300|

Combined median throughput gain: 1.09%. Both individual comparisons improve in the same direction.

Miner-reported search TH/s here is computed from Python time.time() around the search phase, excluding setup; it is not a kernel-only CUDA-event measurement. The region benchmark separately uses CUDA events. Do not extrapolate region TOPS into end-to-end mining TH/s.

Candidate106 was independently measured at372.25ms/grid,189.036TH/s versus89’s369.85ms/grid,190.263TH/s. Thus it is not the winner despite an initially promising region sample.

## Decision

Retain104 as a promising independent prototype; keep the accepted-share production path unchanged. Improvement is approximately1%, not evidence of approaching283–300TH/s. First reductions in launchability/spills came from launch bounds, compact copy loops and R256 specialization. Future changes should be measured against both104 and89, with source/SASS/resource checks; live-share validation is still required before any promotion or multi-GPU test.

Artifacts: ada-v2-kernel.diff,ada-v2-sass.json,ada-v2-final-region-results.json,ada-v2-grid-v104-run{1,2}.json,ada-v2-combined-results.json.
