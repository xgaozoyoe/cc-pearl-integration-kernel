# Ada producer-layout optimization round

GPU 0 only. Production library and mining dispatch were not replaced; no commits were made. V3 salted seeds, rank-penalized targets and proof construction were not changed.

## Outcome

The direct-address producer permutation (v121) gives a small repeatable improvement over frozen v110. Two matched-input A/B runs measured20 stable grids per variant each, for40 total after warmup.

| Full-grid measurement | Frozen v110 | v121 |
|---|---:|---:|
| Median wall time |353.65 ms |351.90 ms |
| Mean wall time |353.62 ms |351.8025 ms |
| p95 wall time |355.60 ms |353.12 ms |
| Grids/sec from median |2.82765 |2.84172 |
| Equivalent end-to-end throughput |198.98 TH/s |199.97 TH/s |

Median throughput gain is0.4973%, with1.75 ms/grid saved. Run1 was199.49→200.40 TH/s; run2 was198.70→199.85 TH/s. Both improved in the same direction. This is a small result; it does not close the gap to300 TH/s. Clocks were not locked and the clock/power policy was not changed.

## Implementation

The winning change preserves the original shared swizzle and all consumer LDSM/native-MMA ownership. For copy chunk i, the producer selects logical chunk `i ^ ((i >> 3) & 3)` within the same warp. The original swizzle applied to that logical chunk produces exactly byte address `16*i`, so v121 writes to that address directly.

This preserves each warp's global address set and the matrices observed by MMA, while adjacent producer lanes write adjacent shared16-byte chunks. v120 used the same permutation but recomputed the swizzled shared address; v121 removes that redundant computation.

The winner retains CTA256x192, warp64x64,12 warps/CTA, K64, two shared-memory stages, and the same INT8 m16n8k32 MMA. It uses168 registers/thread,69,632 shared bytes/CTA, one resident CTA/SM, and zero local bytes. SASS has64 IMMA and16 LDSM.x4 instructions per warp/K64, zero LDL/STL, and740 static instructions versus748 for baseline. The new library's v110 control is instruction-for-instruction identical to the frozen library.

## Controlled alternatives

| Variant | Change | Result |
|---|---|---|
|116 |No swizzle |Regional median~0.377 ms versus baseline~0.301 ms; clearly slower |
|117 |Three XOR bits, shift3 |Full-grid first run199.01→199.46 TH/s; small, not independently confirmed |
|118 |Two XOR bits, shift2 |Regional median~0.316 ms; slower |
|119 |Three XOR bits, shift2 |Regional median~0.3035 ms; no demonstrated full-grid benefit |
|120 |Producer permutation with recomputed shared address |Full-grid198.89→197.36 TH/s; slower |
|121 |Producer permutation with direct shared address |Two full-grid runs confirm small benefit |

Regional CUDA-event samples are diagnostic and not full-grid throughput. The final regional comparison put v121 at0.303445 ms versus v1100.301076 ms, despite the full-grid improvement. Repeated small-region kernels have different cache/launch behavior, so acceptance relies on the full-grid A/B tests. Do not derive mining TH/s from regional TOPS.

## Validation and runtime identity

All six candidates passed complete DP4A transcript comparisons for256x512,512x256,4096x4096,256x768 and768x512, K4096/R256, including partial edge CTAs. A CPU mapping check verified that the producer permutation is warp-local, bijective, and produces the same16-byte chunks.

Nsight Systems traced2048 calls across two full grids to `pearl_ada_v2::kernel<2,64,false,true,false,false,2,false,6>`, confirming that v121 actually ran. Trace durations are identity evidence, not the performance statistics above.

No live accepted share was checked for v121. Keep the production mining library and frozen v110 available until further validation/promotion.

## Next counter check

The earlier profile attributed excessive shared wavefronts to cp.async producers. This round has not yet collected counters for v121; contiguous shared destinations alone do not prove the hardware count fell or explain the speedup.

Run:

```bash
sudo bash tools/profile_ada_swizzle.sh
```

This profiles only GPU0, variants110,117,121, using `/tmp/libpearl-ada-linear-copy.so`, and writes reports to `/tmp/pearl-ada-swizzle-stalls`. The current NVIDIA driver restricts counter access; previous sudo -n attempts required a password. The wrapper has passed shell syntax validation. It leaves clocks unchanged and uses warm-cache regional launches.

Compare cp.async excessive shared wavefronts, LDSM excessive wavefronts, not-issued wait/short-scoreboard samples, and eligible warps. If producer wavefront counts improve without a substantial throughput gain, this is evidence to move effort to MMA issue scheduling instead of expanding the swizzle search.

## Artifacts and rollback

- Full-grid run files: `ada-linear-copy-v121-run1.json`, `ada-linear-copy-v121-run2.json`.
- Combined measurements: `ada-linear-copy-combined-results.json`.
- Alternative full-grid results: `ada-swizzle-v117-grid-results.json`, `ada-ordered-copy-v120-grid-results.json`.
- Regional data: `ada-swizzle-region-results.json`, `ada-ordered-copy-region-results.json`, `ada-linear-copy-region-results.json`.
- Final SASS/resource counts: `ada-linear-copy-sass.json`.
- Runtime dispatch: `ada-v121-launch-kernels.csv`, `ada-v121-launch-verification.log`.
- Round diff: `ada-swizzle-round.diff`.
- Original source snapshots: `/tmp/pearl-ada-swizzle-baseline/`.
- Binary hashes: `ada-swizzle-binaries.json`.

Final candidate: `/tmp/libpearl-ada-linear-copy.so`, v121.

Frozen baseline: `/tmp/libpearl-ada-bpack-fixedk.so`, v110.

Final candidate SHA256: `d502d16187e953510303275f9a3ebf62010eba8f6fc277708282389ed36c9e66`.

Production `libp40cuda.so` remains unchanged, SHA256 `dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6`.

Counter follow-up is now complete: see `ada-swizzle-counter-report.md`. Excessive shared wavefronts were essentially unchanged; the small measured speedup should not be attributed to their removal.
