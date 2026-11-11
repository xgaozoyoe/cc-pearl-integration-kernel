# Ada v2: copy, pipeline, and operand-reuse experiments

GPU 0 only. Production mining dispatch, V3 seeds, target logic and proof construction remain unchanged. No candidate was promoted and no commit was made.

## Result

This round did not produce a stable end-to-end improvement. Keep frozen v110 (`/tmp/libpearl-ada-bpack-fixedk.so`), previously measured at 199.51 TH/s across 40 stable grids. The latest two baseline controls measured 199.80 and 198.75 TH/s.

| Experiment | Baseline median grid | Candidate median grid | Baseline / candidate end-to-end TH/s |
|---|---:|---:|---:|
| v111: ordinary copy-loop unrolling, first isolated build | 352.20 ms | 352.75 ms | 199.80 / 199.49 |
| v113: three-stage packed-B pipeline | 354.05 ms | 356.00 ms | 198.75 / 197.67 |

Each full-grid A/B comparison used matched deterministic inputs and 20 stable grids per variant after warmup. Controls used the frozen v110 library through PEARL_BASELINE_LIBRARY. Clocks/power policy was not changed; clocks were not locked. Effective throughput is 70.368744177664 TH per grid divided by median full-grid seconds.

## Other candidates and resources

All five complete transcript comparisons against DP4A passed for every tested candidate: 256x512, 512x256, 4096x4096, 256x768 and 768x512, K4096/R256.

| Variant in final experimental library | Registers/thread | Threads/CTA | Shared bytes/CTA | Active CTAs/SM | Local bytes/thread |
|---|---:|---:|---:|---:|---:|
| v110 baseline | 168 | 384 | 69,632 | 1 | 0 |
| v111 ordinary copy-loop unroll | 168 | 384 | 69,632 | 1 | 48 |
| v112 fixed copy rounds | 168 | 384 | 69,632 | 1 | 72 |
| v113 three-stage pipeline | 167 | 384 | 98,304 | 1 | 0 |
| v114 fixed copy rounds + three stages | 168 | 384 | 98,304 | 1 | 72 |
| v115 warp 64x96 | 247 | 256 | 69,632 | 1 | 0 |

The first v111 build used zero local bytes; subsequent compilation after adding variants changed its generated code and introduced spills. Its quoted full-grid result belongs to the first isolated library `/tmp/libpearl-ada-copy-unroll.so`, not the final library. No result should be transferred across these builds without checking generated code.

v112 explicitly assigns A's 1,024 16-byte copies to two complete 384-thread rounds plus one 256-thread round; B uses two complete rounds. It introduced spills and showed no convincing regional benefit. v114 likewise spills and has no demonstrated end-to-end benefit; it was not run through a full-grid A/B test.

v115 keeps the 256x192 CTA tile and K64 but expands warp tiles from 64x64 to 64x96. The CTA changes from 12 to 8 warps. Per warp/K64, LDSM.x4 rises from 16 to 20 and IMMA rises from 64 to 96; across the entire CTA, LDSM.x4 falls from 192 to 160 (-16.7%) while IMMA remains 768. Therefore useful MMA per shared operand byte improves by 20%, with unchanged global operand bytes for the same CTA tile.

Despite that reuse improvement, regional CUDA-event median rose from 0.306613 ms (v110) to 0.315317 ms (v115) in the same run. More registers and fewer resident warps offset the reuse benefit. This is regional evidence, not a full-grid measurement of v115. It does not establish instruction-level causality; hardware counter profiling is the next step.

## Profiling handoff

A normal-user Nsight Compute attempt selected the handwritten kernel but failed with ERR_NVGPUCTRPERM. A sudo -n attempt failed because a password is required. No performance-counter report was collected; do not infer measured stall percentages from this round.

Run:

```bash
sudo bash tools/profile_ada_v2_stalls.sh
```

The script uses only GPU 0 and profiles v110 against v115 in `/tmp/libpearl-ada-wide96.so`. It selects the demangled handwritten kernel name, takes 32 warmup launches and profiles three launches per variant, and exports PC sampling, scheduler and warp-state reports to `/tmp/pearl-ada-v2-stalls`. It leaves clocks unchanged. Warm-cache regional profiling diagnoses issue stalls; its timing is not an end-to-end mining result.

The new library's v110 SASS instruction sequence is bit-identical to the frozen baseline, checked in `ada-v110-baseline-sass-check.json`. This preserves a valid control for the reuse experiment.

## Files and rollback

- Full-grid data: `ada-copy-unroll-grid-results.json`, `ada-pipeline3-grid-results.json`.
- Regional measurements: `ada-copy-unroll-region-results.json`, `ada-copy-fixed-region-results.json`, `ada-copy-pipeline-region-results.json`, `ada-wide96-region-results.json`.
- Disassembly/resource records: `ada-copy-unroll-sass.json`, `ada-copy-fixed-sass.json`, `ada-copy-pipeline-sass.json`, `ada-wide96-sass.json`.
- Final round diff: `ada-copy-pipeline-wide-round.diff`.
- Original source snapshots: `/tmp/pearl-ada-copy-baseline/`.
- Known-fast frozen library: `/tmp/libpearl-ada-bpack-fixedk.so`; SHA256 b056dea25bfbe622ea82510cab49e18b7756c01e23965d5209d059d06f89fabd.
- Final experimental library: `/tmp/libpearl-ada-wide96.so`.
- Production libp40cuda.so unchanged; SHA256 dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6.

An external 300 TH/s report implies a 50% throughput gap from 200 TH/s, or about a one-third grid-time reduction. It is not proof of 50% readily available kernel headroom under our clocks, power, dimensions and throughput accounting.
