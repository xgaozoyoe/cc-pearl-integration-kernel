# Ada v121 region-size optimization

The recommended experimental configuration is region8192 with frozen v121:221.04 TH/s effective offline full-grid throughput versus200.00 TH/s at4096, a10.52% gain. No CUDA kernel or production code changed. Accumulators remain resident:168 registers/thread, zero local bytes/spills.

## Full-grid results

GPU0 only. Actual production generation, V3 seeds, noise, full transcript, PoW BLAKE3 and jackpot checks are included. An offline pool stub and zero target prevent share submission; proof/network time is excluded. Mainnet M=N=131072,K4096,R256 is unchanged.

Two runs,20 stable grids/configuration/run after4 warm-ups per block. Configuration order4096→8192→16384→32768, then reversed. Deterministic seed sequences match across sizes within each run. Forty measured grids per size. Timings come from the miner wall log, rounded to0.1ms.

| Region | Median ms/grid | Mean ms | p95 ms | Grids/s | Effective TH/s | Reported search TH/s | Gain |
|---|---:|---:|---:|---:|---:|---:|---:|
|4096|351.85|351.837|353.115|2.8421|200.00|209.25|0.00%|
|8192|318.35|318.337|319.930|3.1412|221.04|232.57|10.52%|
|16384|317.35|317.397|319.015|3.1511|221.74|233.63|10.87%|
|32768|316.00|316.272|318.680|3.1646|222.69|234.67|11.34%|

Effective throughput uses70.368744177664 TMAC per full grid divided by actual wall time. The miner-reported search rate excludes part of setup and is shown separately; neither is the earlier register-only ceiling.

Per-run median throughput:4096 =199.94 /200.02;8192 =221.08 /221.01 TH/s. The improvement repeats in both orders.450W power limit and driver settings were unchanged; clocks were not locked. Telemetry was sampled after each block and buffer release, so it is not a precise in-kernel frequency/power trace.

## Geometry, unchanged main loop

| Region | GEMM launches/grid | CTAs/launch | Waves at128 SMs and1 CTA/SM | Extra N-padding work | Transcript buffer |
|---|---:|---:|---:|---:|---:|
|4096|1024|352|2.75|3.125%|4 MiB|
|8192|256|1376|10.75|0.781%|16 MiB|
|16384|64|5504|43.00|0.781%|64 MiB|
|32768|16|21888|171.00|0.195%|256 MiB|

The identical binary retains CTA256x192, warp64x64,12 warps, K64, two stages,168 registers/thread,69,632 shared bytes/CTA, one CTA/SM and zero local bytes. Only dispatch dimensions and matching region buffers change. The code already supports these dimensions.

Larger regions amortize dispatch/PoW launch overhead, reduce edge padding and change CTA/cache scheduling. Their individual contributions were not isolated, so the measured33.5ms saving must not be attributed solely to CTA tail loss. The8192 configuration captures most of the measured benefit;32768 is only0.74% faster than8192.

## Correctness and operational scope

- For8192,16384 and32768, every transcript word and BLAKE3 digest matched the assembly of frozen-v1214096 subregions with identical A/B/key. Respectively262,144;1,048,576;4,194,304 transcript tiles were checked.
- Rank target, V3 derivation, proof construction and logical16x16 Pearl tile layout were untouched.
- Validation used random full-range signed INT8 operands; the full-grid benchmark used actual production generation and noise.
- This establishes partition invariance of transcripts/digests, not a live accepted-share result for the changed operating configuration.
- With existing FOUND_BATCH=64, larger regions change job-poll cadence.8192 polls four times per grid instead of sixteen for4096.16384 polls once per grid.32768 has only16 regions per grid and never reaches the in-loop64-region job-poll boundary; final flush does not itself poll. Do not deploy32768 with this host loop unchanged.

Prefer8192 as the next candidate: almost all the speedup, smaller buffers, and a functioning in-loop new-job check. No production default or library was changed, and no pool was contacted. Merely adding --region8192 to another production kernel does not establish the same speedup; these results specifically use frozen v121.

## Reproduction

    .venv/bin/python tests/bench_ada_regions.py validate
    .venv/bin/python tests/bench_ada_regions.py bench

The standalone harness selects /tmp/libpearl-ada-linear-copy.so, variant121, only within its own process. Existing files and dispatcher remain unchanged. No CUDA rebuild is needed because this is a launch-granularity experiment.

Raw results: ada-regions-results.json. Correctness: ada-regions-correctness.json. Logs: ada-regions-evidence/. Source diff: ada-regions.patch. No commit.
