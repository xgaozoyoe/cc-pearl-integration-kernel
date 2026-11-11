# MMA consumption-order / LDSM interleaving experiment

GPU 0 only. Isolated /tmp/libpearl-ldsm-order.so; production untouched. No live mining promotion. Full DP4A transcripts match in five cases:256×512,512×256,4096²,256×768,768×512; K4096/R256.

Both candidates preserve CTA256×192, warp4×3, warp tile64×64, K64,2 shared stages, N8 block ordering, one-group-ahead B prefetch and Mode9 transcript. Resources:164 registers/thread,384 threads/CTA,69632 shared bytes/CTA,one resident CTA/SM,local allocation zero. Static K64 body:64 IMMA,24 LDSM,zero LDL/STL.

Variant97 rotates the MMA row order by B group;98 consumes the two 8-column MMA halves alternately by row. These produce different register dependencies and machine encodings, but the filtered MMA/LDSM opcode order remains identical to89. Thus these are consumption-order experiments, not evidence that we changed LDSM issue placement.

## Full-grid measurements

Each comparison runs production,89,candidate,candidate,89,production, with12 grids per block and first2 discarded. Each variant has20 stable grids and matched deterministic inputs. Same clock/power policy within each run; clocks were not locked. Offline zero-target jobs use normal GPU job setup, transcript and PoW, without share submission. Wall timing is parsed at0.1ms resolution.

| Comparison | Variant | Median ms/grid | Mean ms/grid | p95 ms/grid | grids/s | Effective TH/s | Reported CUDA search TH/s* |
|---|---|---:|---:|---:|---:|---:|---:|
|97 vs89|89|370.700|370.430|371.990|2.6976|189.827|198.140|
|97 vs89|97|370.500|370.385|372.815|2.6991|189.929|198.270|
|98 vs89|89|371.500|371.410|373.925|2.6918|189.418|197.655|
|98 vs89|98|371.400|371.490|373.220|2.6925|189.469|197.740|

*Reported search throughput uses the miner’s own CUDA timing denominator, distinct from end-to-end throughput.

Both differences are ~0.03–0.05%, within noise; there is no demonstrated improvement. Keep baseline89. Previously measured best remains about190–191 TH/s, with these runs around189.4–189.9 TH/s under current conditions.

Region-only CUDA event medians:89 0.315453ms,97 0.315843ms,98 0.316880ms. These are small-region measurements and must not be converted to full-grid mining TH/s. Initial batches were slower, so full-grid comparisons are the decisive practical check.

## What this resolves

Changing independent accumulator order alone did not improve performance. Source placement constraints and consumption-order changes still leave ptxas choosing the same load/compute opcode pattern. A controlled issue-placement experiment would require a carefully designed PTX dependency mechanism or assembly scheduling workflow; these experiments do not prove the whole LDSM scheduling direction has no potential. Avoid introducing runtime barriers just to constrain scheduling.

The other actionable profile signal is main-loop arrival skew: ~98% of barrier samples occur after the main-loop barrier. Inspect global-copy ownership and per-warp arrival balance before removing or weakening any required barrier. This is a hypothesis, not yet measured per-warp imbalance.

Evidence: ldsm-order-kernel.diff, ldsm-order-sass.json, ldsm-order-region-results.json and the two full-grid JSON artifacts.
