# Fresh Ada v2 performance retest

Fresh build /tmp/libpearl-ada-v2-retest.so is SHA256-identical to the previously built /tmp/libpearl-ada-v2-no-rg.so:0bd2620677bafa6b141a70a243d5eecd781e439c4ca184d118407b62b7e2395b. Production libp40cuda.so remains unchanged, SHA256 dd21041b17312d177aac619875bbad5d32703da7827d042185a7a783f8531ce6.

GPU0 RTX4090 only; initial0% utilization,34°C,450W power limit. One existing compute context was present but GPU was idle before the test; no existing processes stopped. Clocks not locked. Two interleaved A/B runs,20stable grids each pervariant,40combined. Production/89/104/104/89/production order;12grids/block,discard first2; matched deterministic input seeds. Offline full-grid zero-target jobs include normal generation,noise,transcript and PoW. No pool shares submitted.

All four hand-kernel variants again passed full DP4A transcript comparisons in five cases:256×512,512×256,4096²,256×768,768×512, K4096/R256. Winner104 still reports168registers/thread,384threads,69632shared bytes/CTA,oneCTA/SM,zero local allocation.

|Metric,40stable grids|89 baseline|104 hand Ada v2|
|---|---:|---:|
|Median ms/grid|370.3500|367.7500|
|Mean ms/grid|368.6775|367.7475|
|p95 ms/grid|374.0000|370.5250|
|grids/s|2.7001|2.7192|
|End-to-end TH/s|190.0061|191.3494|
|Miner-reported search TH/s|198.2450|199.9150|

Median throughput advantage0.707%. Both individual runs favor104 by their median, but variability is visible, especially between baseline blocks. This is a small measured advantage under unlocked-clock conditions, not evidence of a large architectural gain. Mean timing advantage is smaller than median advantage.

Miner-reported search throughput uses Python wall time around the search phase, excluding earlier setup; it is not a kernel-only CUDA event rate. Fresh separate region CUDA-event medians:89 0.327657ms,104 0.324997ms. These region measurements should not be extrapolated directly into full-grid throughput.

Conclusion: current hand-written Ada kernel is roughly191–192TH/s end-to-end on GPU0 in this retest. Keep it isolated; no live-share acceptance validation or production promotion occurred. Full data: ada-v2-retest-grid-run{1,2}.json,ada-v2-retest-combined-results.json,ada-v2-retest-region-results.json.
