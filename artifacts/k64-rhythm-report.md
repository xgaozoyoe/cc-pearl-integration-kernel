# K64 software-pipeline experiment

GPU0 only. Isolated library: /tmp/libpearl-k64-rhythm.so. Known working production library unchanged. No live-share test or promotion.

## Implementation

Flatten K64 into eight B operand groups: four per K32 slice. Prologue loads current-slice A and B groups0/1. During group i, source requests B group i+2 after two MMA rows. Variant99 loads A in bulk at the K32 boundary;100 loads next-slice A rows after their final previous-slice MMA consumer. Both preserve the CTA256×192,warp4×3,warp64×64,BK64,2sharedstages,N8 CTA scheduling,original transcript/PoW,original stage barriers and fragment reuse. No operand fragment is reloaded.

Both variants pass complete DP4A transcript comparisons at256×512,512×256,4096²,256×768,768×512 with K4096/R256.

## Machine-code result

Both candidates compile to exactly the same instruction encodings, different from baseline89. Each still has712 static instructions,64 IMMA and24 LDSM in the K64 body,zero LDL/STL. Registers/thread rise164→166;384threads,69632sharedbytes,oneCTA/SM,zero local bytes are unchanged. Only99 was benchmarked end-to-end because100 is identical.

The filtered load/MMA opcode sequence differs in15 positions out of88. Importantly, the source rhythm is not preserved completely. Around the K32 transition, candidate99 issues an A LDSM.x4 followed by a burst of eight B LDSM.x2; subsequent A LDSM.x4 instructions are separated by eight MMAs. Baseline has a different mixture with many B loads separated by four MMAs. Address and other scalar instructions are omitted from this filtered view; this is not a cycle-accurate issue timeline.

Thus this is a valid flattened operand pipeline experiment, but not proof that a uniformly spaced Load(i+2)/MMA(i) schedule has been realized. ptxas still chooses its own placement.

## Timings

Five forward/reverse region batches, each20warmups/150 timed launches; complete-grid test uses matched deterministic inputs, production/89/99/99/89/production order,12grids per block discarding first2,20stable grids per variant. Clocks/power policy unchanged within the run, but clocks were not locked. Wall logs have0.1ms resolution.

| Metric |89 baseline|99 rhythmic source|
|---|---:|---:|
|Median ms/grid|370.9000|375.7000|
|Mean ms/grid|371.1050|375.5400|
|p95 ms/grid|372.9050|377.0050|
|grids/s|2.6961|2.6617|
|Effective TH/s|189.7243|187.3004|
|Median reported CUDA search TH/s|197.985|195.350|
|Region-only CUDA event median ms|0.315412|0.318999|

Full-grid median increases1.29%; effective throughput falls1.28%. Region-only slowdown agrees in direction. These are offline full-grid timings, not accepted live-mining results.

## Decision

Keep89. Do not promote99/100. This experiment does not disprove fixed-cadence software pipelining; it demonstrates that source flattening plus a two-group lead does not enforce a fixed cadence and costs two extra registers here.

A next test should use explicitly bounded rolling operand storage / consumption dependencies rather than separate storage for every group, then verify SASS before timing. Even a source-level ring is not guaranteed to constrain ptxas register renaming. Any added dependency mechanism must be accounted for in instruction count and benchmark cost; avoid runtime barriers merely for instruction ordering. Separately, main-loop warp-arrival skew remains a profiling hypothesis, not yet demonstrated by this experiment.

Artifacts: k64-rhythm-kernel.diff, k64-rhythm-sass.json, k64-rhythm-region-results.json, k64-rhythm-grid-results.json.
