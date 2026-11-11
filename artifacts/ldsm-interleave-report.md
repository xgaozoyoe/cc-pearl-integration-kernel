# LDSM placement experiment

Isolated library /tmp/libpearl-ldsm-interleave.so; GPU 0 only. Baseline89 preserved; production untouched. Candidate95 puts next-group B loading after two MMA rows;96 after three. Empty inline-assembly constraints emit no GPU synchronization instructions.

All three pass complete DP4A transcript comparisons for256×512,512×256,4096×4096,256×768,768×512; K4096/R256. Five forward/reverse passes, each20 warmup and150 timed region launches.

| Variant | Median region ms | Registers | Threads | Shared bytes | Resident CTAs | Local bytes |
|---|---:|---:|---:|---:|---:|---:|
|89|0.315452|164|384|69632|1|0|
|95|0.315421|164|384|69632|1|0|
|96|0.316251|164|384|69632|1|0|

The first pass was noticeably slower for all variants (~0.337ms); do not interpret small timing differences as speedups. More decisively, all three have exactly identical disassembled instruction encodings:712 instructions,64 IMMA,24 LDSM, zero local spill loads/stores. Instruction-encoding SHA256 c9a9847aa989b110f619afb5201486321d5084854158fcff3ecf6d72ca772966. A separate comparison of both encoding words also confirms exact equality.

No full-grid A/B run is justified for identical instruction streams. No new end-to-end TH/s claimed. Existing best stays89 (~190–191TH/s in previous measurements). Source-level placement alone did not alter ptxas scheduling; the compiler already interleaves next-group LDSM with current-group MMA.

Next meaningful experiment needs explicit PTX instruction/dependency control or a different consumption topology that produces different SASS, while preserving load/MMA counts. These candidates remain diagnostic experiments, not promoted live-mining kernels.
