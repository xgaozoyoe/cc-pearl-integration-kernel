# Register-resident IMMA throughput calibration

The current logical accumulator mapping can approach the dense INT8 hardware
ceiling when operands are already in registers. Keep accumulator spill at zero
as an admission criterion for subsequent main-loop candidates. This does not
mean minimizing the total register count at the expense of useful reuse.

Production was not modified; GPU0 only, no pool. Standalone library:
 /tmp/libpearl-imma-roof.so.

## Controls and validation

Same384 threads/CTA,12 warps,64x64 per-warp output ownership,128 accumulators per
thread, and64 m16n8k32 signed INT8 / INT32 accumulator MMA instructions per
logical K64 iteration. Instruction order follows the baseline group/half/row
sequence. Operands are runtime-loaded once and then repeated in registers.
This is a throughput calibration, not a valid Pearl workload or transcript.

The kernel reserves69,632 dynamic shared bytes without accessing them, retaining
one resident CTA/SM. Resource attributes:166 registers/thread, zero local bytes,
zero spills, compared with baseline168 registers/thread. Physical register
numbers and compiler scheduling need not be identical to the full kernel.

SASS main loop0x680–0xaa0 contains exactly64 IMMA, one IADD3, one ISETP and one
branch. No LDSM, global/local loads or stores, barriers, cp.async or clock reads
inside that loop. Every accumulator contributes to an observable checksum.
Positive and signed-negative constant operand cases passed analytic checksum
checks at1,64 and4096 iterations. Timing uses CUDA events plus per-warp clock64
reads outside the loop. All128 SM IDs were observed.

## Long-loop calibration

Each configuration:3-second warm-up followed by two runs of40 samples.
Same450W power setting, sampled core clocks2715–2730MHz; no clock lock.

| CTAs x iterations | Median ms | Mean ms | p95 ms | Executed TMAC/s | MMA/SM/cycle |
|---|---:|---:|---:|---:|---:|
|128 x256|0.301056|0.300950|0.302082|342.39|0.24478|
|128 x1024|1.170400|1.171960|1.175461|352.29|0.24871|
|128 x4096|4.660800|4.656161|4.667392|353.86|0.24967|
|352 x64|0.234240|0.234009|0.234597|302.54|0.23579|

Per-SM cycles cover the earliest warp start through latest warp finish for all
CTAs assigned to that SM, including inter-CTA gaps where applicable. The table
reports the mean of per-SM rates. It is not a globally synchronized SM clock.

NVIDIA lists dense INT8 peak660.6 TOPS at2.52GHz for RTX4090. One MAC is two
operations in TOPS, so this is approximately330.3 TMAC/s, or1024 MAC/SM/cycle.
A warp m16n8k32 instruction performs4096 MACs, hence0.25 such instructions per
SM per cycle. Observed0.24967 is approximately99.87% of that rounded theoretical
rate. This is calibration throughput, not Pearl hash throughput.

Source: https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
Appendix A, dense INT8 throughput, SM count and boost clock. Peak scaling with
observed frequency is an estimate, not a guaranteed application throughput.

## Matched short-region comparison

A separate two-run,40-sample-per-run comparison alternated the frozen baseline
implementation (v121 within the isolated WS library) with the register kernel.
Both execute352 CTAs x12 warps x64 iterations =17,301,504 warp MMA instructions.
Baseline uses actual4096x4096 INT8 matrices and full transcript; the calibration
uses repeated register operands and no transcript. The arithmetic workload
size and logical accumulator organization match; the numerical problem differs.

| Kernel | Median ms | Mean ms | p95 ms | Useful-region equivalent TH/s |
|---|---:|---:|---:|---:|
|Baseline|0.295936|0.295878|0.298035|232.21|
|Register calibration|0.234496|0.234665|0.235530|293.05|

The equivalent column uses4096^3 MACs, excluding N-padding work, to match the
miner's work convention. Executed TMAC/s in the earlier table includes padding.
Neither table measures full-miner end-to-end performance.

Difference:61.44us,20.76% of baseline time; throughput ratio1.2620x.
It is the combined net effect of removing data movement, synchronization,
transcript and some associated code/register lifetimes, not isolated load time.
Prior transcript ablation suggests transcript is not the dominant component,
but this calibration does not independently measure its cost.

Matched-run sampled clocks2715–2745MHz, memory10501MHz and450W power limit.
Alternation reduces drift; it is not a locked-clock comparison.

## Interpretation and next constraints

1. The logical accumulator shape and its independent MMA chains do not
intrinsically limit throughput to approximately230 TMAC/s. With resident
operands, the same logical organization reaches the instruction throughput
ceiling. Physical register-bank effects in the full kernel remain possible.
2. The real short-region kernel retains a measurable operand-supply/synchronization
gap. Focus on feeding MMA while keeping accumulators resident, rather than
adding producer resources that cause spill.
3. Short regions reach approximately302 executed TMAC/s even without operand
loads in the loop, versus354 for long loops. There is also a task-length,
CTA-wave, prologue/epilogue and scheduling gap.352 CTAs on128 SMs give2.75 waves;
tail imbalance alone cannot be assumed to explain the entire gap.
4.293 useful-region TH/s is this calibration's measured result, not a universal
upper bound on future kernels. It also cannot justify a claim of310–330 TH/s
end-to-end. Region/grid scheduling deserves attention alongside the main loop.

Future candidate gate: zero accumulator spill in the timed loop; check ptxas
and SASS, not just registers/thread. Operand/address temporaries may be shortened
or recomputed; accumulator ownership changes must retain bit-exact semantics.
No next mapping or scheduling redesign was implemented in this task.

## Artifacts and reproduction

bash tools/build_ada_imma_roof.sh
.venv/bin/python tests/bench_ada_imma_roof.py
.venv/bin/python tests/compare_ada_imma_roof.py

Raw samples: ada-imma-roof-results.json and ada-imma-roof-matched.json.
SASS, build log and run logs: ada-imma-roof-evidence/.
Reviewable source patch: ada-imma-roof.patch. No commit.
