# CC Pearl Integration Kernel

This repository is a fork of [Muskwak/Open-Pearl-Miner](https://github.com/Muskwak/Open-Pearl-Miner), originally developed as Pascal Pearl Miner. Credit for the original miner, Pascal DP4A and Ampere/Ada tensor-core backends, pool integration, and packaging belongs to the upstream project and its contributors.

Our work focuses on **Pearl certificate V3 integration, the CPU/GPU mining pipeline, and RTX 4090 (Ada / SM89) kernel optimization**. This is a research and integration checkpoint: it includes working miner changes, isolated experimental kernels, benchmark tools, and records of both successful and rejected approaches.

## Current work

### V3 compatibility

- Bind keyed Merkle roots to matrix row counts using the V3 salts in CPU and CUDA noise-seed derivation.
- Pass matrix dimensions through the CUDA C API and PyTorch bindings.
- Use `pearl_mining.penalized_target_bound` for pool and solo target calculation.
- Check GPU seed parity against the CPU reference and validate experimental transcripts and proofs.

### Proof and pool pipeline

- Move CPU proof generation into a separate Linux process so CPU extension work holding the GIL does not block the CUDA launch thread.
- Share A/B-transpose snapshots through `memfd`, with two reusable buffers and bounded backpressure. At the production dimensions these buffers total approximately 2 GiB.
- Match unique submission IDs while processing job and difficulty notifications; make new jobs available to mining while a submission is pending, and surface closed connections for reconnect handling.
- Report snapshot, proof, queue, and submission latency, plus rolling useful-work rates that include setup and HIT pauses. Effective rates account for rank normalization.

The process backend is enabled on supported Linux source runs. `PEARL_PROOF_PROCESS=0` selects the legacy thread backend; frozen builds and platforms without `memfd_create` also use the fallback. Snapshot copies and backpressure can still pause mining.

### Ada / RTX 4090 experiments

The separate SM89 kernel and experiment scripts explore:

- INT8 MMA layouts, shared-memory swizzles, tile sizes, K64/K128 stages, and operand prepacking.
- A streaming, B prefetch, copy scheduling, register lifetimes, and accumulator reuse.
- R128/R256 transcript folding, XOR/REDUX scheduling, and transcript state placement.
- CTA and named barriers, ready/free handoffs, deferred publication, and producer/consumer warp groups.
- Persistent and warp-specialized execution, controlled ablations, and SASS/PC-stall analysis.

Experimental variants are built as separate libraries. The standard miner build does **not** automatically select them. In particular, the R128 trial uses its own runner and kernel; changing the standard miner's rank constant alone is insufficient.

## Recorded results and limitations

These are historical measurements from the checked-in experiment reports, not a fresh benchmark of every variant in this checkout.

| Experiment | Recorded observation | Scope |
|---|---|---|
| [Ada rank comparison](artifacts/ada-rank-report.txt) | R128 reported about **213–214 effective TH/s**, versus **110–111** for matched R256 after rank normalization. Raw compute throughput declined slightly. | Full-grid benchmark without HITs; the report separately records two accepted R128 trial shares. |
| [Proof pipeline isolation](artifacts/pool-pipeline-report.txt) | About **83% higher work rate** with process isolation under the tested injected-HIT frequency; substantially shorter launch-thread stalls. | Controlled workload with a fake submitter, not a prediction of live pool earnings. |
| [A-stream transcript ablation](artifacts/rank128-astream-ablation-profile/conclusions.txt) | Disabling transcript work reduced matched profiled kernel time by roughly **8–9%**. | Diagnostic headroom; the disabled mode is not a valid mining kernel. |
| [Main-loop scheduling](artifacts/mainloop-optimization-report.md) | Several region-level improvements did not produce a repeatable full-grid gain. | Includes slower and rejected candidates. |
| [Stage-management layers](artifacts/rank128-layers-current-report.txt) | Synthetic controls help separate operand delivery, synchronization, and transcript effects. | Timing differences are not independent instruction costs or achieved mining gains. |

Kernel timings, full-grid work rates, rank-normalized rates, and live pool performance measure different things. Reports document their shapes, ranks, validation, and measurement limitations; many runs used unlocked GPU clocks. No blanket speedup or correctness claim applies to all experimental variants.

## Repository map

| Path | Purpose |
|---|---|
| [`csrc/capi/`](csrc/capi/) | Torch-free CUDA C API used by the standalone miner. |
| [`csrc/tensor_hash/`](csrc/tensor_hash/) | GPU Merkle hashing and V3 noise-seed derivation. |
| [`csrc/gemm/`](csrc/gemm/) | Inherited DP4A/tensor-core paths and the isolated Ada v2 kernel. |
| [`python/`](python/) | Miner, pool protocol, CPU reference/proof code, and process-isolated snapshots. |
| [`tests/`](tests/) | Regression tests, transcript validation, CUDA microbenchmarks, and benchmark drivers. |
| [`tools/`](tools/) | Experimental builders, runners, profiling, and analysis scripts. |
| [`artifacts/`](artifacts/) | Curated reports, compact results, patches, and source snapshots. |
| [`packaging/`](packaging/) | Inherited standalone and HiveOS packaging. |

Compiled libraries, raw profiler captures, large PC-stall JSON files, downloaded tools, logs, and local backups are not bundled. The `artifacts/` output directory is ignored for new files; selected evidence is explicitly tracked. Historical reports can reference omitted captures, machine-specific paths, or deployment state at the time of the experiment.

## Build and run from source

The current optimization work targets Linux and RTX 4090. The inherited standard build also contains Pascal (`sm_61`) and Ampere/Ada (`sm_80+`) paths. Windows and HiveOS packaging remain in the repository but have not been revalidated as part of this checkpoint.

You need Python 3.12+, a compatible NVIDIA driver and CUDA toolkit, CUTLASS headers, and `numpy`, `blake3`, and `py-pearl-mining`. The consensus package must expose the V3 APIs used by this checkout, including `penalized_target_bound`. Experimental builders commonly reference CUDA 12.6 and local CUTLASS paths; inspect and adjust those paths for your machine.

```bash
git clone https://github.com/xgaozoyoe/cc-pearl-integration-kernel.git
cd cc-pearl-integration-kernel

python3 -m venv .venv
source .venv/bin/activate
pip install numpy blake3 py-pearl-mining

# CUTLASS_DIR must contain the cutlass/ and cute/ header directories.
CUTLASS_DIR=/path/to/cutlass/include bash packaging/build_capi.sh

# Standard miner path; experimental Ada kernels are built separately.
python packaging/p40_miner_lite_main.py --wallet prl1YOURWALLET --worker rig1
```

The standard builder accepts `NVCC`, `CUTLASS_DIR`, and `GENCODE` overrides. Rebuild `libp40cuda.so` when changing CUDA code, including the V3 seed implementation; an old local binary will not pick up source changes.

### Standard miner options

| Option | Purpose |
|---|---|
| `--wallet prl1YOURWALLET` | Payout address for pool mining. |
| `--worker rig1` | Worker name. |
| `--devices 0,1` | Select GPUs; the default is automatic detection. |
| `--pool HOST:PORT` | Override the configured LuckyPool endpoint. |
| `--region 4096` | Set the sub-output search region. |
| `--solo GATEWAY_HOST:PORT` | Mine through a local Pearl gateway. |

Run the entry point with `--help` for the complete option list. Pool support uses the inherited LuckyPool Stratum protocol. Each selected GPU runs in its own process.

### Isolated R128 trial

After building the standard CUDA library, inspect the compiler/header paths in [`tools/build_ada_rank.py`](tools/build_ada_rank.py), then build and validate the separate SM89 library:

```bash
python tools/build_ada_rank.py
python tests/bench_ada_rank.py validate
```

To explicitly start a single-GPU mining trial:

```bash
python tools/run_rank128.py --gpu 0 --wallet prl1YOURWALLET \
  --worker ada-rank128-test --region 8192
```

The runner defaults to `artifacts/libpearl-rank.so` and supports `--library` for an alternate build. Select an available GPU; do not overlap the trial with another miner on that device. Other experiment families have their own build and validation commands in the corresponding reports.

## Validation

Run the offline CPU/proof/pool regression suite:

```bash
python tests/test_pool_pipeline.py
```

This checks proof-byte parity between thread/process paths, snapshot reuse, response routing, job handoff, connection closure, and wall-rate accounting. It uses local sockets and requires no pool connection or CUDA device.

After rebuilding the standard CUDA library, check GPU V3 seed parity:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/test_v3_gpu_seeds.py
```

The seed test includes production-size matrices and needs sufficient GPU and host memory. Experiment validation may additionally compare CPU/DP4A transcripts, proof verification, edge shapes, and Compute Sanitizer results. Consult each report for what was actually checked.

At the research checkpoint, all six offline pipeline regressions passed, selected Python files parsed successfully, and all 70 shell scripts passed `bash -n`. The full GPU experiment matrix was not rerun for that checkpoint.

## Upstream releases and attribution

[Open-Pearl-Miner releases](https://github.com/Muskwak/Open-Pearl-Miner/releases) belong to the upstream project. They should not be assumed to include this fork's V3 integration, process backend, or Ada experiments. Build this checkout to use the changes described here.

The original miner and packaging are credited to **Muskwak and the Open-Pearl-Miner contributors**. This fork retains the upstream license and developer-fee implementation.

## License and developer fee

See [LICENSE](LICENSE) for the Pascal Pearl Miner License and its full terms. The inherited **2% developer fee** remains in place, directed to the upstream developer's configured address. Distribution must retain that fee as specified by the license; the license also describes a limited personal-use exemption.
