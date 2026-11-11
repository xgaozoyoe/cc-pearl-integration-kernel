# Runtime verification of hand-written Ada v2 dispatch

A fresh Nsight Systems CUDA trace executed two complete offline grids on GPU0 through the same bench_cute_grid.run(104) function used by the full-grid A/B benchmark. tests/verify_ada_v2_launch.py printed the actual ctypes library path and SHA256 before running.

Loaded library: /tmp/libpearl-ada-v2-retest.so
SHA256:0bd2620677bafa6b141a70a243d5eecd781e439c4ca184d118407b62b7e2395b
Variant104 -> launch_pearl_gemm_ada_v2(2) -> pearl_ada_v2::kernel<2>.

Nsight observed2048 invocations of void pearl_ada_v2::kernel<(int)2>(...), exactly1024region launches/grid×2grids. It observed zero invocations of the old cute_pearl_fold or pearl_gemm_only_kernel GEMM/transcript implementations. The unchanged CuTe noise GEMM still appears as expected, along with2048 PoW BLAKE3 calls and normal GPU setup kernels.

Hand-kernel total GPU time641.891ms over two grids; mean313.423µs/region. This trace is for identity verification, not a replacement for the40stable-grid performance benchmark. GPU-kernel time fraction92% is a fraction of summed GPU kernel durations, not full-grid wall time.

Source dispatch: tests/fold_ops.cu:154; direct launcher csrc/gemm/pearl-gemm-ada-v2.cu:172. Raw Nsight report /tmp/pearl-ada-v2-launch-verification.nsys-rep; kernel CSV artifacts/ada-v2-launch-kernels.csv; library metadata/run log artifacts/ada-v2-launch-verification.log.

Why similar performance is plausible: the new implementation preserves CTA256×192,warp64×64,BK64,two shared stages,12warps/CTA,oneCTA/SM,64INT8MMA/warp/K64 and the same operand bytes per MMA. It changes native lane/warp ownership, B LDSM packaging and operand scheduling but retains the major resource/reuse structure. Hand-written PTX is still optimized and scheduled by ptxas. This is an independent implementation of a similar architecture, so modest gains do not imply fallback.
