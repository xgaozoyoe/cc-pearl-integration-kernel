#pragma once
// Experimental variants of the fused CuTe Pearl kernel: int8 tensor-core GEMM + in-mainloop R-block transcript fold.
// Bit-exact with the DP4A reference; ~32.75 TH/s on RTX 4050 (Ada sm_89), vs the hand
// `ldm` kernel's 24.0. Multistage cp.async.cg + ldmatrix; <8,1,1> TiledMMA so each warp
// owns a 16-row band (16 contiguous 16x16 tiles), making the fold a direct in-register
// shfl_xor (no atomics/extra syncs). Namespaced so `using namespace cute` stays local.
//
// Needs the CUTLASS/CuTe include dir on the compile line (-I <cutlass>/include), which
// the p40cuda build (packaging/build_capi.*) and the bench already pass.
#include <cuda_runtime.h>
#include <cstdint>
#include <cute/tensor.hpp>

namespace foldops {
using namespace cute;

static const int HASH_ROT = 13, TLEN = 16;

template <class TiledMMA_, class GCopyA_, class GCopyB_,
          class SmemLayoutA_, class SmemLayoutB_, int bM, int bN, int bK, int Stages, int Mode, int PairStride = 0, bool PaddedCopy = false, bool DisableTranscript = false, int Schedule = 0>
__global__ static void
cute_pearl_fold(int M, int N, int K, int R,
                const int8_t* __restrict__ Aptr, const int8_t* __restrict__ Bptr,
                uint32_t* __restrict__ transcript)
{
    Tensor mA = make_tensor(make_gmem_ptr(Aptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor mB = make_tensor(make_gmem_ptr(Bptr), make_shape(N, K), make_stride(K, Int<1>{}));

    auto cta_tiler = make_shape(Int<bM>{}, Int<bN>{}, Int<bK>{});
    int cta_m = blockIdx.x, cta_n = blockIdx.y;
    if constexpr (PairStride <= -100) {
        constexpr int group = -PairStride-100;
        int linear = blockIdx.x + gridDim.x * blockIdx.y;
        int first_n = linear / (group * gridDim.x) * group;
        int cols = min(group, int(gridDim.y)-first_n);
        int offset = linear % (group * gridDim.x);
        cta_n = first_n + offset % cols;
        cta_m = offset / cols;
    } else if constexpr (PairStride < 0) {
        // Grouped-M scheduling: consume a short band of A rows across N
        // before advancing M. Bijective, including partial final bands.
        constexpr int group = -PairStride;
        int linear = blockIdx.x + gridDim.x * blockIdx.y;
        int first_m = linear / (group * gridDim.y) * group;
        int rows = min(group, int(gridDim.x)-first_m);
        int offset = linear % (group * gridDim.y);
        cta_m = first_m + offset % rows;
        cta_n = offset / rows;
    }
    if constexpr (PairStride > 0) {
        // Bijective wave pairing, including the final partial wave. Encourage
        // concurrently resident CTAs to consume the same B slice through L1.
        int linear = blockIdx.x + gridDim.x * blockIdx.y;
        int base = linear / (2*PairStride) * (2*PairStride);
        int count = min(2*PairStride, int(gridDim.x*gridDim.y)-base);
        int half = (count+1)/2;
        int offset = linear-base;
        int mapped = offset < half ? 2*offset : 2*(offset-half)+1;
        linear = base+mapped;
        cta_m = linear % gridDim.x; cta_n = linear / gridDim.x;
    }
    auto cta_coord = make_coord(cta_m, cta_n, _);
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<Int<1>, X, Int<1>>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X, Int<1>, Int<1>>{});
    Tensor cC = make_identity_tensor(make_shape(Int<bM>{}, Int<bN>{}));

    extern __shared__ char smem_raw[];
    SmemLayoutA_ sA_layout; SmemLayoutB_ sB_layout;
    int8_t* sAp = reinterpret_cast<int8_t*>(smem_raw);
    int8_t* sBp = sAp + cosize(sA_layout);
    Tensor sA = make_tensor(make_smem_ptr(sAp), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(sBp), sB_layout);

    constexpr int TILES = (bM/16) * (bN/16);
    // Modes 3/4/5 eliminate shared transcript; other specialized modes
    // exploit K/R == TLEN, so each slot is written exactly once.
    // Diagnostic modes: 10 samples only the final accumulator; 11 omits
    // warp reduction but retains all intermediate local folds and rT storage.
    __shared__ uint32_t sT[(Mode == 3 || Mode == 4 || Mode == 5 || Mode == 10 || Mode == 11) ? 1 : TILES * TLEN];
    if constexpr (Mode == 0 || Mode == 1 || Mode == 7 || Mode == 8) {
        for (int i = threadIdx.x; i < TILES*TLEN; i += blockDim.x) sT[i] = 0;
    }

    GCopyA_ copyA; GCopyB_ copyB;
    auto thrA = copyA.get_slice(threadIdx.x);
    auto thrB = copyB.get_slice(threadIdx.x);
    Tensor tAgA = thrA.partition_S(gA);  Tensor tAsA = thrA.partition_D(sA);
    Tensor tBgB = thrB.partition_S(gB);  Tensor tBsB = thrB.partition_D(sB);

    TiledMMA_ tiled_mma;
    auto thr_mma = tiled_mma.get_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA);
    Tensor tCsB = thr_mma.partition_B(sB);
    Tensor tCrA = thr_mma.make_fragment_A(tCsA(_,_,_,0));
    Tensor tCrB = thr_mma.make_fragment_B(tCsB(_,_,_,0));
    Tensor tCrC = partition_fragment_C(tiled_mma, make_shape(Int<bM>{}, Int<bN>{}));
    Tensor tCcC = thr_mma.partition_C(cC);
    clear(tCrC);
    constexpr int FT = decltype(size<1>(tCrC))::value * decltype(size<2>(tCrC))::value / 2;
    uint32_t rT[(!DisableTranscript && (Mode == 4 || Mode == 5 || Mode == 11)) ? FT : 1];
    if constexpr (!DisableTranscript && (Mode == 4 || Mode == 5 || Mode == 11)) {
        CUTE_UNROLL
        for (int i = 0; i < FT; ++i) rT[i] = 0;
    }

    auto s2r_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, int8_t>{}, tiled_mma);
    using BLoad=std::conditional_t<Schedule == 6,SM75_U32x4_LDSM_N,SM75_U32x2_LDSM_N>;
    auto s2r_B = make_tiled_copy_B(Copy_Atom<BLoad, int8_t>{}, tiled_mma);
    auto s2rA = s2r_A.get_slice(threadIdx.x);
    auto s2rB = s2r_B.get_slice(threadIdx.x);
    Tensor tXsA = s2rA.partition_S(sA); Tensor tXrA = s2rA.retile_D(tCrA);
    Tensor tXsB = s2rB.partition_S(sB); Tensor tXrB = s2rB.retile_D(tCrB);

    const int k_tiles  = K / bK;
    const int kt_per_R = R / bK;
    const int tiles_w  = N / 16;

    int wr = 0, ww = 0;
    auto stage_copy = [&](int kt, int stage) {
        if constexpr (PaddedCopy) {
            // Exact CTA extents, zero-fill only global edge rows. CuTe's
            // default 128-row copy tiling does not divide a 192-row CTA.
            for (int i=threadIdx.x; i<bM*bK/16; i+=blockDim.x) {
                int row=i/(bK/16), col=i%(bK/16)*16;
                int global_row=cta_m*bM+row;
                const void* src=Aptr+(size_t)(global_row<M?global_row:0)*K+kt*bK+col;
                unsigned dst=__cvta_generic_to_shared(&sA(row,col,stage));
                int bytes=global_row<M?16:0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst),"l"(src),"r"(bytes));
            }
            for (int i=threadIdx.x; i<bN*bK/16; i+=blockDim.x) {
                int row=i/(bK/16), col=i%(bK/16)*16;
                int global_row=cta_n*bN+row;
                const void* src=Bptr+(size_t)(global_row<N?global_row:0)*K+kt*bK+col;
                unsigned dst=__cvta_generic_to_shared(&sB(row,col,stage));
                int bytes=global_row<N?16:0;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst),"l"(src),"r"(bytes));
            }
        } else {
            copy(copyA, tAgA(_,_,_,kt), tAsA(_,_,_,stage));
            copy(copyB, tBgB(_,_,_,kt), tBsB(_,_,_,stage));
        }
    };
    CUTE_UNROLL
    for (int s = 0; s < Stages-1; ++s) {
        stage_copy(s,s);
        cp_async_fence(); ++ww;
    }

    for (int kt = 0; kt < k_tiles; ++kt) {
        cp_async_wait<Stages-2>(); __syncthreads();
        int knext = kt + (Stages-1);
        if constexpr (Schedule == 1) {
            if (knext < k_tiles) stage_copy(knext,ww);
            cp_async_fence();
        }
        if constexpr (Schedule != 5 && Schedule != 7 && Schedule != 8 && Schedule < 9) {
            copy(s2r_A, tXsA(_,_,_,wr), tXrA);
            copy(s2r_B, tXsB(_,_,_,wr), tXrB);
        }
        if constexpr (Schedule != 1) {
            if (knext < k_tiles) stage_copy(knext,ww);
            cp_async_fence();
        }
        if constexpr ((Schedule >= 12 && Schedule <= 14) || Schedule == 16 || Schedule == 17 || Schedule == 18) {
            static_assert(decltype(size<1>(tXrB))::value*2 == decltype(size<2>(tCrC))::value);
            CUTE_UNROLL
            for(int kk=0; kk<size<2>(tCrA); ++kk) {
                if constexpr (Schedule == 16) {
                    if (kk==0) copy(s2r_A,tXsA(_,_,kk,wr),tXrA(_,_,kk));
                } else {
                    copy(s2r_A,tXsA(_,_,kk,wr),tXrA(_,_,kk));
                }
                copy(s2r_B,tXsB(_,0,kk,wr),tXrB(_,0,kk));
                CUTE_UNROLL
                for(int bn=0; bn<size<1>(tXrB); ++bn) {
                    if constexpr (Schedule == 12 || Schedule == 16) {
                        if (bn+1<size<1>(tXrB))
                            copy(s2r_B,tXsB(_,bn+1,kk,wr),tXrB(_,bn+1,kk));
                    }
                    CUTE_UNROLL
                    for(int half=0; half<2; ++half) {
                        int tn=2*bn+half;
                        CUTE_UNROLL
                        for(int tm=0; tm<size<1>(tCrC); ++tm) {
                            if constexpr (Schedule == 17 || Schedule == 18) {
                                // Empty compiler fences: retain the intended LDSM
                                // placement without emitting GPU synchronization.
                                constexpr int split = Schedule == 17 ? 2 : 3;
                                if (half==0 && tm==split && bn+1<size<1>(tXrB)) {
                                    asm volatile("" :: "r"(tCrC(0,tm-1,tn)) : "memory");
                                    copy(s2r_B,tXsB(_,bn+1,kk,wr),tXrB(_,bn+1,kk));
                                    auto packed_next_B = recast<uint32_t>(tXrB(_,bn+1,kk));
                                    asm volatile("" :: "r"(packed_next_B(0)) : "memory");
                                }
                            }
                            if constexpr (Schedule == 14) {
                                if (half==0 && tm==2 && bn+1<size<1>(tXrB))
                                    copy(s2r_B,tXsB(_,bn+1,kk,wr),tXrB(_,bn+1,kk));
                            }
                            gemm(tiled_mma,tCrC(_,tm,tn),tCrA(_,tm,kk),tCrB(_,tn,kk),tCrC(_,tm,tn));
                            if constexpr (Schedule == 16) {
                                // This A row has no remaining consumers in
                                // K32(kk); prefetch the next slice immediately.
                                if (bn+1==size<1>(tXrB) && half==1 && kk+1<size<2>(tCrA))
                                    copy(s2r_A,tXsA(_,tm,kk+1,wr),tXrA(_,tm,kk+1));
                            }
                        }
                        if constexpr (Schedule == 13) {
                            // Half0 is now dead: overlap next B group with
                            // half1 MMAs while limiting extra live operands.
                            if (half==0 && bn+1<size<1>(tXrB))
                                copy(s2r_B,tXsB(_,bn+1,kk,wr),tXrB(_,bn+1,kk));
                        }
                    }
                }
            }
        } else if constexpr (Schedule == 23 || Schedule == 24) {
            // Explicit rolling storage: all K32 slices reuse A slice zero;
            // B groups reuse two or three physical fragment slots.
            constexpr int slots = Schedule == 23 ? 3 : 2;
            constexpr int lead = slots-1;
            constexpr int groups=decltype(size<1>(tXrB))::value;
            static_assert(bK==64 && groups==4);
            copy(s2r_A,tXsA(_,_,0,wr),tXrA(_,_,0));
            CUTE_UNROLL
            for(int p=0;p<lead;++p)
                copy(s2r_B,tXsB(_,p,0,wr),tXrB(_,p,0));
            CUTE_UNROLL
            for(int g=0;g<2*groups;++g) {
                int kk=g/groups, bn=g%groups, slot=g%slots;
                CUTE_UNROLL
                for(int half=0;half<2;++half) {
                    CUTE_UNROLL
                    for(int tm=0;tm<size<1>(tCrC);++tm) {
                        if(half==0 && tm==2 && g+lead<2*groups)
                            copy(s2r_B,tXsB(_,(g+lead)%groups,(g+lead)/groups,wr),
                                        tXrB(_,(g+lead)%slots,0));
                        gemm(tiled_mma,tCrC(_,tm,2*bn+half),tCrA(_,tm,0),
                             tCrB(_,2*slot+half,0),tCrC(_,tm,2*bn+half));
                        if(bn==groups-1 && half==1 && kk==0)
                            copy(s2r_A,tXsA(_,tm,1,wr),tXrA(_,tm,0));
                    }
                }
            }
        } else if constexpr (Schedule == 21 || Schedule == 22) {
            // Register software pipeline spanning both K32 slices of K64.
            // Each B group is loaded once and consumed by eight independent
            // MMAs. Two groups are ready in the prologue; subsequent loads
            // are issued during the current group's MMA consumption.
            constexpr int groups=decltype(size<1>(tXrB))::value;
            constexpr int slices=decltype(size<2>(tCrA))::value;
            static_assert(bK==64 && groups==4 && slices==2);
            copy(s2r_A,tXsA(_,_,0,wr),tXrA(_,_,0));
            copy(s2r_B,tXsB(_,0,0,wr),tXrB(_,0,0));
            copy(s2r_B,tXsB(_,1,0,wr),tXrB(_,1,0));
            CUTE_UNROLL
            for(int g=0; g<groups*slices; ++g) {
                int kk=g/groups, bn=g%groups;
                if constexpr (Schedule == 21) {
                    if(bn==0 && kk>0)
                        copy(s2r_A,tXsA(_,_,kk,wr),tXrA(_,_,kk));
                }
                CUTE_UNROLL
                for(int half=0; half<2; ++half) {
                    int tn=2*bn+half;
                    CUTE_UNROLL
                    for(int tm=0; tm<size<1>(tCrC); ++tm) {
                        if(half==0 && tm==2 && g+2<groups*slices)
                            copy(s2r_B,tXsB(_,(g+2)%groups,(g+2)/groups,wr),
                                        tXrB(_,(g+2)%groups,(g+2)/groups));
                        gemm(tiled_mma,tCrC(_,tm,tn),tCrA(_,tm,kk),tCrB(_,tn,kk),tCrC(_,tm,tn));
                        if constexpr (Schedule == 22) {
                            // Load next A row only after its last old-slice
                            // consumer; no extra A/shared loads are introduced.
                            if(bn==groups-1 && half==1 && kk+1<slices)
                                copy(s2r_A,tXsA(_,tm,kk+1,wr),tXrA(_,tm,kk+1));
                        }
                    }
                }
            }
        } else if constexpr (Schedule == 19 || Schedule == 20) {
            // Reorder independent accumulator chains, retaining one load per
            // operand fragment and the existing group-ahead B prefetch.
            CUTE_UNROLL
            for(int kk=0; kk<size<2>(tCrA); ++kk) {
                copy(s2r_A,tXsA(_,_,kk,wr),tXrA(_,_,kk));
                copy(s2r_B,tXsB(_,0,kk,wr),tXrB(_,0,kk));
                CUTE_UNROLL
                for(int bn=0; bn<size<1>(tXrB); ++bn) {
                    if(bn+1<size<1>(tXrB))
                        copy(s2r_B,tXsB(_,bn+1,kk,wr),tXrB(_,bn+1,kk));
                    CUTE_UNROLL
                    for(int i=0; i<2*size<1>(tCrC); ++i) {
                        constexpr int rows=decltype(size<1>(tCrC))::value;
                        int tm = Schedule == 19 ? (i%rows+bn)%rows : i/2;
                        int half = Schedule == 19 ? i/rows : i%2;
                        int tn=2*bn+half;
                        gemm(tiled_mma,tCrC(_,tm,tn),tCrA(_,tm,kk),tCrB(_,tn,kk),tCrC(_,tm,tn));
                    }
                }
            }
        } else if constexpr (Schedule == 9) {
            static_assert(decltype(size<1>(tXrB))::value*2 == decltype(size<2>(tCrC))::value);
            // Hold one K32 A slice; consume each B atom across all row groups.
            CUTE_UNROLL
            for(int kk=0; kk<size<2>(tCrA); ++kk) {
                copy(s2r_A,tXsA(_,_,kk,wr),tXrA(_,_,kk));
                CUTE_UNROLL
                for(int bn=0; bn<size<1>(tXrB); ++bn) {
                    // One B copy group covers both 8-column MMA halves.
                    copy(s2r_B,tXsB(_,bn,kk,wr),tXrB(_,bn,kk));
                    CUTE_UNROLL
                    for(int half=0; half<2; ++half) {
                    int tn=2*bn+half;
                    CUTE_UNROLL
                    for(int tm=0; tm<size<1>(tCrC); ++tm)
                        gemm(tiled_mma,tCrC(_,tm,tn),tCrA(_,tm,kk),tCrB(_,tn,kk),tCrC(_,tm,tn));
                    }
                }
            }
        } else if constexpr (Schedule == 11) {
            static_assert(decltype(size<1>(tXrB))::value*2 == decltype(size<2>(tCrC))::value);
            // Stream both operands: fewer live registers, but B reloads.
            CUTE_UNROLL
            for(int kk=0; kk<size<2>(tCrA); ++kk)
            CUTE_UNROLL
            for(int tm=0; tm<size<1>(tCrC); ++tm) {
                copy(s2r_A,tXsA(_,tm,kk,wr),tXrA(_,tm,kk));
                CUTE_UNROLL
                for(int bn=0; bn<size<1>(tXrB); ++bn) {
                    copy(s2r_B,tXsB(_,bn,kk,wr),tXrB(_,bn,kk));
                    CUTE_UNROLL
                    for(int half=0; half<2; ++half) {
                    int tn=2*bn+half;
                    gemm(tiled_mma,tCrC(_,tm,tn),tCrA(_,tm,kk),tCrB(_,tn,kk),tCrC(_,tm,tn));
                    }
                }
            }
        } else if constexpr (Schedule == 7 || Schedule == 8) {
            static_assert(bK == 64);
            // Load K32(0), then hide K32(1) loads behind independent
            // accumulator groups. Shared stage lifetime remains unchanged.
            copy(s2r_A,tXsA(_,_,0,wr),tXrA(_,_,0));
            copy(s2r_B,tXsB(_,_,0,wr),tXrB(_,_,0));
            CUTE_UNROLL
            for(int tm=0; tm<size<1>(tCrC); ++tm) {
                if (tm == 1) copy(s2r_A,tXsA(_,_,1,wr),tXrA(_,_,1));
                if (tm == (Schedule == 7 ? 1 : 2))
                    copy(s2r_B,tXsB(_,_,1,wr),tXrB(_,_,1));
                CUTE_UNROLL
                for(int tn=0; tn<size<2>(tCrC); ++tn)
                    gemm(tiled_mma,tCrC(_,tm,tn),tCrA(_,tm,0),tCrB(_,tn,0),tCrC(_,tm,tn));
            }
            gemm(tiled_mma,tCrC,tCrA(_,_,1),tCrB(_,_,1),tCrC);
        } else if constexpr (Schedule == 3 || Schedule == 4) {
            CUTE_UNROLL
            for(int kk=0; kk<size<2>(tCrA); ++kk)
            CUTE_UNROLL
            for(int tm=0; tm<size<1>(tCrC); ++tm)
            CUTE_UNROLL
            for(int tn=0; tn<size<2>(tCrC); ++tn) {
                int ns = Schedule == 3 && (tm&1) ? size<2>(tCrC)-1-tn : tn;
                gemm(tiled_mma,tCrC(_,tm,ns),tCrA(_,tm,kk),tCrB(_,ns,kk),tCrC(_,tm,ns));
            }
        } else if constexpr (Schedule == 5) {
            CUTE_UNROLL
            for(int kk=0; kk<size<2>(tCrA); ++kk) {
                copy(s2r_A,tXsA(_,_,kk,wr),tXrA(_,_,kk));
                copy(s2r_B,tXsB(_,_,kk,wr),tXrB(_,_,kk));
                gemm(tiled_mma,tCrC,tCrA(_,_,kk),tCrB(_,_,kk),tCrC);
            }
        } else {
            gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);
        }
        ww = (ww+1) % Stages; wr = (wr+1) % Stages;

        if constexpr (!DisableTranscript) {
        if ((Mode == 10 ? kt+1 == k_tiles : (kt+1) % kt_per_R == 0)) {
            int rb = kt / kt_per_R;
            const int lane = threadIdx.x & 31;
            CUTE_UNROLL
            for (int tm = 0; tm < size<1>(tCrC); ++tm)
            CUTE_UNROLL
            for (int k = 0; k < size<2>(tCrC)/2; ++k) {
                uint32_t pv = 0;
                CUTE_UNROLL
                for (int v = 0; v < size<0>(tCrC); ++v) {
                    pv ^= (uint32_t)tCrC(v,tm,2*k); pv ^= (uint32_t)tCrC(v,tm,2*k+1);
                }
                if constexpr (Mode == 0 || Mode == 5 || Mode == 8) {
                    CUTE_UNROLL
                    for (int off = 16; off > 0; off >>= 1) pv ^= __shfl_xor_sync(0xffffffffu, pv, off);
                } else if constexpr (Mode != 11) {
                    pv = __reduce_xor_sync(0xffffffffu, pv);
                }
                if constexpr (Mode == 4 || Mode == 5 || Mode == 11) {
                    // Each lane retains one R-boundary result per tile.
                    if (lane == rb) rT[tm * (size<2>(tCrC)/2) + k] = pv;
                } else if (lane == 0) {
                    auto mn = tCcC(0, tm, 2*k);
                    int lt = (get<0>(mn) >> 4) * (bN/16) + (get<1>(mn) >> 4);
                    if constexpr (Mode == 3 || Mode == 10) {
                        int gr = cta_m * (bM/16) + lt/(bN/16);
                        int gc = cta_n * (bN/16) + lt%(bN/16);
                        if (!PaddedCopy || (gr < M/16 && gc < N/16))
                            transcript[(size_t)(gr*tiles_w+gc)*TLEN+rb] = pv;
                    } else if constexpr (Mode == 0 || Mode == 1 || Mode == 7 || Mode == 8) {
                        uint32_t prev = sT[lt*TLEN + rb];
                        sT[lt*TLEN + rb] = ((prev << HASH_ROT) | (prev >> (32-HASH_ROT))) ^ pv;
                    } else {
                        sT[lt*TLEN + rb] = pv;
                    }
                }
            }
        }
        }
    }

    const int trow0 = cta_m * (bM/16);
    const int tcol0 = cta_n * (bN/16);
    if constexpr (DisableTranscript) {
        // One observable checksum per warp, once after the complete GEMM.
        // Every accumulator contributes; no transcript tiles or intermediate
        // R-boundary samples are produced. Output is NEVER a mining transcript.
        uint32_t checksum=0;
        CUTE_UNROLL
        for (int i=0; i<size(tCrC); ++i) checksum ^= uint32_t(tCrC(i));
        checksum=__reduce_xor_sync(0xffffffffu,checksum);
        if ((threadIdx.x&31)==0)
            transcript[(size_t)(cta_n*gridDim.x+cta_m)*(blockDim.x/32)+(threadIdx.x/32)]=checksum;
    } else if constexpr (Mode == 4 || Mode == 5 || Mode == 11) {
        int lane = threadIdx.x & 31;
        if (lane < TLEN) {
            CUTE_UNROLL
            for (int tm=0; tm<size<1>(tCrC); ++tm)
            CUTE_UNROLL
            for (int k=0; k<size<2>(tCrC)/2; ++k) {
                auto mn = tCcC(0,tm,2*k);
                int gr=trow0+(get<0>(mn)>>4), gc=tcol0+(get<1>(mn)>>4);
                if (!PaddedCopy || (gr<M/16 && gc<N/16))
                    transcript[(size_t)(gr*tiles_w+gc)*TLEN+lane] = rT[tm*(size<2>(tCrC)/2)+k];
            }
        }
    } else if constexpr (Mode != 3 && Mode != 10) {
        __syncthreads();
        if constexpr (Mode == 7 || Mode == 8 || Mode == 9) {
            // Adjacent lanes write adjacent words, instead of strided tiles.
            for (int i=threadIdx.x; i<TILES*TLEN; i+=blockDim.x) {
                int lt=i/TLEN, t=i%TLEN;
                int gtile=(trow0+lt/(bN/16))*tiles_w+tcol0+lt%(bN/16);
                if (!PaddedCopy || (trow0+lt/(bN/16)<M/16 && tcol0+lt%(bN/16)<N/16))
                    transcript[(size_t)gtile*TLEN+t]=sT[i];
            }
        } else {
            for (int lt=threadIdx.x; lt<TILES; lt+=blockDim.x) {
                int gtile=(trow0+lt/(bN/16))*tiles_w+tcol0+lt%(bN/16);
                CUTE_UNROLL
                for (int t=0;t<TLEN;++t) transcript[(size_t)gtile*TLEN+t]=sT[lt*TLEN+t];
            }
        }
    }
}

} // namespace foldops
