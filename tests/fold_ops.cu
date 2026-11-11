// Isolated tuning library; never changes the production dispatcher or binary.
#include <cuda_runtime.h>
#include "../csrc/gemm/pearl_cute_fold.cuh"
#include "fold_ops.cuh"
extern void launch_pearl_gemm_only(const int8_t*, const int8_t*, int, int, int, int,
                                  uint32_t*, int, cudaStream_t);
extern void launch_pearl_blake3(const uint32_t*, int, int, const uint32_t*,
                               const uint32_t*, uint8_t*, int*, int*, cudaStream_t);
#ifdef PEARL_ADA_V2
extern cudaError_t launch_pearl_gemm_ada_v2(int,const int8_t*,const int8_t*,int,int,int,int,uint32_t*,int*,cudaStream_t);
#endif

// Keep warp ownership of 16-row bands. Match global-copy thread count to MMA.
template<int BM, int BN, int S, int Carve = -1, bool KeepL1 = false, int BK = 32, int Mode = -1, int WM = BM/16, int WN = 1, int PairStride = 0, bool PaddedCopy = false, bool DisableTranscript = false, int Schedule = 0>
cudaError_t variant(const int8_t* a, const int8_t* b, int m, int n, int k, int r,
                    uint32_t* out, int* info = nullptr, cudaStream_t stream = nullptr) {
    using namespace cute;
    using Atom = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    // Keep both 8-column MMA halves of each transcript tile in one warp.
    using NPerm = Layout<Shape<_8,Int<WN>,_2>,Stride<_1,_16,_8>>;
    using Perm = std::conditional_t<WN == 1, Underscore, NPerm>;
    using MMA = decltype(make_tiled_mma(Atom{}, Layout<Shape<Int<WM>,Int<WN>,_1>>{},
                                       make_tile(_, Perm{}, _)));
    using SA = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<BM>,Int<BK>>,Stride<Int<BK>,_1>>{}));
    using SB = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<BN>,Int<BK>>,Stride<Int<BK>,_1>>{}));
    using LA = decltype(tile_to_shape(SA{},make_shape(Int<BM>{},Int<BK>{},Int<S>{})));
    using LB = decltype(tile_to_shape(SB{},make_shape(Int<BN>{},Int<BK>{},Int<S>{})));
    constexpr int threads = WM*WN*32;
    using CopyOp = std::conditional_t<KeepL1, SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>;
    using Copy = decltype(make_tiled_copy(Copy_Atom<CopyOp,int8_t>{},
        Layout<Shape<Int<PaddedCopy ? 128 : threads/2>,_2>,Stride<_2,_1>>{}, Layout<Shape<_1,_16>>{}));
    auto kern = [] {
        if constexpr (Mode == -1) return pcute::cute_pearl_fold<MMA,Copy,Copy,LA,LB,BM,BN,BK,S>;
        else return foldops::cute_pearl_fold<MMA,Copy,Copy,LA,LB,BM,BN,BK,S,Mode,PairStride,PaddedCopy,DisableTranscript,Schedule>;
    }();
    constexpr int smem = cosize(LA{}) + cosize(LB{});
    static bool initialized = false;
    if (!initialized) {
        auto err = cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
        if (err != cudaSuccess) return err;
        if (Carve >= 0) {
            err = cudaFuncSetAttribute(kern,cudaFuncAttributePreferredSharedMemoryCarveout,Carve);
            if (err != cudaSuccess) return err;
        }
        initialized = true;
    }
    if (info) {
        // Carveout variants share the same kernel specialization; restore the
        // requested policy before each experiment to avoid cross-run bias.
        auto policy_err = cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, Carve);
        if (policy_err != cudaSuccess) return policy_err;
        cudaFuncAttributes attrs;
        auto err=cudaFuncGetAttributes(&attrs,kern); if(err!=cudaSuccess)return err;
        int blocks=0;
        err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kern,threads,smem);
        if(err!=cudaSuccess)return err;
        info[0]=attrs.numRegs;info[1]=threads;info[2]=smem+attrs.sharedSizeBytes;
        info[3]=blocks;info[4]=attrs.localSizeBytes;
        return cudaSuccess;
    }
    kern<<<dim3(PaddedCopy?(m+BM-1)/BM:m/BM,PaddedCopy?(n+BN-1)/BN:n/BN),threads,smem,stream>>>(m,n,k,r,a,b,out);
    return cudaGetLastError();
}

static cudaError_t dispatch(int id,const int8_t* a,const int8_t* b,int m,int n,int k,int r,uint32_t* t,int* info=nullptr,cudaStream_t stream=nullptr) {
    if (!info && (k != 4096 || r != 256)) return cudaErrorInvalidValue;
    switch(id) {
#define V(ID,M,N,S,C) case ID:return variant<M,N,S,C>(a,b,m,n,k,r,t,info);
        V(0,128,256,3,-1)
#if !defined(PEARL_FOCUS_SQUARE) && !defined(PEARL_PURE_BENCH_ONLY)
#define OP(ID,MODE,STAGE,BLOCKK) case ID:return variant<128,256,STAGE,-1,false,BLOCKK,MODE>(a,b,m,n,k,r,t,info);
        OP(1,0,3,32) // unmodified clone sanity check
        OP(2,1,3,32) // native XOR reduction
        OP(3,2,3,32) // native XOR, no shared initialization / old-value load
        OP(4,3,3,32) // native XOR, direct global output
        OP(5,4,3,32) // native XOR, lane-distributed register transcript
        OP(6,5,3,32) // shuffle XOR, lane-distributed register transcript
        OP(7,7,3,32) // native XOR + coalesced shared transcript epilogue
        OP(8,8,3,32) // shuffle XOR + coalesced epilogue
        OP(9,9,3,32) // native XOR, no init/load + coalesced epilogue
        OP(10,4,4,32)
        OP(11,4,3,64)
        OP(12,9,4,32)
        OP(13,9,3,64)
        case 14:return variant<128,256,3,-1,false,32,1,4,2>(a,b,m,n,k,r,t,info);
        case 15:return variant<128,256,3,-1,false,32,4,4,2>(a,b,m,n,k,r,t,info);
        case 16:return variant<128,256,4,-1,false,32,4,4,2>(a,b,m,n,k,r,t,info);
        case 17:return variant<128,256,3,-1,false,64,4,4,2>(a,b,m,n,k,r,t,info);
        case 18:return variant<128,256,3,-1,false,64,9,4,2>(a,b,m,n,k,r,t,info);
        case 19:return variant<128,256,4,-1,false,64,4,4,2>(a,b,m,n,k,r,t,info);
#endif
#ifndef PEARL_REUSE_FOCUS
        case 20:return variant<128,256,3,-1,false,64,4,2,4>(a,b,m,n,k,r,t,info,stream);
        case 48:return variant<128,256,3,-1,false,64,4,2,4,0,false,true>(a,b,m,n,k,r,t,info,stream);
        case 49:return variant<128,256,3,-1,false,64,4,2,4,0,false,false,1>(a,b,m,n,k,r,t,info);
        case 50:return variant<128,256,3,-1,false,64,4,2,4,0,false,false,3>(a,b,m,n,k,r,t,info);
        case 51:return variant<128,256,3,-1,false,64,4,2,4,0,false,false,4>(a,b,m,n,k,r,t,info);
        case 52:return variant<128,256,3,-1,false,64,4,2,4,0,false,false,5>(a,b,m,n,k,r,t,info);
        case 53:return variant<128,256,3,-1,false,64,4,2,4,0,false,false,6>(a,b,m,n,k,r,t,info);
        case 54:return variant<128,256,2,100,false,128,4,2,4,0,false,false,5>(a,b,m,n,k,r,t,info);
        case 55:return variant<128,256,2,100,false,128,4,2,4>(a,b,m,n,k,r,t,info);
        case 56:return variant<128,256,3,-1,false,64,4,2,4,0,false,false,7>(a,b,m,n,k,r,t,info);
        case 57:return variant<128,256,3,-1,false,64,4,2,4,0,false,false,8>(a,b,m,n,k,r,t,info);
        case 58:return variant<128,256,3,-1,false,64,4,2,4,-1>(a,b,m,n,k,r,t,info);
        case 59:return variant<128,256,3,-1,false,64,4,2,4,-4>(a,b,m,n,k,r,t,info);
        case 60:return variant<128,256,3,-1,false,64,4,2,4,-8>(a,b,m,n,k,r,t,info);
        case 61:return variant<128,256,3,-1,false,64,4,2,4,-16>(a,b,m,n,k,r,t,info);
        case 62:return variant<128,256,3,-1,false,64,4,2,4,-2>(a,b,m,n,k,r,t,info);
        case 63:return variant<128,256,3,-1,false,64,4,2,4,-12>(a,b,m,n,k,r,t,info);
        case 64:return variant<128,256,3,-1,false,64,4,2,4,-24>(a,b,m,n,k,r,t,info);
        case 65:return variant<128,256,3,-1,false,64,4,2,4,-102>(a,b,m,n,k,r,t,info);
        case 66:return variant<128,256,3,-1,false,64,4,2,4,-104>(a,b,m,n,k,r,t,info);
#endif
        case 67:return variant<128,256,3,-1,false,64,4,2,4,-108>(a,b,m,n,k,r,t,info);
#ifndef PEARL_REUSE_FOCUS
        case 68:return variant<256,128,3,-1,false,64,4,4,2,-108>(a,b,m,n,k,r,t,info);
        case 69:return variant<256,128,3,-1,false,64,4,2,4,-108>(a,b,m,n,k,r,t,info);
        case 70:return variant<256,128,3,-1,false,64,4,8,1,-108>(a,b,m,n,k,r,t,info);
        case 71:return variant<128,128,3,100,false,64,4,2,2,-108>(a,b,m,n,k,r,t,info);
        case 72:return variant<128,128,3,100,false,64,4,1,4,-108>(a,b,m,n,k,r,t,info);
        case 73:return variant<128,128,3,100,false,64,4,4,1,-108>(a,b,m,n,k,r,t,info);
        case 74:return variant<128,320,4,-1,false,32,4,2,5,-108,true>(a,b,m,n,k,r,t,info);
        case 75:return variant<128,320,4,-1,false,32,3,2,5,-108,true>(a,b,m,n,k,r,t,info);
        case 76:return variant<128,320,3,-1,false,32,9,2,5,-108,true>(a,b,m,n,k,r,t,info);
        case 77:return variant<128,320,4,-1,false,32,9,2,5,-108,true>(a,b,m,n,k,r,t,info);
#endif
        case 78:return variant<128,320,3,-1,false,32,9,2,5,-108,true,false,9>(a,b,m,n,k,r,t,info);
        case 79:return variant<128,320,3,-1,false,32,9,2,5,-108,true,false,11>(a,b,m,n,k,r,t,info);
        case 80:return variant<128,320,3,-1,false,32,4,2,5,-108,true,false,9>(a,b,m,n,k,r,t,info);
        case 81:return variant<128,256,3,-1,false,64,4,2,4,-108,false,false,9>(a,b,m,n,k,r,t,info);
        case 82:return variant<256,192,3,-1,false,32,9,4,3,-108,true,false,9>(a,b,m,n,k,r,t,info);
        case 83:return variant<256,192,3,-1,false,32,9,4,3,-108,true,false,11>(a,b,m,n,k,r,t,info);
        case 84:return variant<256,192,3,-1,false,32,4,4,3,-108,true,false,9>(a,b,m,n,k,r,t,info);
        case 85:return variant<256,192,4,-1,false,32,9,4,3,-108,true,false,9>(a,b,m,n,k,r,t,info);
        case 86:return variant<256,192,3,100,false,64,9,4,3,-108,true,false,9>(a,b,m,n,k,r,t,info);
        case 87:return variant<256,192,3,100,false,64,9,4,3,-108,true,false,11>(a,b,m,n,k,r,t,info);
        case 88:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,9>(a,b,m,n,k,r,t,info);
        case 89:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,12>(a,b,m,n,k,r,t,info);
        case 90:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,13>(a,b,m,n,k,r,t,info);
        case 91:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,14>(a,b,m,n,k,r,t,info);
        case 92:return variant<256,192,3,100,false,64,9,4,3,-108,true,false,12>(a,b,m,n,k,r,t,info);
        case 93:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,16>(a,b,m,n,k,r,t,info);
        case 94:return variant<256,192,3,100,false,64,9,4,3,-108,true,false,16>(a,b,m,n,k,r,t,info);
        case 95:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,17>(a,b,m,n,k,r,t,info);
        case 96:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,18>(a,b,m,n,k,r,t,info);
        case 97:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,19>(a,b,m,n,k,r,t,info);
        case 98:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,20>(a,b,m,n,k,r,t,info);
        case 99:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,21>(a,b,m,n,k,r,t,info);
        case 100:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,22>(a,b,m,n,k,r,t,info);
        case 101:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,23>(a,b,m,n,k,r,t,info);
        case 102:return variant<256,192,2,100,false,64,9,4,3,-108,true,false,24>(a,b,m,n,k,r,t,info);
#ifdef PEARL_ADA_V2
        case 103:return launch_pearl_gemm_ada_v2(1,a,b,m,n,k,r,t,info,nullptr);
        case 104:return launch_pearl_gemm_ada_v2(2,a,b,m,n,k,r,t,info,nullptr);
        case 105:return launch_pearl_gemm_ada_v2(0,a,b,m,n,k,r,t,info,nullptr);
        case 106:return launch_pearl_gemm_ada_v2(3,a,b,m,n,k,r,t,info,nullptr);
        case 107:return launch_pearl_gemm_ada_v2(4,a,b,m,n,k,r,t,info,nullptr);
        case 108:return launch_pearl_gemm_ada_v2(5,a,b,m,n,k,r,t,info,nullptr);
        case 109:return launch_pearl_gemm_ada_v2(2,a,b,m,n,k,r,t,info,nullptr);
        case 110:return launch_pearl_gemm_ada_v2(6,a,b,m,n,k,r,t,info,nullptr);
        case 111:return launch_pearl_gemm_ada_v2(7,a,b,m,n,k,r,t,info,nullptr);
        case 112:return launch_pearl_gemm_ada_v2(8,a,b,m,n,k,r,t,info,nullptr);
        case 113:return launch_pearl_gemm_ada_v2(9,a,b,m,n,k,r,t,info,nullptr);
        case 114:return launch_pearl_gemm_ada_v2(10,a,b,m,n,k,r,t,info,nullptr);
        case 115:return launch_pearl_gemm_ada_v2(11,a,b,m,n,k,r,t,info,nullptr);
        case 116:return launch_pearl_gemm_ada_v2(12,a,b,m,n,k,r,t,info,nullptr);
        case 117:return launch_pearl_gemm_ada_v2(13,a,b,m,n,k,r,t,info,nullptr);
        case 118:return launch_pearl_gemm_ada_v2(14,a,b,m,n,k,r,t,info,nullptr);
        case 119:return launch_pearl_gemm_ada_v2(15,a,b,m,n,k,r,t,info,nullptr);
        case 120:return launch_pearl_gemm_ada_v2(16,a,b,m,n,k,r,t,info,nullptr);
        case 121:return launch_pearl_gemm_ada_v2(17,a,b,m,n,k,r,t,info,nullptr);
#endif
#if !defined(PEARL_FOCUS_SQUARE) && !defined(PEARL_PURE_BENCH_ONLY)
        case 21:return variant<128,256,4,-1,false,32,4,2,4>(a,b,m,n,k,r,t,info);
        case 22:return variant<128,256,3,-1,false,64,4,1,8>(a,b,m,n,k,r,t,info);
        // Diagnostic only, never usable as proof-of-work transcripts.
        case 23:return variant<128,256,3,-1,false,64,10,8,1>(a,b,m,n,k,r,t,info);
        case 24:return variant<128,256,3,-1,false,64,10,2,4>(a,b,m,n,k,r,t,info);
        case 25:return variant<128,256,3,-1,false,64,11,8,1>(a,b,m,n,k,r,t,info);
        case 26:return variant<128,256,3,-1,false,64,11,2,4>(a,b,m,n,k,r,t,info);
        // Smaller accumulator footprints to test two resident CTAs per SM.
        case 27:return variant<128,128,3,100,false,32,4,2,4>(a,b,m,n,k,r,t,info);
        case 28:return variant<128,128,3,100,false,64,4,2,4>(a,b,m,n,k,r,t,info);
        case 29:return variant<64,256,3,100,false,32,4,1,8>(a,b,m,n,k,r,t,info);
        case 30:return variant<128,256,3,-1,false,32,4,2,4>(a,b,m,n,k,r,t,info);
        case 31:return variant<128,128,3,50,true,32,4,2,4>(a,b,m,n,k,r,t,info);
        case 32:return variant<128,128,3,50,true,32,4,2,4,132>(a,b,m,n,k,r,t,info);
        case 33:return variant<128,128,3,50,true,32,4,2,4,128>(a,b,m,n,k,r,t,info);
        case 34:return variant<128,128,3,100,true,64,4,2,4,132>(a,b,m,n,k,r,t,info);
        case 35:return variant<128,256,3,50,true,32,4,2,4>(a,b,m,n,k,r,t,info);
        case 36:return variant<128,256,3,100,true,64,4,2,4>(a,b,m,n,k,r,t,info);
        case 37:return variant<128,128,3,50,false,32,4,2,4,132>(a,b,m,n,k,r,t,info);
        case 38:return variant<128,128,4,50,true,32,4,2,4,132>(a,b,m,n,k,r,t,info);
#endif
#ifndef PEARL_PURE_BENCH_ONLY
        case 39:return variant<192,192,3,-1,false,32,4,3,4,0,true>(a,b,m,n,k,r,t,info);
        case 40:return variant<192,192,3,-1,false,64,4,3,4,0,true>(a,b,m,n,k,r,t,info);
        case 41:return variant<192,192,4,-1,false,32,4,3,4,0,true>(a,b,m,n,k,r,t,info);
        case 42:return variant<192,192,3,-1,false,32,4,4,3,0,true>(a,b,m,n,k,r,t,info);
        case 43:return variant<192,192,3,-1,false,32,4,4,2,0,true>(a,b,m,n,k,r,t,info);
        case 44:return variant<192,192,4,-1,false,32,4,4,2,0,true>(a,b,m,n,k,r,t,info);
        case 45:return variant<192,192,3,-1,false,64,9,4,2,0,true>(a,b,m,n,k,r,t,info);
        case 46:return variant<192,192,3,-1,false,32,9,4,2,0,true>(a,b,m,n,k,r,t,info);
        case 47:return variant<192,192,3,-1,false,32,4,2,4,0,true>(a,b,m,n,k,r,t,info);
#endif
#undef OP
#undef V
        case -1: launch_pearl_gemm_only(a,b,m,n,k,r,t,1,0);return cudaGetLastError();
        default:return cudaErrorInvalidValue;
    }
}
extern "C" int trial_fold(int id,const int8_t* a,const int8_t* b,int m,int n,int k,int r,uint32_t* t) {
    return dispatch(id,a,b,m,n,k,r,t);
}
extern "C" int trial_info(int id,int* info) {return dispatch(id,nullptr,nullptr,4096,4096,4096,256,nullptr,info);}
extern "C" int trial_bench(int id,const int8_t* a,const int8_t* b,int m,int n,int k,int r,uint32_t* t,int iters,float* ms) {
    cudaEvent_t start,end;
    cudaError_t err=cudaEventCreate(&start);if(err!=cudaSuccess)return err;
    err=cudaEventCreate(&end);if(err!=cudaSuccess){cudaEventDestroy(start);return err;}
    cudaEventRecord(start);
    for(int i=0;i<iters;i++){err=dispatch(id,a,b,m,n,k,r,t);if(err!=cudaSuccess)break;}
    if(err==cudaSuccess){cudaEventRecord(end);err=cudaEventSynchronize(end);}
    if(err==cudaSuccess){err=cudaEventElapsedTime(ms,start,end);*ms/=iters;}
    cudaEventDestroy(start);cudaEventDestroy(end);return err;
}
extern "C" int trial_pow(int id,const int8_t* a,const int8_t* b,int m,int n,int k,int r,const uint32_t* key,const uint32_t* target,
    uint32_t* t,uint8_t* dig,int* found,int* coord) {
    if ((id >= 23 && id <= 26) || id == 48) return cudaErrorInvalidValue;
    auto err=dispatch(id,a,b,m,n,k,r,t);if(err!=cudaSuccess)return err;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,0);
    return cudaGetLastError();
}
