// Isolated tuning library; never changes the production dispatcher or binary.
#include <cuda_runtime.h>
#include "../csrc/gemm/pearl_cute_fold.cuh"
extern void launch_pearl_gemm_only(const int8_t*, const int8_t*, int, int, int, int,
                                  uint32_t*, int, cudaStream_t);
extern void launch_pearl_blake3(const uint32_t*, int, int, const uint32_t*,
                               const uint32_t*, uint8_t*, int*, int*, cudaStream_t);

// Keep warp ownership of 16-row bands. Match global-copy thread count to MMA.
template<int BM, int BN, int S, int Carve = -1, bool KeepL1 = false, int BK = 32>
cudaError_t variant(const int8_t* a, const int8_t* b, int m, int n, int k, int r,
                    uint32_t* out, int* info = nullptr) {
    using namespace cute;
    using Atom = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;
    using MMA = decltype(make_tiled_mma(Atom{}, Layout<Shape<Int<BM/16>,_1,_1>>{}));
    using SA = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<BM>,Int<BK>>,Stride<Int<BK>,_1>>{}));
    using SB = decltype(composition(Swizzle<2,4,3>{}, Layout<Shape<Int<BN>,Int<BK>>,Stride<Int<BK>,_1>>{}));
    using LA = decltype(tile_to_shape(SA{},make_shape(Int<BM>{},Int<BK>{},Int<S>{})));
    using LB = decltype(tile_to_shape(SB{},make_shape(Int<BN>{},Int<BK>{},Int<S>{})));
    constexpr int threads = BM/16*32;
    using CopyOp = std::conditional_t<KeepL1, SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>>;
    using Copy = decltype(make_tiled_copy(Copy_Atom<CopyOp,int8_t>{},
        Layout<Shape<Int<threads/2>,_2>,Stride<_2,_1>>{}, Layout<Shape<_1,_16>>{}));
    auto kern = pcute::cute_pearl_fold<MMA,Copy,Copy,LA,LB,BM,BN,BK,S>;
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
    kern<<<dim3(m/BM,n/BN),threads,smem>>>(m,n,k,r,a,b,out);
    return cudaGetLastError();
}

static cudaError_t dispatch(int id,const int8_t* a,const int8_t* b,int m,int n,int k,int r,uint32_t* t,int* info=nullptr) {
    switch(id) {
#define V(ID,M,N,S,C) case ID:return variant<M,N,S,C>(a,b,m,n,k,r,t,info);
        V(0,128,256,3,-1)
        V(1,128,128,3,-1)
        V(2,64,256,3,-1)
        V(3,64,128,3,-1)
        V(4,128,128,4,-1)
        V(5,128,256,2,-1)
        V(6,128,256,3,100)
        V(7,128,128,3,100)
        V(8,64,256,3,100)
        V(9,64,256,4,100)
        V(10,128,256,4,-1)
        V(11,128,256,5,-1)
        V(12,256,128,3,-1)
        V(13,256,128,4,-1)
        V(14,128,128,2,-1)
        case 15:return variant<128,256,3,-1,true>(a,b,m,n,k,r,t,info);
        case 16:return variant<128,256,2,-1,false,64>(a,b,m,n,k,r,t,info);
        case 17:return variant<128,256,3,-1,false,64>(a,b,m,n,k,r,t,info);
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
    auto err=dispatch(id,a,b,m,n,k,r,t);if(err!=cudaSuccess)return err;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,0);
    return cudaGetLastError();
}
