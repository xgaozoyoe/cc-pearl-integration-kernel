// Offline-only SM89 producer/consumer experiment. Original source is included
// to share the exact operand mapping and MMA helpers without touching it.
#include "../csrc/gemm/pearl-gemm-ada-v2.cu"
namespace pearl_ada_ws {
using namespace pearl_ada_v2;
template<int Count>
__device__ __forceinline__ void arrive(int id) {
    asm volatile("bar.arrive %0, %1;" :: "r"(id),"n"(Count) : "memory");
}
template<int Count>
__device__ __forceinline__ void wait(int id) {
    asm volatile("bar.sync %0, %1;" :: "r"(id),"n"(Count) : "memory");
}
template<int Producers>
__device__ __forceinline__ void produce(const int8_t* a,const int8_t* b,
    int8_t* sa,int8_t* sb,int cm,int cn,int kt,int m,int n) {
    int pt=threadIdx.x-384;
    #pragma unroll 1
    for(int i=pt;i<BM*64/16;i+=Producers*32) {
        int logical=i^((i>>3)&3),row=logical/4,col=(logical%4)*16;
        int gr=cm*BM+row,valid=gr<m?16:0;
        auto src=a+(size_t)(gr<m?gr:0)*4096+kt*64+col;
        unsigned dst=__cvta_generic_to_shared(sa+i*16);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
    }
    #pragma unroll 1
    for(int i=pt;i<BN*64/16;i+=Producers*32) {
        int logical=i^((i>>3)&3),row=logical/4,col=(logical%4)*16;
        int gr=cn*BN+row,valid=gr<n?16:0;
        auto src=b+(size_t)(gr<n?gr:0)*4096+kt*64+col;
        unsigned dst=__cvta_generic_to_shared(sb+i*16);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
}
template<int Producers,int Stages=2>
__global__ void __launch_bounds__(384+Producers*32,1)
kernel(const int8_t* a,const int8_t* b,int m,int n,unsigned* out) {
    constexpr int Threads=384+Producers*32;
    constexpr int Lead=2,TileK=64,Layout=6,Groups=4,WarpN=64,k=4096;
    constexpr bool Planar=false,PackedB=true;
    extern __shared__ int8_t shared[];
    int8_t* sa=shared;int8_t* sb=shared+Stages*BM*64;
    unsigned* transcript=reinterpret_cast<unsigned*>(shared+Stages*(BM+BN)*64);
    int linear=blockIdx.x+gridDim.x*blockIdx.y;
    int first=linear/(8*gridDim.x)*8;
    int cols=min(8,int(gridDim.y)-first),off=linear%(8*gridDim.x);
    int cm=off/cols,cn=first+off%cols;
    if(threadIdx.x>=384) {
        for(int kt=0;kt<64;++kt) {
            int stage=kt%Stages;
            if(kt>=Stages) wait<Threads>(2+2*stage);
            produce<Producers>(a,b,sa+stage*BM*64,sb+stage*BN*64,cm,cn,kt,m,n);
            arrive<Threads>(1+2*stage);
        }
    } else {
        int warp=threadIdx.x/32,lane=threadIdx.x%32;
        int wm=warp%4,wn=warp/4;
        unsigned c[4][4][2][4]={};
        Fragment ar[4],br[3];
        for(int kt=0;kt<64;++kt) {
            int stage=kt%Stages;
            wait<Threads>(1+2*stage);
            int8_t* aa=sa+stage*BM*64;
            int8_t* bb=sb+stage*BN*64;
        #pragma unroll
        for(int row=0;row<4;++row) ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,0,lane,BM);
        #pragma unroll
        for(int p=0;p<Lead;++p) br[p]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+p*16,0,lane,BN);
        #pragma unroll
        for(int g=0;g<Groups*(TileK/32);++g) {
            constexpr int slots=Lead+1;
            int slice=g/Groups,group=g%Groups,slot=g%slots;
            #pragma unroll
            for(int half=0;half<2;++half) {
                #pragma unroll
                for(int row=0;row<4;++row) {
                    if(half==0 && row==2 && g+Lead<Groups*(TileK/32) && (TileK==64 || kt*TileK+((g+Lead)/Groups)*32<k))
                        br[(g+Lead)%slots]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+((g+Lead)%Groups)*16,((g+Lead)/Groups)*32,lane,BN);
                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);
                    if(slice+1<TileK/32 && (TileK==64 || kt*TileK+(slice+1)*32<k) && group==Groups-1 && half==1)
                        ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,(slice+1)*32,lane,BM);
                }
            }
        }
        if constexpr(TileK==64) {
        if((kt+1)%4==0) {
            int rb=(kt+1)/4-1;
            #pragma unroll
            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    }
                    x=__reduce_xor_sync(0xffffffffu,x);
                    if(lane==0) transcript[((wm*4+row)*(BN/16)+wn*Groups+group)*16+rb]=x;
                }
            }
        }
        }
            arrive<Threads>(2+2*stage);
        }
    }
    __syncthreads();
    for(int i=threadIdx.x;i<WORDS;i+=Threads) {
        int tile=i/16,gr=cm*(BM/16)+tile/(BN/16),gc=cn*(BN/16)+tile%(BN/16);
        if(gr<m/16 && gc<n/16) out[((size_t)gr*(n/16)+gc)*16+i%16]=transcript[i];
    }
}
template<int Producers,int Stages=2>
cudaError_t launch(const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int* info) {
    constexpr int Threads=384+Producers*32,smem=Stages*(BM+BN)*64+WORDS*4;
    static bool initialized=false;
    cudaError_t e;
    if(!initialized) {
        e=cudaFuncSetAttribute(kernel<Producers,Stages>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
        if(e!=cudaSuccess)return e;
        e=cudaFuncSetAttribute(kernel<Producers,Stages>,cudaFuncAttributePreferredSharedMemoryCarveout,100);
        if(e!=cudaSuccess)return e;
        initialized=true;
    }
    if(info) {
        cudaFuncAttributes attr;e=cudaFuncGetAttributes(&attr,kernel<Producers,Stages>);if(e!=cudaSuccess)return e;
        int active;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel<Producers,Stages>,Threads,smem);
        if(e!=cudaSuccess)return e;
        info[0]=attr.numRegs;info[1]=Threads;info[2]=smem+attr.sharedSizeBytes;info[3]=active;info[4]=attr.localSizeBytes;
        return cudaSuccess;
    }
    kernel<Producers,Stages><<<dim3((m+BM-1)/BM,(n+BN-1)/BN),Threads,smem>>>(a,b,m,n,out);
    return cudaGetLastError();
}
}
extern "C" int ws_run(int id,const int8_t* a,const int8_t* b,int m,int n,int k,int r,unsigned* out,int* info) {
    if(m<=0 || n<=0 || m%16 || n%16 || k!=4096 || r!=256)return cudaErrorInvalidValue;
    if(id==0)return pearl_ada_v2::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,k,r,out,info,nullptr);
    if(id==1)return pearl_ada_ws::launch<1>(a,b,m,n,out,info);
    if(id==2)return pearl_ada_ws::launch<2>(a,b,m,n,out,info);
    return cudaErrorInvalidValue;
}
extern "C" int ws_bench(int id,const int8_t* a,const int8_t* b,int m,int n,int k,int r,unsigned* out,int iters,float* ms) {
    cudaEvent_t start,end;auto e=cudaEventCreate(&start);if(e!=cudaSuccess)return e;
    e=cudaEventCreate(&end);if(e!=cudaSuccess){cudaEventDestroy(start);return e;}
    cudaEventRecord(start);
    for(int i=0;i<iters;++i){e=(cudaError_t)ws_run(id,a,b,m,n,k,r,out,nullptr);if(e!=cudaSuccess)break;}
    if(e==cudaSuccess){cudaEventRecord(end);e=cudaEventSynchronize(end);}
    if(e==cudaSuccess){e=cudaEventElapsedTime(ms,start,end);*ms/=iters;}
    cudaEventDestroy(start);cudaEventDestroy(end);return e;
}
