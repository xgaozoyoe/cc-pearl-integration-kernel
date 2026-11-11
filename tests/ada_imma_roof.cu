// Offline throughput calibration only. Not a Pearl/mining kernel.
// Same 12 warps, 64x64 accumulator ownership, and 64 m16n8k32 per K64.
#include <cuda_runtime.h>
#include <cstdint>
__global__ __launch_bounds__(384,1)
void imma_roof(const unsigned* input,unsigned* sink,unsigned long long* times,
               unsigned* smids,int iterations) {
    unsigned a[4][4],b[3][4],c[4][4][2][4]={};
    int tid=threadIdx.x, lane=tid%32;
    #pragma unroll
    for(int r=0;r<4;++r)
        #pragma unroll
        for(int j=0;j<4;++j) a[r][j]=input[tid*28+r*4+j];
    #pragma unroll
    for(int r=0;r<3;++r)
        #pragma unroll
        for(int j=0;j<4;++j) b[r][j]=input[tid*28+16+r*4+j];
    unsigned long long start=clock64();
    #pragma unroll 1
    for(int k=0;k<iterations;++k) {
        #pragma unroll
        for(int g=0;g<8;++g) {
            #pragma unroll
            for(int h=0;h<2;++h) {
                #pragma unroll
                for(int r=0;r<4;++r) {
                    unsigned* d=c[r][g%4][h];
                    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                      : "+r"(d[0]),"+r"(d[1]),"+r"(d[2]),"+r"(d[3])
                      : "r"(a[r][0]),"r"(a[r][1]),"r"(a[r][2]),"r"(a[r][3]),
                        "r"(b[g%3][2*h]),"r"(b[g%3][2*h+1]));
                }
            }
        }
    }
    unsigned long long end=clock64();
    // All 128 accumulators observable; nonzero runtime input prevents dead MMA.
    unsigned checksum=0;
    #pragma unroll
    for(int r=0;r<4;++r)
        #pragma unroll
        for(int g=0;g<4;++g)
            #pragma unroll
            for(int h=0;h<2;++h)
                #pragma unroll
                for(int j=0;j<4;++j) checksum+=c[r][g][h][j];
    sink[blockIdx.x*384+tid]=checksum;
    if(lane==0) {
        int i=(blockIdx.x*12+tid/32)*2;
        times[i]=start;times[i+1]=end;
    }
    if(tid==0) {
        unsigned sm;asm("mov.u32 %0, %%smid;" : "=r"(sm));
        smids[blockIdx.x]=sm;
    }
}
extern "C" int roof_run(const unsigned* input,unsigned* sink,unsigned long long* times,
 unsigned* smids,int blocks,int iterations,int* info,float* ms) {
    constexpr int shared=69632;
    static bool initialized=false;
    cudaError_t e;
    if(!initialized) {
        e=cudaFuncSetAttribute(imma_roof,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
        if(e!=cudaSuccess)return e;
        e=cudaFuncSetAttribute(imma_roof,cudaFuncAttributePreferredSharedMemoryCarveout,100);
        if(e!=cudaSuccess)return e; initialized=true;
    }
    if(info) {
        cudaFuncAttributes attr;
        e=cudaFuncGetAttributes(&attr,imma_roof);if(e!=cudaSuccess)return e;
        int active;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,imma_roof,384,shared);
        if(e!=cudaSuccess)return e;
        info[0]=attr.numRegs;info[1]=attr.localSizeBytes;info[2]=shared;info[3]=active;
        return cudaSuccess;
    }
    if(blocks<=0||iterations<=0)return cudaErrorInvalidValue;
    cudaEvent_t begin,end;
    e=cudaEventCreate(&begin);if(e!=cudaSuccess)return e;
    e=cudaEventCreate(&end);if(e!=cudaSuccess){cudaEventDestroy(begin);return e;}
    cudaEventRecord(begin);
    imma_roof<<<blocks,384,shared>>>(input,sink,times,smids,iterations);
    e=cudaGetLastError();
    if(e==cudaSuccess)e=cudaEventRecord(end);
    if(e==cudaSuccess)e=cudaEventSynchronize(end);
    if(e==cudaSuccess)e=cudaEventElapsedTime(ms,begin,end);
    cudaEventDestroy(begin);cudaEventDestroy(end);return e;
}
