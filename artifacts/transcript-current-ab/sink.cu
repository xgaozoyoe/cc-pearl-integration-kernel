#include <cuda_runtime.h>
extern "C" __global__ void sink(const signed char*a,const signed char*b,int m,int n,int k,int r,unsigned*out) {
 unsigned v=clock();
 out[(blockIdx.y*gridDim.x+blockIdx.x)*384u+threadIdx.x]=v;
}
