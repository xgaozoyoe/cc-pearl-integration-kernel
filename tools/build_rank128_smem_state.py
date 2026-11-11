"""Isolated register vs explicitly shared-memory accumulator experiment."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text()
s=s[s.index('namespace rank128_astream_candidate_baseline'):s.index('template<int TileK,bool Planar,bool UnrollCopy')]
s=s.replace('rank128_astream_candidate_baseline','smem_state_trial').replace('BM=256, BN=192, THREADS=384','BM=128, BN=64, THREADS=128')
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s
s+=r'''
constexpr int TK=128, AB=2*(BM+BN)*TK;
__device__ __forceinline__ void copy_stage(const int8_t* a,const int8_t* b,int8_t* sa,int8_t* sb,int cm,int cn,int kt,int m,int n) {
 for(int i=threadIdx.x;i<(BM+BN)*TK/16;i+=THREADS) {
  int row=i/(TK/16),col=i%(TK/16)*16;
  bool isa=row<BM;int rr=isa?row:row-BM;
  int gr=isa?cm*BM+rr:cn*BN+rr;
  bool valid=gr<(isa?m:n);
  const int8_t* src=(isa?a:b)+(size_t)(valid?gr:0)*4096+kt*TK+col;
  unsigned dst=__cvta_generic_to_shared((isa?sa:sb)+offset<TK,false,6>(rr,col,isa?BM:BN));
  asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(src),"r"(valid?16:0));
 }
 asm volatile("cp.async.commit_group;");
}
template<bool Shared>
__global__ void __launch_bounds__(THREADS) kernel(const int8_t* a,const int8_t* b,int m,int n,unsigned* out) {
 extern __shared__ int8_t mem[];
 int8_t* sa=mem;int8_t* sb=mem+2*BM*TK;
 unsigned* transcript=reinterpret_cast<unsigned*>(mem+AB);
 volatile unsigned* state=reinterpret_cast<unsigned*>(mem+AB+WORDS*4);
 int warp=threadIdx.x/32,lane=threadIdx.x%32,cm=blockIdx.x,cn=blockIdx.y;
 unsigned acc[Shared?1:2][Shared?1:4][2][4]={};
 copy_stage(a,b,sa,sb,cm,cn,0,m,n);
 for(int kt=0;kt<4096/TK;++kt) {
  asm volatile("cp.async.wait_group 0;");__syncthreads();
  int cur=kt%2,nxt=1-cur;
  if(kt+1<4096/TK)copy_stage(a,b,sa+nxt*BM*TK,sb+nxt*BN*TK,cm,cn,kt+1,m,n);
  #pragma unroll
  for(int row=0;row<2;++row) {
   #pragma unroll
   for(int group=0;group<4;++group) {
    unsigned (*c)[4]=acc[Shared?0:row][Shared?0:group];
    if constexpr(Shared) {
     #pragma unroll
     for(int h=0;h<2;++h) {
      #pragma unroll
      for(int v=0;v<4;++v)c[h][v]=kt==0?0:state[(((warp*2+row)*4+group)*8+h*4+v)*32+lane];
     }
    }
    #pragma unroll
    for(int slice=0;slice<TK/32;++slice) {
     Fragment av=load16x32<TK,false,6>(sa+cur*BM*TK,warp*32+row*16,slice*32,lane,BM);
     Fragment bv=loadB16x32<TK,false,true,6>(sb+cur*BN*TK,group*16,slice*32,lane,BN);
     mma<true>(c[0],av,bv,0);mma<true>(c[1],av,bv,1);
    }
    unsigned x=0;
    #pragma unroll
    for(int h=0;h<2;++h) {
     #pragma unroll
     for(int v=0;v<4;++v) {
      x^=c[h][v];
      if constexpr(Shared) if(kt+1<4096/TK)state[(((warp*2+row)*4+group)*8+h*4+v)*32+lane]=c[h][v];
     }
    }
    x=__reduce_xor_sync(0xffffffffu,x);
    if(lane==0) {
     int idx=((warp*2+row)*4+group)*16+kt%16;
     if(kt<16)transcript[idx]=x;
     else {unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^x;}
    }
   }
  }
 }
 __syncthreads();
 for(int i=threadIdx.x;i<WORDS;i+=THREADS) {
  int tile=i/16,gr=cm*(BM/16)+tile/(BN/16),gc=cn*(BN/16)+tile%(BN/16);
  if(gr<m/16&&gc<n/16)out[((size_t)gr*(n/16)+gc)*16+i%16]=transcript[i];
 }
}
template<bool Shared>
int launch(const int8_t* a,const int8_t* b,int m,int n,unsigned* t,int* info) {
 if(m<=0||n<=0||m%16||n%16)return cudaErrorInvalidValue;
 constexpr int bytes=AB+WORDS*4+(Shared?BM*BN*4:0);
 cudaError_t e=cudaFuncSetAttribute(kernel<Shared>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);if(e)return e;
 if(info) {
  cudaFuncAttributes attr;e=cudaFuncGetAttributes(&attr,kernel<Shared>);if(e)return e;
  int active;e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel<Shared>,THREADS,bytes);if(e)return e;
  info[0]=attr.numRegs;info[1]=THREADS;info[2]=bytes+attr.sharedSizeBytes;info[3]=active;info[4]=attr.localSizeBytes;return 0;
 }
 kernel<Shared><<<dim3((m+BM-1)/BM,(n+BN-1)/BN),THREADS,bytes>>>(a,b,m,n,t);
 return cudaGetLastError();
}
}
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int smem_state_fold(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* t,int* info) {
 if(v==0)return smem_state_trial::launch<false>(a,b,m,n,t,info);
 if(v==1)return smem_state_trial::launch<true>(a,b,m,n,t,info);
 return cudaErrorInvalidValue;
}
extern "C" int smem_state_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const unsigned* key,const unsigned* target,unsigned* t,uint8_t* dig,int* found,int* coord) {
 auto e=smem_state_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-smem-state-source.cu';path.write_text(s)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-smem-state-experimental.so')],check=True)
