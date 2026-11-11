#include <cuda_runtime.h>
#include <cstdint>
namespace smem_state_trial {
constexpr int BM=128, BN=64, THREADS=128;
constexpr int WORDS=(BM/16)*(BN/16)*16;
struct Fragment { unsigned v[4]; };

template<int Layout=0>
__device__ __forceinline__ unsigned swizzle(unsigned p) {
    static_assert(Layout>=0 && Layout<=6);
    // Layout5 keeps this original swizzle, but permutes producer lanes.
    if constexpr(Layout==1) return p; // Control: no swizzle.
    if constexpr(Layout==2) return p ^ ((p >> 3) & 0x70u);
    if constexpr(Layout==3) return p ^ ((p >> 2) & 0x30u);
    if constexpr(Layout==4) return p ^ ((p >> 2) & 0x70u);
    return p ^ ((p >> 3) & 0x30u);
}
template<int TileK,bool Planar,int Layout=0>
__device__ __forceinline__ unsigned offset(int row,int col,int height) {
    unsigned p=Planar ? (col/32)*height*32+row*32+col%32 : row*TileK+col;
    return swizzle<Layout>(p);
}
template<int TileK,bool Planar,int Layout=0>
__device__ __forceinline__ Fragment load16x32(const signed char* base,int row,int col,int lane,int height) {
    // Four 8x8 matrices of 16-bit pairs. Registers are low-row/low-K,
    // high-row/low-K, low-row/high-K, high-row/high-K respectively.
    unsigned addr=__cvta_generic_to_shared(base+offset<TileK,Planar,Layout>(row+(lane&15),col+(lane>>4)*16,height));
    Fragment x;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
        : "=r"(x.v[0]),"=r"(x.v[1]),"=r"(x.v[2]),"=r"(x.v[3]) : "r"(addr));
    return x;
}
template<int TileK,bool Planar,bool Packed,int Layout=0>
__device__ __forceinline__ Fragment loadB16x32(const signed char* base,int row,int col,int lane,int height) {
    if constexpr(!Packed) return load16x32<TileK,Planar,Layout>(base,row,col,lane,height);
    else {
        // Matrix order: Nlow/Klow, Nlow/Khigh, Nhigh/Klow, Nhigh/Khigh.
        // Each adjacent output pair is now a complete MMA B operand.
        unsigned addr=__cvta_generic_to_shared(base+offset<TileK,Planar,Layout>(
            row+(lane&7)+(lane>>4)*8,col+((lane>>3)&1)*16,height));
        Fragment x;
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
            : "=r"(x.v[0]),"=r"(x.v[1]),"=r"(x.v[2]),"=r"(x.v[3]) : "r"(addr));
        return x;
    }
}
template<bool Packed=false>
__device__ __forceinline__ void mma(unsigned* c,const Fragment& a,const Fragment& b,int half) {
    // An 8-row B half uses registers (0,2) or (1,3).
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
        : "r"(a.v[0]),"r"(a.v[1]),"r"(a.v[2]),"r"(a.v[3]),
          "r"(b.v[Packed?2*half:half]),"r"(b.v[Packed?2*half+1:half+2]));
}

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
