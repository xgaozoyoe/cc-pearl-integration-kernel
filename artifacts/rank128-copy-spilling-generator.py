"""Isolated v121 vs128x192 CTA /32x96 warp, targeting two CTAs per SM."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=(root/"artifacts/ada-rank-frozen-source.cu").read_text()
original="#include <cuda_runtime.h>\n#include <cstdint>\n"+original[original.index("namespace pearl_rank128"):original.index("void launch_pearl_blake3")]
base=original.replace("pearl_rank128","rank128_copy_baseline")
new=original.replace("pearl_rank128","rank128_copy_plan")
helper=r"""
struct CopyPlan {
    const signed char *a,*b;
    unsigned offA,offB,dstA,dstB;
    int remA,remB,col;
    __device__ __forceinline__ CopyPlan(const signed char* aa,const signed char* bb,
        signed char* sa,signed char* sb,int cm,int cn,int m,int n):a(aa),b(bb) {
        int i=threadIdx.x,logical=i^((i>>3)&3);
        int row=logical/4;col=(logical%4)*16;
        int grA=cm*256+row,grB=cn*192+row;
        offA=grA*4096+col;offB=grB*4096+col;
        remA=m-grA;remB=n-grB;
        dstA=__cvta_generic_to_shared(sa+i*16);
        dstB=__cvta_generic_to_shared(sb+i*16);
    }
    __device__ __forceinline__ void issue(int stage,int kt) const {
        #pragma unroll
        for(int j=0;j<5;++j) {
            if(j!=2 || threadIdx.x<256) {
                constexpr unsigned stride=96*4096;
                bool isA=j<3;
                int round=isA?j:j-3;
                bool valid=(isA?remA:remB)>round*96;
                unsigned offset=valid?(isA?offA:offB)+round*stride:col;
                const signed char* p=(isA?a:b)+offset+kt*64;
                unsigned d=(isA?dstA:dstB)+round*384*16+stage*(isA?256:192)*64;
                int bytes=valid?16:0;
                asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(d),"l"(p),"r"(bytes));
            }
        }
        asm volatile("cp.async.commit_group;");
    }
};
"""
pos=new.index('template<int Lead')
new=new[:pos]+helper+new[pos:]
needle='    Fragment ar[4], br[Lead+1];'
assert new.count(needle)==1
new=new.replace(needle,needle+'\n    CopyPlan copy(a,b,sa,sb,cm,cn,m,n);')
for old,replacement in [
 ('stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+p*BM*TileK,sb+p*BN*TileK,cm,cn,p,m,n,k);','copy.issue(p,p);'),
 ('stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);','copy.issue(next,kt+Stages-1);')]:
    assert new.count(old)==1
    new=new.replace(old,replacement)
source=base+new+"""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int copy_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return rank128_copy_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1)return rank128_copy_plan::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int copy_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=copy_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/rank128-copy-generated.cu").write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/rank128-copy-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-rank128-copy.so"],check=True)
