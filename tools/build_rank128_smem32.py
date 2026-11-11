"""R128 control vs native 32-bit shared addresses for LDSM."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=(root/"artifacts/ada-rank-frozen-source.cu").read_text()
original="#include <cuda_runtime.h>\n#include <cstdint>\n"+original[original.index("namespace pearl_rank128"):original.index("void launch_pearl_blake3")]
base=original.replace("pearl_rank128","rank128_smem32_baseline")
new=original.replace("pearl_rank128","rank128_smem32_native")
# LDSM only: use shared-space uint32 addresses instead of generic pointers.
a=new.index('template<int TileK,bool Planar,int Layout=0>\n__device__ __forceinline__ Fragment load16x32')
b=new.index('template<bool Packed=false>',a)
chunk=new[a:b].replace('const signed char* base','unsigned base')
chunk=chunk.replace('__cvta_generic_to_shared(base+offset', '(base+offset')
new=new[:a]+chunk+new[b:]
new=new.replace('signed char* aa=sa+current*BM*TileK;', 'unsigned aa=__cvta_generic_to_shared(sa)+current*BM*TileK;')
new=new.replace('signed char* bb=sb+current*BN*TileK;', 'unsigned bb=__cvta_generic_to_shared(sb)+current*BN*TileK;')
source=base+new+"""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int smem32_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return rank128_smem32_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1)return rank128_smem32_native::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int smem32_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=smem32_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/rank128-smem32-generated.cu").write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/rank128-smem32-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-rank128-smem32.so"],check=True)
