"""R128 all-warps-compute with 7/5 warp partition only during copy."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=(root/"artifacts/ada-rank-frozen-source.cu").read_text()
original="#include <cuda_runtime.h>\n#include <cstdint>\n"+original[original.index("namespace pearl_rank128"):original.index("void launch_pearl_blake3")]
base=original.replace("pearl_rank128","rank128_copy75_baseline")
new=original.replace("pearl_rank128","rank128_copy75_partition")
start=new.index('    if constexpr(FixedCopy) {')
end=new.index('\n}\n\ntemplate<int Lead',start)
new=new[:start]+r"""
    static_assert(TileK==64 && !Planar && Layout==6 && !FixedCopy);
    constexpr int AThreads=7*32, BThreads=5*32;
    const bool isA=threadIdx.x<AThreads;
    const int first=isA?int(threadIdx.x):int(threadIdx.x)-AThreads;
    const int stride=isA?AThreads:BThreads;
    const int chunks=(isA?BM:BN)*TileK/16;
    const int row0=isA?cm*BM:cn*BN;
    const int rows=isA?m:n;
    const signed char* input=isA?a:b;
    signed char* output=isA?sa:sb;
    #pragma unroll 1
    for(int i=first;i<chunks;i+=stride) {
        int logical=i^((i>>3)&3);
        int gr=row0+logical/4;
        int gc=kt*64+(logical%4)*16;
        const signed char* src=input+(size_t)(gr<rows?gr:0)*k+gc;
        unsigned dst=__cvta_generic_to_shared(output+i*16);
        int valid=gr<rows?16:0;
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dst),"l"(src),"r"(valid));
    }
    asm volatile("cp.async.commit_group;");
"""+new[end:]
source=base+new+"""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int copy75_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return rank128_copy75_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1)return rank128_copy75_partition::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int copy75_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=copy75_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/rank128-copy75-generated.cu").write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/rank128-copy75-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-rank128-copy75.so"],check=True)
