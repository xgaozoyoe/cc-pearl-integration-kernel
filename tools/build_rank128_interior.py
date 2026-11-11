"""R128 complete-tile fast path plus original boundary kernel."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=(root/"artifacts/ada-rank-frozen-source.cu").read_text()
original="#include <cuda_runtime.h>\n#include <cstdint>\n"+original[original.index("namespace pearl_rank128"):original.index("void launch_pearl_blake3")]
base=original.replace("pearl_rank128","rank128_interior_baseline")
new=original.replace("pearl_rank128","rank128_interior_full")
new=new.replace('bool valid_row=gr<m && (TileK==64 || gc<k);','bool valid_row=true;')
new=new.replace('bool valid_row=gr<n && (TileK==64 || gc<k);','bool valid_row=true;')
new=new.replace('(gr<m?gr:0)','gr').replace('(gr<n?gr:0)','gr')
new=new.replace('dim3((m+BM-1)/BM,(n+BN-1)/BN)','dim3(m/BM,n/BN)')
tail=original.replace("pearl_rank128","rank128_interior_tail")
tail=tail.replace('int cm=offset/cols,cn=first+offset%cols;', 'int cm=offset/cols,cn=first+offset%cols+n/BN;')
tail=tail.replace('dim3((m+BM-1)/BM,(n+BN-1)/BN)','dim3(m/BM,1)')
source=base+new+tail+"""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int interior_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return rank128_interior_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1) {
        // Fast path is valid only for complete M tiles and at least one full N tile.
        if(m%256 || n<192) return rank128_interior_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
        auto e=rank128_interior_full::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
        if(e!=cudaSuccess || info || n%192==0)return e;
        return rank128_interior_tail::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);
    }
    if(v==2 && info)return rank128_interior_tail::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int interior_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=interior_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/rank128-interior-generated.cu").write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/rank128-interior-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-rank128-interior.so"],check=True)
