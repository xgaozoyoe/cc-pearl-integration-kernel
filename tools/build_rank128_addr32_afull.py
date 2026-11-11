"""R128 single-launch full-A-tile specialization, bounded B unchanged."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=(root/"artifacts/ada-rank-frozen-source.cu").read_text()
original="#include <cuda_runtime.h>\n#include <cstdint>\n"+original[original.index("namespace pearl_rank128"):original.index("void launch_pearl_blake3")]
base=original.replace("pearl_rank128","rank128_addr32_afull_baseline")
new=original.replace("pearl_rank128","rank128_addr32_afull_fast")
# Region offsets fit in uint32; preserve original bounds and tile geometry.
new=new.replace('(size_t)(gr<m?gr:0)*k+gc','(unsigned(gr<m?gr:0)*unsigned(k)+unsigned(gc))')
new=new.replace('(size_t)(gr<n?gr:0)*k+gc','(unsigned(gr<n?gr:0)*unsigned(k)+unsigned(gc))')
new=new.replace('bool valid_row=gr<m && (TileK==64 || gc<k);','bool valid_row=true;')
new=new.replace('unsigned(gr<m?gr:0)','unsigned(gr)').replace('(gr<m?gr:0)','gr')
source=base+new+"""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int addr32_afull_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return rank128_addr32_afull_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1 && (m%256 || m>1048576 || n>1048576))return rank128_addr32_afull_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1)return rank128_addr32_afull_fast::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int addr32_afull_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=addr32_afull_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/rank128-addr32_afull-generated.cu").write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/rank128-addr32_afull-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-rank128-addr32_afull.so"],check=True)
