"""Isolated v121 vs128x192 CTA /32x96 warp, targeting two CTAs per SM."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=(root/"csrc/gemm/pearl-gemm-ada-v2.cu").read_text()
original=original[:original.index("cudaError_t launch_pearl_gemm_ada_v2")]
base=original.replace("pearl_ada_v2","pearl_dual_baseline")
new=original.replace("pearl_ada_v2","pearl_dual_cta")
new=new.replace("BM=256, BN=192, THREADS=384","BM=128, BN=192, THREADS=256")
new=new.replace("__launch_bounds__(Wide?256:THREADS,1)","__launch_bounds__(256,2)")
new=new.replace("WarpN=Wide?96:64","WarpN=96")
new=new.replace("unsigned c[4][Groups]","unsigned c[2][Groups]")
new=new.replace("Fragment ar[4]","Fragment ar[2]").replace("Fragment aa_rows[4]","Fragment aa_rows[2]")
new=new.replace("row<4","row<2").replace("wm*64","wm*32").replace("wm*4+row","wm*2+row")
new=new.replace("half==0 && row==2","half==0 && row==1")
source=base+new+"""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int dual_cta_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return pearl_dual_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,256,t,info,nullptr);
    if(v==1)return pearl_dual_cta::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,256,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int dual_cta_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=dual_cta_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/pearl-dual-cta-generated.cu").write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/pearl-dual-cta-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-dual-cta.so"],check=True)
