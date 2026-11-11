"""R128 control vs row-paired MMA with staggered next-slice A loads."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
original=(root/"artifacts/ada-rank-frozen-source.cu").read_text()
original="#include <cuda_runtime.h>\n#include <cstdint>\n"+original[original.index("namespace pearl_rank128"):original.index("void launch_pearl_blake3")]
base=original.replace("pearl_rank128","rank128_operand_baseline")
new=original.replace("pearl_rank128","rank128_operand_early_a")
# Only rewrite the Lead2 steady-state group loop. Each A fragment is reused
# for both N halves before loading its next K-slice, spreading four A loads
# across eight MMA instead of concentrating them in the final four MMA.
a=new.index('        for(int g=0;g<Groups*(TileK/32);++g)')
b=new.index('            if constexpr(TileK!=64) {\n        if(group==Groups-1',a)
chunk=new[a:b]
old="""            for(int half=0;half<2;++half) {
                #pragma unroll
                for(int row=0;row<4;++row) {"""
replacement="""            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int half=0;half<2;++half) {"""
assert chunk.count(old)==1
chunk=chunk.replace(old,replacement).replace('half==0 && row==2','half==0 && row==1')
chunk=chunk.replace('for(int row=0;row<4;++row) {\n                #pragma unroll', 'for(int row=0;row<4;++row) {\n                Fragment nextA;\n                #pragma unroll')
load='load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,(slice+1)*32,lane,BM)'
condition='slice+1<TileK/32 && (TileK==64 || kt*TileK+(slice+1)*32<k) && group==Groups-1'
point='                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);'
assert chunk.count(point)==1
chunk=chunk.replace(point, '                    if(half==0 && '+condition+') nextA='+load+';\n'+point)
assert chunk.count('ar[row]='+load+';')==1
chunk=chunk.replace('ar[row]='+load+';', 'ar[row]=nextA;')
new=new[:a]+chunk+new[b:]
source=base+new+"""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int operand_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    if(v==0)return rank128_operand_baseline::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    if(v==1)return rank128_operand_early_a::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int operand_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=operand_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/rank128-operand-generated.cu").write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/rank128-operand-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-rank128-operand.so"],check=True)
