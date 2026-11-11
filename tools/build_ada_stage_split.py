"""Isolated v121 derivative: split next-stage A/B copy issue around current MMA."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/"csrc/gemm/pearl-gemm-ada-v2.cu").read_text()
s=s[:s.index("cudaError_t launch_pearl_gemm_ada_v2")]
s=s.replace("pearl_ada_v2","pearl_stage_split")
s=s.replace("bool FixedCopy=false,int Layout=0>\n","bool FixedCopy=false,int Layout=0,int Part=0>\n")
# Keep fixed/unrolled controls untouched; only the ordinary loops are partitioned.
a=s.index("    #pragma unroll (UnrollCopy ? 3 : 1)")
b=s.index("    #pragma unroll (UnrollCopy ? 2 : 1)",a)
end=s.index('    asm volatile("cp.async.commit_group;");',b)
s=s[:a]+"    if constexpr(Part!=2) {\n"+s[a:b]+"    }\n    if constexpr(Part!=1) {\n"+s[b:end]+"    }\n"+s[end:]
s=s.replace("bool Wide=false,int Layout=0>\n","bool Wide=false,int Layout=0,int Split=0>\n")
s=s.replace("Stages,Wide,Layout>","Stages,Wide,Layout,Split>")
old="stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);"
assert s.count(old)==1
s=s.replace(old,old.replace("Layout>","Layout,(Split?1:0)>"))
needle="            int slice=g/Groups,group=g%Groups,slot=g%slots;"
assert s.count(needle)==1
s=s.replace(needle,needle+"""
            if constexpr(Split!=0) {
                if(g==Split && kt+Stages-1<(k+TileK-1)/TileK)
                    stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout,2>(
                        a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);
            }""")
s+="""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int stage_split_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    using namespace pearl_stage_split;
    if(v==0)return launch<2,64,false,true,false,false,2,false,6,0>(a,b,m,n,4096,256,t,info,nullptr);
    if(v==1)return launch<2,64,false,true,false,false,2,false,6,2>(a,b,m,n,4096,256,t,info,nullptr);
    if(v==2)return launch<2,64,false,true,false,false,2,false,6,4>(a,b,m,n,4096,256,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int stage_split_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=stage_split_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
Path("/tmp/pearl-stage-split-generated.cu").write_text(s)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared","/tmp/pearl-stage-split-generated.cu",
 str(root/"csrc/gemm/pearl_blake3_sm61.cu"),"-o","/tmp/libpearl-stage-split.so"],check=True)
