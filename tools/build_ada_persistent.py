"""Generate an isolated v121 derivative; leave the source and production intact."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
source=(root/"csrc/gemm/pearl-gemm-ada-v2.cu").read_text()
source=source[:source.index("cudaError_t launch_pearl_gemm_ada_v2")]
source=source.replace("pearl_ada_v2","pearl_persistent")
source=source.replace("bool Wide=false,int Layout=0>\n","bool Wide=false,int Layout=0,int Schedule=0>\n")
old="""    int linear=blockIdx.x+gridDim.x*blockIdx.y;
    int first=linear/(8*gridDim.x)*8;
    int cols=min(8,int(gridDim.y)-first), offset=linear%(8*gridDim.x);
    int cm=offset/cols,cn=first+offset%cols;"""
new="""    constexpr bool Persistent=Schedule==1||Schedule==2;
    if constexpr(Schedule!=0) { m=8192; n=8192; }
    const int tiles_m=(m+BM-1)/BM,tiles_n=(n+BN-1)/BN;
    for(int work=(Persistent?int(blockIdx.x):0);
        work<(Persistent?tiles_m*tiles_n:1);
        work+=(Persistent?int(gridDim.x):1)) {
    int cm,cn;
    if constexpr(Schedule==0 || Schedule==3) {
        int linear=blockIdx.x+gridDim.x*blockIdx.y;
        int first=linear/(8*gridDim.x)*8;
        int cols=min(8,int(gridDim.y)-first),offset=linear%(8*gridDim.x);
        cm=offset/cols;cn=first+offset%cols;
    } else if constexpr(Schedule==1) {
        int first=work/(8*tiles_m)*8;
        int cols=min(8,tiles_n-first),offset=work%(8*tiles_m);
        cm=offset/cols;cn=first+offset%cols;
    } else {
        int mfirst=work/(16*tiles_n)*16;
        int rows=min(16,tiles_m-mfirst),local=work%(16*tiles_n);
        int first=local/(8*rows)*8;
        int cols=min(8,tiles_n-first),offset=local%(8*rows);
        cm=mfirst+offset/cols;cn=first+offset%cols;
    }"""
assert source.count(old)==1
source=source.replace(old,new)
needle="\n}\ntemplate<int Lead"
assert source.count(needle)==1
source=source.replace(needle,"\n    } // end persistent task loop\n}\ntemplate<int Lead")
source=source.replace("    constexpr int Threads=Wide?256:THREADS;", "    constexpr bool Persistent=Schedule==1||Schedule==2;\n    constexpr int Threads=Wide?256:THREADS;")
# Prevent epilogue index constants from being hoisted across every task/MMA loop.
source=source.replace("    for(int i=threadIdx.x;i<WORDS;i+=Threads) {", """    int output_tid=threadIdx.x;
    if constexpr(Schedule==1 || Schedule==2) asm volatile("mov.u32 %0, %%tid.x;" : "=r"(output_tid));
    for(int i=output_tid;i<WORDS;i+=Threads) {""")
source=source.replace("Stages,Wide,Layout>","Stages,Wide,Layout,Schedule>")
source=source.replace("<<<dim3((m+BM-1)/BM,(n+BN-1)/BN),Threads,smem,stream>>>",
"""<<<(Persistent?dim3(min(128,((m+BM-1)/BM)*((n+BN-1)/BN))):dim3((m+BM-1)/BM,(n+BN-1)/BN)),Threads,smem,stream>>>""")
source+="""
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int persistent_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
    using namespace pearl_persistent;
    if(v!=0 && (m!=8192 || n!=8192))return cudaErrorInvalidValue;
    if(v==0)return launch<2,64,false,true,false,false,2,false,6,0>(a,b,m,n,4096,256,t,info,nullptr);
    if(v==1)return launch<2,64,false,true,false,false,2,false,6,1>(a,b,m,n,4096,256,t,info,nullptr);
    if(v==3)return launch<2,64,false,true,false,false,2,false,6,3>(a,b,m,n,4096,256,t,info,nullptr);
    if(v==2)return launch<2,64,false,true,false,false,2,false,6,2>(a,b,m,n,4096,256,t,info,nullptr);
    return cudaErrorInvalidValue;
}
extern "C" int persistent_pow(int v,const int8_t* a,const int8_t* b,int m,int n,
 const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
    auto e=persistent_fold(v,a,b,m,n,t,nullptr);if(e)return e;
    launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);
    return cudaGetLastError();
}
"""
path=Path("/tmp/pearl-persistent-generated.cu");path.write_text(source)
subprocess.run(["/usr/local/cuda-12.6/bin/nvcc","-std=c++20","-O3","-lineinfo",
 "-gencode","arch=compute_89,code=sm_89","-Xcompiler","-fPIC","--ptxas-options=-v",
 "--expt-relaxed-constexpr","-I"+str(root/"csrc/gemm"),"-I"+str(root/"csrc"),
 "-I/home/xgao/cutlass/include","-shared",str(path),str(root/"csrc/gemm/pearl_blake3_sm61.cu"),
 "-o","/tmp/libpearl-persistent.so"],check=True)
