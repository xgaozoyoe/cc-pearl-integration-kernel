"""Offline layered diagnostic, generated from frozen accepted kernel; no mining ABI."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/ada-rank-frozen-source.cu').read_text()
s=s[s.index('namespace pearl_rank128'):s.index('void launch_pearl_blake3')]
def remove(source,needle):
 start=source.index(needle);op=source.index('{',start);end=op+1;depth=1
 while depth:
  depth+=(source[end]=='{')-(source[end]=='}');end+=1
 return source[:start]+source[end:]
s=remove(s,'            if constexpr(TileK!=64) {\n        if(group==Groups-1')
s=remove(s,'        if constexpr(TileK==64) {')
a=s.index('    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;')
b=s.index('\n}\ntemplate<int Lead',a)
s=s[:a]+'''
    unsigned checksum=0;
    #pragma unroll
    for(int row=0;row<4;++row)
    #pragma unroll
    for(int g=0;g<Groups;++g)
    #pragma unroll
    for(int h=0;h<2;++h)
    #pragma unroll
    for(int v=0;v<4;++v) checksum+=c[row][g][h][v];
    out[linear*Threads+threadIdx.x]=checksum;
'''+s[b:]
shared=s.replace("p<Stages-1;++p", "p<Stages;++p")
start=shared.index('    for(int kt=0;')
end=shared.index('        if constexpr(Lead==0',start)
shared=shared[:start]+'''
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    // Read-only shared tile: no overwrite and thus no loop handoff required.
    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {
        signed char* aa=sa+(kt%Stages)*BM*TileK;
        signed char* bb=sb+(kt%Stages)*BN*TileK;
'''+shared[end:]
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n',s.replace('pearl_rank128','layers_full'),shared.replace('pearl_rank128','layers_shared')]
parts.append('''
extern "C" int layers(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int* info) {
 if(v==0)return layers_full::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,0);
 if(v==1)return layers_shared::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,0);
 return cudaErrorInvalidValue;
}
extern "C" int layers_time(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int repeats,float* ms) {
 cudaEvent_t begin,end;cudaError_t e=cudaEventCreate(&begin);if(e)return e;
 e=cudaEventCreate(&end);if(e){cudaEventDestroy(begin);return e;}
 e=cudaEventRecord(begin);
 for(int i=0;!e && i<repeats;++i)e=(cudaError_t)layers(v,a,b,m,n,out,nullptr);
 if(!e)e=cudaEventRecord(end);if(!e)e=cudaEventSynchronize(end);
 if(!e)e=cudaEventElapsedTime(ms,begin,end);
 cudaEventDestroy(begin);cudaEventDestroy(end);return e;
}
''')
path=root/'artifacts/rank128-layers-source.cu';path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(path),str(root/'tests/ada_imma_roof.cu'),'-o','/tmp/libpearl-rank128-layers.so'],check=True)
