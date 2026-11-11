"""Compile frozen R128 full/pure controls without modifying mining libraries."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/ada-rank-frozen-source.cu').read_text()
s=s[s.index('namespace pearl_rank128'):s.index('void launch_pearl_blake3')]

def remove_block(source,needle):
    start=source.index(needle);opening=source.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[:start]+source[end:]

parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
parts.append(s.replace('pearl_rank128','rank128_full'))
pure=remove_block(s,'            if constexpr(TileK!=64) {\n        if(group==Groups-1')
pure=remove_block(pure,'        if constexpr(TileK==64) {')
start=pure.index('    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;')
end=pure.index('\n}\ntemplate<int Lead',start)
pure=pure[:start]+'''
    // Consume EVERY MMA accumulator. One observable uint32 per lane/CTA.
    unsigned checksum=0;
    #pragma unroll
    for(int row=0;row<4;++row)
    #pragma unroll
    for(int group=0;group<Groups;++group)
    #pragma unroll
    for(int half=0;half<2;++half)
    #pragma unroll
    for(int v=0;v<4;++v) checksum^=c[row][group][half][v];
    out[(blockIdx.x+gridDim.x*blockIdx.y)*Threads+threadIdx.x]=checksum;
'''+pure[end:]
parts.append(pure.replace('pearl_rank128','rank128_pure'))
parts.append('''
extern "C" int ablation(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int* info) {
 if(v==0)return rank128_full::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,0);
 if(v==1)return rank128_pure::launch<2,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,0);
 return cudaErrorInvalidValue;
}
extern "C" int event_time(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int repeats,float* ms) {
 cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
 cudaEventRecord(begin);
 for(int i=0;i<repeats;++i){auto e=ablation(v,a,b,m,n,out,nullptr);if(e)return e;}
 cudaEventRecord(end);auto e=cudaEventSynchronize(end);
 if(e==cudaSuccess)e=cudaEventElapsedTime(ms,begin,end);
 cudaEventDestroy(begin);cudaEventDestroy(end);return e;
}
''')
generated=Path('/tmp/rank128-ablation.cu');generated.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo',
 '-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v',
 '-shared',str(generated),'-o','/tmp/libpearl-rank128-ablation.so'],check=True)
