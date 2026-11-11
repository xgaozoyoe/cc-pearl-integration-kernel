"""Current A-stream full/pure/local/redux controls. Never a mining library."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
def remove_block(s,needle):
 a=s.index(needle);b=s.index('{',a);depth=1;e=b+1
 while depth:
  depth+=(s[e]=='{')-(s[e]=='}');e+=1
 return s[:a]+s[e:]
# Remove dead generic-K fold in diagnostic variants as well.
base=remove_block(s,'            if constexpr(TileK!=64) {\n        if(group==Groups-1')
start=base.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)')
end=base.index('    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;',start)
# This span includes fold, closing K loop, and precedes final output barrier.
finish=base.index('\n}\ntemplate<int Lead',end)
loop_prefix=base[:start]
final_sink='''
    unsigned final_xor=0;
    #pragma unroll
    for(int row=0;row<4;++row)
    #pragma unroll
    for(int group=0;group<Groups;++group)
    #pragma unroll
    for(int half=0;half<2;++half)
    #pragma unroll
    for(int v=0;v<4;++v) final_xor^=c[row][group][half][v];
    out[(blockIdx.x+gridDim.x*blockIdx.y)*Threads+threadIdx.x]=final_xor SINK;
'''
variants=[s]
# Keep a runtime-gated fold body to avoid allocator changes caused by deleting
# the entire boundary. Launch always passes zero in this diagnostic namespace.
pure=base[:end]+'    if(enabled) {\n'+base[end:finish]+'    } else {\n'+final_sink.replace(' SINK','')+'    }'+base[finish:]
pure=pure.replace('int r,unsigned* out) {','int r,unsigned* out,int enabled) {')
pure=pure.replace('if((kt+1)%2==0) {','if(enabled && (kt+1)%2==0) {')
pure=pure.replace('cudaStream_t stream) {','cudaStream_t stream,int enabled=0) {')
pure=pure.replace('(a,b,m,n,k,r,t);','(a,b,m,n,k,r,t,enabled);')
variants.append(pure)
for reduce in (False,True):
 prefix=loop_prefix.replace('    unsigned c[4][Groups][2][4]={};','    unsigned observed=0;\n    unsigned c[4][Groups][2][4]={};')
 fold='''
        if((kt+1)%2==0) {
            #pragma unroll
            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int group=0;group<Groups;++group) {
                    unsigned x=0;
                    #pragma unroll
                    for(int half=0;half<2;++half)
                    #pragma unroll
                    for(int v=0;v<4;++v) x^=c[row][group][half][v];
                    REDUCE
                    observed=((observed<<1)|(observed>>31))^x;
                }
            }
        }
    }
'''.replace('REDUCE','x=__reduce_xor_sync(0xffffffffu,x);' if reduce else '')
 variants.append(prefix+fold+final_sink.replace(' SINK',' ^ observed')+base[finish:])
names=['full','pure','local','redux'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for name,v in zip(names,variants):
 parts.append(v.replace(ns,'astream_ablation_'+name))
 (root/f'artifacts/rank128-astream-ablation-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),v.splitlines(True),fromfile='current-baseline',tofile=name)))
parts.append('extern "C" int astream_ablation(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int* info) {')
for v,name in enumerate(names):parts.append(f'if(v=={v})return astream_ablation_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,0);')
parts.append('if(v==4)return astream_ablation_pure::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,0,1);')
parts.append('return cudaErrorInvalidValue;}')
path=root/'artifacts/rank128-astream-ablation-source.cu';path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(path),'-o',str(root/'artifacts/libpearl-rank128-astream-ablation.so')],check=True)
