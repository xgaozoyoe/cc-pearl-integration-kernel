"""Isolated late transcript-state preloads, current A-stream, SM89."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['baseline','tail1','tail2']
for v,name in enumerate(names):
 new=s
 if v:
  start=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)');tail=new[start:]
  pre=f'''            unsigned early[{v}]={{}};
            if(lane==0 && rb>=16) {{
                #pragma unroll
                for(int j=0;j<{v};++j) {{
                    int idx=((wm*4+3)*(BN/16)+wn*Groups+Groups-{v}+j)*16+(rb%16);
                    unsigned addr=__cvta_generic_to_shared(transcript+idx);
                    // Real memory load, deliberately issued before the fold body.
                    asm volatile("ld.shared.u32 %0,[%1];" : "=r"(early[j]) : "r"(addr) : "memory");
                }}
            }}
'''
  tail=tail.replace('            int rb=(kt+1)/2-1;','            int rb=(kt+1)/2-1;\n'+pre,1)
  tail=tail.replace('unsigned prev=transcript[idx];',f'unsigned prev=(row==3 && group>=Groups-{v})?early[group-(Groups-{v})]:transcript[idx];',1)
  new=new[:start]+tail
 (root/f'artifacts/rank128-state-prefetch-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='A-stream baseline',tofile=name)))
 parts.append(new.replace(ns,'state_prefetch_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int state_prefetch_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return state_prefetch_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int state_prefetch_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=state_prefetch_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
path=root/'artifacts/rank128-state-prefetch-source.cu';path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-state-prefetch-experimental.so')],check=True)
