"""Partial lane-owned transcript state, current R128 A-stream, isolated ABI."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['baseline','tail1','tail2']
for count,name in enumerate(names):
 new=s
 if count:
  new=new.replace('    unsigned c[4][Groups][2][4]={};','    unsigned saved_state=0;\n    unsigned c[4][Groups][2][4]={};')
  start=new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)');tail=new[start:]
  tail=tail.replace('                    if(lane==0) {',f'''                    if(row==3 && group>=Groups-{count}) {{
                        // Two tiles share one register across disjoint half-warps.
                        int owner=(group-(Groups-{count}))*16+(rb%16);
                        // Every owning lane's first update starts from zero.
                        // Pure predicated selection avoids per-owner branches.
                        unsigned updated=((saved_state<<13)|(saved_state>>19))^x;
                        asm("{{{{ .reg .pred p; setp.eq.u32 p,%3,%4; selp.b32 %0,%1,%2,p; }}}}"
                            : "=r"(saved_state) : "r"(updated),"r"(saved_state),"r"(lane),"r"(owner));
                    }} else if(lane==0) {{''',1)
  tail=tail.replace('    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;',f'''    // Publish once before the existing coalesced output barrier.
    if(lane<{count*16}) {{
        int group=Groups-{count}+lane/16;
        int idx=((wm*4+3)*(BN/16)+wn*Groups+group)*16+lane%16;
        transcript[idx]=saved_state;
    }}
    __syncthreads();
    for(int i=threadIdx.x;i<WORDS;''',1)
  new=new[:start]+tail
 (root/f'artifacts/rank128-state-register-select-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='A-stream baseline',tofile=name)))
 parts.append(new.replace(ns,'state_register_select_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int state_register_select_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return state_register_select_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int state_register_select_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=state_register_select_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
path=root/'artifacts/rank128-state-register-select-source.cu';path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-state-register-select-experimental.so')],check=True)
