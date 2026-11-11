"""Delay boundary fold into the following K64 iteration, before each C update."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();ns='rank128_astream_candidate_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);s=s[a:b]
base=s.replace(ns,'rank128_fold_handoff_baseline');new=s.replace(ns,'rank128_fold_handoff_fast')
# Preserve the final K4096 boundary: there is no following iteration to host it.
needle='if((kt+1)%2==0) {';assert new.count(needle)==1
new=new.replace(needle,'if(kt+1==k/TileK) {')
# Fold previous boundary after issuing the NEXT future global->shared copy,
# before loading current A/B registers. This crosses the K-loop boundary.
start=base.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)')
opening=base.index('{',start);depth=1;end=opening+1
while depth:
 depth+=(base[end]=='{')-(base[end]=='}');end+=1
fold=base[start:end].replace('if((kt+1)%2==0)', 'if(kt>0 && kt%2==0)').replace('int rb=(kt+1)/2-1;', 'int rb=kt/2-1;')
point='        signed char* aa=sa+current*BM*TileK;'
assert new.count(point)==1;new=new.replace(point,fold+'\n'+point)
source='#include <cuda_runtime.h>\n#include <cstdint>\n'+base+'\n'+new+r'''
void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int fold_handoff_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
 if(m%256||m>1048576||n>1048576)return cudaErrorInvalidValue;
 if(v==0)return rank128_fold_handoff_baseline::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 if(v==1)return rank128_fold_handoff_fast::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);
 return cudaErrorInvalidValue;
}
extern "C" int fold_handoff_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
 auto e=fold_handoff_fold(v,a,b,m,n,t,nullptr);if(e)return e;
 launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}
'''
path=root/'artifacts/rank128-fold-handoff-source.cu';path.write_text(source)
a=base.replace('rank128_fold_handoff_baseline','rank128_trial');b=new.replace('rank128_fold_handoff_fast','rank128_trial')
(root/'artifacts/rank128-fold-handoff-kernel.diff').write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='current-astream.cu',tofile='cross-iteration-fold.cu')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I'+str(root/'csrc/gemm'),'-I'+str(root/'csrc'),'-I/home/xgao/cutlass/include','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-fold-handoff-experimental.so')],check=True)
