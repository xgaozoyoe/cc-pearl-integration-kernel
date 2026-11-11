"""Full-correctness copy/wait placement controls; fixed256x192,K64,2stages."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];ns='rank128_astream_candidate_fast'
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
copy='''        if(kt+Stages-1<(k+TileK-1)/TileK)
            stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);
'''
assert s.count(copy)==1
names=['baseline','latecopy','earlywait'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for mode,name in enumerate(names):
 new=s
 if mode==1:
  new=new.replace(copy,'')
  point='    }\n    __syncthreads();\n    for(int i=threadIdx.x;i<WORDS;'
  assert new.count(point)==1
  new=new.replace(point,copy+point)
 if mode==2:
  start=new.index('        if constexpr(Stages==2) asm volatile("cp.async.wait_group 0;");')
  stop=new.index('        __syncthreads();',start)
  new=new[:start]+new[stop:]
  new=new.replace('    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {','    static_assert(Stages==2);\n    asm volatile("cp.async.wait_group 0;");\n    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {')
  point='        if constexpr(TileK==64) {\n        if((kt+1)%2==0) {'
  assert new.count(point)==1
  new=new.replace(point,'        asm volatile("cp.async.wait_group 0;");\n'+point)
 (root/f'artifacts/rank128-handoff-control-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='current-A-stream',tofile=name)))
 parts.append(new.replace(ns,'handoff_control_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_control_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_control_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_control_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_control_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/rank128-handoff-control-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-handoff-control.so')],check=True)
