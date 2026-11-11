"""Controlled padded shared-stage addressing; isolated SM89 experiment."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/handoff-lean-source.cu').read_text()
def section(ns):
 a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);return s[a:b]
base=section('handoff_lean_baseline');control=section('handoff_lean_toggle')
names=['baseline','control','padded','rolling'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for name in names:
 old=base if name=='baseline' else control;new=old
 if name in ('padded','rolling'):
  new=new.replace('constexpr int WORDS=', 'constexpr int SharedBN=256;\nconstexpr int WORDS=')
  new=new.replace('Stages*(BM+BN)*TileK','Stages*(BM+SharedBN)*TileK')
  for var in ['p','next','current']:
   new=new.replace('sb+'+var+'*BN*TileK','sb+'+var+'*SharedBN*TileK')
 if name=='rolling':
  point='    unsigned own_bar=4u+unsigned(threadIdx.x>=192);'
  assert new.count(point)==1;new=new.replace(point,'    unsigned stage_byte=0;\n'+point)
  new=new.replace('int current=kt%Stages,next=(kt+Stages-1)%Stages;', 'unsigned next_byte=stage_byte^16384u;')
  new=new.replace('sa+next*BM*TileK','sa+next_byte').replace('sb+next*SharedBN*TileK','sb+next_byte')
  new=new.replace('sa+current*BM*TileK','sa+stage_byte').replace('sb+current*SharedBN*TileK','sb+stage_byte')
  new=new.replace('        own_bar^=2u;', '        own_bar^=2u;\n        stage_byte^=16384u;')
 ns='handoff_lean_baseline' if name=='baseline' else 'handoff_lean_toggle'
 (root/f'artifacts/handoff-aligned-{name}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='validated-'+ns,tofile=name)))
 parts.append(new.replace(ns,'handoff_aligned_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_aligned_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_aligned_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_aligned_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_aligned_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-aligned-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-aligned.so')],check=True)
