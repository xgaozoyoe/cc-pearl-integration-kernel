"""Controlled nanosleep backoff in the validated SM89 split barrier wait."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
source=(root/'artifacts/transcript-handoff-source.cu').read_text()
def section(ns):
 a=source.index('namespace '+ns);b=source.index('// namespace '+ns,a)+len('// namespace '+ns)
 return source[a:b]
base=section('transcript_handoff_baseline');overlap=section('transcript_handoff_overlap')
names=['baseline','spin','sleep32','sleep128'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for name in names:
 old=base if name=='baseline' else overlap
 new=old
 if name.startswith('sleep'):
  point='    } while(!done);'
  assert new.count(point)==1
  new=new.replace(point,'        if(!done) __nanosleep('+name[5:]+');\n'+point)
 ns='transcript_handoff_baseline' if name=='baseline' else 'transcript_handoff_overlap'
 parts.append(new.replace(ns,'handoff_backoff_'+name))
 (root/f'artifacts/handoff-backoff-{name}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='validated-'+ns,tofile=name)))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_backoff_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return handoff_backoff_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_backoff_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_backoff_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-backoff-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-backoff.so')],check=True)
