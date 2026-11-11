"""Fixed shared vs read-only ping-pong vs real stream, isolated diagnostic."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-layers-current-source.cu').read_text()
def extract(name):
 ns='layers_current_'+name
 return s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['fixed','pingpong','global','full']
for v,name in enumerate(names):
 old=extract('shared_sync' if v<2 else name);new=old
 if v==1:
  new=new.replace('p<Stages-1','p<Stages')
  new=new.replace('signed char* aa=sa;','signed char* aa=sa+(kt%Stages)*BM*TileK;')
  new=new.replace('signed char* bb=sb;','signed char* bb=sb+(kt%Stages)*BN*TileK;')
 (root/f'artifacts/rank128-stage-rotation-{name}.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='fixed-stage control',tofile=name)))
 parts.append(new.replace('layers_current_'+('shared_sync' if v<2 else name),'stage_rotation_'+name))
parts.append('extern "C" int stage_rotation(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int* info) {\nif(m<=0 || n<=0 || m%256 || n%16)return cudaErrorInvalidValue;')
for v,name in enumerate(names):parts.append(f'if(v=={v})return stage_rotation_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,nullptr);')
parts.append('return cudaErrorInvalidValue;}')
p=root/'artifacts/rank128-stage-rotation-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(p),'-o',str(root/'artifacts/libpearl-rank128-stage-rotation.so')],check=True)
