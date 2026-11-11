"""Offline real-B fragment lookahead on the accepted lean handoff checkpoint."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/handoff-lean-source.cu').read_text();ns='handoff_lean_toggle'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);base=s[a:b]
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['control','early','two_ahead','compact']
for name in names:
 new=base
 if name=='early':
  old='if(half==0 && row==1 && g+Lead<';assert new.count(old)==1;new=new.replace(old,'if(half==0 && row==0 && g+Lead<')
 if name=='compact':
  new=new.replace('Fragment ar[4], br[Lead+1];','Fragment ar[4], br[Lead];').replace('constexpr int slots=Lead+1;','constexpr int slots=Lead;')
  a=new.index('                    if(half==0 && row==1 && g+Lead<');b=new.index('\n',new.index('br[(g+Lead)%slots]=',a))+1
  load=new[a:b];new=new[:a]+new[b:]
  mma='                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);\n';assert new.count(mma)==1
  new=new.replace(mma,mma+load.replace('half==0 && row==1','half==1 && row==3'))
 (root/f'artifacts/handoff-realb-{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='lean-toggle',tofile=name)))
 parts.append(new.replace(ns,'handoff_realb_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_realb_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):
 lead=1 if v<2 else 2
 parts.append(f'if(v=={v})return handoff_realb_{name}::launch<{lead},64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_realb_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_realb_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-realb-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-realb.so')],check=True)
