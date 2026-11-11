"""Current A-stream full-transcript small tiles; production is never replaced."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1];ns='rank128_astream_candidate_fast'
s=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
names=['baseline','tile128','tile64'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for mode,name in enumerate(names):
 new=s
 if mode:
  bm=128 if mode==1 else 64;warpN=64 if mode==1 else 32;warpRows=4 if mode==1 else 2
  new=new.replace('BM=256, BN=192, THREADS=384',f'BM={bm}, BN=128, THREADS=256')
  new=new.replace('__launch_bounds__(Wide?256:THREADS,1)',f'__launch_bounds__(256,{2 if mode==1 else 3})')
  new=new.replace('WarpN=Wide?96:64',f'WarpN={warpN}')
  new=new.replace('unsigned c[4][Groups]','unsigned c[2][Groups]').replace('Fragment ar[4]','Fragment ar[2]').replace('Fragment aa_rows[4]','Fragment aa_rows[2]')
  new=new.replace('row<4','row<2').replace('wm*64','wm*32').replace('wm*4+row','wm*2+row')
  new=new.replace('wm=warp%4,wn=warp/4',f'wm=warp%{warpRows},wn=warp/{warpRows}')
  # FixedCopy is false for every ABI launch. Optimized copywalk uses generic
  # THREADS strides: A bounds eliminate excess rounds; B128 uses 2x256 copies.
 (root/f'artifacts/rank128-small-current-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='A-stream baseline',tofile=name)))
 parts.append(new.replace(ns,'small_current_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int small_current_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):parts.append(f'if(v=={v})return small_current_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int small_current_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=small_current_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/rank128-small-current-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-small-current.so')],check=True)
