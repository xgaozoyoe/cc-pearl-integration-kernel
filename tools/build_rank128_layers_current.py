"""Current A-stream diagnostic layers, isolated from all production libraries."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-astream-ablation-source.cu').read_text();ns='astream_ablation_pure'
s=s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
full=(root/'artifacts/rank128-astream-candidate-source.cu').read_text();fns='rank128_astream_candidate_fast'
full=full[full.index('namespace '+fns):full.index('// namespace '+fns)+len('// namespace '+fns)]
names=['registers','shared','shared_sync','global','full'];parts=['#include <cuda_runtime.h>\n#include <cstdint>\n']
for mode,name in enumerate(names):
 new=s
 if mode<3:
  start=new.index('    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {')
  stop=new.index('        if constexpr(Lead==0 || Lead==3)',start)
  setup='''    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
'''
  if mode==0:
   setup+='''    #pragma unroll
    for(int row=0;row<4;++row) ar[row]=load16x32<TileK,Planar,Layout>(sa,wm*64+row*16,0,lane,BM);
    #pragma unroll
    for(int group=0;group<2;++group) br[group]=loadB16x32<TileK,Planar,PackedB,Layout>(sb,wn*WarpN+group*16,0,lane,BN);
'''
  setup+='''    #pragma unroll 1
    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {
'''
  if mode==2:setup+='        __syncthreads();\n'
  setup+='        signed char* aa=sa;\n        signed char* bb=sb;\n'
  new=new[:start]+setup+new[stop:]
  if mode==0:
   start=new.index('        if constexpr(Lead==0 || Lead==3)')
   stop=new.index('        if constexpr(TileK==64) {\n        if(enabled',start)
   loop='''        #pragma unroll
        for(int g=0;g<8;++g) {
            #pragma unroll
            for(int row=0;row<4;++row) {
                #pragma unroll
                for(int half=0;half<2;++half)
                    mma<PackedB>(c[row][g%4][half],ar[row],br[g%2],half);
            }
        }
'''
   new=new[:start]+loop+new[stop:]
 if mode==4:new=full.replace(fns,ns)
 else:new=new.replace('final_xor^=c[','final_xor+=c[')
 (root/f'artifacts/rank128-layers-current-{name}.diff').write_text(''.join(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='same-kernel transcript-off',tofile=name)))
 parts.append(new.replace(ns,'layers_current_'+name))
parts.append('extern "C" int layers_current(int v,const int8_t* a,const int8_t* b,int m,int n,unsigned* out,int* info) {\nif(m<=0 || n<=0 || m%256 || n%16)return cudaErrorInvalidValue;')
for v,name in enumerate(names):parts.append(f'if(v=={v})return layers_current_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,out,info,nullptr);')
parts.append('return cudaErrorInvalidValue;}')
p=root/'artifacts/rank128-layers-current-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(p),'-o',str(root/'artifacts/libpearl-rank128-layers-current.so')],check=True)
