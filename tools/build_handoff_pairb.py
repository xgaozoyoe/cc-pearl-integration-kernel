"""Offline real-B fragment lookahead on the accepted lean handoff checkpoint."""
from pathlib import Path
import subprocess,difflib
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/handoff-lean-source.cu').read_text();ns='handoff_lean_toggle'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);base=s[a:b]
parts=['#include <cuda_runtime.h>\n#include <cstdint>\n'];names=['control','paired']
for name in names:
 new=base
 if name=='paired':
  start=new.index('        if constexpr(Lead==0 || Lead==3) {')
  end=new.index("        // Publish this group's",start)
  # Replace only operand consumption, preserving the K64 fold and handoff below.
  old=new[start:end]
  assert 'if constexpr(TileK!=64)' in old
  new=new[:start]+"""        static_assert(TileK==64 && Groups==4);
        #pragma unroll
        for(int slice=0;slice<2;++slice) {
            Fragment ap[4];
            #pragma unroll
            for(int row=0;row<4;++row)
                ap[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,slice*32,lane,BM);
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                Fragment bp[2];
                #pragma unroll
                for(int j=0;j<2;++j)
                    bp[j]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+(2*pair+j)*16,slice*32,lane,BN);
                #pragma unroll
                for(int row=0;row<4;++row) {
                    #pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int j=0;j<2;++j)
                            mma<PackedB>(c[row][2*pair+j][half],ap[row],bp[j],half);
                    }
                }
            }
        }
"""+new[end:]
 (root/f'artifacts/handoff-pairb-{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='lean-toggle',tofile=name)))
 parts.append(new.replace(ns,'handoff_pairb_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int handoff_pairb_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):
 lead=1
 parts.append(f'if(v=={v})return handoff_pairb_{name}::launch<{lead},64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int handoff_pairb_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=handoff_pairb_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
p=root/'artifacts/handoff-pairb-source.cu';p.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(p),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-handoff-pairb.so')],check=True)
