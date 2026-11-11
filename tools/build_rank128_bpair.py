"""Isolated B-group interleaving experiment; never replace production DSO."""
from pathlib import Path
import difflib
import subprocess

root = Path(__file__).resolve().parents[1]
ns = 'rank128_astream_candidate_fast'
s = (root/'artifacts/rank128-astream-candidate-source.cu').read_text()
s = s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
parts = ['#include <cuda_runtime.h>\n#include <cstdint>\n']
names = ['baseline', 'row', 'half']
for mode, name in enumerate(names):
    new = s
    if mode:
        start = new.index('        #pragma unroll\n        for(int row=0;row<4;++row) ar[row]=')
        stop = new.index('        if constexpr(TileK==64) {\n        if((kt+1)%2==0)', start)
        body = '''        static_assert(TileK==64 && Groups==4 && Lead==1);
        #pragma unroll
        for(int row=0;row<4;++row)
            ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,0,lane,BM);
        #pragma unroll
        for(int slice=0;slice<2;++slice) {
            #pragma unroll
            for(int pair=0;pair<2;++pair) {
                br[0]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+pair*32,slice*32,lane,BN);
                br[1]=loadB16x32<TileK,Planar,PackedB,Layout>(bb,wn*WarpN+pair*32+16,slice*32,lane,BN);
                #pragma unroll
                for(int row=0;row<4;++row) {
                    LOOPS
                    if(slice==0 && pair==1)
                        ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,32,lane,BM);
                }
            }
        }
        }
'''
        loops = '''#pragma unroll
                    for(int side=0;side<2;++side) {
                        #pragma unroll
                        for(int half=0;half<2;++half)
                            mma<PackedB>(c[row][pair*2+side][half],ar[row],br[side],half);
                    }'''
        if mode == 2:
            loops = '''#pragma unroll
                    for(int half=0;half<2;++half) {
                        #pragma unroll
                        for(int side=0;side<2;++side)
                            mma<PackedB>(c[row][pair*2+side][half],ar[row],br[side],half);
                    }'''
        new = new[:start]+body.replace('LOOPS', loops)+new[stop:]
    (root/f'artifacts/rank128-bpair-{name}.diff').write_text(''.join(difflib.unified_diff(
        s.splitlines(True),new.splitlines(True),fromfile='A-stream baseline',tofile=name)))
    parts.append(new.replace(ns, 'bpair_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int bpair_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):
    parts.append(f'if(v=={v})return bpair_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int bpair_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=bpair_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
path = root/'artifacts/rank128-bpair-source.cu'
path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-bpair-experimental.so')],check=True)
