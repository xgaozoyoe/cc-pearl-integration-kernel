"""Isolated staggered B half-load experiment; never replace production DSO."""
from pathlib import Path
import difflib
import subprocess

root = Path(__file__).resolve().parents[1]
ns = 'rank128_astream_candidate_fast'
s = (root/'artifacts/rank128-astream-candidate-source.cu').read_text()
s = s[s.index('namespace '+ns):s.index('// namespace '+ns)+len('// namespace '+ns)]
parts = ['#include <cuda_runtime.h>\n#include <cstdint>\n']
names = ['baseline', 'late', 'early']
for mode, name in enumerate(names):
    new = s
    if mode:
        helper = r'''template<int TileK,bool Planar,int Layout>
__device__ __forceinline__ void loadBhalf(Fragment& x,int half,const signed char* base,int row,int col,int lane,int height) {
    unsigned addr=__cvta_generic_to_shared(base+offset<TileK,Planar,Layout>(row+half*8+(lane&7),col+((lane>>3)&1)*16,height));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];"
        : "=r"(x.v[half*2]),"=r"(x.v[half*2+1]) : "r"(addr));
}
'''
        at = new.index('template<bool Packed=false>')
        new = new[:at]+helper+new[at:]
        start = new.index('                    if(half==0 && row==1 && g+Lead<')
        stop = new.index('                    if(half==0 && slice+1',start)
        low = 'row==1 && half==0' if mode==1 else 'row==0 && half==1'
        high = 'row==3 && half==0' if mode==1 else 'row==2 && half==1'
        code = ''
        for h,condition in enumerate((low,high)):
            code += f'''                    if(({condition}) && g+Lead<Groups*(TileK/32))
                        loadBhalf<TileK,Planar,Layout>(br[(g+Lead)%slots],{h},bb,wn*WarpN+((g+Lead)%Groups)*16,((g+Lead)/Groups)*32,lane,BN);
'''
        new = new[:start]+code+new[stop:]
    (root/f'artifacts/rank128-bsplit-{name}.diff').write_text(''.join(difflib.unified_diff(
        s.splitlines(True),new.splitlines(True),fromfile='A-stream baseline',tofile=name)))
    parts.append(new.replace(ns, 'bsplit_'+name))
parts.append('''void launch_pearl_blake3(const uint32_t*,int,int,const uint32_t*,const uint32_t*,uint8_t*,int*,int*,cudaStream_t);
extern "C" int bsplit_fold(int v,const int8_t* a,const int8_t* b,int m,int n,uint32_t* t,int* info) {
if(m<=0 || n<=0 || m%256 || n%16 || m>1048576 || n>1048576)return cudaErrorInvalidValue;
''')
for v,name in enumerate(names):
    parts.append(f'if(v=={v})return bsplit_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,info,nullptr);')
parts.append('''return cudaErrorInvalidValue;
}
extern "C" int bsplit_pow(int v,const int8_t* a,const int8_t* b,int m,int n,const uint32_t* key,const uint32_t* target,uint32_t* t,uint8_t* dig,int* found,int* coord) {
auto e=bsplit_fold(v,a,b,m,n,t,nullptr);if(e)return e;
launch_pearl_blake3(t,(m/16)*(n/16),n,key,target,dig,found,coord,nullptr);return cudaGetLastError();
}''')
path = root/'artifacts/rank128-bsplit-source.cu'
path.write_text('\n'.join(parts))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-gencode','arch=compute_89,code=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','--expt-relaxed-constexpr','-I/home/xgao/cutlass/include','-Icsrc','-Icsrc/gemm','-shared',str(path),str(root/'csrc/gemm/pearl_blake3_sm61.cu'),'-o',str(root/'artifacts/libpearl-rank128-bsplit-experimental.so')],check=True)
