"""Invariant shared destination base + PTX immediate offsets for A cp.async."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-copywalk-fulla-candidate-source.cu').read_text();ns='rank128_copyfollow_fulla_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[a:b]
base=s.replace(ns,'rank128_copyimm_baseline')
new=s.replace(ns,'rank128_copyimm_fast')
helper='''
template<int Offset>
__device__ __forceinline__ void copy_a_immediate(unsigned dst,const signed char* src) {
 asm volatile("cp.async.cg.shared.global [%0+%2],[%1],16;" :: "r"(dst),"l"(src),"n"(Offset));
}
'''
pos=new.index('template<int TileK,bool Planar,bool UnrollCopy')
new=new[:pos]+helper+new[pos:]
a=new.index('    #pragma unroll\n    for(int j=0;j<3;++j) if(threadIdx.x+j*THREADS')
b=new.index('    #pragma unroll\n    for(int j=0;j<2;++j)',a)
new=new[:a]+'''
    if(threadIdx.x<BM*64/16) {
        copy_a_immediate<0>(dstA,a+offA);
        offA+=unsigned(THREADS/4)*unsigned(k);
    }
    if(threadIdx.x+THREADS<BM*64/16) {
        copy_a_immediate<THREADS*16>(dstA,a+offA);
        offA+=unsigned(THREADS/4)*unsigned(k);
    }
    if(threadIdx.x+2*THREADS<BM*64/16)
        copy_a_immediate<2*THREADS*16>(dstA,a+offA);
'''+new[b:]
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32','copyimm')
tail=tail.replace('    if(v==0)return','    if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;\n    if(v==0)return')
exec(tail)
