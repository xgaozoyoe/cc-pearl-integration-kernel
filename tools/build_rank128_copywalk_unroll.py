"""Copy-loop induction: exploit 384-thread stride preserving Layout6 swizzle."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/ada-rank-frozen-source.cu').read_text()
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[s.index('namespace pearl_rank128'):s.index('void launch_pearl_blake3')]
base=s.replace('pearl_rank128','rank128_copywalk_unroll_baseline')
new=s.replace('pearl_rank128','rank128_copywalk_unroll_fast')
start=new.index('    #pragma unroll (UnrollCopy ? 3 : 1)')
end=new.index('    asm volatile("cp.async.commit_group;");',start)
new=new[:start]+'''
    static_assert(TileK==64 && Layout==6 && !Planar);
    // 384 is a multiple of128: the swizzled low7 chunk bits never change.
    int logical=int(threadIdx.x)^(int(threadIdx.x>>3)&3);
    int row=logical/4,col=(logical%4)*16;
    unsigned dstA=__cvta_generic_to_shared(sa+threadIdx.x*16);
    unsigned dstB=__cvta_generic_to_shared(sb+threadIdx.x*16);
    int grA=cm*BM+row,grB=cn*BN+row;
    unsigned offA=unsigned(grA)*unsigned(k)+unsigned(kt*64+col);
    unsigned offB=unsigned(grB)*unsigned(k)+unsigned(kt*64+col);
    #pragma unroll
    for(int j=0;j<3;++j) if(threadIdx.x+j*THREADS<BM*64/16) {
        bool valid=grA<m;
        const signed char* src=a+(valid?offA:0u);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dstA),"l"(src),"r"(valid?16:0));
        grA+=THREADS/4;offA+=unsigned(THREADS/4)*unsigned(k);dstA+=THREADS*16;
    }
    #pragma unroll
    for(int j=0;j<2;++j) {
        bool valid=grB<n;
        const signed char* src=b+(valid?offB:0u);
        asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;" :: "r"(dstB),"l"(src),"r"(valid?16:0));
        grB+=THREADS/4;offB+=unsigned(THREADS/4)*unsigned(k);dstB+=THREADS*16;
    }
'''+new[end:]
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32','copywalk_unroll')
exec(tail)
