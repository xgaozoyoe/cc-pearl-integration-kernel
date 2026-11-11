"""Interleave future B copies with MMA, keeping next-stage commit after both A/B."""
from pathlib import Path
import subprocess,argparse
p=argparse.ArgumentParser();p.add_argument('group',type=int,choices=[2,4]);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-copywalk-fulla-candidate-source.cu').read_text();ns='rank128_copyfollow_fulla_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[a:b]
name='copycadence'+str(args.group)
base=s.replace(ns,'rank128_'+name+'_baseline')
new=s.replace(ns,'rank128_'+name+'_fast')
new=new.replace('bool FixedCopy=false,int Layout=0>','bool FixedCopy=false,int Layout=0,bool CopyA=true,bool CopyB=true>')
new=new.replace('    #pragma unroll\n    for(int j=0;j<3;++j)', '    if constexpr(CopyA)\n    #pragma unroll\n    for(int j=0;j<3;++j)')
new=new.replace('    #pragma unroll\n    for(int j=0;j<2;++j)', '    if constexpr(CopyB)\n    #pragma unroll\n    for(int j=0;j<2;++j)')
# Final commit in helper; unused FixedCopy branch remains unchanged.
pos=new.index('    asm volatile("cp.async.commit_group;");',new.index('static_assert(TileK==64 && Layout==6'))
new=new[:pos]+new[pos:].replace('    asm volatile("cp.async.commit_group;");','    if constexpr(CopyB) asm volatile("cp.async.commit_group;");',1)
call='stage_copy<TileK,Planar,UnrollCopy,FixedCopy,Layout>(a,b,sa+next*BM*TileK,sb+next*BN*TileK,cm,cn,kt+Stages-1,m,n,k);'
assert new.count(call)==1
new=new.replace(call,call.replace('FixedCopy,Layout>','FixedCopy,Layout,true,false>'))
point='        for(int g=0;g<Groups*(TileK/32);++g) {'
new=new.replace(point,point+f'''\n            if(g=={args.group} && kt+Stages-1<(k+TileK-1)/TileK)
                '''+call.replace('FixedCopy,Layout>','FixedCopy,Layout,false,true>'))
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32',name)
# The source is an aligned-A specialization; reject unsupported sizes for this offline ABI.
tail=tail.replace('    if(v==0)return', '    if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;\n    if(v==0)return')
exec(tail)
