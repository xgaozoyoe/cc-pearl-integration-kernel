"""Revisit staggered next-slice A loads on the new copywalk-fulla loop."""
from pathlib import Path
import argparse,subprocess
p=argparse.ArgumentParser();p.add_argument('lead',type=int,choices=[1]);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-copycadence4-source.cu').read_text();ns='rank128_copycadence4_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[a:b]
name='astream_overlap'
base=s.replace(ns,'rank128_'+name+'_baseline');new=s.replace(ns,'rank128_'+name+'_fast')
stable=(root/'artifacts/rank128-copywalk-fulla-candidate-source.cu').read_text()
stable_ns='rank128_copyfollow_fulla_fast'
a=stable.index('namespace '+stable_ns);b=stable.index('// namespace '+stable_ns,a)+len('// namespace '+stable_ns)
base=('#include <cuda_runtime.h>\n#include <cstdint>\n'+stable[a:b]).replace(stable_ns,'rank128_'+name+'_baseline')
transform=(root/'tools/build_rank128_operand.py').read_text()
exec(transform[transform.index("a=new.index('        for(int g="):transform.index('source=base+new')])
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32',name)
tail=tail.replace('    if(v==0)return','    if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;\n    if(v==0)return')
tail=tail.replace('rank128_'+name+'_fast::launch<2,','rank128_'+name+'_fast::launch<'+str(args.lead)+',')
exec(tail)
