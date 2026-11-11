"""Compare against the measured copywalk winner, not the retired baseline."""
from pathlib import Path
import subprocess,argparse
p=argparse.ArgumentParser();p.add_argument('candidate',choices=['unroll2']);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-copywalk-fulla-candidate-source.cu').read_text();ns='rank128_copyfollow_fulla_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[a:b]
name='copyfollow_'+args.candidate
base=s.replace(ns,'rank128_'+name+'_baseline')
new=s.replace(ns,'rank128_'+name+'_fast')
new=new.replace('    for(int kt=0;', '    #pragma unroll 2\n    for(int kt=0;')
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32',name)
tail=tail.replace('    if(v==0)return','    if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;\n    if(v==0)return')
exec(tail)
