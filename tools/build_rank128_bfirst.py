"""Remove provably redundant first B copy bounds for N%192==0 or >=96."""
from pathlib import Path
import subprocess
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-copywalk-fulla-candidate-source.cu').read_text();ns='rank128_copyfollow_fulla_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[a:b]
base=s.replace(ns,'rank128_bfirst_baseline')
new=s.replace(ns,'rank128_bfirst_fast').replace('bool valid=grB<n;', 'bool valid=(j==0) || grB<n;')
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32','bfirst')
tail=tail.replace('    if(v==0)return','    if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;\n    if(v==0)return')
tail=tail.replace('(m>1048576 || n>1048576)','(n%192>0 && n%192<96)')
exec(tail)
