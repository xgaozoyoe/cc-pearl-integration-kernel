"""Compare against the measured copywalk winner, not the retired baseline."""
from pathlib import Path
import subprocess,argparse
p=argparse.ArgumentParser();p.add_argument('candidate',choices=['fulla','stages3']);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-copywalk-candidate-source.cu').read_text();ns='rank128_copywalk_unroll_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[a:b]
name='copyfollow_'+args.candidate
base=s.replace(ns,'rank128_'+name+'_baseline')
new=s.replace(ns,'rank128_'+name+'_fast')
if args.candidate=='fulla':
 new=new.replace('bool valid=grA<m;','bool valid=true;')
 new=new.replace('a+(valid?offA:0u)','a+offA')
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32',name)
if args.candidate=='fulla':
 tail=tail.replace('(m>1048576 || n>1048576)','(m%256 || m>1048576 || n>1048576)')
else:
 tail=tail.replace('rank128_'+name+'_fast::launch<2,64,false,true,false,false,2,false,6>', 'rank128_'+name+'_fast::launch<2,64,false,true,false,false,3,false,6>')
exec(tail)
