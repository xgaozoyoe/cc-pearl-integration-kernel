"""Prefetch only next-slice A row0 one B group before its final old-slice use."""
from pathlib import Path
import argparse,subprocess
p=argparse.ArgumentParser();p.add_argument('lead',type=int,choices=[1,2]);args=p.parse_args()
root=Path(__file__).resolve().parents[1]
s=(root/'artifacts/rank128-copywalk-fulla-candidate-source.cu').read_text();ns='rank128_copyfollow_fulla_fast'
a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s[a:b]
name='aheadrow'+str(args.lead)
base=s.replace(ns,'rank128_'+name+'_baseline');new=s.replace(ns,'rank128_'+name+'_fast')
new=new.replace('Fragment ar[4], br[Lead+1];','Fragment ar[4], br[Lead+1], nextA0;')
point='                    mma<PackedB>(c[row][group][half],ar[row],br[slot],half);'
assert new.count(point)==1
new=new.replace(point,'''                    if(g==Groups-2 && half==1 && row==3)
                        nextA0=load16x32<TileK,Planar,Layout>(aa,wm*64,32,lane,BM);
'''+point)
load='ar[row]=load16x32<TileK,Planar,Layout>(aa,wm*64+row*16,(slice+1)*32,lane,BM);'
assert new.count(load)==1
new=new.replace(load,'{ if(row==0) ar[row]=nextA0; else '+load+' }')
template=(root/'tools/build_rank128_addr32.py').read_text()
tail=template[template.index('source=base+new'):].replace('addr32',name)
tail=tail.replace('    if(v==0)return','    if(m%256 || m>1048576 || n>1048576)return cudaErrorInvalidValue;\n    if(v==0)return')
tail=tail.replace('rank128_'+name+'_fast::launch<2,','rank128_'+name+'_fast::launch<'+str(args.lead)+',')
exec(tail)
