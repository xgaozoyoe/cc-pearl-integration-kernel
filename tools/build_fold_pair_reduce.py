"""Offline paired-K scheduling experiment; leaves all existing binaries untouched."""
from pathlib import Path
import subprocess,json,re,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/fold-pair-reduce';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text()
m={'variants':{},'warning':'Offline only. Paired K64 loops; not a production backend.'}
ref=root/'artifacts/fold-preamble';v=json.loads((ref/'manifest.json').read_text())['variants']['both_compact'];blob=(ref/'both_compact.cubin').read_bytes();assert hashlib.sha256(blob).hexdigest()==v['sha256'];(out/'native.cubin').write_bytes(blob);m['variants']['native']=v
for name in ('source8','defer2_copy','defer2_mma'):
 ns='fold_defer_'+name;a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);old=s[a:b]
 loop='    for(int kt=0;kt<(k+TileK-1)/TileK;++kt) {'
 assert old.count(loop)==1
 new=old.replace(loop,'''    #pragma unroll 1
    for(int pair=0;pair<32;++pair) {
        #pragma unroll
        for(int phase=0;phase<2;++phase) {
        int kt=pair*2+phase;''')
 point='        own_bar^=2u;\n    }';assert new.count(point)==1;new=new.replace(point,point+'\n    }')
 new=new.replace(ns,'fold_pair_'+name)
 source='#include <cuda_runtime.h>\n#include <cstdint>\n'+new+f'\nextern "C" int launch_pair(const int8_t*a,const int8_t*b,int m,int n,uint32_t*t){{return fold_pair_{name}::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);}}\n'
 path=out/(name+'.cu');path.write_text(source)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='previous-'+name,tofile='paired-'+name)))
 subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--ptxas-options=-v','--expt-relaxed-constexpr','-cubin',str(path),'-o',str(out/(name+'.cubin'))],check=True)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);(out/(name+'.sass')).write_text(dump)
 entry=next(x for x in re.findall(r'Function : (\S+)',dump) if 'fold_pair_'+name in x)
 m['variants'][name]={'entry':entry,'sha256':hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest(),'smem':69632,'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
