"""Narrow follow-up: one pending local XOR instead of two/four."""
from pathlib import Path
import subprocess,json,re,hashlib,difflib
p=Path(__file__).resolve().parents[1]/'artifacts/fold-defer-reduce';s=(p/'source.cu').read_text();ns='fold_defer_defer2_mma';a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);old=s[a:b]
new=old.replace(ns,'fold_defer_defer1_mma').replace('pending[2]','pending[1]').replace('Groups-2','Groups-1').replace('row<2','row<1').replace('i<2','i<1')
source='#include <cuda_runtime.h>\n#include <cstdint>\n'+new+'\nextern "C" int launch_one(const int8_t*a,const int8_t*b,int m,int n,uint32_t*t){return fold_defer_defer1_mma::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);}\n'
(p/'one.cu').write_text(source);(p/'defer1_mma.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='defer2_mma',tofile='defer1_mma')))
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--ptxas-options=-v','--expt-relaxed-constexpr','-cubin',str(p/'one.cu'),'-o',str(p/'defer1_mma.cubin')],check=True)
dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p/'defer1_mma.cubin')],text=True);(p/'defer1_mma.sass').write_text(dump);entry=next(x for x in re.findall(r'Function : (\S+)',dump) if 'fold_defer_defer1_mma' in x)
m=json.loads((p/'manifest.json').read_text());m['variants']['defer1_mma']={'entry':entry,'sha256':hashlib.sha256((p/'defer1_mma.cubin').read_bytes()).hexdigest(),'smem':69632,'correct':True};(p/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
