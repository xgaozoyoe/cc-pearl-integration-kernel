"""Offline region reuse experiment. Production libraries are never touched."""
from pathlib import Path
import subprocess,json,re,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/region-free';out.mkdir(exist_ok=True)
parent=root/'artifacts/ready-free'
meta=json.loads((parent/'manifest.json').read_text());manifest={'variants':{}}
for name in ['native','source_control']:
 v=meta['variants'][name];b=(parent/(name+'.cubin')).read_bytes();assert hashlib.sha256(b).hexdigest()==v['sha256'];(out/(name+'.cubin')).write_bytes(b);manifest['variants'][name]=v
base=(parent/'split_deferred.cu').read_text()
for name,defer in [('region_early_wait',False),('region_deferred',True)]:
 s=base
 start=s.index('__device__ __forceinline__ void rf_wait');end=s.index('template<int Lead,int TileK=64',start)
 helpers=s[start:end];s=s[:start]+s[end:]
 point=s.index('template<int TileK,bool Planar,bool UnrollCopy=')
 s=s[:point]+helpers+s[point:]
 s=s.replace('int cm,int cn,int kt,int m,int n,int k) {','int cm,int cn,int kt,int m,int n,int k,unsigned bars=0) {')
 s=s.replace('if(threadIdx.x<4)', 'if(threadIdx.x<16)')
 s=s.replace('asm volatile("mbarrier.init.shared.b64 [%0],384;" :: "r"(addr) : "memory");','unsigned count=threadIdx.x<2?384:((threadIdx.x-2)%7<4?96:128);\n        asm volatile("mbarrier.init.shared.b64 [%0],%1;" :: "r"(addr),"r"(count) : "memory");')
 s=s.replace('            if(kt>0) rf_wait(bars+16+8*next,unsigned((kt-1)/2)&1u);','')
 s=s.replace('cm,cn,kt+1,m,n,k);','cm,cn,kt+1,m,n,k,bars);')
 s=s.replace('rf_free(bars+16+8*current);','rf_free(bars+16+56*current+8*wm);\n        rf_free(bars+16+56*current+32+8*wn);')
 # At copy of Kt, destination held K(t-2). Wait for its readers, not their tails.
 helper='__device__ __forceinline__ void region_wait(unsigned bars,int kt,int row,bool b) {\n if(kt>=2) rf_wait(bars+16+56*(kt&1)+8*((b?4:0)+row/64),unsigned((kt-2)/2)&1u);\n}\n'
 point=s.index('template<int TileK,bool Planar,bool UnrollCopy=');s=s[:point]+helper+s[point:]
 needle='    int logical=int(threadIdx.x)^(int(threadIdx.x>>3)&3);'
 if not defer:
  pre='    if(kt>=2) {\n        #pragma unroll\n        for(int j=0;j<3;++j) if(threadIdx.x+j*THREADS<BM*64/16)\n            region_wait(bars,kt,(threadIdx.x+j*THREADS)/4,false);\n        #pragma unroll\n        for(int j=0;j<2;++j) region_wait(bars,kt,(threadIdx.x+j*THREADS)/4,true);\n    }\n'
  s=s.replace(needle,pre+needle)
 else:
  s=s.replace('        bool valid=true;', '        region_wait(bars,kt,(threadIdx.x+j*THREADS)/4,false);\n        bool valid=true;')
  s=s.replace('        bool valid=grB<n;', '        region_wait(bars,kt,(threadIdx.x+j*THREADS)/4,true);\n        bool valid=grB<n;')
 s=s.replace('ready_free_split_deferred','region_free_'+name)
 path=out/(name+'.cu');path.write_text(s)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='whole_stage_split_deferred',tofile=name)))
 p=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(path),'-o',str(out/(name+'.cubin'))],capture_output=True,text=True)
 (out/(name+'.build.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);(out/(name+'.sass')).write_text(sass)
 manifest['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest(),smem=69760,correct=True)
 print(name,p.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
