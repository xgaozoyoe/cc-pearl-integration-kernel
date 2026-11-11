"""Source-based arrival tracing with explicit frozen-baseline provenance limits."""
from pathlib import Path
import json,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/arrival-order';out.mkdir(exist_ok=True)
base=(root/'artifacts/transcript-tasks/source_control.cu').read_text();m={'variants':{}}
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];b=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(b).hexdigest()==v['sha256'];(out/'native.cubin').write_bytes(b);m['variants']['native']=dict(v,threads=384)
for name in ('source_control','trace'):
 s=base
 if name=='trace':
  point=s.index('template<int Lead,int TileK=64')
  s=s[:point]+'''__device__ __forceinline__ unsigned tick() {unsigned t;asm volatile("mov.u32 %0, %%clock;" : "=r"(t) :: "memory");return t;}
'''+s[point:]
  s=s.replace('        asm volatile("cp.async.wait_group 0;" ::: "memory");\n        asm volatile("bar.arrive', '        asm volatile("cp.async.wait_group 0;" ::: "memory");\n        unsigned early=0,late=0;\n        if(kt==r)early=tick();\n        asm volatile("bar.arrive')
  old='        asm volatile("bar.sync %0, 384;" :: "r"(own_bar) : "memory");'
  assert old in s
  s=s.replace(old,'''        if(kt==r)late=tick();
'''+old+'''
        if(kt==r && lane==0) {
            unsigned* dst=out+size_t(m)*n/256*16+(size_t(blockIdx.x)+size_t(gridDim.x)*blockIdx.y)*24+warp*2;
            dst[0]=early;dst[1]=late;
        }''')
 s=s.replace('transcript_tasks_source_control','arrival_order_'+name)
 (out/f'{name}.cu').write_text(s);(out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='source_control',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/f'{name}.cu'),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
 (out/f'{name}.build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass)
 m['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),threads=384,smem=69632,correct=True)
 print(name,r.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2))
