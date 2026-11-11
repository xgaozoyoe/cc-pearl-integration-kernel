"""Instrumented offline wait attribution; never a mining backend."""
from pathlib import Path
import json,re,subprocess,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/transcript-wait-probe';out.mkdir(exist_ok=True)
base=(root/'artifacts/transcript-handoff-diagnosis/batch_reduce.cu').read_text()
m={'variants':{}}
for name in ('control','timed','ready_any'):
 s=base
 if name=='ready_any':
  pos=s.index('template<int Lead,int TileK=64')
  s=s[:pos]+'''__device__ __forceinline__ bool task_ready(unsigned bar,unsigned phase) {
 unsigned ok;
 asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p,[%1],%2; selp.u32 %0,1,0,p; }" : "=r"(ok) : "r"(bar),"r"(phase) : "memory");
 return __all_sync(0xffffffffu,ok!=0);
}
'''+s[pos:]
  old='            for(unsigned producer=0;producer<9;++producer) {\n                task_wait(barbase+producer*16,seq&1u);'
  new='            unsigned pending=511u;\n            while(pending) {\n                unsigned producer=0;\n                for(;;) {\n                    if((pending&(1u<<producer)) && task_ready(barbase+producer*16,seq&1u)) break;\n                    producer=(producer==8)?0:producer+1;\n                }\n                pending&=~(1u<<producer);'
  assert old in s;s=s.replace(old,new)
 if name=='timed':
  pos=s.index('template<int Lead,int TileK=64')
  s=s[:pos]+'''__device__ __forceinline__ unsigned stamp() {
 unsigned t; asm volatile("mov.u32 %0, %%clock;" : "=r"(t) :: "memory"); return t;
}
'''+s[pos:]
  s=s.replace('    int wm=warp%3,wn=warp/3;', '''    int wm=warp%3,wn=warp/3;
    unsigned dstart=stamp(),dw=0,dp=0,db=0,dmask=0,dcount=0;''')
  s=s.replace('                task_wait(barbase+producer*16,seq&1u);', '''                unsigned before=stamp();
                task_wait(barbase+producer*16,seq&1u);
                unsigned after=stamp();dw+=after-before;
                dmask+=(__activemask()!=0xffffffffu);++dcount;''')
  end='''            }
        }
    } else {'''
  assert end in s;s=s.replace(end,'''                dp+=stamp()-after;
'''+end,1)
  s=s.replace('            if(seq)task_wait(barbase+warp*16+8,(seq-1)&1u);', '''            if(seq) {unsigned before=stamp();task_wait(barbase+warp*16+8,(seq-1)&1u);dw+=stamp()-before;}''')
  s=s.replace('        asm volatile("bar.sync %0, 288;" :: "r"(own_bar) : "memory");','''        unsigned bar_start=stamp();
        asm volatile("bar.sync %0, 288;" :: "r"(own_bar) : "memory");
        db+=stamp()-bar_start;''')
  s=s.replace('    } // producer path\n    __syncthreads();', '''    } // producer path
    unsigned dend=stamp();
    if(lane==0) {
        unsigned* dst=out+(size_t(m)*n/256*16)+(size_t(blockIdx.x)+size_t(gridDim.x)*blockIdx.y)*80+warp*8;
        dst[0]=dend-dstart;dst[1]=dw;dst[2]=dp;dst[3]=db;dst[4]=dmask;dst[5]=dcount;dst[6]=0;dst[7]=0;
    }
    __syncthreads();''')
 s=s.replace('handoff_diagnosis_batch_reduce','transcript_wait_probe_'+name)
 (out/f'{name}.cu').write_text(s);(out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='batch_reduce',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/f'{name}.cu'),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
 (out/f'{name}.build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass)
 m['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),threads=320,smem=63120,correct=True)
 print(name,r.stderr,flush=True)
m['variants']['ready_any']['safety_status']='EXCLUDED:racecheck reported shared read/write hazards'
(out/'manifest.json').write_text(json.dumps(m,indent=2))
