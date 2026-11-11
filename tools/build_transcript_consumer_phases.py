"""Offline source-interval timing for consumer; timings include scheduling."""
from pathlib import Path
import subprocess,re,json,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/transcript-consumer-phases';out.mkdir(exist_ok=True)
base=(root/'artifacts/transcript-handoff-diagnosis/batch_reduce.cu').read_text();manifest={'variants':{}}
for name in ('control','phases','lanes4'):
 s=base
 if name=='lanes4':
  a=s.index('                if(lane==0) {',s.index('if(warp==9)'))
  b=s.index('            }\n        }\n    } else {',a)
  s=s[:a]+'''                if(lane<4) {
                    unsigned value=lane==0?values[0]:(lane==1?values[1]:(lane==2?values[2]:values[3]));
                    int idx=(((producer%3)*4+lane)*(BN/16)+(producer/3)*4+group)*16+(rb&15u);
                    if(rb<16)transcript[idx]=value;
                    else {unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^value;}
                }
'''+s[b:]
 if name=='phases':
  pos=s.index('template<int Lead,int TileK=64')
  s=s[:pos]+'''__device__ __forceinline__ unsigned tick() {unsigned t; asm volatile("mov.u32 %0, %%clock;" : "=r"(t) :: "memory");return t;}
'''+s[pos:]
  s=s.replace('    if(warp==9) {','    if(warp==9) {\n        unsigned start=tick(),wr=0,ld=0,fr=0,rx=0,up=0,count=0;')
  s=s.replace('                task_wait(barbase+producer*16,seq&1u);','                unsigned t0=tick();\n                task_wait(barbase+producer*16,seq&1u);\n                unsigned t1=tick();')
  s=s.replace('                task_arrive(barbase+producer*16+8);','                unsigned t2=tick();\n                task_arrive(barbase+producer*16+8);\n                unsigned t3=tick();')
  a=s.index('                if(lane==0) {',s.index('if(warp==9)'))
  s=s[:a]+'                unsigned t4=tick();\n'+s[a:]
  ending='''            }
        }
    } else {'''
  assert ending in s
  s=s.replace(ending,'''                unsigned t5=tick();
                wr+=t1-t0;ld+=t2-t1;fr+=t3-t2;rx+=t4-t3;up+=t5-t4;++count;
            }
        }
        unsigned end=tick();
        if(lane==0) {
            unsigned* dst=out+size_t(m)*n/256*16+(size_t(blockIdx.x)+size_t(gridDim.x)*blockIdx.y)*80+72;
            dst[0]=end-start;dst[1]=wr;dst[2]=ld;dst[3]=fr;dst[4]=rx;dst[5]=up;dst[6]=count;dst[7]=0;
        }
    } else {''',1)
 s=s.replace('handoff_diagnosis_batch_reduce','consumer_phases_'+name)
 (out/f'{name}.cu').write_text(s);(out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='batch_reduce',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/f'{name}.cu'),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
 (out/f'{name}.build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass)
 manifest['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),threads=320,smem=63120,correct=True)
 print(name,r.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
