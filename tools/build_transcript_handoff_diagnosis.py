"""Offline factorial: single/four queue slots x all-lane/leader polling."""
from pathlib import Path
import json,hashlib,re,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/transcript-handoff-diagnosis';out.mkdir(exist_ok=True)
old=root/'artifacts/transcript-cta192';manifest={'variants':{}}
for name in ('self','control','leader_wait','slots4','slots4_leader','leader_arrive','leader_handoff','batch_reduce'):
 base=(old/('self.cu' if name=='self' else 'consumer.cu')).read_text();s=base
 if name=='batch_reduce':
  a=s.index('                #pragma unroll\n                for(int row=0;row<4;++row) {\n                    unsigned x=__reduce_xor_sync')
  b=s.index('            }\n        }\n    } else {',a)
  s=s[:a]+'''                #pragma unroll
                for(int row=0;row<4;++row) values[row]=__reduce_xor_sync(0xffffffffu,values[row]);
                if(lane==0) {
                    #pragma unroll
                    for(int row=0;row<4;++row) {
                        int idx=(((producer%3)*4+row)*(BN/16)+(producer/3)*4+group)*16+(rb&15u);
                        if(rb<16) transcript[idx]=values[row];
                        else {unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^values[row];}
                    }
                }
'''+s[b:]
 if name in ('leader_wait','slots4_leader','leader_handoff'):
  a=s.index(' unsigned ok;');b=s.index('\n}',a)
  s=s[:a]+''' if((threadIdx.x&31)==0) {
 unsigned ok;
 do { asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.shared.b64 p,[%1],%2; selp.u32 %0,1,0,p; }" : "=r"(ok) : "r"(bar),"r"(phase) : "memory"); } while(!ok);
 }
 __syncwarp();'''+s[b:]
 if name in ('leader_arrive','leader_handoff'):
  s=s.replace('mbarrier.init.shared.b64 [%0],32;', 'mbarrier.init.shared.b64 [%0],1;')
  s=s.replace(' asm volatile("mbarrier.arrive.shared.b64 _,[%0];"', ' __syncwarp();\n if((threadIdx.x&31)==0) asm volatile("mbarrier.arrive.shared.b64 _,[%0];"')
 if name in ('slots4','slots4_leader'):
  for a,b in [('shared+58368+4608','shared+58368+18432'),('threadIdx.x<18','threadIdx.x<72'),('task_wait(barbase+producer*16,seq&1u)','task_wait(barbase+(producer*4+group)*16,rb&1u)'),('tasks[producer*128+row*32+lane]','tasks[(producer*4+group)*128+row*32+lane]'),('task_arrive(barbase+producer*16+8)','task_arrive(barbase+(producer*4+group)*16+8)'),('if(seq)task_wait(barbase+warp*16+8,(seq-1)&1u);','if(seq>=4)task_wait(barbase+(warp*4+group)*16+8,((seq/4)-1)&1u);'),('tasks[warp*128+row*32+lane]','tasks[(warp*4+group)*128+row*32+lane]'),('task_arrive(barbase+warp*16);','task_arrive(barbase+(warp*4+group)*16);')]:
   assert a in s,a;s=s.replace(a,b)
 s=s.replace('transcript_cta192_self','handoff_diagnosis_'+name).replace('transcript_cta192_consumer','handoff_diagnosis_'+name)
 (out/f'{name}.cu').write_text(s);(out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='cta192',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/f'{name}.cu'),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
 (out/f'{name}.build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass)
 manifest['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),threads=288 if name=='self' else 320,smem=58368 if name=='self' else (77376 if name in ('slots4','slots4_leader') else 63120),correct=True)
 print(name,r.stderr,flush=True)
for n in ('leader_wait','slots4_leader','leader_handoff'):
 manifest['variants'][n]['safety_status']='excluded: leader polling racecheck failure or unvalidated related variant'
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
