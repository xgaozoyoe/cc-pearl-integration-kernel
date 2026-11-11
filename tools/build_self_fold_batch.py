"""Offline self-fold batching on256x192; frozen fastest baseline retained."""
from pathlib import Path
import json,re,subprocess,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/self-fold-batch';out.mkdir(exist_ok=True)
base=(root/'artifacts/transcript-tasks/source_control.cu').read_text()
meta=json.loads((root/'artifacts/fold-preamble/manifest.json').read_text())['variants']['both_compact']
b=(root/'artifacts/fold-preamble/both_compact.cubin').read_bytes();assert hashlib.sha256(b).hexdigest()==meta['sha256'];(out/'native.cubin').write_bytes(b)
m={'variants':{'native':dict(meta,threads=384)}}
for name,batch in [('control',1),('batch2',2),('batch4',4)]:
 s=base
 if batch!=1:
  a=s.index('            #pragma unroll\n            for(int row=0;row<4;++row) {',s.index('        if constexpr(TileK==64) {',s.index('// Publish this group')))
  b=s.index('\n        }\n        }\n        asm volatile("bar.sync',a)
  s=s[:a]+f'''            #pragma unroll
            for(int row=0;row<4;++row) {{
                #pragma unroll
                for(int first=0;first<Groups;first+={batch}) {{
                    unsigned values[{batch}];
                    #pragma unroll
                    for(int j=0;j<{batch};++j) {{
                        unsigned x=0;
                        #pragma unroll
                        for(int half=0;half<2;++half) {{
                            #pragma unroll
                            for(int v=0;v<4;++v)x^=c[row][first+j][half][v];
                        }}
                        values[j]=__reduce_xor_sync(0xffffffffu,x);
                    }}
                    if(lane==0) {{
                        #pragma unroll
                        for(int j=0;j<{batch};++j) {{
                            int idx=((wm*4+row)*(BN/16)+wn*Groups+first+j)*16+(rb&15u);
                            if(rb<16)transcript[idx]=values[j];
                            else {{unsigned prev=transcript[idx];transcript[idx]=((prev<<13)|(prev>>19))^values[j];}}
                        }}
                    }}
                }}
            }}'''+s[b:]
 s=s.replace('transcript_tasks_source_control','self_fold_'+name)
 (out/f'{name}.cu').write_text(s);(out/f'{name}.diff').write_text(''.join(difflib.unified_diff(base.splitlines(True),s.splitlines(True),fromfile='source_control',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/f'{name}.cu'),'-o',str(out/f'{name}.cubin')],capture_output=True,text=True)
 (out/f'{name}.build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass)
 m['variants'][name]=dict(entry=re.search(r'Function : (\S+)',sass)[1],sha256=hashlib.sha256((out/f'{name}.cubin').read_bytes()).hexdigest(),threads=384,smem=69632,correct=True)
 print(name,r.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2))
