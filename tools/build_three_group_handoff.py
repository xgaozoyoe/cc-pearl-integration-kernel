"""Offline 8+4 versus 4+4+4 split publication; no production writes."""
from pathlib import Path
import subprocess,re,json,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/three-group-handoff';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();a=s.index('namespace fold_defer_source8');b=s.index('} // namespace fold_defer_source8',a);base=s[a:b]
m={'variants':{},'warning':'Offline source protocol experiment. Source controls must be compared with frozen native; compiler scheduling may differ.'}
v=json.loads((root/'artifacts/fold-preamble/manifest.json').read_text())['variants']['both_compact'];blob=(root/'artifacts/fold-preamble/both_compact.cubin').read_bytes();assert hashlib.sha256(blob).hexdigest()==v['sha256'];(out/'native.cubin').write_bytes(blob);m['variants']['native']=v
for name in ('source_control','three_groups'):
 new=base
 if name=='three_groups':
  old='    unsigned own_bar=4u+unsigned(threadIdx.x>=256);'
  assert new.count(old)==1
  new=new.replace(old,'''    const unsigned group=threadIdx.x/128;
    unsigned own_bar=4u+group;
    unsigned peer0=4u+(group==2u?0u:group+1u);
    unsigned peer1=4u+(group==0u?2u:group-1u);
    // IDs4/5/6 alternate with8/9/10. ID0 is prologue/epilogue.
    // Every warp publishes to both other groups after copies and reads.
    // Its own late sync completes the384-participant barrier for its group.''')
  new=new.replace('        const unsigned peer_bar=own_bar^1u;','')
  new=new.replace('        asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");','        asm volatile("bar.arrive %0, 384; bar.arrive %1, 384;" :: "r"(peer0), "r"(peer1) : "memory");')
  new=new.replace('        own_bar^=2u;','        own_bar^=12u; peer0^=12u; peer1^=12u;')
  new=new.replace('// IDs4/5 and6/7 alternate. Same two-group protocol; no division,\n        // parity shift or add is needed to select the barriers each iteration.','// Three receiving groups; two phase-specific ID sets.')
 ns='three_handoff_'+name;new=new.replace('fold_defer_source8',ns)
 new+='\ntemplate __global__ void kernel<1,64,false,true,false,false,2,false,6>(const signed char*,const signed char*,int,int,int,int,unsigned*);\n}\n'
 (out/(name+'.cu')).write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+new)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='source8',tofile=name)))
 r=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/(name+'.cu')),'-o',str(out/(name+'.cubin'))],capture_output=True,text=True)
 (out/(name+'.build.log')).write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);(out/(name+'.sass')).write_text(dump)
 e=re.search(r'Function : (\S+)',dump)[1];m['variants'][name]=dict(entry=e,sha256=hashlib.sha256((out/(name+'.cubin')).read_bytes()).hexdigest(),smem=69632,correct=True)
 print(name,r.stderr,flush=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
# Restore early publication for the three-group candidate; compiler otherwise
# emits both arrivals after all64 MMA. Preserve tail order and both read waits.
import struct
parent='three_groups';name='three_early';v=m['variants'][parent];blob=(out/(parent+'.cubin')).read_bytes();e=v['entry']
sh=struct.unpack_from('<Q',blob,40)[0];sz,n,st=struct.unpack_from('<HHH',blob,58);hs=[struct.unpack_from('<IIQQQQIIQQ',blob,sh+i*sz) for i in range(n)];h=hs[st];ss=blob[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
r=rows((out/(parent+'.sass')).read_text())
assert r[0x16c0]=='DEPBAR.LE SB0, 0x0'
assert r[0x17d0]=='BAR.ARV R159, 0x180' and r[0x17e0]=='BAR.ARV R158, 0x180'
assert all(r[pc].startswith('IMMA.') for pc in range(0x16d0,0x17d0,16))
mask=0
for pc in [0x1660,0x16b0]:
 assert r[pc].startswith('LDSM.')
 w=int.from_bytes(blob[off+pc:off+pc+16],'little');sb=(w>>110)&7;assert sb<6;mask|=1<<sb
order=[0x17d0,0x17e0]+list(range(0x16d0,0x17d0,16));d=bytearray(blob);expected=dict(r)
for j,old in enumerate(order):
 pc=0x16d0+j*16;w=int.from_bytes(blob[off+old:off+old+16],'little')
 if j==0:w|=mask<<116
 d[off+pc:off+pc+16]=w.to_bytes(16,'little');expected[pc]=r[old]
assert d[:off+0x16d0]==blob[:off+0x16d0] and d[off+0x17f0:]==blob[off+0x17f0:]
(out/(name+'.cubin')).write_bytes(d);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True);assert rows(dump)==expected;(out/(name+'.sass')).write_text(dump)
(out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{p:x} {x}\n' for p,x in r.items()],[f'{p:x} {x}\n' for p,x in expected.items()],fromfile=parent,tofile=name)))
m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'publication_wait_mask':mask};(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
