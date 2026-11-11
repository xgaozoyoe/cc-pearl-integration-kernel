"""Full CTA handoff correctness candidate, fixed native SASS."""
from pathlib import Path
import json,struct,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/cta-handoff';out.mkdir(exist_ok=True);src=root/'artifacts/fold-preamble';v=json.loads((src/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(src/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((src/'both_compact.sass').read_text());ops=[pc for pc,x in r.items() if x.startswith('IMMA.')];assert len(ops)==64
m={'description':'Correctness candidate: replace paired named-barrier handoff with one full CTA barrier at the same end-of-iteration location. No code motion, no cp.async changes.','variants':{}}
mask=((1<<17)-1)<<105
nop=int.from_bytes(base[off+0x3700:off+0x3710],'little')&~mask
bar0=int.from_bytes(base[off+0x2a50:off+0x2a60],'little')&~mask
assert r[0x2a50]=='BAR.SYNC.DEFER_BLOCKING 0x0'
for name in ('native','cta_sync'):
 data=bytearray(base);changes=[]
 if name!='native':
  for pc,opcode in [(0x16d0,nop),(0x2a00,bar0)]:
   w=int.from_bytes(base[off+pc:off+pc+16],'little');ctl=(w>>105)&0x1ffff
   # No R161/R164 operand-read producers remain. Preserve wait masks/static stalls.
   ctl=(ctl&~((7<<5)|(7<<8)))|(7<<5)|(7<<8)
   data[off+pc:off+pc+16]=(opcode|(ctl<<105)).to_bytes(16,'little')
   changes.append(dict(pc=hex(pc),original=r[pc],replacement='NOP' if pc==0x16d0 else r[0x2a50]))
 assert all(off+0x16d0<=i<off+0x16e0 or off+0x2a00<=i<off+0x2a10 for i,(a,b) in enumerate(zip(base,data)) if a!=b)
 p=out/(name+'.cubin');p.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump)
 expected=dict(r)
 if name!='native':expected[0x16d0]='NOP';expected[0x2a00]=r[0x2a50]
 assert got==expected
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native',tofile=name))+json.dumps(changes,indent=2)+'\n')
 m['variants'][name]={**v,'sha256':hashlib.sha256(data).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
