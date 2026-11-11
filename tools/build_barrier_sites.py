"""Single-site CTA barrier ablations of the frozen binary, offline only."""
from pathlib import Path
import json,struct,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/barrier-sites';out.mkdir(exist_ok=True);src=root/'artifacts/fold-preamble';v=json.loads((src/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(src/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((src/'both_compact.sass').read_text());ops=[pc for pc,x in r.items() if x.startswith('IMMA.')];assert len(ops)==64
m={'warning':'Deliberately racy offline diagnostic. Remove exactly one prologue or epilogue CTA barrier; all other barriers and copy waits retained. Never mine with this binary.','variants':{}}
mask=((1<<17)-1)<<105
nop=int.from_bytes(base[off+0x3700:off+0x3710],'little')&~mask
sites={'native':(), 'no_initial':(0xba0,), 'no_final':(0x2a50,)}
for name in sites:
 data=bytearray(base);changes=[]
 if name!='native':
  for pc in sites[name]:
   assert r[pc].startswith('BAR.')
   w=int.from_bytes(base[off+pc:off+pc+16],'little');ctl=(w>>105)&0x1ffff
   # Preserve wait masks and static stalls; remove only producer annotations
   # of removed BAR instructions. Existing downstream waits may remain satisfied.
   ctl=(ctl&~((7<<5)|(7<<8)))|(7<<5)|(7<<8)
   data[off+pc:off+pc+16]=(nop|(ctl<<105)).to_bytes(16,'little')
   changes.append(dict(pc=hex(pc),original=r[pc],replacement='NOP',waits_and_static_stalls_preserved=True))
 assert all(any(off+pc<=i<off+pc+16 for pc in sites[name]) for i,(a,b) in enumerate(zip(base,data)) if a!=b)
 p=out/(name+'.cubin');p.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump)
 assert got=={pc:('NOP' if pc in sites[name] else x) for pc,x in r.items()}
 assert sum(x.startswith('IMMA.') for x in got.values())==64
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native',tofile=name))+json.dumps(changes,indent=2)+'\n')
 m['variants'][name]={**v,'sha256':hashlib.sha256(data).hexdigest(),'correct':name=='native'}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
