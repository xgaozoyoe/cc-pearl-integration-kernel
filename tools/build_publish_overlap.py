"""Offline SASS scheduling controls: delay peer-ID work and overlap B3 wait with B2 MMA."""
from pathlib import Path
import struct,json,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/publish-overlap';out.mkdir(exist_ok=True)
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(a,16):x for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text());start=0x1620;end=0x1700;original=list(range(start,end,16))
id_late=[x for x in original if x!=0x1620];id_late.insert(id_late.index(0x16c0),0x1620)
overlap=[x for x in id_late if x not in [0x16e0,0x16f0]];k=overlap.index(0x16d0);overlap[k:k]=[0x16e0,0x16f0]
orders={'native':original,'id_late':id_late,'overlap_b2':overlap}
m={'description':'Offline only. Original instructions and resources; overlap_b2 adds SB1 wait to first B2 IMMA, retains both LDSM completion waits on BAR.ARV.','variants':{}}
for name,order in orders.items():
 assert sorted(order)==original
 assert order.index(0x1650)<order.index(0x1660) and order.index(0x16a0)<order.index(0x16b0)
 assert order.index(0x1620)<order.index(0x16c0)<order.index(0x16d0)
 d=bytearray(base);expected=dict(r)
 for j,old in enumerate(order):
  pc=start+j*16;word=int.from_bytes(base[off+old:off+old+16],'little')
  if name=='overlap_b2' and old==0x16e0:
   assert (word>>116)&63==0
   word|=2<<116 # B2 LDSM write scoreboard1; do not wait for B3 scoreboard2 yet.
  d[off+pc:off+pc+16]=word.to_bytes(16,'little');expected[pc]=r[old]
 assert d[:off+start]==base[:off+start] and d[off+end:]==base[off+end:]
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);assert rows(dump)==expected
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in expected.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True,'order':[hex(x) for x in order]}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
