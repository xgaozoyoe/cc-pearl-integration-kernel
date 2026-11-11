"""Drain old B operands earlier without new registers/instructions or barrier changes."""
from pathlib import Path
import struct,json,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/publish-ldsm';out.mkdir(exist_ok=True);p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(a,16):x for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text());start=0x1570;end=0x16e0;original=list(range(start,end,16))
orders={'native':original,
 'drain_b0':[0x1570,0x1600,0x1590,0x1630,0x15a0,0x1640,0x15b0,0x1650,0x1580,0x15c0,0x1660,0x15d0,0x15e0,0x15f0,0x1670,0x1610,0x1680,0x1620,0x1690,0x16a0,0x16b0,0x16c0,0x16d0],
 'drain_b1':[0x1570,0x1580,0x1590,0x15a0,0x15b0,0x15c0,0x15d0,0x15e0,0x1670,0x1680,0x1690,0x16a0,0x15f0,0x1600,0x1610,0x1620,0x16b0,0x1630,0x1640,0x1650,0x1660,0x16c0,0x16d0]}
manifest={'variants':{},'description':'Same instruction multiset; reorder23instructions before early publication. Both LDSM completion waits retained; fixed native baseline.'}
for name,order in orders.items():
 assert sorted(order)==original
 # Both old B halves must finish before their quartet is overwritten by LDSM.
 assert all(order.index(a)<order.index(0x1660) for a in [0x1600,0x1630,0x1640,0x1650])
 assert all(order.index(a)<order.index(0x16b0) for a in [0x1580,0x15c0,0x15d0,0x15e0,0x1670,0x1680,0x1690,0x16a0])
 assert order.index(0x15b0)<order.index(0x1660) and order.index(0x1610)<order.index(0x16b0)
 assert order.index(0x1610)<order.index(0x1620)<order.index(0x16d0)
 assert order[-2:]==[0x16c0,0x16d0]
 d=bytearray(base);expected=dict(r)
 for j,old in enumerate(order):
  pc=start+j*16;d[off+pc:off+pc+16]=base[off+old:off+old+16];expected[pc]=r[old]
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump);assert got==expected
 assert sorted(x for x in got.values() if 'IMMA.' in x)==sorted(x for x in r.values() if 'IMMA.' in x)
 assert d[:off+start]==base[:off+start] and d[off+end:]==base[off+end:]
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native',tofile=name)))
 manifest['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
