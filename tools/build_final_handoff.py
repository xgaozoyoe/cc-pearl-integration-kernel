"""Skip only the final K iteration's paired named handoff; retain final CTA output barrier."""
from pathlib import Path
import struct,json,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/final-handoff';out.mkdir(exist_ok=True);p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text());assert not any(re.search(r'\bP6\b',x) for x in r.values())
assert r[0xcb0]=='PLOP3.LUT P2, PT, PT, PT, UP0, 0x80, 0x0'
assert not any(re.search(r'\bP2\b',x) for pc,x in r.items() if 0xcb0<pc<0xdc0)
assert r[0xdc0]=='@P2 BRA 0x1090'
m={'description':'Correctness candidate:skip only kt63 handoff after final A/B reads; no future shared-stage reuse. Final CTA barrier retains transcript publication. No added instructions.','variants':{}}
for name in ['native','predicate_control','skip_final']:
 d=bytearray(base);expected=dict(r)
 if name!='native':
  # Preserve the existing final-iteration predicate in unused P6 instead of P2.
  w=int.from_bytes(d[off+0xcb0:off+0xcc0],'little');assert (w>>81)&7==2;w^=1<<83;d[off+0xcb0:off+0xcc0]=w.to_bytes(16,'little');expected[0xcb0]=r[0xcb0].replace('P2,','P6,',1)
  w=int.from_bytes(d[off+0xdc0:off+0xdd0],'little');assert (w>>12)&15==2;w=(w&~(15<<12))|(6<<12);d[off+0xdc0:off+0xdd0]=w.to_bytes(16,'little');expected[0xdc0]='@P6 BRA 0x1090'
 if name=='skip_final':
  for pc in [0x16d0,0x2a00]:
   w=int.from_bytes(d[off+pc:off+pc+16],'little');assert (w>>12)&15==7;w=(w&~(15<<12))|(14<<12);d[off+pc:off+pc+16]=w.to_bytes(16,'little');expected[pc]='@!P6 '+r[pc]
 path=out/(name+'.cubin');path.write_bytes(d);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);assert rows(dump)==expected
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{p:x} {x}\n' for p,x in r.items()],[f'{p:x} {x}\n' for p,x in expected.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
