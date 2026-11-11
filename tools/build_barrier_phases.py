"""Offline phase-selective barrier ablation; no code motion or production writes."""
from pathlib import Path
import json,struct,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/barrier-phases';out.mkdir(exist_ok=True)
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text());assert not any(re.search(r'\bP6\b',x) for x in r.values())
assert r[0x10a0]=='ISETP.NE.AND P2, PT, RZ, UR6, PT'
assert r[0x17e0]=='@!P2 BRA 0x2a00'
manifest={'warning':'Racy offline diagnostic, never mine. Remove both arrival and sync for a selected K parity; remaining barrier phases retain all384participants.','variants':{}}
for name in ['native','predicate_control','no_plain_handoff','no_fold_handoff']:
 data=bytearray(base);expected=dict(r)
 if name!='native':
  # ISETP dual result: retain P2=fold; replace discarded complement PT with unused P6=!fold.
  data[off+0x10a0+84//8]^=1<<(84%8);expected[0x10a0]='ISETP.NE.AND P2, P6, RZ, UR6, PT'
 if name.startswith('no_'):
  # Remove plain rounds: execute barriers only when !P6 (fold).
  # Remove fold rounds: execute only when P6 (plain). Same predicate at both sites.
  pred=14 if name=='no_plain_handoff' else 6
  for pc in [0x16d0,0x2a00]:
   w=int.from_bytes(data[off+pc:off+pc+16],'little');assert (w>>12)&15==7
   data[off+pc:off+pc+16]=((w&~(15<<12))|(pred<<12)).to_bytes(16,'little')
   expected[pc]=('@!P6 ' if pred==14 else '@P6 ')+r[pc]
 assert all(any(off+pc<=i<off+pc+16 for pc in [0x10a0,0x16d0,0x2a00]) for i,(a,b) in enumerate(zip(base,data)) if a!=b)
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump);assert got==expected
 assert [x for x in got.values() if 'IMMA.' in x]==[x for x in r.values() if 'IMMA.' in x]
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native',tofile=name)))
 manifest['variants'][name]={**v,'sha256':hashlib.sha256(data).hexdigest(),'correct':name in ['native','predicate_control']}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Built4 variants; only ISETP complement and barrier predication changed. All instruction positions and control fields preserved.')
