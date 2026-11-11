"""Restore matched early publication in compiled copy-plan candidates.
Both publish after completion of last LDSM + cp.async, with seven IMMA left.
Use dead R165 for peer ID. No new instruction or register allocation.
"""
from pathlib import Path
import json,struct,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-prepare'
m=json.loads((out/'manifest.json').read_text());s=(out/'compiled.sass').read_text()
for parent,start,idpc,before in [('late',0x15a0,0x1800,0x17f0),('early_tail',0x16f0,0x17c0,0x1770)]:
 name=parent+'_pub';entry=m['variants'][parent]['entry'];base=(out/(parent+'.cubin')).read_bytes();assert hashlib.sha256(base).hexdigest()==m['variants'][parent]['sha256']
 sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];names=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if names[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
 sec=s.split('Function : '+entry)[1].split('Function : ')[0];r={int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)}
 arv=0x1870;assert r[arv]=='BAR.ARV R96, 0x180'
 assert not any(re.search(r'\bR165\b',x) for pc,x in r.items() if pc>=start)
 for pc,x in r.items():
  if start<=pc<=arv and 'IMMA.' in x:
   regs=list(map(int,re.findall(r'\bR(\d+)',x)));assert not any(v<=165<v+w for v,w in zip(regs,[4,4,2,4]))
 assert all(not any(t in r[pc] for t in ('BRA','BSSY','BSYNC','CALL')) for pc in range(start,arv,16))
 ids=[pc for pc in range(start,arv+16,16) if pc not in (idpc,arv)];ids.insert(0,idpc);ids.insert(ids.index(before),arv)
 expected=dict(r);data=bytearray(base)
 for i,old in enumerate(ids):
  pc=start+16*i;w=bytearray(base[off+old:off+old+16]);x=r[old]
  if old==idpc:assert w[2]==96;w[2]=165;x=x.replace('R96,','R165,',1)
  if old==arv:assert w[4]==96;w[4]=165;x=x.replace('R96,','R165,',1)
  data[off+pc:off+pc+16]=w;expected[pc]=x
 p=out/(name+'.cubin');p.write_bytes(data);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump)
 sec=dump.split('Function : '+entry)[1].split('Function : ')[0];got={int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)};assert got==expected
 assert [x for x in got.values() if 'IMMA.' in x]==[x for x in r.values() if 'IMMA.' in x]
 apc=next(pc for pc,x in got.items() if x.startswith('BAR.ARV'))
 tail=sum('IMMA.' in x for pc,x in got.items() if apc<pc<=arv);assert tail==7,tail
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{p:04x} {x}\n' for p,x in r.items()],[f'{p:04x} {x}\n' for p,x in got.items()],fromfile=parent,tofile=name)))
 m['variants'][name]={**m['variants'][parent],'sha256':hashlib.sha256(data).hexdigest(),'mma_after_arrival':tail}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
