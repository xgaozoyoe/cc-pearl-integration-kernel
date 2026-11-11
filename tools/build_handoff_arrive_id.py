"""Offline fixed-SASS free-register ID preparation and earlier arrival.
R164 is dead after original0x1610 until unconditional definition at0xcd0 next K.
"""
from pathlib import Path
import re,struct,json,hashlib,subprocess,difflib,collections
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-arrive-id';out.mkdir(exist_ok=True)
src=root/'artifacts/handoff-arrive-tail';meta=json.loads((src/'manifest.json').read_text());entry=meta['entry']
base=(src/'baseline.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==meta['variants']['baseline']['sha256']
shoff=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,shoff+i*sz) for i in range(n)];h=hs[st];names=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if names[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+entry)[1].split('Function : ')[0])}
r=rows((src/'baseline.sass').read_text());ref=rows((src/'tail9.sass').read_text())
def word(pc):return base[off+pc:off+pc+16]
def ctrl(pc):return int.from_bytes(word(pc),'little')>>105&0x1ffff
assert r[0x1610]=='LOP3.LUT R153, R164, 0x30, R153, 0x78, !PT'
assert r[0xcd0]=='IMAD R164, R96, 0x40, R99'
assert not any(re.search(r'\bR164\b',x) for pc,x in r.items() if pc>0x1610 or pc<0xcd0)
# Audit implicit MMA register vectors as well as explicit scalar uses.
for pc,x in r.items():
 if pc>=0x1620 and x.startswith('IMMA.'):
  regs=list(map(int,re.findall(r'\bR(\d+)',x)))
  assert not any(v<=164<v+width for v,width in zip(regs,[4,4,2,4]))
assert r[0x1720]=='LOP3.LUT R96, R161, 0x1, RZ, 0x3c, !PT'
assert r[0x17d0]=='BAR.ARV R96, 0x180'
assert ((ctrl(0x1650)>>5)&7)==1 and ((ctrl(0x16a0)>>5)&7)==2
assert ctrl(0x16c0)>>11==2 and ctrl(0x1700)>>11==4
assert r[0x16b0]=='DEPBAR.LE SB0, 0x0'
assert all(not ('LDSM' in x or 'LDGSTS' in x) for pc,x in r.items() if 0x16b0<=pc<0x17e0)
manifest=dict(entry=entry,variants={},warning='Offline correct-arithmetic binary experiment; do not overwrite production library.')
for name,before in [('tail9',None),('id_only9',0x1740),('id11',0x1710),('id16',0x16c0)]:
 if before is None:
  data=bytearray((src/'tail9.cubin').read_bytes());expected=dict(ref)
 else:
  data=bytearray(base);ids=[pc for pc in range(0x1620,0x17e0,16) if pc not in (0x1720,0x17d0)]
  ids.insert(0,0x1720);ids.insert(ids.index(before),0x17d0)
  expected=dict(r);new_pcs={old:0x1620+16*i for i,old in enumerate(ids)}
  assert new_pcs[0x16b0]<new_pcs[0x17d0]
  assert new_pcs[0x1720]<new_pcs[0x16a0]
  assert sum(ctrl(pc)&15 for pc in ids[:ids.index(0x17d0)])>=9
  for i,old in enumerate(ids):
   new=0x1620+16*i;w=bytearray(word(old));ins=r[old]
   if old==0x1720:
    assert w[2]==96;w[2]=164;ins=ins.replace('R96,','R164,',1)
   if old==0x17d0:
    assert w[4]==96;w[4]=164;ins=ins.replace('R96,','R164,',1)
   if name=='id16' and old in (0x17d0,0x16c0,0x1700):
    # Consume both pending LDSM completions on arrival, then remove the
    # old waits so they do not spuriously wait on arrival's new read-SB1.
    bits=int.from_bytes(w,'little');mask=6 if old==0x17d0 else 0
    w=bytearray(((bits&~(63<<116))|(mask<<116)).to_bytes(16,'little'))
   data[off+new:off+new+16]=w;expected[new]=ins
  assert data[:off+0x1620]==base[:off+0x1620] and data[off+0x17e0:]==base[off+0x17e0:]
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump);assert rr==expected
 assert sum(x.startswith('IMMA.') for x in rr.values())==64
 assert [x for x in rr.values() if x.startswith('IMMA.')]==[x for x in r.values() if x.startswith('IMMA.')]
 assert len(rr)==len(r)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in ref.items()],[f'{pc:04x} {x}\n' for pc,x in rr.items()],fromfile='validated-tail9',tofile=name)))
 apc=next(pc for pc,x in rr.items() if x.startswith('BAR.ARV'))
 manifest['variants'][name]=dict(entry=entry,sha256=hashlib.sha256(data).hexdigest(),smem=69632,correct=True,arrival_pc=hex(apc),mma_after_arrival=sum(x.startswith('IMMA.') for pc,x in rr.items() if apc<pc<0x17e0))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
