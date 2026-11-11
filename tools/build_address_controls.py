"""Correct-arithmetic fixed-SASS address anti-dependency controls on fastest native."""
from pathlib import Path
import struct,re,json,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/address-controls';out.mkdir(exist_ok=True);src=root/'artifacts/fold-preamble';v=json.loads((src/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(src/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((src/'both_compact.sass').read_text())
assert not any(re.search(r'\bR154\b',x) for pc,x in r.items() if 0xf60<=pc<0x1500)
# No implicit vector instruction accesses reserved register154 either.
for pc,x in r.items():
 if not 0xf60<=pc<0x1500:continue
 if x.startswith('IMMA.'):
  regs=list(map(int,re.findall(r'\bR(\d+)',x)));assert not any(a<=154<a+w for a,w in zip(regs,[4,4,2,4]))
 if x.startswith('LDSM.'):assert int(re.search(r'R(\d+)',x)[1])+3<154
assert r[0x1110]=='IADD3 R152, R164.reuse, 0x420, RZ'
assert r[0xf60]=='IMAD R96, R157, 0x1000, R142'
m={'variants':{}}
for name,ld,cp,release in [('native',False,False,False),('ld_rename',True,False,False),('ld_release',True,False,True),('cp_rename',False,True,False),('cp_release',False,True,True),('both_release',True,True,True)]:
 data=bytearray(base);expected=dict(r);changes=[]
 def reg(pc,byte,old,new,text):
  w=bytearray(data[off+pc:off+pc+16]);assert w[byte]==old,(hex(pc),byte,w[byte]);w[byte]=new;data[off+pc:off+pc+16]=w;expected[pc]=text
 def wait(pc,clear=0,add=0):
  w=int.from_bytes(data[off+pc:off+pc+16],'little');mask=w>>116&63;assert mask&clear==clear
  new=(mask&~clear)|add;w=(w&~(63<<116))|(new<<116);data[off+pc:off+pc+16]=w.to_bytes(16,'little');changes.append(dict(pc=hex(pc),old_wait=mask,new_wait=new))
 if cp:
  reg(0xf60,2,96,154,r[0xf60].replace('R96,','R154,',1))
  reg(0xfa0,3,96,154,r[0xfa0].replace('R96','R154'))
  reg(0xfc0,2,96,154,r[0xfc0].replace('R96','R154'))
  reg(0xfc0,3,96,154,r[0xfc0].replace('R96','R154'))
  reg(0x1010,3,96,154,r[0x1010].replace('P2, R96,','P2, R154,'))
  if release:wait(0xf60,clear=2);wait(0xf70,add=2) # next true write of cp.async source addressR141
 if ld:
  reg(0x1110,2,152,154,r[0x1110].replace('R152,','R154,',1))
  reg(0x14d0,8,152,154,r[0x14d0].replace('R152','R154'))
  reg(0x14f0,3,152,154,r[0x14f0].replace('R152','R154'))
  if release:wait(0x1110,clear=16);wait(0x1150,add=16) # keep original read completion by first use of loadedA
 assert data[:off+0xf60]==base[:off+0xf60] and data[off+0x1500:]==base[off+0x1500:]
 p=out/(name+'.cubin');p.write_bytes(data);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump);assert got==expected,[(hex(pc),got.get(pc),x) for pc,x in expected.items() if got.get(pc)!=x]
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='fastest-native',tofile=name))+json.dumps(changes,indent=2)+'\n')
 m['variants'][name]={**v,'sha256':hashlib.sha256(data).hexdigest(),'wait_changes':changes}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
