"""Delay BAR.ARV register-read scoreboard wait, retain shared synchronization."""
from pathlib import Path
import struct,re,json,hashlib,subprocess
root=Path(__file__).resolve().parents[1];out=root/'artifacts/fold-read-wait';out.mkdir(exist_ok=True)
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256'];e=v['entry']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(a,16):x for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text())
def word(pc):return int.from_bytes(base[off+pc:off+pc+16],'little')
assert r[0x16d0]=='BAR.ARV R164, 0x180' and (word(0x16d0)>>113&7)==1
assert word(0x16d0)>>116&63==6 # preserve final LDSM completion
for pc in range(0x17e0,0x1940,16):
 assert 'R164' not in r[pc]
 assert word(pc)>>110&7==7 and word(pc)>>113&7==7
assert r[0x18e0]=='@!P2 BRA 0x2320' and r[0x2330].startswith('REDUX')
assert word(0x2320)>>110&7==7 and word(0x2320)>>113&7==7
m={'variants':{}}
for name,changes in [('native',[]),('prep_wait',[(0x17f0,2,0)]),('redux_wait',[(0x17f0,2,0),(0x1840,2,0),(0x1940,0,2),(0x2330,0,2)])]:
 d=bytearray(base)
 for pc,old,new in changes:
  w=word(pc);assert w>>116&63==old
  d[off+pc:off+pc+16]=((w&~(63<<116))|(new<<116)).to_bytes(16,'little')
 (out/f'{name}.cubin').write_bytes(d)
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass);assert rows(sass)==r
 (out/f'{name}.diff').write_text('\n'.join(f'{pc:#06x} {r[pc]}: waitmask {old:#x} -> {new:#x}' for pc,old,new in changes)+'\n')
 m['variants'][name]=dict(v,sha256=hashlib.sha256(d).hexdigest(),threads=384,correct=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2))
print('Built3variants;instruction text and all controls except specified wait masks unchanged.')
