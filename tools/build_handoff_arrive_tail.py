"""Fixed-SASS early publication after every shared read and copy has completed.
Only rotate the existing BAR.ARV across independent tail IMMA instructions.
Offline driver harness only; never replace the mining backend.
"""
from pathlib import Path
import re,struct,json,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-arrive-tail';out.mkdir(exist_ok=True)
base=(root/'artifacts/handoff-abcd/B.cubin').read_bytes()
assert hashlib.sha256(base).hexdigest()==json.loads((root/'artifacts/handoff-abcd/manifest.json').read_text())['variants']['B']['sha256']
shoff=struct.unpack_from('<Q',base,40)[0];size,num,strings=struct.unpack_from('<HHH',base,58)
hs=[struct.unpack_from('<IIQQQQIIQQ',base,shoff+i*size) for i in range(num)];h=hs[strings];names=base[h[4]:h[4]+h[5]]
sec,h=next((names[h[0]:].split(b'\0')[0].decode(),h) for h in hs if names[h[0]:].startswith(b'.text.') and b'handoff_lean_toggle6kernel' in names[h[0]:].split(b'\0')[0]);entry=sec[6:];off=h[4]
sass=(root/'artifacts/handoff-abcd/B.sass').read_text()
def rows(s):return {int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+entry)[1].split('Function : ')[0])}
r=rows(sass)
def word(pc):return base[off+pc:off+pc+16]
def ctrl(pc):return (int.from_bytes(word(pc),'little')>>105)&0x1ffff
assert r[0x1650]=='LDSM.16.M88.4 R148, [R149+UR5]'
assert r[0x16a0]=='LDSM.16.M88.4 R152, [R153+UR5]'
assert ((ctrl(0x1650)>>5)&7)==1 and ((ctrl(0x16a0)>>5)&7)==2
assert ctrl(0x16c0)>>11==2 and ctrl(0x1700)>>11==4
assert r[0x16b0]=='DEPBAR.LE SB0, 0x0'
assert r[0x1720]=='LOP3.LUT R96, R161, 0x1, RZ, 0x3c, !PT'
assert r[0x17d0]=='BAR.ARV R96, 0x180'
# Last shared read:16a0, retired at1700. No later shared instruction before arrival.
assert not any('LDSM' in ins or re.search(r'\b(?:LDS|STS|LDGSTS)\.',ins) for pc,ins in r.items() if 0x16b0<=pc<0x17d0)
assert (ctrl(0x1720)&15)+(ctrl(0x1730)&15)>=9 # Keep the original scalar latency guard.
manifest=dict(entry=entry,source_sha256=hashlib.sha256(base).hexdigest(),variants={})
for name,dest in [('baseline',0x17d0),('tail5',0x1780),('tail9',0x1740)]:
 data=bytearray(base);expected=dict(r)
 for pc in range(dest,0x17d0,16):
  assert r[pc].startswith('IMMA.') and 'R96' not in r[pc]
  c=ctrl(pc);assert (c>>5)&7==7 and (c>>8)&7==7 and c>>11==0
 if dest!=0x17d0:
  block=base[off+dest:off+0x17d0];data[off+dest:off+0x17e0]=word(0x17d0)+block
  expected[dest]=r[0x17d0]
  for pc in range(dest,0x17d0,16):expected[pc+16]=r[pc]
 # Exact instruction-word multiset and all PCs outside the rotated block retained.
 assert sorted(data[off+pc:off+pc+16] for pc in range(dest,0x17e0,16))==sorted(word(pc) for pc in range(dest,0x17e0,16))
 assert data[:off+dest]==base[:off+dest] and data[off+0x17e0:]==base[off+0x17e0:]
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump);assert rr==expected
 assert sum(s.startswith('IMMA.') for s in rr.values())==64
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {s}\n' for pc,s in r.items()],[f'{pc:04x} {s}\n' for pc,s in rr.items()],fromfile='baseline',tofile=name)))
 manifest['variants'][name]=dict(entry=entry,sha256=hashlib.sha256(data).hexdigest(),smem=69632,correct=True,arrival_pc=hex(dest),mma_after_arrival=(0x17d0-dest)//16)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
