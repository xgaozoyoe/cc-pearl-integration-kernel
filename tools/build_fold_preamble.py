"""Fixed-SASS correct fold preamble simplification, current8+4 baseline."""
from pathlib import Path
import struct,re,json,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/fold-preamble';out.mkdir(exist_ok=True)
src=root/'artifacts/handoff-groups';meta=json.loads((src/'manifest.json').read_text());entry=meta['variants']['group8']['entry'];base=(src/'group8.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==meta['variants']['group8']['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
def rows(s):
 s=s.split('Function : '+entry)[1].split('Function : ')[0]
 return {int(a,16):x for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
r=rows((src/'group8.sass').read_text())
def word(pc):return base[off+pc:off+pc+16]
def ctrl(w):return int.from_bytes(w,'little')>>105&0x1ffff
def wait(w,mask):
 v=int.from_bytes(w,'little');return ((v&~(63<<116))|(mask<<116)).to_bytes(16,'little')
def relative(w,old,new,target):
 v=int.from_bytes(w,'little');delta=(v>>32)&0xffffffff;delta=delta-(1<<32) if delta>>31 else delta
 assert old+16+delta==target
 d=target-new-16;assert d>=0
 return ((v&~(0xffffffff<<32))|(d<<32)).to_bytes(16,'little')
assert r[0x17f0]=='S2R R96, SR_TID.X'
assert r[0x1860]=='ULOP3.LUT UR6, UR6, 0x3ffffff0, URZ, 0xc0, !UPT'
assert r[0x1880]=='SHF.R.U32.HI R96, RZ, 0x5, R96'
assert r[0x3700]=='NOP'
# rb in[0,31], so signed remainder by16 equals bitwiseAND15.
a=bytearray(word(0x1860));assert a[3]==6;a[3]=5
v=int.from_bytes(a,'little');v=(v&~(0xffffffff<<32))|(15<<32);andword=v.to_bytes(16,'little')
manifest={'entry':entry,'variants':{}}
for name in ('baseline','mod_nop','mod_compact','both_compact'):
 data=bytearray(base);expected=dict(r)
 if name=='mod_nop':
  for pc in (0x1830,0x1840,0x1860,0x1870):
   w=andword if pc==0x1830 else word(0x3700);data[off+pc:off+pc+16]=w
   expected[pc]='ULOP3.LUT UR6, UR5, 0xf, URZ, 0xc0, !UPT' if pc==0x1830 else 'NOP'
 elif name!='baseline':
  ids=[pc for pc in range(0x17f0,0x1930,16) if pc not in (0x1840,0x1860,0x1870)]
  if name=='both_compact':ids.remove(0x17f0)
  for pc in range(0x17f0,0x1930,16):data[off+pc:off+pc+16]=word(0x3700);expected[pc]='NOP'
  for i,old in enumerate(ids):
   pc=0x17f0+i*16;w=word(old);x=r[old]
   if old==0x1830:w=andword;x='ULOP3.LUT UR6, UR5, 0xf, URZ, 0xc0, !UPT'
   if old in (0x1810,0x1920):w=relative(w,old,pc,0x2a00 if old==0x1810 else 0x2320)
   if name=='both_compact':
    if i==0:w=wait(w,2) # retain original early arrival-read completion fence
    if old==0x1880:
     q=bytearray(w);assert q[8]==96;q[8]=162;w=bytes(q);x=x.replace('0x5, R96','0x5, R162')
   data[off+pc:off+pc+16]=w;expected[pc]=x
  # First-half path branches directly to2320. Second-half falls through
  # this extra branch, bypassing vacated slots. Net saving2/3 or3/4 instr.
  pc=0x17f0+len(ids)*16;w=relative(word(0x2310),0x2310,pc,0x29f0)
  v=int.from_bytes(w,'little');v=(v&~(0xffffffff<<32))|((0x1930-pc-16)<<32)
  data[off+pc:off+pc+16]=v.to_bytes(16,'little');expected[pc]='BRA 0x1930'
 assert data[:off+0x17f0]==base[:off+0x17f0] and data[off+0x1930:]==base[off+0x1930:]
 p=out/(name+'.cubin');p.write_bytes(data);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump);assert got==expected
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native8',tofile=name)))
 manifest['variants'][name]={'entry':entry,'sha256':hashlib.sha256(data).hexdigest(),'smem':69632,'correct':True}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
