raise SystemExit("Disabled: rejected structural diagnostic; see artifacts/b-handoff-factorial/REJECTED.txt")
"""GPU0 offline B-wait x safe stage-wait relocation. No production mutations."""
from pathlib import Path
import struct,json,hashlib,re,subprocess,difflib,shutil
root=Path(__file__).resolve().parents[1];out=root/'artifacts/b-handoff-factorial';out.mkdir(exist_ok=True)
src=root/'artifacts/nonzero-dependency';meta=json.loads((src/'manifest.json').read_text());entry=meta['variants']['native']['entry']
base=(src/'native.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==meta['variants']['native']['sha256']
shoff=struct.unpack_from('<Q',base,40)[0];sz,num,si=struct.unpack_from('<HHH',base,58);hs=[list(struct.unpack_from('<IIQQQQIIQQ',base,shoff+i*sz)) for i in range(num)];h=hs[si];strings=base[h[4]:h[4]+h[5]]
idx=next(i for i,h in enumerate(hs) if strings[h[0]:].split(b'\0')[0].decode()=='.text.'+entry);off=hs[idx][4];oldsize=hs[idx][5];cut=off+oldsize;extra=0x300
phoff=struct.unpack_from('<Q',base,32)[0];phsz,phnum=struct.unpack_from('<HH',base,54)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+entry)[1].split('Function : ')[0])}
r=rows((src/'native.sass').read_text())
def get(data,pc):return int.from_bytes(data[off+pc:off+pc+16],'little')
def put(data,pc,w):data[off+pc:off+pc+16]=w.to_bytes(16,'little')
nop=get(base,0x3700)
prefix=list(range(0xbb0,0xdc0,16));assert len(prefix)==33
assert all(not any(t in r[pc] for t in ('LDSM','LDG','STS','STG','BAR.','BRA','CALL')) for pc in prefix)
assert all('R161' not in r[pc] for pc in prefix)
def branch(pc,target):
 w=get(base,0x2a40);mask=((1<<50)-1)<<32
 # SM89 BRA carries signed byte displacement from the next instruction.
 return (w&~mask)|(((target-pc-16)&((1<<50)-1))<<32)
assert branch(0x2a40,0xbb0)==get(base,0x2a40)
def expand(binary):
 d=bytearray(binary[:cut])+bytearray(nop.to_bytes(16,'little')*(extra//16))+bytearray(binary[cut:])
 def shifted(x):return x+extra if x>=cut else x
 struct.pack_into('<Q',d,32,shifted(phoff));struct.pack_into('<Q',d,40,shifted(shoff))
 for i,old in enumerate(hs):
  h=old.copy()
  if i==idx:h[5]+=extra
  elif h[4]>=cut:h[4]+=extra
  struct.pack_into('<IIQQQQIIQQ',d,shifted(shoff)+i*sz,*h)
  if old[1]==2: # symbol table: grow target function symbol, keeping all values/indices
   for at in range(old[4],old[4]+old[5],old[9]):
    sym=list(struct.unpack_from('<IBBHQQ',base,at))
    if sym[3]==idx and sym[1]&15==2:
     sym[5]+=extra;struct.pack_into('<IBBHQQ',d,shifted(at),*sym)
 for i in range(phnum):
  q=list(struct.unpack_from('<IIQQQQQQ',base,phoff+i*phsz))
  if q[2]>=cut:q[2]+=extra
  elif q[2]+q[5]>=cut and q[5]:q[5]+=extra;q[6]+=extra
  struct.pack_into('<IIQQQQQQ',d,shifted(phoff)+i*phsz,*q)
 return d
manifest={'warning':'Offline altered arithmetic; safe waiting relocation with final handoff preserved. Shared memory accesses remain after the stage barrier.','variants':{}}
for name,bcut,late in [('native',False,False),('control',False,False),('b_only',True,False),('handoff_only',False,True),('both',True,True)]:
 if name=='native':d=bytearray(base)
 else:
  original=(src/('a_as_b_release.cubin' if bcut else 'a_as_b.cubin')).read_bytes()
  d=expand(original)
  put(d,0x2a00,branch(0x2a00,oldsize));put(d,0x2a40,branch(0x2a40,0xdc0))
  pos=oldsize
  w=get(base,0x2a10);w=(w&~(31<<105))|(12<<105);put(d,pos,w);pos+=16
  guard=pos;pos+=16
  seq=prefix+[0x2a00] if late else [0x2a00]+prefix
  for pc in seq:put(d,pos,get(base,pc));pos+=16
  put(d,pos,branch(pos,0x2a10));pos+=16
  final=pos;put(d,pos,get(base,0x2a00));pos+=16
  put(d,pos,branch(pos,0x2a10))
  w=branch(guard,final);put(d,guard,(w&~(15<<12))|(10<<12))
 p=out/(name+'.cubin');p.write_bytes(d)
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(sass);got=rows(sass)
 assert sum(x.startswith('IMMA.') for x in got.values())==64
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native',tofile=name)))
 manifest['variants'][name]={**meta['variants']['native'],'sha256':hashlib.sha256(d).hexdigest(),'correct':name=='native','b_wait_release':bcut,'late_handoff':late}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');shutil.copy2(src/'liboracle.so',out/'liboracle.so')
