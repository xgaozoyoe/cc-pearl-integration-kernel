raise SystemExit('Disabled: relocated control timed out; see artifacts/deferred-prefix/REJECTED.txt')
"""Frozen SASS delayed blocking with audited loop/exit predicates. Offline only."""
from pathlib import Path
import json,struct,re,subprocess,hashlib,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/deferred-prefix';out.mkdir(exist_ok=True)
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,si=struct.unpack_from('<HHH',base,58);hs=[list(struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz)) for i in range(n)];h=hs[si];ss=base[h[4]:h[4]+h[5]];idx=next(i for i,h in enumerate(hs) if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e);off=hs[idx][4];size=hs[idx][5];cut=off+size;extra=0x400
ph=struct.unpack_from('<Q',base,32)[0];psz,pn=struct.unpack_from('<HH',base,54)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text())
def word(pc):return int.from_bytes(base[off+pc:off+pc+16],'little')
def put(d,pc,w):d[off+pc:off+pc+16]=w.to_bytes(16,'little')
def branch(pc,target,pred=7):
 w=word(0x2a40);w=(w&~(((1<<50)-1)<<32))|(((target-pc-16)&((1<<50)-1))<<32)
 return (w&~(15<<12))|(pred<<12)
def expand():
 d=bytearray(base[:cut])+bytearray(word(0x3700).to_bytes(16,'little')*(extra//16))+bytearray(base[cut:])
 def shift(x):return x+extra if x>=cut else x
 struct.pack_into('<Q',d,32,shift(ph));struct.pack_into('<Q',d,40,shift(sh))
 for i,old in enumerate(hs):
  h=old.copy()
  if i==idx:h[5]+=extra
  elif h[4]>=cut:h[4]+=extra
  struct.pack_into('<IIQQQQIIQQ',d,shift(sh)+i*sz,*h)
  if old[1]==2:
   for at in range(old[4],old[4]+old[5],old[9]):
    sym=list(struct.unpack_from('<IBBHQQ',base,at))
    if sym[3]==idx and sym[1]&15==2:sym[5]+=extra;struct.pack_into('<IBBHQQ',d,shift(at),*sym)
 for i in range(pn):
  q=list(struct.unpack_from('<IIQQQQQQ',base,ph+i*psz))
  if q[2]>=cut:q[2]+=extra
  elif q[2]+q[5]>=cut and q[5]:q[5]+=extra;q[6]+=extra
  struct.pack_into('<IIQQQQQQ',d,shift(ph)+i*psz,*q)
 return d
prefix=list(range(0xbb0,0xdc0,16));assert len(prefix)==33
assert not any(any(t in r[pc] for t in ['BAR.','BRA','CALL','LDSM','LDGSTS','STS']) for pc in prefix)
assert r[0xcb0].startswith('PLOP3.LUT P2') and r[0xdc0]=='@P2 BRA 0x1090'
assert all('R161' not in r[pc] for pc in prefix)
# Entry: termination check, final-case branch, original barrier, old-ID-consumed toggle,
# optional branch to next instruction (forces deferred barrier blocking),33instruction prefix,copy-path branch.
start=size;prefix_start=start+5*16;endprefix=prefix_start+33*16;final=endprefix+16
m={'description':'No-fold/fold work unchanged. All64named syncs retained. Continuing path prepares next iteration without clobbering its P2 before dc0.','variants':{}}
for name in ['native','relocated_control','deferred_prefix']:
 d=bytearray(base) if name=='native' else expand()
 if name!='native':
  put(d,0x2a00,branch(0x2a00,start))
  # Predicate latency:12cycles, yield bit cleared (12/13+yield encoding invalid).
  w=word(0x2a10);w=(w&~(31<<105))|(12<<105);put(d,start,w)
  put(d,start+16,branch(start+16,final,pred=10)) # !P2 means final iteration
  put(d,start+32,word(0x2a00));put(d,start+48,word(0x2a20))
  put(d,start+64,branch(start+64,prefix_start) if name=='relocated_control' else word(0x3700))
  for i,pc in enumerate(prefix):put(d,prefix_start+i*16,word(pc))
  put(d,endprefix,branch(endprefix,0xdc0)) # preserve new P2, do NOT pass original2a10
  put(d,final,word(0x2a00));put(d,final+16,branch(final+16,0x2a10))
  assert d[off:off+0x2a00]==base[off:off+0x2a00]
  assert d[off+0x2a10:cut]==base[off+0x2a10:cut]
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump)
 assert [x for x in rr.values() if 'IMMA.' in x]==[x for x in r.values() if 'IMMA.' in x]
 if name!='native':
  assert [rr[prefix_start+i*16] for i in range(33)]==[r[pc] for pc in prefix]
  assert rr[endprefix]=='BRA 0xdc0'
  assert rr[final+16]=='BRA 0x2a10'
  assert rr[start+16]==f'@!P2 BRA {hex(final)}'
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in rr.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('Built',list(m['variants']),'prefix',hex(prefix_start),'final',hex(final))
