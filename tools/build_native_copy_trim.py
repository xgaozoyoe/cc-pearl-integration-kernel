"""Fixed-SASS copy control experiments on frozen fastest baseline."""
from pathlib import Path
import json,struct,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/native-copy-trim';out.mkdir(exist_ok=True)
src=root/'artifacts/fold-preamble';v=json.loads((src/'manifest.json').read_text())['variants']['both_compact'];entry=v['entry'];base=(src/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+entry)[1].split('Function : ')[0])}
r=rows((src/'both_compact.sass').read_text())
def word(pc):return int.from_bytes(base[off+pc:off+pc+16],'little')
assert r[0xdd0]=='S2R R98, SR_TID.X';assert r[0x40]=='S2R R162, SR_TID.X'
assert r[0xe40]=='ISETP.GT.U32.AND P4, PT, R98.reuse, 0x3ff, PT'
assert r[0xe80]=='ISETP.GT.U32.AND P2, PT, R96, 0x3ff, PT'
assert r[0x1620]=='LOP3.LUT R164, R161, 0x1, RZ, 0x3c, !PT'
assert not any(re.match(r'(?:@\S+ )?\S+ R162[, ]',x) for pc,x in r.items() if 0x40<pc<0x2a50)
m={'variants':{},'warning':'GPU0 offline diagnostic; fixed384-thread CTA only; never replace production.'}
for name in ('native','tid_move','tid_bounds'):
 data=bytearray(base);expected=dict(r)
 def put(pc,w,x):data[off+pc:off+pc+16]=w.to_bytes(16,'little');expected[pc]=x
 if name!='native':
  # Keep all original instruction addresses. Substitute one ALU move for S2R.
  w=word(0x1620);w=(w&~((255<<16)|(255<<24)|(0xffffffff<<32)))|(98<<16)|(162<<24)
  # S2R had write-scoreboard1. Fixed-latency LOP3 needs no scoreboard.
  control=(word(0xdd0)>>105)&0x1ffff;control=(control&~(7<<5))|(7<<5)
  w=(w&~(0x1ffff<<105))|(control<<105)
  put(0xdd0,w,'LOP3.LUT R98, R162, 0x0, RZ, 0x3c, !PT')
 if name=='tid_bounds':
  # tid<=383 and tid+384<=767, so comparisons against1023 are alwaysfalse.
  # Keep original control delays in NOP slots, but no obsolete S2R completion wait.
  for pc in (0xe40,0xe80):
   ctrl=(word(pc)>>105)&0x1ffff;ctrl &= ~(63<<11)
   put(pc,(word(0x3700)&~(0x1ffff<<105))|(ctrl<<105),'NOP')
  for pc in (0xeb0,0xec0,0xed0,0xef0,0xf00,0xf40,0xf50,0xf70,0xfb0,0x1000):
   assert r[pc].startswith(('@!P4 ','@!P2 '))
   put(pc,(word(pc)&~(15<<12))|(7<<12),r[pc].split(' ',1)[1])
 p=out/(name+'.cubin');p.write_bytes(data);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump);assert got==expected,[(hex(pc),got.get(pc),x) for pc,x in expected.items() if got.get(pc)!=x]
 assert data[:off+0xdd0]==base[:off+0xdd0] and data[off+0x1010:]==base[off+0x1010:]
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='fastest-native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(data).hexdigest()}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
