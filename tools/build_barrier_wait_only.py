"""Racy offline diagnostic:retain arrivals,remove main-loop blocking only."""
from pathlib import Path
import struct,json,hashlib,re,subprocess
root=Path(__file__).resolve().parents[1];out=root/'artifacts/barrier-wait-only';out.mkdir(exist_ok=True)
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def word(pc):return int.from_bytes(base[off+pc:off+pc+16],'little')
def rows(s):return {int(a,16):x for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text());assert r[0x16d0]=='BAR.ARV R164, 0x180';assert r[0x2a00]=='BAR.SYNC.DEFER_BLOCKING R161, 0x180'
# Reuse proven arrival opcode, but keep original sync ID operand and controls.
w=word(0x16d0);old=word(0x2a00);assert (w>>32)&255==164 and (old>>32)&255==161
w=(w&~(255<<32))|(161<<32);controlmask=((1<<17)-1)<<105;w=(w&~controlmask)|(old&controlmask)
m={'warning':'Deliberately racy,offline only;no production entrypoint. Results NOT valid Pearl outputs.','variants':{}}
for name in ['native','arrive_only','no_named']:
 d=bytearray(base)
 if name=='arrive_only':d[off+0x2a00:off+0x2a10]=w.to_bytes(16,'little')
 if name=='no_named':
  parent=root/'artifacts/barrier-removal';vv=json.loads((parent/'manifest.json').read_text())['variants']['no_named_barriers'];d=bytearray((parent/'no_named_barriers.cubin').read_bytes());assert hashlib.sha256(d).hexdigest()==vv['sha256']
 (out/f'{name}.cubin').write_bytes(d);sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass);rr=rows(sass)
 if name=='arrive_only':
  assert rr[0x2a00]=='BAR.ARV R161, 0x180',rr[0x2a00]
  assert all(rr[pc]==x for pc,x in r.items() if pc!=0x2a00)
 assert [x for x in rr.values() if 'IMMA.' in x]==[x for x in r.values() if 'IMMA.' in x]
 (out/f'{name}.diff').write_text('\n'.join(f'{pc:#x}: {r[pc]} -> {x}' for pc,x in rr.items() if x!=r[pc])+'\n')
 m['variants'][name]=dict(v,sha256=hashlib.sha256(d).hexdigest(),correct=name=='native')
(out/'manifest.json').write_text(json.dumps(m,indent=2));print('Built native,arrive_only,no_named')
