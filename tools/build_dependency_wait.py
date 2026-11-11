"""Diagnostic only: relocate three dead B-operand waits in AB-zero cubin.
Control encoding reference: cloudcores/CuAssembler CuControlCode.py/CuSMVersion.py.
Never use this deliberately incorrect arithmetic for mining.
"""
from pathlib import Path
import json,struct,re,hashlib,subprocess
root=Path(__file__).resolve().parents[1];src=root/'artifacts/dependency-ablation';out=root/'artifacts/dependency-wait';out.mkdir(exist_ok=True)
meta=json.loads((src/'manifest.json').read_text());entry=meta['entry'];binary=(src/'ab_zero.cubin').read_bytes()
assert hashlib.sha256(binary).hexdigest()==meta['variants']['ab_zero']['sha256']
shoff=struct.unpack_from('<Q',binary,40)[0];size,count,strings=struct.unpack_from('<HHH',binary,58)
hs=[struct.unpack_from('<IIQQQQIIQQ',binary,shoff+i*size) for i in range(count)];h=hs[strings];names=binary[h[4]:h[4]+h[5]]
offset=next(h[4] for h in hs if names[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
def control(pc):return (int.from_bytes(binary[offset+pc:offset+pc+16],'little')>>105)&0x1ffff
sec=(src/'ab_zero.sass').read_text().split('Function : '+entry)[1].split('Function : ')[0]
rows={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)}
moves=[(0x1190,0x11a0,0x11e0),(0x1240,0x1250,0x1290),(0x1300,0x1310,0x1420)]
for producer,first,last in moves:
 assert rows[producer].startswith('LDSM.16.M88.4 R148,')
 assert (control(producer)>>5)&7==1
 assert control(first)>>11==2 and control(last)>>11==0
 for pc in range(first,last,16):
  ins=rows[pc];c=control(pc)
  assert (c>>5)&7==7 and (c>>8)&7==7,(hex(pc),'new scoreboard producer')
  assert not any(x in ins for x in ('BRA','BAR.','LDSM','LDG','STS','STG'))
  regs=set(map(int,re.findall(r'\bR(\d+)',ins)))
  if ins.startswith('IMMA.'):
   assert 'RZ' in ins
   regs|={r+j for r in tuple(regs) for j in range(4)}
  assert not regs.intersection(range(148,152)),(hex(pc),ins)
 assert re.match(r'(SHF\S*|IADD3) R148,',rows[last])
result=dict(entry=entry,variants={},moves=[dict(producer=hex(a),old_wait=hex(b),new_wait=hex(c)) for a,b,c in moves],warning='Wrong arithmetic diagnostic. Only three first-half B waits relocated; not complete dependency removal.')
for name,base,selected in [('baseline',(src/'baseline.cubin').read_bytes(),[]),('ab_zero',binary,[]),('ab_wait2',binary,moves[:2]),('ab_wait3',binary,moves)]:
 data=bytearray(base)
 for _,first,last in selected:
  for pc,mask in [(first,0),(last,2)]:
   at=offset+pc;old=int.from_bytes(data[at:at+16],'little');new=(old&~(63<<116))|(mask<<116)
   assert (old^new)==2<<116
   data[at:at+16]=new.to_bytes(16,'little')
 assert sum(a!=b for a,b in zip(data,base))==2*len(selected)
 (out/(name+'.cubin')).write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'.cubin'))],text=True)
 (out/(name+'.sass')).write_text(dump)
 if selected:
  section=dump.split('Function : '+entry)[1].split('Function : ')[0]
  assert {int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',section)}==rows
 result['variants'][name]=dict(sha256=hashlib.sha256(data).hexdigest(),wait_moves=len(selected))
(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
