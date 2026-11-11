"""Isolated SM89 cubin scheduling experiment; no production ABI/library writes.
Shorten only a reviewed late-K64 all-IMMA window. Validate full transcript before timing.
"""
from pathlib import Path
import json,struct,re,hashlib,subprocess
root=Path(__file__).resolve().parents[1];src=root/'artifacts/dependency-wait';out=root/'artifacts/imma-cadence';out.mkdir(exist_ok=True)
meta=json.loads((src/'manifest.json').read_text());entry=meta['entry'];binary=(src/'baseline.cubin').read_bytes()
assert hashlib.sha256(binary).hexdigest()==meta['variants']['baseline']['sha256']
shoff=struct.unpack_from('<Q',binary,40)[0];size,count,strings=struct.unpack_from('<HHH',binary,58)
hs=[struct.unpack_from('<IIQQQQIIQQ',binary,shoff+i*size) for i in range(count)];h=hs[strings];names=binary[h[4]:h[4]+h[5]]
offset=next(h[4] for h in hs if names[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
sec=(src/'baseline.sass').read_text().split('Function : '+entry)[1].split('Function : ')[0]
rows={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)}
def word(pc):return int.from_bytes(binary[offset+pc:offset+pc+16],'little')
selected=list(range(0x1680,0x1730,16));written=set();audit=[]
for pc in selected:
 ins=rows[pc];assert ins.startswith('IMMA.16832.S8.S8 ')
 d,a,b,c=map(int,re.findall(r'\bR(\d+)',ins));ctrl=(word(pc)>>105)&0x1ffff
 assert ctrl&15==8 and (ctrl>>5)&7==7 and (ctrl>>8)&7==7
 reads=set(range(a,a+4))|set(range(b,b+2))|set(range(c,c+4));dest=set(range(d,d+4))
 assert not (reads|dest)&written,(hex(pc),'in-window dependency')
 assert not dest&(set(range(a,a+4))|set(range(b,b+2)))
 written|=dest;audit.append(dict(pc=hex(pc),instruction=ins,wait_mask=ctrl>>11,original_stall=8))
# Leave the last four eight-cycle gaps and the final IMMA untouched.
assert all((word(pc)>>105)&15==8 for pc in range(0x1730,0x1770,16))
manifest=dict(entry=entry,variants={},audit=audit,caveat='Experimental scheduling; not deployed. Static in-window independence is not a proof of all machine latency constraints.')
for name,stall in [('baseline',8),('tail4',4),('tail1',1)]:
 data=bytearray(binary)
 if stall!=8:
  for pc in selected:
   old=word(pc);new=(old&~(15<<105))|(stall<<105);assert ((old^new)&~(15<<105))==0
   data[offset+pc:offset+pc+16]=new.to_bytes(16,'little')
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump)
 sec2=dump.split('Function : '+entry)[1].split('Function : ')[0]
 assert {int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec2)}==rows
 manifest['variants'][name]=dict(sha256=hashlib.sha256(data).hexdigest(),stall=stall,changed_bytes=sum(x!=y for x,y in zip(data,binary)))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest['variants'],indent=2))
