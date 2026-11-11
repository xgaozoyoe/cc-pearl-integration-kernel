"""Preserve sleep32's shorter register polling schedule; replace only sleep with NOP."""
from pathlib import Path
import json,struct,re,hashlib,subprocess
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-poll-nop';binary=(out/'handoff-backoff-source.sm_89.cubin').read_bytes();sass=(root/'artifacts/handoff-backoff.sass').read_text()
shoff=struct.unpack_from('<Q',binary,40)[0];size,count,strings=struct.unpack_from('<HHH',binary,58);hs=[struct.unpack_from('<IIQQQQIIQQ',binary,shoff+i*size) for i in range(count)];h=hs[strings];names=binary[h[4]:h[4]+h[5]];sections={names[h[0]:].split(b'\0')[0].decode():h for h in hs}
manifest=dict(variants={},caveat='Correct arithmetic experiment. Only a non-synchronizing sleep is replaced by NOP; shorter polling schedule preserved.')
for name,orig in [('baseline','baseline'),('spin','spin'),('poll_nop','sleep32')]:
 sec=next(x for x in sass.split('Function : ')[1:] if 'handoff_backoff_'+orig+'6kernel' in x.splitlines()[0]);entry=sec.splitlines()[0].strip();offset=sections['.text.'+entry][4];rows={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)};data=bytearray(binary)
 if name=='poll_nop':
  sleeps=[pc for pc,ins in rows.items() if ins=='NANOSLEEP 0x20'];assert sleeps==[0x2b60]
  pc=sleeps[0];npc=next(pc for pc,ins in rows.items() if ins=='NOP');mask=((1<<17)-1)<<105
  old=int.from_bytes(data[offset+pc:offset+pc+16],'little');nop=int.from_bytes(data[offset+npc:offset+npc+16],'little');new=(nop&~mask)|(old&mask)
  data[offset+pc:offset+pc+16]=new.to_bytes(16,'little')
  assert [i for i,(a,b) in enumerate(zip(data,binary)) if a!=b] and all(offset+pc<=i<offset+pc+16 for i,(a,b) in enumerate(zip(data,binary)) if a!=b)
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump)
 sec2=dump.split('Function : '+entry)[1].split('Function : ')[0];got={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec2)}
 assert got=={pc:('NOP' if name=='poll_nop' and pc==0x2b60 else ins) for pc,ins in rows.items()}
 manifest['variants'][name]=dict(entry=entry,smem=69632 if name=='baseline' else 69640,sha256=hashlib.sha256(data).hexdigest())
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
