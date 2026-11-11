"""Wrong-arithmetic copy/stage-handoff ablation; never a mining backend."""
from pathlib import Path
import json,struct,re,hashlib,subprocess
root=Path(__file__).resolve().parents[1];src=root/'artifacts/ldsm-bypass';out=root/'artifacts/copy-bypass';out.mkdir(exist_ok=True)
meta=json.loads((src/'manifest.json').read_text());entry=meta['entry'];binary=(src/'no_ab.cubin').read_bytes()
assert hashlib.sha256(binary).hexdigest()==meta['variants']['no_ab']['sha256']
shoff=struct.unpack_from('<Q',binary,40)[0];size,count,strings=struct.unpack_from('<HHH',binary,58)
hs=[struct.unpack_from('<IIQQQQIIQQ',binary,shoff+i*size) for i in range(count)];h=hs[strings];names=binary[h[4]:h[4]+h[5]]
offset=next(h[4] for h in hs if names[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
sec=(src/'no_ab.sass').read_text().split('Function : '+entry)[1].split('Function : ')[0]
rows={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)}
def word(pc):return int.from_bytes(binary[offset+pc:offset+pc+16],'little')
controlmask=((1<<17)-1)<<105;nop=word(next(pc for pc,ins in rows.items() if ins=='NOP'))&~controlmask
copies={pc for pc,ins in rows.items() if 'LDGSTS.' in ins};commits={pc for pc,ins in rows.items() if ins=='LDGDEPBAR'};waits={pc for pc,ins in rows.items() if ins.startswith('DEPBAR.LE')}
assert copies=={0x930,0x990,0xa70,0xa80,0xa90,0xf20,0xf90,0xff0,0x1030,0x1040}
assert commits=={0xab0,0x1050} and waits=={0xc30}
assert [pc for pc,ins in rows.items() if ins.startswith('BAR.SYNC')]==[0xc60,0x29d0]
assert not any(ins.startswith('LDSM.') for ins in rows.values())
# Explicit copy-address read-barrier consumers only. Keep 0xb60 waitmask:
# it is a loop header reached both from prologue and from transcript code.
clear={0x9f0:2,0xa30:4,0xac0:2,0xae0:4,0xb00:8,0xf50:2,0xfd0:2,0x10b0:2}
for pc,mask in clear.items():assert ((word(pc)>>116)&63)==mask
manifest=dict(entry=entry,variants={},caveat='Wrong arithmetic. Copy address arithmetic, NOP slots and fixed stall counts remain. Final transcript barrier retained. Loop-header waits retained for other predecessors.')
for name,remove_bar in [('baseline',False),('no_ab',False),('no_copy',False),('no_copy_bar',True)]:
 data=bytearray((src/'baseline.cubin').read_bytes() if name=='baseline' else binary);chosen=set();changes=[]
 if name.startswith('no_copy'):
  chosen=copies|commits|waits|({0xc60} if remove_bar else set())
  for pc in sorted(chosen|set(clear)):
   old=word(pc);ctrl=(old>>105)&0x1ffff
   if pc in clear:ctrl&=~(clear[pc]<<11)
   if pc in chosen:
    ctrl=(ctrl&~((7<<5)|(7<<8)))|(7<<5)|(7<<8)
    new=nop|(ctrl<<105)
   else:new=(old&~controlmask)|(ctrl<<105)
   data[offset+pc:offset+pc+16]=new.to_bytes(16,'little');changes.append(dict(pc=hex(pc),original=rows[pc],removed=pc in chosen,cleared_wait_mask=clear.get(pc,0)))
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump)
 sec2=dump.split('Function : '+entry)[1].split('Function : ')[0];got={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec2)}
 if name!='baseline':assert got=={pc:('NOP' if pc in chosen else ins) for pc,ins in rows.items()}
 assert got[0x29d0]==rows[0x29d0] and data[offset+0x29d0:offset+0x29e0]==binary[offset+0x29d0:offset+0x29e0]
 assert sum(ins.startswith('IMMA.') for ins in got.values())==64
 manifest['variants'][name]=dict(sha256=hashlib.sha256(data).hexdigest(),changes=changes)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print({n:len(v['changes']) for n,v in manifest['variants'].items()})
