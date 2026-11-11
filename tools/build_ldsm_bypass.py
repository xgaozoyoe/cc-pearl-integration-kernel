"""Wrong-arithmetic LDSM-work/dependency ablation, isolated cubins only."""
from pathlib import Path
import json,struct,re,hashlib,subprocess
root=Path(__file__).resolve().parents[1];src=root/'artifacts/dependency-wait';out=root/'artifacts/ldsm-bypass';out.mkdir(exist_ok=True)
meta=json.loads((src/'manifest.json').read_text());entry=meta['entry'];binary=(src/'ab_zero.cubin').read_bytes()
assert hashlib.sha256(binary).hexdigest()==meta['variants']['ab_zero']['sha256']
shoff=struct.unpack_from('<Q',binary,40)[0];size,count,strings=struct.unpack_from('<HHH',binary,58)
hs=[struct.unpack_from('<IIQQQQIIQQ',binary,shoff+i*size) for i in range(count)];h=hs[strings];names=binary[h[4]:h[4]+h[5]]
offset=next(h[4] for h in hs if names[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
sec=(src/'ab_zero.sass').read_text().split('Function : '+entry)[1].split('Function : ')[0]
rows={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)}
def word(pc):return int.from_bytes(binary[offset+pc:offset+pc+16],'little')
controlmask=((1<<17)-1)<<105
nop=word(next(pc for pc,ins in rows.items() if ins=='NOP'))&~controlmask
loads={pc:int(re.match(r'LDSM.16.M88.4 R(\d+)',ins)[1]) for pc,ins in rows.items() if ins.startswith('LDSM.')}
assert len(loads)==16
assert sum(r>=148 for r in loads.values())==8
# Loaded values have no non-IMMA use before their overwrite in the compute body.
# Track removed register writes independently of scoreboard tracking.
manifest=dict(entry=entry,variants={},caveat='Wrong arithmetic. Removes load work AND dependencies, not a latency-only experiment. Fixed stalls, copy pipeline, CTA barriers remain.')
for name,chosen in [('baseline',[]),('ab_zero',[]),('no_b',[pc for pc,r in loads.items() if r>=148]),('no_ab',list(loads))]:
 data=bytearray((src/'baseline.cubin').read_bytes() if name=='baseline' else binary);changes=[];pending=[False]*6;dead=set()
 if chosen:
  for pc,ins in rows.items():
   if not 0x1060<=pc<=0x1770:continue
   old=word(pc);ctrl=(old>>105)&0x1ffff;wait=ctrl>>11;wr=(ctrl>>5)&7;rd=(ctrl>>8)&7
   cleared=sum(1<<i for i in range(6) if pending[i] and wait&(1<<i))
   ctrl&=~(cleared<<11)
   regs=list(map(int,re.findall(r'\bR(\d+)',ins)))
   if pc in chosen:
    assert rd==7
    dead.update(range(loads[pc],loads[pc]+4));ctrl=(ctrl&~(7<<5))|(7<<5)
    if wr<6:pending[wr]=True
    new=nop|(ctrl<<105)
   else:
    if wr<6:pending[wr]=False
    if rd<6:pending[rd]=False
    if ins.startswith('LDSM.'):
     dest=set(range(regs[0],regs[0]+4));reads=set(regs[1:])
    elif ins.startswith('IMMA.'):
     assert len(regs)==2 and re.search(r'RZ(?:\.reuse)?\.ROW',ins)
     dest=set(range(regs[0],regs[0]+4));reads=set(range(regs[1],regs[1]+4))
    else:
     m=re.match(r'\S+ R(\d+),',ins);dest={int(m[1])} if m else set();reads=set(regs[1:] if m else regs)
    assert not reads&dead,(hex(pc),ins,reads&dead)
    dead-=dest;new=(old&~controlmask)|(ctrl<<105)
   if new!=old:
    data[offset+pc:offset+pc+16]=new.to_bytes(16,'little');changes.append(dict(pc=hex(pc),removed_load=pc in chosen,cleared_wait_mask=cleared))
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump)
 sec2=dump.split('Function : '+entry)[1].split('Function : ')[0];got={int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec2)}
 if name!='baseline':
  assert got=={pc:('NOP' if pc in chosen else ins) for pc,ins in rows.items()}
 assert sum(ins.startswith('IMMA.') for ins in got.values())==64
 manifest['variants'][name]=dict(sha256=hashlib.sha256(data).hexdigest(),removed_ldsm=len(chosen),changes=changes)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print({n:v['removed_ldsm'] for n,v in manifest['variants'].items()})
