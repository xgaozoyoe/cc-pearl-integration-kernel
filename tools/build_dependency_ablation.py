"""Patch only calibrated IMMA operand bytes in isolated baseline cubin copies.
Wrong arithmetic, diagnostic only. All instruction scheduling bits preserved.
"""
from pathlib import Path
import struct,re,json,hashlib,subprocess
root=Path(__file__).resolve().parents[1];out=root/'artifacts/dependency-ablation'
source=out/'rank128-handoff-control-source.sm_89.cubin';original=source.read_bytes()
assert original[:6]==b'\x7fELF\x02\x01'
shoff=struct.unpack_from('<Q',original,40)[0];size,count,strings=struct.unpack_from('<HHH',original,58)
headers=[struct.unpack_from('<IIQQQQIIQQ',original,shoff+i*size) for i in range(count)]
h=headers[strings];names=original[h[4]:h[4]+h[5]]
sections={names[h[0]:].split(b'\0')[0].decode():h for h in headers}
section=next(n for n in sections if n.startswith('.text.') and 'handoff_control_baseline6kernel' in n)
entry=section[6:];offset=sections[section][4]
sass=(out/'original.sass').read_text();sec=next(x for x in sass.split('Function : ')[1:] if entry in x.splitlines()[0])
rows=re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',sec)
ops=[(int(pc,16),[int(r) for r in re.findall(r'\bR(\d+)',ins)]) for pc,ins in rows if ins.startswith('IMMA.16832.S8.S8 ')]
assert len(ops)==64
# Cross-check every source instruction against disassembler register numbers.
for pc,r in ops:
 assert len(r)==4
 assert [original[offset+pc+i] for i in (2,3,4,8)]==r,(pc,r)
variants={'baseline':(),'c_zero':(8,),'a_zero':(3,),'b_zero':(4,),'ab_zero':(3,4)}
meta=dict(entry=entry,source_sha256=hashlib.sha256(original).hexdigest(),variants={},caveat='Wrong arithmetic; all original wait/stall/reuse control bits remain. A/B input substitution does NOT remove encoded LDSM scoreboard waits.')
for name,fields in variants.items():
 binary=bytearray(original)
 for pc,r in ops:
  for field in fields:binary[offset+pc+field]=255
 changed=[i for i,(a,b) in enumerate(zip(original,binary)) if a!=b]
 assert len(changed)==64*len(fields)
 path=out/(name+'.cubin');path.write_bytes(binary)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True)
 (out/(name+'.sass')).write_text(dump)
 patched=next(x for x in dump.split('Function : ')[1:] if entry in x.splitlines()[0])
 got=re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',patched)
 assert len(got)==len(rows)
 for (pc,ins),(pc2,ins2) in zip(rows,got):
  assert pc==pc2
  if not ins.startswith('IMMA.'):assert ins==ins2
  else:
   regs=re.findall(r'\bR(?:\d+|Z)',ins2)
   origregs=re.findall(r'\bR(?:\d+|Z)',ins)
   for j,field in enumerate((2,3,4,8)):assert regs[j]==('RZ' if field in fields else origregs[j])
 meta['variants'][name]=dict(changed_bytes=len(changed),sha256=hashlib.sha256(binary).hexdigest(),operand_fields=list(fields))
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
