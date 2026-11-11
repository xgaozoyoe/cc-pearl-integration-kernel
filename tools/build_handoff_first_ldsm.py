"""Move ready-stage LDSM ahead of future-copy preparation, offline SM89 only.
All named barriers and original instructions retained; no production writes.
"""
from pathlib import Path
s=(Path(__file__).with_name('build_deferred_prefix_v3.py')).read_text().split('prefix=list(')[0].replace('artifacts/deferred-prefix-v3','artifacts/handoff-first-ldsm')
exec(compile(s,'first_ldsm_helpers','exec'))
start,end=0xdc0,0x10d0
original=list(range(start,end,16))
m={'description':'Same frozen baseline, instruction count, registers, copies, MMA and barriers. First A or first A/B LDSM moved ahead of copy preparation AFTER original handoff.','variants':{}}
for name,moved in [('native',[]),('first_a',[0x1090]),('first_ab',[0x1090,0x10c0])]:
 order=moved+[pc for pc in original if pc not in moved]
 mapping={old:start+i*16 for i,old in enumerate(order)}
 d=bytearray(base)
 for old,pc in mapping.items():
  w=word(old)
  if old==0xdc0 and moved:
   # Early LDSM already executed on both paths. Skip only global copy.
   target=mapping[0x10a0]
   w=(w&~(((1<<50)-1)<<32))|(((target-pc-16)&((1<<50)-1))<<32)
  put(d,pc,w)
 assert sorted(order)==original
 assert d[:off+start]==base[:off+start] and d[off+end:]==base[off+end:]
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump)
 for old,pc in mapping.items():
  assert rr[pc]==(f'@P2 BRA {hex(mapping[0x10a0])}' if old==0xdc0 and moved else r[old])
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{p:x} {x}\n' for p,x in r.items()],[f'{p:x} {x}\n' for p,x in rr.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True,'mapping':{hex(k):hex(v) for k,v in mapping.items()}}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('Built',list(m['variants']))
