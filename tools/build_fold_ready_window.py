"""Frozen SM89 first-fold local work before/after copy wait; offline only."""
from pathlib import Path
s=Path(__file__).with_name('build_deferred_prefix_v3.py').read_text().split('prefix=list(')[0].replace('artifacts/deferred-prefix-v3','artifacts/fold-ready-window')
exec(compile(s,'fold_ready_helpers','exec'))
assert r[0x1600].startswith('IMMA.16832.S8.S8 R16,')
assert r[0x1870]=='LOP3.LUT R96, R34, R33, R32, 0x96, !PT'
assert r[0x1890]=='LOP3.LUT R96, R16, R35, R96, 0x96, !PT'
assert r[0x18a0]=='LOP3.LUT R96, R18, R17, R96, 0x96, !PT'
# R165 is dead from last prefix use through the fold. All other instructions
# unchanged; explicitly retain stalls and dependency controls.
assert not any('R165' in x for pc,x in r.items() if 0x1600<=pc<=0x18d0)
local=[]
for pc in (0x1870,0x1890,0x18a0):
 w=word(pc);assert (w>>16)&255==96
 w=(w&~(255<<16))|(165<<16)
 if pc!=0x1870:
  assert (w>>64)&255==96;w=(w&~(255<<64))|(165<<64)
 # Original first XOR had an intervening address instruction before its consumer.
 # Preserve at least that dependency spacing when making the chain contiguous.
 w=(w&~(31<<105))|(6<<105)
 local.append(w)
m={'description':'Same fastest native cubin.3 local XOR before vs after cp.wait, scratch existing deadR165. Control shares branch/NOP overhead. Not a complete early REDUX experiment.','variants':{}}
for name in ('native','after_wait','before_wait'):
 d=bytearray(base) if name=='native' else expand()
 if name!='native':
  put(d,0x16c0,branch(0x16c0,size))
  seq=([word(0x16c0)]+local) if name=='after_wait' else (local+[word(0x16c0)])
  for i,w in enumerate(seq):put(d,size+i*16,w)
  put(d,size+len(seq)*16,branch(size+len(seq)*16,0x16d0))
  for pc in (0x1870,0x1890,0x18a0):
   mask=((1<<17)-1)<<105
   put(d,pc,(word(0x3700)&~mask)|(word(pc)&mask))
  w=word(0x18c0);assert (w>>32)&255==96
  put(d,0x18c0,(w&~(255<<32))|(165<<32))
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump)
 if name!='native':
  assert rr[0x18c0]=='LOP3.LUT R137, R19, R165, RZ, 0x3c, !PT'
  assert sum('R165' in rr[size+i*16] for i in range(4))==3
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:x} {x}\n' for pc,x in r.items()],[f'{pc:x} {x}\n' for pc,x in rr.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print('Built',list(m['variants']))
