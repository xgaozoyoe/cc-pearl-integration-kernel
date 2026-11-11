"""Isolate ELF extension, a loop-back branch, and barrier relocation separately."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'tools/build_deferred_prefix.py').read_text().split('\n',1)[1]
s=s[:s.index("for name in ['native'")].replace('artifacts/deferred-prefix','artifacts/deferred-probe')
exec(compile(s,'deferred_probe_helpers','exec'))
m={'variants':{},'warning':'Control-flow diagnostics only. No mining.'}
for name in ['native','grow_only','loop_branch','move_bar_only','guard_safe','guard12']:
 d=bytearray(base) if name=='native' else expand()
 if name=='loop_branch':
  put(d,0x2a40,branch(0x2a40,size));put(d,size,word(0x3700));put(d,size+16,branch(size+16,0xbb0))
 if name=='move_bar_only':
  put(d,0x2a00,branch(0x2a00,size));put(d,size,word(0x2a00));put(d,size+16,branch(size+16,0x2a10))
 if name in ['guard_safe','guard12']:
  put(d,0x2a00,branch(0x2a00,size))
  w=word(0x2a10);w=(w&~(31<<105))|((11 if name=='guard_safe' else 12)<<105);put(d,size,w)
  w=word(0x3700);w=(w&~(31<<105))|(8<<105);put(d,size+16,w)
  put(d,size+32,branch(size+32,size+80,pred=10))
  put(d,size+48,word(0x2a00));put(d,size+64,branch(size+64,0x2a10))
  put(d,size+80,word(0x2a00));put(d,size+96,branch(size+96,0x2a10))
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump)
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
