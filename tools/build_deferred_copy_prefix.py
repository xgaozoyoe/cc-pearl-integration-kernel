"""Delay named handoff across next-K common prefix AND copy-address preparation."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'tools/build_deferred_prefix_v3.py').read_text().split('prefix=list(')[0].replace('artifacts/deferred-prefix-v3','artifacts/deferred-copy-prefix').replace('extra=0x400','extra=0x800')
exec(compile(s,'deferred_copy_helpers','exec'))
prefix=list(range(0xbb0,0xdc0,16));prep=list(range(0xdd0,0xf40,16));assert len(prefix)==33 and len(prep)==23
assert all(not any(op in r[pc] for op in ['BAR.','BRA','CALL','LDSM','LDGSTS','STS']) for pc in prefix)
assert all('LDS' not in r[pc] or r[pc]=='@!PT LDS RZ, [RZ]' for pc in prep)
assert all('R161' not in r[pc] for pc in prefix+prep)
assert r[0xf40].startswith('@!P4 LDGSTS') and r[0x1090].startswith('LDSM')
dummy=word(0x2a10)^(1<<83);packet=[word(0x2a00),dummy,word(0x2a20)]
m={'description':'Shared writes/reads remain after synchronization. Late sync spans33common+23copy-prep instructions; paths preserve final K copy skip and final exit.','variants':{}}
for name in ['native','matched_control','late_copy_sync']:
 d=bytearray(base) if name=='native' else expand()
 if name!='native':
  pos=size
  def emit(w):
   global pos
   pc=pos;put(d,pos,w);pos+=16;return pc
  put(d,0x2a00,branch(0x2a00,size))
  w=word(0x2a10);emit((w&~(31<<105))|(12<<105))
  w=word(0x3700);emit((w&~(31<<105))|(8<<105))
  end_guard=emit(word(0x3700))
  if name=='matched_control':
   for w in packet:emit(w)
  for pc in prefix:emit(word(pc))
  copy_guard=emit(word(0x3700))
  for pc in prep:emit(word(pc))
  if name=='late_copy_sync':
   for w in packet:emit(w)
  pc=pos;emit(branch(pc,0xf40))
  no_copy=pos
  if name=='late_copy_sync':
   for w in packet:emit(w)
  pc=pos;emit(branch(pc,0x1090))
  final=pos;emit(word(0x2a00));pc=pos;emit(branch(pc,0x2a10))
  put(d,end_guard,branch(end_guard,final,pred=10)) # !P2: old iteration was final
  put(d,copy_guard,branch(copy_guard,no_copy,pred=2)) # P2: next iteration has no future copy
  assert pos<=size+extra
  assert d[off:off+0x2a00]==base[off:off+0x2a00] and d[off+0x2a10:cut]==base[off+0x2a10:cut]
 path=out/(name+'.cubin');path.write_bytes(d);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump)
 assert [x for x in rr.values() if 'IMMA.' in x]==[x for x in r.values() if 'IMMA.' in x]
 if name!='native':
  assert rr[copy_guard]==f'@P2 BRA {hex(no_copy)}'
  assert rr[end_guard]==f'@!P2 BRA {hex(final)}'
  assert all(rr[pc]==x for pc,x in r.items() if pc!=0x2a00)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{p:x} {x}\n' for p,x in r.items()],[f'{p:x} {x}\n' for p,x in rr.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
