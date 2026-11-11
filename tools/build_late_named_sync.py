"""Actually move named sync after33address instructions, with matched control."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'tools/build_deferred_prefix_v3.py').read_text();s=s[:s.index("for name in ['native'")].replace('artifacts/deferred-prefix-v3','artifacts/late-named-sync')
exec(compile(s,'late_named_helpers','exec'))
assert not any(re.search(r'\bP6\b',x) for x in r.values())
# Preserve BAR->independent instruction->ID toggle scheduling without clobbering next-copy P2.
dummy=word(0x2a10)^(1<<83)
body=size+48;end=body+(33+3)*16;final=end+16
m={'variants':{},'description':'Same33prefix instructions and barrier packet, reordered only. All rendezvous retained; old ID consumed before toggle. P6 dummy prevents copy-path P2 clobber.'}
for name in ['native','matched_control','late_sync']:
 d=bytearray(base) if name=='native' else expand()
 if name!='native':
  put(d,0x2a00,branch(0x2a00,size))
  w=word(0x2a10);put(d,size,(w&~(31<<105))|(12<<105))
  w=word(0x3700);put(d,size+16,(w&~(31<<105))|(8<<105))
  put(d,size+32,branch(size+32,final,pred=10))
  packet=[word(0x2a00),dummy,word(0x2a20)]
  seq=packet+[word(pc) for pc in prefix] if name=='matched_control' else [word(pc) for pc in prefix]+packet
  for j,w in enumerate(seq):put(d,body+j*16,w)
  put(d,end,branch(end,0xdc0));put(d,final,word(0x2a00));put(d,final+16,branch(final+16,0x2a10))
  assert d[off:off+0x2a00]==base[off:off+0x2a00] and d[off+0x2a10:cut]==base[off+0x2a10:cut]
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump)
 assert [x for x in rr.values() if 'IMMA.' in x]==[x for x in r.values() if 'IMMA.' in x]
 if name!='native':
  pp=body+48 if name=='matched_control' else body;bp=body if name=='matched_control' else body+33*16
  assert [rr[pp+j*16] for j in range(33)]==[r[pc] for pc in prefix]
  assert rr[bp+16]=='PLOP3.LUT P6, PT, PT, PT, UP1, 0x80, 0x0'
  assert rr[bp]==r[0x2a00] and rr[bp+32]==r[0x2a20]
  assert rr[end]=='BRA 0xdc0' and rr[final+16]=='BRA 0x2a10'
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in rr.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('Built',list(m['variants']))
