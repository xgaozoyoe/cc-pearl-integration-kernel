"""Frozen hot loop, mixed slow/early two-group membership computed only in prologue."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'tools/build_deferred_prefix_v3.py').read_text().split('prefix=list(')[0]
s=s.replace("out=root/'artifacts/deferred-prefix-v3'", "out=root/'artifacts/handoff-mixed'")
exec(compile(s,'elf_helpers','exec'))
m={'variants':{},'description':'Same frozen hot-loop bytes. Change only initialization grouping; masked_control keeps original8+4.'}
for name,mask in [('native',None),('masked_control',256),('pair01',192),('pair02',160)]:
 d=bytearray(base) if mask is None else expand()
 if mask is not None:
  put(d,0xa60,branch(0xa60,size))
  # R165 is dead here and is initialized again at0xbf0 before any use.
  w=word(0xbd0);w=(w&~(255<<16))|(165<<16);w=(w&~(0xffffffff<<32))|(mask<<32)
  w=(w&~(31<<105))|(6<<105);put(d,size,w) # R165 = tid & mask
  w=word(0xa60);w=(w&~(255<<24))|(165<<24);w&=~(0xffffffff<<32)
  put(d,size+16,w) # P2 = (R165 > 0)
  put(d,size+32,branch(size+32,0xa70))
  assert d[off+0xa70:cut]==base[off+0xa70:cut]
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump)
 if mask is not None:
  assert rr[size]==f'LOP3.LUT R165, R162, {hex(mask)}, RZ, 0xc0, !PT',rr[size]
  assert rr[size+16]=='ISETP.GT.U32.AND P2, PT, R165, 0x0, PT',rr[size+16]
  assert all(rr[pc]==x for pc,x in r.items() if pc!=0xa60)
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{p:x} {x}\n' for p,x in r.items()],[f'{p:x} {x}\n' for p,x in rr.items()],fromfile='native',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True,'groups':[[w for w in range(12) if not ((w*32)&(mask or 256))],[w for w in range(12) if (w*32)&(mask or 256)]]}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
