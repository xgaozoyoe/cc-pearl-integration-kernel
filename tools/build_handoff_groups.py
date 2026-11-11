"""Offline cohort-size sweep. No production writes or hot-loop changes."""
from pathlib import Path
import struct, json, hashlib, subprocess, re, difflib
root=Path(__file__).resolve().parents[1]
src=root/'artifacts/handoff-arrive-id'; out=root/'artifacts/handoff-groups'; out.mkdir(exist_ok=True)
meta=json.loads((src/'manifest.json').read_text()); entry=meta['entry']
def rows(s):
 s=s.split('Function : '+entry)[1].split('Function : ')[0]
 return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
manifest={'entry':entry,'variants':{},'warning':'Offline only. Barrier cohorts change; MMA/data ownership unchanged.'}
for name,parent,split in [('tail9','tail9',6),('id16','id16',6),('group4','id16',4),('group8','id16',8),('group3','id16',3),('group9','id16',9)]:
 base=(src/(parent+'.cubin')).read_bytes(); assert hashlib.sha256(base).hexdigest()==meta['variants'][parent]['sha256']
 shoff=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58)
 hs=[struct.unpack_from('<IIQQQQIIQQ',base,shoff+i*sz) for i in range(n)]
 h=hs[st]; names=base[h[4]:h[4]+h[5]]
 off=next(h[4] for h in hs if names[h[0]:].split(b'\0')[0].decode()=='.text.'+entry)
 expected=rows((src/(parent+'.sass')).read_text()); data=bytearray(base)
 assert expected[0xa60]=='ISETP.GT.U32.AND P2, PT, R162, 0xbf, PT'
 # Immediate32 in bits32..63; only the warp-uniform thread threshold changes.
 pos=off+0xa60; w=int.from_bytes(base[pos:pos+16],'little'); assert (w>>32)&0xffffffff==191
 data[pos:pos+16]=((w&~(0xffffffff<<32))|((split*32-1)<<32)).to_bytes(16,'little')
 expected[0xa60]=expected[0xa60].replace('0xbf',hex(split*32-1))
 path=out/(name+'.cubin');path.write_bytes(data)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True)
 (out/(name+'.sass')).write_text(dump); got=rows(dump); assert got==expected
 assert data[:pos]==base[:pos] and data[pos+16:]==base[pos+16:]
 before=rows((src/(parent+'.sass')).read_text())
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{p:04x} {x}\n' for p,x in before.items()],[f'{p:04x} {x}\n' for p,x in got.items()],fromfile=parent,tofile=name)))
 manifest['variants'][name]={**meta['variants'][parent],'sha256':hashlib.sha256(data).hexdigest(),'groups':[split,12-split],'parent':parent}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
