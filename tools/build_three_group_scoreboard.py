"""Remove stale load waits after early arrivals; optional separate BAR read scoreboards."""
from pathlib import Path
import struct,json,hashlib,subprocess,re
root=Path(__file__).resolve().parents[1];src=root/'artifacts/three-group-handoff';out=root/'artifacts/three-group-scoreboard';out.mkdir(exist_ok=True);m=json.loads((src/'manifest.json').read_text());result={'variants':{}}
for name,parent in [('native','native'),('three_early','three_early'),('clear_tail','three_early'),('split_read_sb','three_early')]:
 v=m['variants'][parent];base=(src/(parent+'.cubin')).read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256'];d=bytearray(base)
 sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+v['entry'])
 changes=[]
 def field(pc,bit,width,old,new):
  pos=off+pc;w=int.from_bytes(d[pos:pos+16],'little');assert (w>>bit)&((1<<width)-1)==old,(pc,bit)
  w=(w&~(((1<<width)-1)<<bit))|(new<<bit);d[pos:pos+16]=w.to_bytes(16,'little');changes.append(dict(pc=hex(pc),bit=bit,width=width,old=old,new=new))
 if name in ['clear_tail','split_read_sb']:
  # First arrival already waits for BOTH final LDSM writes. SB1 then tracks BAR ID reads.
  field(0x16f0,116,6,2,0);field(0x1730,116,6,4,0)
 if name=='split_read_sb':
  field(0x16e0,113,3,1,2) # Second BAR ID read uses SB2 instead of reusing SB1.
  field(0x2a20,116,6,0,4) # Wait for that read before toggling R158.
 path=out/(name+'.cubin');path.write_bytes(d)
 dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(dump)
 before=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(src/(parent+'.cubin'))],text=True)
 def instructions(s):return re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)
 assert instructions(dump)==instructions(before) # Only control words change.
 result['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'control_changes':changes}
 (out/(name+'.diff')).write_text(json.dumps(changes,indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
