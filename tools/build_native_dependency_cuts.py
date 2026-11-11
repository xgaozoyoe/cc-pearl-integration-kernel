"""Deliberately wrong arithmetic: cut IMMA inputs in exact fastest SASS, no compiler."""
from pathlib import Path
import json,struct,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/native-dependency-cuts';out.mkdir(exist_ok=True);src=root/'artifacts/fold-preamble';v=json.loads((src/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(src/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((src/'both_compact.sass').read_text());ops=[pc for pc,x in r.items() if x.startswith('IMMA.')];assert len(ops)==64
def control(pc):return int.from_bytes(base[off+pc:off+pc+16],'little')>>105&0x1ffff
moves=[(0x11d0,0x11e0,0x1220),(0x1280,0x1290,0x12d0),(0x1340,0x1350,0x1410)]
for prod,first,last in moves:
 assert r[prod].startswith('LDSM.16.M88.4 R148,') and (control(prod)>>5)&7==1
 assert control(first)>>11==2 and control(last)>>11==0
 for pc in range(first,last,16):
  x=r[pc];c=control(pc);assert (c>>5)&7==7 and (c>>8)&7==7
  if x.startswith('IMMA.'):
   regs=list(map(int,re.findall(r'\bR(\d+)',x)));reads=set(range(regs[1],regs[1]+4))|set(range(regs[3],regs[3]+4));writes=set(range(regs[0],regs[0]+4))
  else:
   assert not any(op in x for op in ('BRA','BAR.','LDSM','LDG','STS','STG'))
   regs=list(map(int,re.findall(r'\bR(\d+)',x)));writes=set(regs[:1]);reads=set(regs[1:])
  assert not (reads|writes)&set(range(148,152)),(hex(pc),x)
 assert re.match(r'(?:SHF\S*|IADD3|LOP3\S*) R14[89],',r[last])
variants=[('native',(),False),('a_zero',(3,),False),('b_zero',(4,),False),('ab_zero',(3,4),False),('b_release',(4,),True),('bc_zero',(4,8),False),('bc_release',(4,8),True)]
m={'warning':'INVALID PEARL arithmetic except native. Offline diagnostic only. RZ means literal zero; no compiler or initialization changes. Preserve all copy/load/MMA/output work and control flow.','variants':{},'moves':moves}
for name,fields,release in variants:
 data=bytearray(base);expected=dict(r);changes=[]
 for pc in ops:
  regs=list(map(int,re.findall(r'\bR(\d+)',r[pc])));assert [base[off+pc+i] for i in (2,3,4,8)]==regs
  for field in fields:data[off+pc+field]=255
 if release:
  for prod,first,last in moves:
   for pc,mask in ((first,0),(last,2)):
    w=int.from_bytes(data[off+pc:off+pc+16],'little');old=w>>116&63;w=(w&~(63<<116))|(mask<<116);data[off+pc:off+pc+16]=w.to_bytes(16,'little');changes.append(dict(pc=hex(pc),old_wait=old,new_wait=mask))
 p=out/(name+'.cubin');p.write_bytes(data);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump);assert got.keys()==r.keys()
 for pc,x in got.items():
  if pc not in ops:assert x==r[pc]
  else:
   oldregs=re.findall(r'\bR(?:\d+|Z)',r[pc]);newregs=re.findall(r'\bR(?:\d+|Z)',x)
   assert newregs==['RZ' if field in fields else oldregs[j] for j,field in enumerate((2,3,4,8))]
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native',tofile=name))+json.dumps(changes,indent=2)+'\n')
 m['variants'][name]={**v,'sha256':hashlib.sha256(data).hexdigest(),'correct':name=='native','zero_fields':fields,'wait_changes':changes}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
