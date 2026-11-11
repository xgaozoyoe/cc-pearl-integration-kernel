"""Nonzero A-derived B operand diagnostic, fixed native SASS + separate source oracle."""
from pathlib import Path
import json,struct,re,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/nonzero-dependency';out.mkdir(exist_ok=True);src=root/'artifacts/fold-preamble';v=json.loads((src/'manifest.json').read_text())['variants']['both_compact'];e=v['entry'];base=(src/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((src/'both_compact.sass').read_text());ops=[pc for pc,x in r.items() if x.startswith('IMMA.')];assert len(ops)==64
moves=[(0x11e0,0x1220),(0x1290,0x12d0),(0x1350,0x1410)]
m={'warning':'Incorrect Pearl arithmetic except native. B operand uses current A fragment half, not zero. Keep original B memory work.','variants':{}}
for name in ('native','a_as_b','a_as_b_release'):
 data=bytearray(base);changes=[]
 if name!='native':
  for pc in ops:
   regs=list(map(int,re.findall(r'\bR(\d+)',r[pc])));assert [base[off+pc+i] for i in (2,3,4,8)]==regs
   assert regs[1]%4==0 and regs[2] in (148,150,152,154)
   data[off+pc+4]=regs[1]+(regs[2]&2)
 if name.endswith('release'):
  for first,last in moves:
   for pc,oldmask,newmask in ((first,2,0),(last,0,2)):
    w=int.from_bytes(data[off+pc:off+pc+16],'little');assert w>>116&63==oldmask;data[off+pc:off+pc+16]=((w&~(63<<116))|(newmask<<116)).to_bytes(16,'little');changes.append(dict(pc=hex(pc),old_wait=oldmask,new_wait=newmask))
 p=out/(name+'.cubin');p.write_bytes(data);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);got=rows(dump);assert got.keys()==r.keys()
 for pc,x in got.items():
  if pc not in ops or name=='native':assert x==r[pc]
  else:
   before=list(map(int,re.findall(r'\bR(\d+)',r[pc])));after=list(map(int,re.findall(r'\bR(\d+)',x)));assert after==[before[0],before[1],before[1]+(before[2]&2),before[3]]
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in got.items()],fromfile='native',tofile=name))+json.dumps(changes,indent=2)+'\n')
 m['variants'][name]={**v,'sha256':hashlib.sha256(data).hexdigest(),'correct':name=='native'}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();ns='fold_defer_source8';a=s.index('namespace '+ns);b=s.index('// namespace '+ns,a)+len('// namespace '+ns);s=s[a:b].replace(ns,'nonzero_oracle')
old='"r"(b.v[Packed?2*half:half]),"r"(b.v[Packed?2*half+1:half+2])';new='"r"(a.v[Packed?2*half:half]),"r"(a.v[Packed?2*half+1:half+2])';assert s.count(old)==1;s=s.replace(old,new)
s='#include <cuda_runtime.h>\n#include <cstdint>\n'+s+'\nextern "C" int oracle(const int8_t*a,const int8_t*b,int m,int n,unsigned*t){return nonzero_oracle::launch<1,64,false,true,false,false,2,false,6>(a,b,m,n,4096,128,t,nullptr,nullptr);}\n'
p=out/'oracle.cu';p.write_text(s)
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-arch=sm_89','-Xcompiler','-fPIC','--ptxas-options=-v','-shared',str(p),'-o',str(out/'liboracle.so')],check=True)
