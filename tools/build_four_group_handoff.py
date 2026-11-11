"""Offline four-group protocol; preserve production and operand mapping."""
from pathlib import Path
import subprocess,re,json,hashlib,difflib,struct
root=Path(__file__).resolve().parents[1];out=root/'artifacts/four-group-handoff';out.mkdir(exist_ok=True)
s=(root/'artifacts/fold-defer-reduce/source.cu').read_text();a=s.index('namespace fold_defer_source8');b=s.index('} // namespace fold_defer_source8',a);base=s[a:b];m={'variants':{}}
for name,parent in [('native','native'),('source_control','source_control')]:
 p=root/'artifacts/three-group-handoff';v=json.loads((p/'manifest.json').read_text())['variants'][parent];blob=(p/(parent+'.cubin')).read_bytes();assert hashlib.sha256(blob).hexdigest()==v['sha256'];(out/(name+'.cubin')).write_bytes(blob);m['variants'][name]=v
for name,group in [('contiguous4','threadIdx.x/96'),('interleaved4','(threadIdx.x/32)%4')]:
 new=base.replace('    unsigned own_bar=4u+unsigned(threadIdx.x>=256);',f'''    unsigned own_bar=4u+({group});
    unsigned peer0=own_bar^1u,peer1=own_bar^2u,peer2=own_bar^3u;
    // IDs4..7 and12..15 alternate. Each warp early-arrives to three peers;
    // own late sync plus all peer early arrivals totals384 participants.''')
 new=new.replace('        const unsigned peer_bar=own_bar^1u;','')
 new=new.replace('        asm volatile("bar.arrive %0, 384;" :: "r"(peer_bar) : "memory");','        asm volatile("bar.arrive %0, 384; bar.arrive %1, 384; bar.arrive %2, 384;" :: "r"(peer0), "r"(peer1), "r"(peer2) : "memory");')
 new=new.replace('        own_bar^=2u;','        own_bar^=8u; peer0^=8u; peer1^=8u; peer2^=8u;')
 new=new.replace('fold_defer_source8','four_handoff_'+name)
 new+='\ntemplate __global__ void kernel<1,64,false,true,false,false,2,false,6>(const signed char*,const signed char*,int,int,int,int,unsigned*);\n}\n'
 (out/(name+'.cu')).write_text('#include <cuda_runtime.h>\n#include <cstdint>\n'+new)
 (out/(name+'.source.diff')).write_text(''.join(difflib.unified_diff(base.splitlines(True),new.splitlines(True),fromfile='source8',tofile=name)))
 result=subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-std=c++20','-O3','-lineinfo','-arch=sm_89','--cubin','--ptxas-options=-v',str(out/(name+'.cu')),'-o',str(out/(name+'-late.cubin'))],capture_output=True,text=True)
 (out/(name+'.build.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr;print(name,result.stderr,flush=True)
 blob=(out/(name+'-late.cubin')).read_bytes();dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/(name+'-late.cubin'))],text=True);(out/(name+'-late.sass')).write_text(dump)
 e=re.search(r'Function : (\S+)',dump)[1];m['variants'][name]=dict(entry=e,sha256=hashlib.sha256(blob).hexdigest(),smem=69632,correct=True)
 # Early-arrival patch is applied after inspecting the emitted tail; do not
 # benchmark these late files as the early-publication candidates.
(out/'compiled-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
# Use the spill-free contiguous binary for BOTH memberships.
v=m['variants']['contiguous4'];blob=(out/'contiguous4-late.cubin').read_bytes();e=v['entry']
sh=struct.unpack_from('<Q',blob,40)[0];sz,n,st=struct.unpack_from('<HHH',blob,58);hs=[struct.unpack_from('<IIQQQQIIQQ',blob,sh+i*sz) for i in range(n)];h=hs[st];ss=blob[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(pc,16):x for pc,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
r=rows((out/'contiguous4-late.sass').read_text())
def w(pc):return int.from_bytes(blob[off+pc:off+pc+16],'little')
def put(d,pc,x):d[off+pc:off+pc+16]=x.to_bytes(16,'little')
first=next(pc for pc,x in r.items() if x.startswith('BAR.ARV'));start=max(pc for pc,x in r.items() if pc<first and x.startswith('DEPBAR.'))+16
assert first==0x17a0 and start==0x16a0
assert all(r[pc].startswith('IMMA.') for pc in range(start,first,16))
loads=[pc for pc,x in r.items() if pc<start and x.startswith('LDSM.')][-2:];mask=0
for pc in loads:mask|=1<<((w(pc)>>110)&7)
assert mask==6
order=[first,first+16,first+32]+list(range(start,first,16));d=bytearray(blob)
for j,old in enumerate(order):
 x=w(old)
 if j==0:x|=mask<<116
 # The first arrival already retires both load scoreboards. Do not make
 # tail MMA wait on their subsequent reuse for arrival operand reads.
 if old<first:
  waits=(x>>116)&63;assert waits&~mask==0;x&=~(63<<116)
 put(d,start+16*j,x)
early=bytes(d)
for name in ['contiguous4','interleaved4']:
 d=bytearray(early)
 if name=='interleaved4':
  padding=[pc for pc,x in r.items() if x=='NOP'];pad=padding[-4];assert padding[-4:]==list(range(pad,pad+64,16))
  brpc=next(pc for pc,x in r.items() if x.startswith('BRA '))
  def branch(pc,target):return (w(brpc)&~(((1<<50)-1)<<32))|(((target-pc-16)&((1<<50)-1))<<32)
  def scalar(x):return (x&~(0x1ffff<<105))|((6|(7<<5)|(7<<8))<<105)
  assert r[0x9f0].startswith('LEA.HI R161')
  put(d,0x9f0,branch(0x9f0,pad))
  x=w(0xc70);assert r[0xc70]=='SHF.R.U32.HI R96, RZ, 0x5, R162';x=(x&~(255<<16))|(161<<16);put(d,pad,scalar(x))
  lp=next(pc for pc,x in r.items() if x.startswith('LOP3.LUT') and ', RZ, 0xc0, !PT' in x)
  x=w(lp);x=(x&~((255<<16)|(255<<24)|(0xffffffff<<32)))|(161<<16)|(161<<24)|(3<<32);put(d,pad+16,scalar(x))
  # Known immediate IADD3 encoding from frozen native (same SM89 ISA).
  nb=(out/'native.cubin').read_bytes();nm=m['variants']['native'];sh2=struct.unpack_from('<Q',nb,40)[0];sz2,n2,st2=struct.unpack_from('<HHH',nb,58);hh=[struct.unpack_from('<IIQQQQIIQQ',nb,sh2+i*sz2) for i in range(n2)];tt=nb[hh[st2][4]:hh[st2][4]+hh[st2][5]];no=next(z[4] for z in hh if tt[z[0]:].split(b'\0')[0].decode()=='.text.'+nm['entry'])
  x=int.from_bytes(nb[no+0x1570:no+0x1580],'little');x=(x&~((255<<16)|(255<<24)|(0xffffffff<<32)))|(161<<16)|(161<<24)|(4<<32);put(d,pad+32,scalar(x));put(d,pad+48,branch(pad+48,0xa00))
  assert d[off+0xa00:off+pad]==early[off+0xa00:off+pad]
 path=out/(name+'.cubin');path.write_bytes(d);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);rr=rows(dump);(out/(name+'.sass')).write_text(dump)
 assert sum(x.startswith('IMMA.') for x in rr.values())==64
 assert sum(x.startswith('BAR.ARV') for x in rr.values())==3
 if name=='interleaved4':
  assert rr[pad]=='SHF.R.U32.HI R161, RZ, 0x5, R162'
  assert rr[pad+16]=='LOP3.LUT R161, R161, 0x3, RZ, 0xc0, !PT',rr[pad+16]
  assert rr[pad+32].replace('.reuse','')=='IADD3 R161, R161, 0x4, RZ',rr[pad+32]
 (out/(name+'.sass.diff')).write_text(''.join(difflib.unified_diff([f'{p:x} {x}\n' for p,x in r.items()],[f'{p:x} {x}\n' for p,x in rr.items()],fromfile='compiled-late',tofile=name)))
 m['variants'][name]={**v,'sha256':hashlib.sha256(d).hexdigest(),'correct':True,'publication_wait_mask':mask}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
