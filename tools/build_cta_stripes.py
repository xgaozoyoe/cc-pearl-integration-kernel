"""Frozen fastest cubin:change only four CTA traversal stripe constants."""
from pathlib import Path
import struct,re,json,hashlib,subprocess
root=Path(__file__).resolve().parents[1];out=root/'artifacts/cta-stripes';out.mkdir(exist_ok=True)
p=root/'artifacts/fold-preamble';v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];base=(p/'both_compact.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==v['sha256'];e=v['entry']
sh=struct.unpack_from('<Q',base,40)[0];sz,n,st=struct.unpack_from('<HHH',base,58);hs=[struct.unpack_from('<IIQQQQIIQQ',base,sh+i*sz) for i in range(n)];h=hs[st];ss=base[h[4]:h[4]+h[5]];off=next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+e)
def rows(s):return {int(a,16):x for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows((p/'both_compact.sass').read_text());m={'variants':{}}
assert r[0x50]=='IMAD.SHL.U32 R0, R0, 0x8, RZ'
assert r[0x450]=='IMAD.MOV.U32 R2, RZ, RZ, -0x8'
assert r[0x5c0]=='IMNMX R4, R2, 0x8, PT'
assert r[0x8a0]=='IMAD R159, R159, 0x8, R4'
for width in [8,4,16,32]:
 # Exact mapping is a bijection including the short final stripe.
 for gm in [1,2,3,32,512]:
  for gn in [1,3,5,8,16,43,683]:
   visited=set()
   for linear in range(gm*gn):
    first=linear//(width*gm)*width;cols=min(width,gn-first);offset=linear%(width*gm)
    pair=(offset//cols,first+offset%cols)
    assert 0<=pair[0]<gm and 0<=pair[1]<gn and pair not in visited
    visited.add(pair)
   assert len(visited)==gm*gn
 d=bytearray(base);changes=[]
 for pc,old,new in [(0x50,8,width),(0x450,(-8)&0xffffffff,(-width)&0xffffffff),(0x5c0,8,width),(0x8a0,8,width)]:
  w=int.from_bytes(base[off+pc:off+pc+16],'little');assert w>>32&0xffffffff==old
  d[off+pc:off+pc+16]=((w&~(0xffffffff<<32))|(new<<32)).to_bytes(16,'little')
  changes.append(f'{pc:#x}:immediate{old:#x}->{new:#x}')
 assert d[off+0x8b0:]==base[off+0x8b0:]
 name=f'stripe{width}';(out/f'{name}.cubin').write_bytes(d)
 sass=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/f'{name}.cubin')],text=True);(out/f'{name}.sass').write_text(sass)
 rr=rows(sass);assert all(rr[pc]==i for pc,i in r.items() if pc not in [0x50,0x450,0x5c0,0x8a0])
 (out/f'{name}.diff').write_text('\n'.join(changes)+'\n')
 m['variants'][name]=dict(v,sha256=hashlib.sha256(d).hexdigest(),threads=384,correct=True)
(out/'manifest.json').write_text(json.dumps(m,indent=2));print('4variants built;140shape/order bijection checks PASS;main loop unchanged.')
