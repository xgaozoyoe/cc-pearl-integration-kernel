"""Offline fixed-SASS diagnostics. C/C0 are deliberately invalid Pearl arithmetic."""
from pathlib import Path
import struct,re,json,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/handoff-abcd'
out.mkdir(exist_ok=True)
if not (out/'handoff-lean-source.sm_89.cubin').exists():
 subprocess.run(['/usr/local/cuda-12.6/bin/cuobjdump','-xelf','all','../libpearl-handoff-lean.so'],cwd=out,check=True)
binary=(out/'handoff-lean-source.sm_89.cubin').read_bytes()
assert hashlib.sha256(binary).hexdigest()=='46fd8653295a23a2215aaad91dbfdf41c98fe87359af09201dfbcb21f1687bad', 'Unexpected input; audit patch offsets again'
shoff=struct.unpack_from('<Q',binary,40)[0];size,count,strings=struct.unpack_from('<HHH',binary,58)
hs=[struct.unpack_from('<IIQQQQIIQQ',binary,shoff+i*size) for i in range(count)];h=hs[strings];names=binary[h[4]:h[4]+h[5]]
secs={names[h[0]:].split(b'\0')[0].decode():h for h in hs}
dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(out/'handoff-lean-source.sm_89.cubin')],text=True)
def entry(ns):return next(n[6:] for n in secs if n.startswith('.text.') and ns+'6kernel' in n)
aentry=entry('handoff_lean_baseline');bentry=entry('handoff_lean_toggle');offset=secs['.text.'+bentry][4]
def rows(s,e):return {int(pc,16):ins for pc,ins in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s.split('Function : '+e)[1].split('Function : ')[0])}
r=rows(dump,bentry)
def word(data,pc):return int.from_bytes(data[offset+pc:offset+pc+16],'little')
def put(data,pc,w):data[offset+pc:offset+pc+16]=w.to_bytes(16,'little')
def ctrl(pc):return (word(binary,pc)>>105)&0x1ffff
immas=[pc for pc,s in r.items() if s.startswith('IMMA.')];assert len(immas)==64
nop=next(word(binary,pc) for pc,s in r.items() if s=='NOP')
zero=bytearray(binary)
for pc in immas:
 regs=re.findall(r'\bR(\d+|Z)',r[pc]);assert len(regs)==4
 for i,b in enumerate((2,3,4,8)):assert zero[offset+pc+b]==(255 if regs[i]=='Z' else int(regs[i]))
 zero[offset+pc+4]=255
zr=dict(r)
for pc in immas:zr[pc]=', '.join(part if i!=2 else re.sub(r'R\d+', 'RZ', part) for i,part in enumerate(r[pc].split(', ')))
moves=[(0x11d0,0x11e0,0x1220),(0x1280,0x1290,0x12d0),(0x1340,0x1350,0x1410)]
c=bytearray(zero)
for producer,first,last in moves:
 assert r[producer].startswith('LDSM.16.M88.4 R148,') and (ctrl(producer)>>5)&7==1
 assert ctrl(first)>>11==2 and ctrl(last)>>11==0
 for pc in range(first,last,16):
  s=zr[pc];ct=ctrl(pc);assert (ct>>5)&7==7 and (ct>>8)&7==7
  assert not any(x in s for x in ('BRA','BAR.','LDSM','LDG','STS','STG'))
  regs=set(map(int,re.findall(r'\bR(\d+)',s)))
  if s.startswith('IMMA.'):regs|={x+j for x in tuple(regs) for j in range(4)}
  assert not regs.intersection(range(148,152)),(hex(pc),s)
 assert re.match(r'(SHF\S*|IADD3|LOP3\S*) R14[89],',r[last])
 for pc,mask in ((first,0),(last,2)):put(c,pc,(word(c,pc)&~(63<<116))|(mask<<116))
d=bytearray(binary);removed=[0xe40,0xe60,0xe80]
for pc in removed:
 assert r[pc].startswith(('ISETP.','IADD3'))
 mask=((1<<17)-1)<<105;put(d,pc,(nop&~mask)|(word(d,pc)&mask))
for pred,pcs in [('!P4',[0xeb0,0xec0,0xed0,0xf40,0xf70]),('!P2',[0xef0,0xf00,0xf50,0xfb0,0x1000])]:
 for pc in pcs:
  assert r[pc].startswith('@'+pred+' ')
  w=word(d,pc);assert (w>>12)&15==8+int(pred[-1]);put(d,pc,(w&~(15<<12))|(7<<12))
manifest=dict(warning='C and C0 have intentionally wrong arithmetic; never use for mining. C relocates only three B scoreboard waits. D keeps NOP slots and fixed stalls.',source_sha256=hashlib.sha256(binary).hexdigest(),variants={},wait_moves=moves)
for name,data,e in [('A',binary,aentry),('B',binary,bentry),('C0',zero,bentry),('C',c,bentry),('D',d,bentry)]:
 path=out/(name+'.cubin');path.write_bytes(data)
 s=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(path)],text=True);(out/(name+'.sass')).write_text(s)
 rr=rows(s,e);assert sum(x.startswith('IMMA.') for x in rr.values())==64
 if name in ('C0','C'):
  for pc in r:
   if pc not in immas:assert rr[pc]==r[pc]
   else:assert rr[pc]==zr[pc],(rr[pc],zr[pc])
 if name=='D':
  for pc in removed:assert rr[pc]=='NOP'
  for pc in r:
   if pc not in removed:
    expected=r[pc]
    if pc in [0xeb0,0xec0,0xed0,0xf40,0xf70,0xef0,0xf00,0xf50,0xfb0,0x1000]:
     expected=expected.removeprefix('@!P4 ').removeprefix('@!P2 ')
    assert rr[pc]==expected
 manifest['variants'][name]=dict(entry=e,sha256=hashlib.sha256(data).hexdigest(),smem=69632,correct=name in ('A','B','D'),static_imma=64,static_ldsm=sum(x.startswith('LDSM.') for x in rr.values()))
 (out/(name+'.diff')).write_text('\n'.join(difflib.unified_diff([f'{pc:x}: {s}' for pc,s in r.items()],[f'{pc:x}: {s}' for pc,s in rr.items()],fromfile='B',tofile=name))+'\n')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Built A/B/C0/C/D; IMMA and instruction preservation verified.')
