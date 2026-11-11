"""Fixed-SASS transcript ablation of current validated8+4; offline only."""
from pathlib import Path
import re,json,struct,hashlib,subprocess,difflib
root=Path(__file__).resolve().parents[1];out=root/'artifacts/transcript-current-ab';out.mkdir(exist_ok=True)
ref=root/'artifacts/handoff-groups';meta=json.loads((ref/'manifest.json').read_text());entry=meta['variants']['group8']['entry'];base=(ref/'group8.cubin').read_bytes();assert hashlib.sha256(base).hexdigest()==meta['variants']['group8']['sha256']
def textoff(b,name):
 sh=struct.unpack_from('<Q',b,40)[0];sz,n,st=struct.unpack_from('<HHH',b,58);hs=[struct.unpack_from('<IIQQQQIIQQ',b,sh+i*sz) for i in range(n)];h=hs[st];ss=b[h[4]:h[4]+h[5]]
 return next(h[4] for h in hs if ss[h[0]:].split(b'\0')[0].decode()=='.text.'+name)
off=textoff(base,entry)
def rows(s):
 s=s.split('Function : '+entry)[1].split('Function : ')[0]
 return {int(a,16):x for a,x in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
r=rows((ref/'group8.sass').read_text())
assert r[0x17e0]=='@!P2 BRA 0x2a00';assert r[0x2a50]=='BAR.SYNC.DEFER_BLOCKING 0x0'
# Fold doesn't define architectural accumulators or persistent iteration state.
assert not any(re.search(r'\bR120\b',x.split(',',1)[0]) for pc,x in r.items() if 0x17f0<=pc<0x2a00)
# Sink snippet independently compiled using identical argument layout, no branches
# before EXIT. Replace CLOCK read with final physical accumulatorR120.
subprocess.run(['/usr/local/cuda-12.6/bin/nvcc','-O3','-arch=sm_89','-cubin',str(out/'sink.cu'),'-o',str(out/'sink.cubin')],check=True)
sink=(out/'sink.cubin').read_bytes();so=textoff(sink,'sink');snippet=bytearray(sink[so:so+0xc0])
idpc=next(pc for pc,x in r.items() if x=='LOP3.LUT R164, R161, 0x1, RZ, 0x3c, !PT')
mv=bytearray(base[off+idpc:off+idpc+16]);assert mv[2]==164 and mv[3]==161
mv[2]=4;mv[3]=120
w=int.from_bytes(mv,'little');assert (w>>32)&0xffffffff==1
w &= ~(0xffffffff<<32) # R120 XOR0 = R120
w=(w&~(31<<105))|(8<<105);snippet[0x20:0x30]=w.to_bytes(16,'little')
manifest={'entry':entry,'variants':{},'warning':'Diagnostic-only. Sink modes produce no Pearl transcript/proof; never mine.'}
for name in ('full','fold_sink','gemm_sink'):
 data=bytearray(base)
 if name!='full':data[off+0x2a60:off+0x2a60+len(snippet)]=snippet
 if name=='gemm_sink':
  # Same branch target, now unconditional. All64IMMA,16LDSM,copy,wait,
  # BAR.ARV and final BAR.SYNC instruction words unchanged.
  pos=off+0x17e0;w=int.from_bytes(data[pos:pos+16],'little');assert w>>12&15==10
  data[pos:pos+16]=((w&~(15<<12))|(7<<12)).to_bytes(16,'little')
 p=out/(name+'.cubin');p.write_bytes(data);dump=subprocess.check_output(['/usr/local/cuda-12.6/bin/cuobjdump','--dump-sass',str(p)],text=True);(out/(name+'.sass')).write_text(dump);rr=rows(dump)
 assert sum('IMMA.' in x for x in rr.values())==64
 for pc,x in r.items():
  if pc<0x2a60 and pc!=0x17e0:assert data[off+pc:off+pc+16]==base[off+pc:off+pc+16]
 if name!='full':assert rr[0x2a80]=='LOP3.LUT R4, R120, 0x0, RZ, 0x3c, !PT' and rr[0x2b10]=='EXIT'
 if name=='gemm_sink':assert rr[0x17e0]=='BRA 0x2a00'
 (out/(name+'.diff')).write_text(''.join(difflib.unified_diff([f'{pc:04x} {x}\n' for pc,x in r.items()],[f'{pc:04x} {x}\n' for pc,x in rr.items()],fromfile='native8',tofile=name)))
 manifest['variants'][name]={'entry':entry,'sha256':hashlib.sha256(data).hexdigest(),'smem':69632,'correct':name=='full','sink':name!='full'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
