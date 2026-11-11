"""Static value provenance for16 transcript REDUX blocks on frozen fastest kernel."""
from pathlib import Path
import json,re,hashlib
root=Path(__file__).resolve().parents[1];p=root/'artifacts/fold-preamble';out=root/'artifacts/fold-ready-window'
v=json.loads((p/'manifest.json').read_text())['variants']['both_compact'];assert hashlib.sha256((p/'both_compact.cubin').read_bytes()).hexdigest()==v['sha256']
s=(p/'both_compact.sass').read_text().split('Function : '+v['entry'])[1].split('Function : ')[0]
r={int(a,16):b for a,b in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;',s)}
last={};mma=[]
for pc,x in r.items():
 if 0x1090<=pc<0x17e0 and x.startswith('IMMA.'):
  dst=int(re.search(r' R(\d+),',x)[1]);mma.append(pc)
  for reg in range(dst,dst+4):last[reg]=pc
state={reg:({reg},set()) for reg in last};blocks=[]
for pc,x in r.items():
 if not 0x17f0<=pc<0x2320:continue
 regs=[int(n) for n in re.findall(r'\bR(\d+)\b',x)]
 if x.startswith('REDUX.XOR'):
  origins,ops=state[regs[0]]
  assert len(origins)==8,(hex(pc),origins)
  final=max(last[n] for n in origins)
  blocks.append(dict(redux=hex(pc),accumulators=sorted(origins),last_mma=hex(final),first_local_xor=hex(min(ops)),remaining_mma_after_producer=sum(p>final for p in mma),blocks_on=[{'pc':hex(p),'instruction':i} for p,i in r.items() if final<p<min(ops) and ('DEPBAR' in i or 'BAR.ARV' in i or 'BRA' in i)]))
 elif x.startswith('LOP3.LUT') and (', 0x96,' in x or ', 0x3c,' in x):
  origin=set();ops={pc}
  for reg in regs[1:]:
   o,chain=state.get(reg,(set(),set()));origin|=o;ops|=chain
  state[regs[0]]=(origin,ops)
 elif re.match(r'(?:@!?P\d+ )?(?:\w+\.)*\w+ R\d+,',x):
  if regs:state[regs[0]]=(set(),set())
assert len(blocks)==16,len(blocks)
lines=['Static provenance only: last MMA issue PC is NOT its result-ready cycle.', 'Block labels are REDUX traversal order,not source row/column indices.', 'block last_MMA first_XOR REDUX remaining_MMA accumulator_registers']
for i,b in enumerate(blocks):lines.append(f"{i:2} {b['last_mma']} {b['first_local_xor']} {b['redux']} {b['remaining_mma_after_producer']:2} {b['accumulators']}")
lines+=['All local XOR/REDUX starts after the final MMA in this frozen baseline.',
'For block0:producer1600;3localXOR1870,1890,18a0;finalXOR18c0;REDUX1940 (other transcript path2320).',
'Current experiment moves only the3localXOR across copy wait16c0;remaining REDUX and stores unchanged.',
'This is a narrow ready-work diagnostic,not a full transcript-warp or16-block interleave implementation.',
'Previously build_rank128_fold_local*.py and fold_interleave.py explored related source-level changes on older baselines;this frozen-SASS matched wait-boundary comparison is narrower.']
(out/'dependency-map.json').write_text(json.dumps(blocks,indent=2)+'\n');(out/'dependency-map.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
