"""Exact 16-byte producer coverage of baseline 16x32 LDSM fragments."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
source=(root/'tools/analyze_stage_ownership.py').read_text().split('reads = []')[0]
source=source.replace("Path(__file__).resolve().parents[1]",'root')
exec(compile(source,'ownership_helpers','exec'))
rows=[]
for warp in range(12):
 for operand,start in [('A',(warp%4)*64),('B',(warp//4)*64)]:
  for fragment in range(4):
   for half in range(2):
    keys={(operand,row,col) for row in range(start+16*fragment,start+16*(fragment+1)) for col in range(32*half,32*(half+1),16)}
    producers=sorted({owners[k] for k in keys})
    assert len(keys)==32 and len(producers)==2
    rows.append(dict(warp=warp,operand=operand,fragment=fragment,k_half=half,producers=producers,local_wait_sufficient=producers==[warp]))
assert not any(r['local_wait_sufficient'] for r in rows)
out=root/'artifacts/handoff-first-ldsm'
(out/'ownership.json').write_text(json.dumps(rows,indent=2)+'\n')
print('All192 warp/operand/fragment/K-half combinations need2 producer warps; none is certified by local cp.wait alone.')
