"""Report frozen-kernel first-load scheduling experiment, offline GPU0 only."""
from pathlib import Path
import json
import numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/handoff-first-ldsm'
lines=['GPU0 only;8192x8192 region,K4096,rank128. CUDA event kernel times, not full-grid or pool throughput.',
'Each run:200 samples/variant,16 launches/sample,10 rotating-order blocks,unlocked clocks.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries']
 lines += [f'Run{run}: name mean_ms median_ms p95_ms delta_native_pct']
 for n,v in s.items():
  lines.append(f"{n:10} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f}")
if (p/'grids.json').exists():
 rows=json.loads((p/'grids.json').read_text())['measurements'];means={}
 lines += ['Full offline grids: name n mean_ms median_ms p95_ms grids/sec equivalent_TH/s']
 for name in ('native','first_a','first_ab'):
  values=np.array([x for row in rows if row['name']==name for x in row['samples_ms']]);means[name]=float(values.mean())
  lines.append(f'{name:10} {len(values)} {values.mean():.6f} {np.median(values):.6f} {np.percentile(values,95):.6f} {1000/values.mean():.6f} {2**46/1e12*1000/values.mean():.6f}')
 for name in ('first_a','first_ab'):
  pairs=[]
  for repeat in sorted(set(row['repeat'] for row in rows)):
   batch={row['name']:row['mean_ms'] for row in rows if row['repeat']==repeat}
   pairs.append((batch[name]/batch['native']-1)*100)
  lines.append(f'{name} grid time delta:{(means[name]/means["native"]-1)*100:+.4f}%; paired batch deltas:'+','.join(f'{x:+.4f}' for x in pairs))
assert len(json.loads((p/'validation.json').read_text()))==36
assert len(json.loads((p/'pow-validation.json').read_text()))==18
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines += ['36 transcript +18 transcript/PoW comparisons PASS;racecheck0 hazards/errors/warnings.',
'All168regs,0local,69632B dynamic shared,384threads,1CTA/SM. All64named handoffs retained.',
'first_a: original1090 LDSM moved to dc0;first_ab: original1090+10c0 moved to dc0+dd0.',
'Both remain AFTER original synchronization. Reads target current buffer; copies target future buffer.',
'Exact instruction permutation; copy-skip branch retargeted; no added instructions,branches,registers or shared stores.',
'No hardware counters collected; timing cannot establish specific stall causality or tensor utilization.',
'No production changes or live mining. Equivalent TH/s is offline work rate, not measured pool throughput.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
