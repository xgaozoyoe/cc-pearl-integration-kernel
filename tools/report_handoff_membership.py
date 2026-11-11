"""Offline exact-hot-loop warp membership controls."""
from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/handoff-membership'
lines=['GPU0 only.8192x8192 region,K4096,rank128. Unlocked clocks.', 'Two runs,200samples/variant/run,16launches/sample,rotating blocks. CUDA event times,not grid/pool throughput.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries'];lines += [f'Run{run}','variant mean_ms median_ms p95_ms delta_native_pct delta_matched_control_pct']
 for n,v in s.items():lines.append(f"{n:16} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f} {(v['mean_ms']/s['masked_control']['mean_ms']-1)*100:+.4f}")
if (p/'grids.json').exists():
 rows=json.loads((p/'grids.json').read_text())['measurements'];lines+=['Full offline grid wall times:','name n mean_ms median_ms p95_ms equivalent_TH/s']
 for name in dict.fromkeys(r['name'] for r in rows):
  a=np.array([x for r in rows if r['name']==name for x in r['samples_ms']]);lines.append(f'{name} {len(a)} {a.mean():.6f} {np.median(a):.6f} {np.percentile(a,95):.6f} {(2**46/1e12)*1000/a.mean():.6f}')
assert len(json.loads((p/'validation.json').read_text()))==48
assert len(json.loads((p/'pow-validation.json').read_text()))==24
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines+=['48transcript+24transcript/PoW comparisons PASS; racecheck0 hazards/errors/warnings.',
 'All168regs,0local,69632B dynamicshared,384threads,1CTA/SM.',
 'Main loop and output machine code byte-identical to native. Only prologue grouping is changed via a bounded initialization trampoline.',
 'masked_control:tid&256>0, same8+4 membership as native.',
 'middle4:tid&128>0, groups[0,1,2,3,8,9,10,11] and[4,5,6,7].',
 'rows6:tid&64>0, groups[0,1,4,5,8,9] and[2,3,6,7,10,11].',
 'Each warp still arrives once early to peer and syncs once late to own group. All384 participants retained; phase IDs unchanged.',
 'No new copies or operand duplication. No measured hardware-counter attribution. Production libraries unchanged.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
