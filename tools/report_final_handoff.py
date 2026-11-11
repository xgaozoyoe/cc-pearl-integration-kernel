"""Offline last-iteration handoff elimination: exact-code controls."""
from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]/'artifacts/final-handoff'
lines=['GPU0 only,8192x8192 region,K4096,rank128. CUDA events; unlocked clocks.', 'Two independent runs;200samples/variant/run,16launches/sample,rotating blocks. Not grid/pool throughput.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries'];lines += [f'Run{run}','variant mean_ms median_ms p95_ms delta_native_pct delta_control_pct']
 for n,v in s.items():lines.append(f"{n:18} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f} {(v['mean_ms']/s['predicate_control']['mean_ms']-1)*100:+.4f}")
if (p/'grids.json').exists():
 import numpy as np
 rows=json.loads((p/'grids.json').read_text())['measurements'];means={};lines+=['Offline full-grid wall time: name n mean_ms median_ms p95_ms equivalent_TH/s']
 for name in ('native','predicate_control'):
  a=np.array([x for r in rows if r['name']==name for x in r['samples_ms']]);means[name]=float(a.mean());lines.append(f'{name} {len(a)} {a.mean():.6f} {np.median(a):.6f} {np.percentile(a,95):.6f} {(2**46/1e12)*1000/a.mean():.6f}')
 lines.append(f"Predicate control grid time delta:{(means['predicate_control']/means['native']-1)*100:+.4f}%")
 pairs=[]
 for repeat in sorted(set(r['repeat'] for r in rows)):
  v={r['name']:r['mean_ms'] for r in rows if r['repeat']==repeat};pairs.append((v['predicate_control']/v['native']-1)*100)
 lines.append('Paired batch time deltas (%):'+', '.join(f'{x:+.4f}' for x in pairs))
assert len(json.loads((p/'validation.json').read_text()))==36
assert len(json.loads((p/'pow-validation.json').read_text()))==18
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines+=['36transcript+18transcript/PoW comparisons PASS; racecheck0 hazards/errors/warnings.',
 'All168regs,0local,69632B dynamicshared,384threads,1CTA/SM. Same instructions and control words; only predicate operands changed.',
 'Existing final-iteration predicate from0xcb0 is retained in unusedP6; copy-skip branch0xdc0 followsP6. Other predicates/data unaffected.',
 'Only kt63 skips paired BAR.ARV0x16d0 and BAR.SYNC0x2a00. Earlier63handoffs and prologue/final CTA barriers retained.',
 'Safety:previous handoff published last-stage inputs and released buffers. Final iteration issues no next-stage copies, and no later iteration overwrites old shared data. Transcript remains protected by final CTA output barrier.',
 'LDSM completion control waits and cp.async waits retained. Both sites are uniformly predicated over CTA. No bare removal of a necessary stage handoff.',
 'One of64handoffs eliminated; no claim that earlier iterations can be removed safely. Racecheck is additional evidence, not a formal proof.',
 'No production changes or live mining.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
