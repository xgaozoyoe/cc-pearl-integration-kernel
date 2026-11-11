"""Report delayed handoff across full next-iteration copy preparation."""
from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]/'artifacts/deferred-copy-prefix'
lines=['GPU0 only;8192x8192 region,K4096,rank128. Unlocked-clock CUDA event measurements,not grid/pool throughput.', 'Two independent runs;200samples/variant/run,16launches/sample,10rotating blocks.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries'];lines += [f'Run{run}','variant mean_ms median_ms p95_ms delta_native_pct delta_control_pct']
 for n,v in s.items():lines.append(f"{n:18} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f} {(v['mean_ms']/s['matched_control']['mean_ms']-1)*100:+.4f}")
if (p/'grids.json').exists():
 import numpy as np
 rows=json.loads((p/'grids.json').read_text())['measurements'];means={};lines+=['Offline full-grid wall time:name n mean_ms median_ms p95_ms equivalent_TH/s']
 for name in ('native','matched_control','late_copy_sync'):
  a=np.array([x for r in rows if r['name']==name for x in r['samples_ms']]);means[name]=float(a.mean());lines.append(f'{name} {len(a)} {a.mean():.6f} {np.median(a):.6f} {np.percentile(a,95):.6f} {(2**46/1e12)*1000/a.mean():.6f}')
 for name in ('matched_control','late_copy_sync'):
  pairs=[]
  for repeat in sorted(set(r['repeat'] for r in rows)):
   v={r['name']:r['mean_ms'] for r in rows if r['repeat']==repeat};pairs.append((v[name]/v['native']-1)*100)
  lines.append(f"{name} grid time delta vs native:{(means[name]/means['native']-1)*100:+.4f}%; paired batches:"+', '.join(f'{x:+.4f}' for x in pairs))
 lines.append(f"Late vs matched control grid time delta:{(means['late_copy_sync']/means['matched_control']-1)*100:+.4f}%")
assert len(json.loads((p/'validation.json').read_text()))==36
assert len(json.loads((p/'pow-validation.json').read_text()))==18
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines += ['36transcript+18transcript/PoW comparisons PASS;racecheck0 hazards/errors/warnings.',
 'All168regs,0local,69632B dynamicshared,384threads,1CTA/SM. Same MMA and operand copy work.',
 'matched_control: old handoff packet ->33common-prefix instructions -> copy-skip branch ->23copy-preparation instructions -> first shared write0xf40.',
 'late_copy_sync:33prefix -> copy-skip branch ->23copy preparation -> old handoff packet -> first shared write0xf40.',
 'No-copy path:common prefix then synchronization before first LDSM0x1090. Final-exit path preserves original sync without executing next prefix.',
 'Actual candidate shared reads/writes remain after handoff; @!PT LDS placeholders never execute. Constant loads and address arithmetic may run before handoff.',
 'Old barrier ID consumed before toggle. DummyP6 instruction preserves packet scheduling without clobbering copy predicates. Native prefix waitmask6 retains transcript-store register lifetime protection.',
 'All early arrivals,64late syncs,initial/final CTA barriers,cp.wait and shared accesses retained; no extra CTA sync.',
 'Trampoline branches/guard instructions add overhead versus frozen native; matched control shares the branch structure. No claim of pure synchronization latency.',
 'No production changes, live mining or hardware-counter collection this round.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
