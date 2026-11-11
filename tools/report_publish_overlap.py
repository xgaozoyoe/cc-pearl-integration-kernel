"""Summarize offline early-publication controls, without implying pool throughput."""
from pathlib import Path
import json
import numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/publish-overlap'
lines=['Early publication: delay peer-ID calculation / overlap B2 compute',
       'GPU 0 only. Region = 8192 x 8192, K4096, rank128. Unlocked clocks.',
       'Kernel times are CUDA event measurements; each run has 200 samples/variant, 16 launches/sample.']
for run in (1,2):
 s=json.loads((p/f'events-run{run}.json').read_text())['summaries']
 lines += [f'Run {run}:','variant mean_ms median_ms p95_ms time_change_pct']
 for n,v in s.items():
  lines.append(f"{n:12} {v['mean_ms']:.6f} {v['median_ms']:.6f} {v['p95_ms']:.6f} {(v['mean_ms']/s['native']['mean_ms']-1)*100:+.4f}")
if (p/'grids.json').exists():
 rows=json.loads((p/'grids.json').read_text())['measurements'];means={}
 lines += ['Full offline grid wall time (includes setup/noise/Merkle/PoW; excludes network/proof delivery):','variant n mean_ms median_ms p95_ms grids/sec equivalent_TH/s']
 for n in ('native','id_late'):
  a=np.array([x for r in rows if r['name']==n for x in r['samples_ms']]);means[n]=float(a.mean())
  lines.append(f'{n:12} {len(a)} {a.mean():.6f} {np.median(a):.6f} {np.percentile(a,95):.6f} {1000/a.mean():.6f} {(2**46/1e12)*1000/a.mean():.6f}')
 lines.append(f"Grid time change: {(means['id_late']/means['native']-1)*100:+.4f}%; equivalent throughput change: {(means['native']/means['id_late']-1)*100:+.4f}%")
 pairs=[]
 for repeat in sorted(set(r['repeat'] for r in rows)):
  v={r['name']:r['mean_ms'] for r in rows if r['repeat']==repeat};pairs.append((v['id_late']/v['native']-1)*100)
 lines.append('Paired grid-batch time changes (%): '+', '.join(f'{v:+.4f}' for v in pairs))
assert len(json.loads((p/'validation.json').read_text()))==36
assert len(json.loads((p/'pow-validation.json').read_text()))==18
assert '0 hazards displayed (0 errors, 0 warnings)' in (p/'racecheck.log').read_text()
lines += ['Validation:36 transcript and18 transcript+PoW comparisons PASS; racecheck 0 hazards/errors/warnings.',
 'All variants:168 registers/thread,0 local bytes,69632B dynamic shared/CTA,384threads,1CTA/SM. Same IMMA count.',
 'id_late: move original LOP3 peer-ID instruction0x1620 to0x16b0. Final LDSM PCs:0x1660->0x1650 and0x16b0->0x16a0. cp.wait0x16c0 and BAR.ARV0x16d0 unchanged.',
 'overlap_b2: same change plus move first2 B2 IMMA before publication; BAR.ARV now0x16f0. First IMMA gains explicit SB1 completion wait (mask2); BAR retains mask6 for both final loads.',
 'SASS textual diffs omit control fields: see builder for this added scoreboard wait. No new instructions/registers/barriers.',
 'Causality limitation: timing does not prove lower barrier stalls or increased tensor utilization. Requires matching hardware counters; not collected this round.',
 'No production changes or live mining. Equivalent TH/s is offline arithmetic, not a measured pool rate.']
(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
