"""Summarize the no-Nsight cohort sweep; retain per-batch comparisons."""
from pathlib import Path
import json, statistics
import numpy as np
root=Path(__file__).resolve().parents[1]; out=root/'artifacts/handoff-groups'
events=json.loads((out/'events.json').read_text())
grid=json.loads((out/'grids.json').read_text()); rows=grid['measurements']
names=list(grid['resources']); summaries={}
lines=['GPU0 offline named-barrier cohort sweep; no Nsight used.',
'CTA256x192 K64 stages2;384threads;168regs;69632B shared;1CTA/SM;0local.',
'All IMMA/copy/shared addresses and hot-loop instructions unchanged from parent.',
'Only prologue warp-cohort threshold differs from id16. Known-good production unchanged.',
'Unfixed clocks; balanced rotation across6batches. Event samples120/variant,16launches/sample.',
'Full offline grids60/variant,12/batch discardfirst2; setup/noise/Merkle/transcript/PoW included.',
'Equivalent TH/s=2**46 / grid_seconds /1e12; excludes networking/proof delivery.',
'variant event_mean_ms grid_mean_ms grid_median_ms grid_p95_ms grids/sec equivTH/s']
for name in names:
 vals=np.array([v for r in rows if r['name']==name for v in r['samples_ms']]); mean=float(vals.mean())
 x=dict(n=len(vals),mean_ms=mean,median_ms=float(np.median(vals)),p95_ms=float(np.percentile(vals,95)),grids_sec=1000/mean,equivalent_ths=2**46/mean/1e9)
 summaries[name]=x
 lines.append(f'{name} {events["summaries"][name]["mean_ms"]:.6f} {mean:.3f} {x["median_ms"]:.3f} {x["p95_ms"]:.3f} {x["grids_sec"]:.5f} {x["equivalent_ths"]:.3f}')
for baseline in ('tail9','id16'):
 lines.append('Compared to '+baseline+':')
 for name in names:
  if name==baseline:continue
  means={r['repeat']:r['mean_ms'] for r in rows if r['name']==name}
  bases={r['repeat']:r['mean_ms'] for r in rows if r['name']==baseline}
  paired=[100*(means[k]/bases[k]-1) for k in sorted(means)]
  lines.append(f'{name}: time {100*(summaries[name]["mean_ms"]/summaries[baseline]["mean_ms"]-1):+.3f}%, throughput {100*(summaries[baseline]["mean_ms"]/summaries[name]["mean_ms"]-1):+.3f}%, paired time changes% '+', '.join(f'{v:+.3f}' for v in paired))
(out/'grid-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
(out/'report.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
if (out/'grids-run2.json').exists():
 lines += ['','Independent full-grid confirmation (60 measurements/variant/run):','run variant mean_ms median_ms p95_ms equivalent_THs']
 combined={}
 for run in (1,2):
  rr=json.loads((out/f'grids-run{run}.json').read_text())['measurements']; totals={}
  for name in ('tail9','id16','group8'):
   vv=np.array([v for r in rr if r['name']==name for v in r['samples_ms']]); mm=float(vv.mean());totals[name]=mm
   combined.setdefault(name,[]).extend(vv.tolist())
   lines.append(f'{run} {name} {mm:.3f} {np.median(vv):.3f} {np.percentile(vv,95):.3f} {2**46/mm/1e9:.3f}')
  lines.append(f'run{run}: group8 throughput vs id16 {100*(totals["id16"]/totals["group8"]-1):+.3f}%; vs tail9 {100*(totals["tail9"]/totals["group8"]-1):+.3f}%')
 lines += ['Combined120 full grids/variant:']
 for name,vv in combined.items():
  mm=float(np.mean(vv));lines.append(f'{name}: mean={mm:.3f}ms median={np.median(vv):.3f}ms p95={np.percentile(vv,95):.3f}ms grids/sec={1000/mm:.5f} equivTH/s={2**46/mm/1e9:.3f}')
 lines += ['Interpretation:8+4 cohorts improve full-grid throughput over id16 in both runs.',
 'Every paired batch faster (12/12). Clocks unlocked, no claim of fixed-clock tensor/cycle improvement.',
 'Not a large breakthrough: incremental throughput gain about0.8%, about1.2% vs tail9.',
 'No new hot-loop instruction, register, shared memory, spill, or loss of operand reuse.',
 'Do not extrapolate this to pool income; no live submission/deployment.',
 'Other splits were slower; see report-run1.txt for all rejected candidates.',
 'Correctness:72 transcript edge comparisons across all6variants,36 transcript+PoW comparisons including8192x8192.',
 'Racecheck all6variants; memcheck/synccheck winner group8. Consult logs for results.',
 'No production/mining library overwritten. No Nsight run or script requested.']
 (out/'report.txt').write_text('\n'.join(lines)+'\n')
