from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/fold-preamble';e=json.loads((p/'events.json').read_text());g=json.loads((p/'grids.json').read_text());rows=g['measurements'];names=list(g['resources']);summary={}
lines=['GPU0 current8+4 fold preamble simplification; no Nsight.',
'Kernel8192x8192 region:160event samples/variant,16launches/sample,8rotating blocks.',
'Full offline grids:80samples/variant,8rotating blocks,12grids/block discardfirst2.',
'Clocks unlocked. Full grids include setup/noise/Merkle/transcript/PoW; no network/proof delivery.',
'variant kernel_mean_ms grid_mean_ms grid_median_ms grid_p95_ms grids/sec equivalent_THs']
for n in names:
 v=np.array([x for r in rows if r['name']==n for x in r['samples_ms']]);mean=float(v.mean());x=dict(n=len(v),mean_ms=mean,median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),grids_sec=1000/mean,equivalent_ths=2**46/mean/1e9);summary[n]=x
 lines.append(f'{n} {e["summaries"][n]["mean_ms"]:.6f} {mean:.3f} {x["median_ms"]:.3f} {x["p95_ms"]:.3f} {x["grids_sec"]:.5f} {x["equivalent_ths"]:.3f}')
base={r['repeat']:r['mean_ms'] for r in rows if r['name']=='baseline'}
for n in (n for n in names if n != "baseline"):
 vals={r['repeat']:r['mean_ms'] for r in rows if r['name']==n};paired=[100*(vals[k]/base[k]-1) for k in sorted(vals)]
 lines.append(f'{n}: time {100*(summary[n]["mean_ms"]/summary["baseline"]["mean_ms"]-1):+.3f}%; throughput {100*(summary["baseline"]["mean_ms"]/summary[n]["mean_ms"]-1):+.3f}%; paired time% '+', '.join(f'{x:+.3f}' for x in paired))
lines += ['Same168registers,69632Bshared,384threads,1CTA/SM,0spill.',
'All main-loop MMA,LDSM,copy and earlyhandoff instruction words unchanged.',
'mod_nop removes3uniformALU operations but retains3NOP issue slots.',
'mod_compact/both_compact bypass vacated slots; extra branch executed on second-half folds.',
'Address bases and per-tile immediate offsets were already shared; no large repeated-address opportunity found.',
'No new source recompilation, no production library changes.']
(p/'grid-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if (p/'grids-run2.json').exists():
 combined={n:[] for n in ('baseline','both_compact')}
 lines += ['', 'Independent confirmation:run name mean_ms median_ms p95_ms equivalent_THs']
 for run in (1,2):
  rr=json.loads((p/f'grids-run{run}.json').read_text())['measurements'];means={}
  for n in combined:
   vv=[v for r in rr if r['name']==n for v in r['samples_ms']];combined[n].extend(vv);mean=float(np.mean(vv));means[n]=mean
   lines.append(f'{run} {n} {mean:.3f} {np.median(vv):.3f} {np.percentile(vv,95):.3f} {2**46/mean/1e9:.3f}')
  lines.append(f'run{run} throughput gain {100*(means["baseline"]/means["both_compact"]-1):.3f}%')
 lines.append('Combined160grids/variant:')
 for n,vv in combined.items():
  mean=float(np.mean(vv));lines.append(f'{n}: mean={mean:.3f}ms median={np.median(vv):.3f}ms p95={np.percentile(vv,95):.3f}ms equivTH/s={2**46/mean/1e9:.3f}')
 lines.append('Unfixed-clock offline result; do not claim pool-income gain or fixed-cycle attribution.')
 (p/'report.txt').write_text('\n'.join(lines)+'\n')
