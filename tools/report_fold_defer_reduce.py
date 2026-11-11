from pathlib import Path
import json,numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/fold-defer-reduce';g=json.loads((p/'grids.json').read_text());rows=g['measurements'];names=list(g['resources']);summary={}
lines=['Cross-K local XOR snapshot + delayed reduction: GPU0 offline; no Nsight.',
'60full grids/variant,6rotating batches,12/block discardfirst2. Unfixed clocks.',
'Full grids include setup/noise/Merkle/GEMM/transcript/PoW; exclude network/proof delivery.',
'variant mean_ms median_ms p95_ms grids/sec equivalent_THs resource_regs/local/static_smem/CTAs']
for n in names:
 v=np.array([x for r in rows if r['name']==n for x in r['samples_ms']]);mean=float(v.mean());x=dict(n=len(v),mean_ms=mean,median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),grids_sec=1000/mean,equivalent_ths=2**46/mean/1e9);summary[n]=x
 lines.append(f'{n} {mean:.3f} {x["median_ms"]:.3f} {x["p95_ms"]:.3f} {x["grids_sec"]:.5f} {x["equivalent_ths"]:.3f} {g["resources"][n]}')
for base in ('native','source8'):
 for n in names:
  if n==base:continue
  bas={r['repeat']:r['mean_ms'] for r in rows if r['name']==base};vs={r['repeat']:r['mean_ms'] for r in rows if r['name']==n};paired=[100*(vs[k]/bas[k]-1) for k in sorted(vs)]
  lines.append(f'{n} vs {base}: time {100*(summary[n]["mean_ms"]/summary[base]["mean_ms"]-1):+.3f}%; paired changes% '+', '.join(f'{v:+.3f}' for v in paired))
lines += ['', 'SASS confirms next-K IMMA followed by saved prior-K REDUX; local XOR computed before crossing boundary.',
'Not merely a source-code reorder; see sass-audit.txt. No Nsight cycle-level overlap attribution.',
'Source variants lose native early-publication tuning; use source8 comparison as well as native.',
'Extra guards/address prep/control flow and compiler spills are present; no net-gain claim.',
'Keep current production and native candidate unchanged. See design.txt for exact schedules and limits.']
(p/'grid-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
