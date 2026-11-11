from pathlib import Path
import json,re,numpy as np
p=Path(__file__).resolve().parents[1]/'artifacts/handoff-prepare'
e=json.loads((p/'events-publish.json').read_text());g=json.loads((p/'grids.json').read_text());rows=g['measurements'];names=list(g['resources']);out={}
lines=['Next-K address preparation: GPU0 offline results, no Nsight.',
'200 region event samples/variant,16launches/sample;60full grids/variant,6rotating batches.',
'Unfixed clocks; full grids include setup/noise/Merkle/transcript/PoW, not networking/proof delivery.',
'variant event_mean_ms grid_mean_ms median_ms p95_ms grids/sec equivalent_THs regs/local/static_smem/CTAs_per_SM']
for n in names:
 v=np.array([x for r in rows if r['name']==n for x in r['samples_ms']]);mean=float(v.mean());out[n]=dict(mean_ms=mean,median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),n=len(v),grids_sec=1000/mean,equivalent_ths=2**46/mean/1e9)
 lines.append(f'{n} {e["summaries"][n]["mean_ms"]:.6f} {mean:.3f} {np.median(v):.3f} {np.percentile(v,95):.3f} {1000/mean:.5f} {2**46/mean/1e9:.3f} {g["resources"][n]}')
for base in ('native8','late_pub'):
 for n in names:
  if n==base:continue
  means={r['repeat']:r['mean_ms'] for r in rows if r['name']==n};refs={r['repeat']:r['mean_ms'] for r in rows if r['name']==base}
  paired=[100*(means[k]/refs[k]-1) for k in sorted(means)]
  lines.append(f'{n} vs {base}: time {100*(out[n]["mean_ms"]/out[base]["mean_ms"]-1):+.3f}%; throughput {100*(out[base]["mean_ms"]/out[n]["mean_ms"]-1):+.3f}%; paired time deltas% '+', '.join(f'{x:+.3f}' for x in paired))
lines += ['','Interpretation: judge net benefit against native8, then early_tail_pub against late_pub.',
'SASS: both CopyPlan candidates have7IMMA after publication vs native8 16.',
'Early preparation is real but source recompilation changed LDSM/MMA scheduling.',
'No claim that isolated address arithmetic is the cause of the entire performance delta.',
'early_wait rejected:40Bstack,36Bspillstores+36Bspillloads despite168register allocation.',
'No production changes or deployment. Existing8+4 remains baseline unless a net winner is measured.',
'See design.txt, compiled.sass, *_pub.sass, *.diff, validation and sanitizer logs.']
(p/'grid-summary.json').write_text(json.dumps(out,indent=2)+'\n');(p/'report.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
