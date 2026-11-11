"""Summarize actual copy/wait placement controls, no causal stall estimates."""
from pathlib import Path
import json,numpy as np
root=Path(__file__).resolve().parents[1];out={}
for kind in ('events','results'):
 p=root/f'artifacts/rank128-handoff-control-{kind}.json'
 if not p.exists():continue
 d=json.loads(p.read_text());stats={}
 for v,name in enumerate(('baseline','latecopy','earlywait')):
  a=np.array([x for r in d['measurements'] if r['variant']==v for x in r['samples_ms']]);mean=float(a.mean())
  stats[name]=dict(samples=len(a),mean_ms=mean,median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)))
  if kind=='results':stats[name].update(grids_sec=1000/mean,equivalent_ths=2**46/mean/1e9)
 for name in ('latecopy','earlywait'):
  stats[name].update(time_change_pct=100*(stats[name]['mean_ms']/stats['baseline']['mean_ms']-1),throughput_change_pct=100*(stats['baseline']['mean_ms']/stats[name]['mean_ms']-1))
 out[kind]=dict(resources=d['resources'],summaries=stats)
(root/'artifacts/rank128-handoff-control-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
