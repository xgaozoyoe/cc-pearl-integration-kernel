"""Summarize matched offline split-B measurements; no live TH/s claim."""
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
out = {}
for kind, filename in [('kernel', 'events'), ('grid', 'results')]:
    source = json.loads((root/f'artifacts/rank128-bsplit-{filename}.json').read_text())
    rows = source['measurements']
    stats = {}
    for v in (0, 1):
        a = np.array([x for row in rows if row['variant'] == v for x in row['samples_ms']])
        stats[v] = dict(samples=len(a), mean_ms=float(a.mean()),
                        median_ms=float(np.median(a)), p95_ms=float(np.percentile(a, 95)))
        if kind == 'grid':
            stats[v].update(grids_sec=1000/float(a.mean()), equivalent_ths=2**46/float(a.mean())/1e9)
    out[kind] = dict(variants=stats, time_increase_pct=100*(stats[1]['mean_ms']/stats[0]['mean_ms']-1),
                     throughput_change_pct=100*(stats[0]['mean_ms']/stats[1]['mean_ms']-1),
                     resources=source['resources'])
    out[kind]['paired_time_increase_pct'] = [100*(next(r['mean_ms'] for r in rows if r['repeat']==i and r['variant']==1)/next(r['mean_ms'] for r in rows if r['repeat']==i and r['variant']==0)-1) for i in range(6)]
(root/'artifacts/rank128-bsplit-summary.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
